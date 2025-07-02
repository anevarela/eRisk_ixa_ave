import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Start of script")

class DepressionModel(nn.Module):
    def __init__(self, text_model_name="bert-base-multilingual-cased", hidden_size=768):
        super(DepressionModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.similarity_fc = nn.Linear(21, 128)
        self.classifier = nn.Linear(hidden_size + 128, 21)

    def forward(self, input_ids, attention_mask, similarity):
        """
        :param input_ids: Tokenized sentence (BERT input).
        :param attention_mask: Mask for valid tokens.
        :param similarity: Precomputed similarity scores.
        :return: Predictions for all 21 symptoms.
        """
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state.mean(dim=1)  # Pooling
        similarity_features = self.similarity_fc(similarity)  # Transform similarity scores
        combined_features = torch.cat((text_features, similarity_features), dim=1) # Concatenate both outputs
        logits = self.classifier(combined_features)  # Final output for all 21 symptoms

        return logits


from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

import re

with open("../data/BDI_simplified.txt", "r", encoding="utf-8") as f:
    # print("reading txt file")
    raw_questions = f.read().strip().lower().split("\n\n")  # Splitting symptoms

# print("preparing symptom severity list")
def clean_text(text):
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\.\s*$", "", text)
    return text.strip()  
questions = [list(map(clean_text, symptom.split("\n"))) for symptom in raw_questions] # Splitting by severity

model_sim = SentenceTransformer("mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es")

flattened_questions = [q for sublist in questions for q in sublist]  # Flatten list
embeddings = model_sim.encode(flattened_questions, convert_to_numpy=True)

symptom_ranges = []
start = 0
for sublist in questions:
    symptom_ranges.append((start, start + len(sublist)))
    start += len(sublist)

severity_weights1 = [1.0, 1.0, 1.5, 2, 2.5]  
severity_weights2 = [1.0, 1.0, 1.5, 1.5, 2, 2, 2.5, 2.5]  

def similarity(sentence):
    """
    Compute weighted cosine similarity scores between a sentence and all 21 symptoms.
    
    :param sentence: Input sentence to compare.
    :return: Array of 21 similarity scores (one per symptom).
    """
    sentence_embedding = model_sim.encode([sentence], convert_to_numpy=True) 
    similarities = cosine_similarity(embeddings, sentence_embedding).flatten() 

    weighted_scores = []
    
    for (start, end) in symptom_ranges:
        symptom_similarities = similarities[start:end]  

        if end-start > 5:
            weighted_score = np.sum(symptom_similarities * severity_weights2) / np.sum(severity_weights2)  # Normalize
            weighted_scores.append(weighted_score)
        else:
            weighted_score = np.sum(symptom_similarities * severity_weights1) / np.sum(severity_weights2)  # Normalize
            weighted_scores.append(weighted_score)
    
    return np.array(weighted_scores)  # Shape: (21,)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize
model = DepressionModel()

from collections import OrderedDict

checkpoint = torch.load("trained_model", map_location="cpu")

new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    new_key = k.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def preprocess_test_data(trec_folder):
    """
    Reads and processes training data.
    
    :param trec_folder: Path to folder containing .trec sentence files.
    :return: List of (sentence_id, sentence_text)
    """

    user_labeled_sentences = {}  # Store labeled sentences per user (for classification)

    for filename in os.listdir(trec_folder):
        if filename.endswith(".trec"):
            file_path = os.path.join(trec_folder, filename)
            user_id = file_path[:-5]
            user_labeled_sentences[user_id] = []

            with open(file_path, "r", encoding="utf-8") as file:
                current_sentence_id, current_text = None, None

                for line in file:
                    line = line.strip()
                    if line.startswith("<DOCNO>"):
                        current_sentence_id = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
                    elif line.startswith("<TEXT>"):
                        current_text = line.replace("<TEXT>", "").replace("</TEXT>", "").strip()
                        if current_sentence_id and current_text:
                            user_labeled_sentences[user_id].append((current_sentence_id, current_text))
                            current_sentence_id, current_text = None, None
    
    return user_labeled_sentences

test_data = preprocess_test_data("../data/erisk25-t1-dataset/")
test_users = list(test_data.keys())
test_data_str = [sample for user in test_users for sample in test_data.get(user, [])]

# ---- similarity in batches ----

from tqdm import tqdm

BATCH_SIZE = 1024 
filtered_data = []

logging.info("Computing similarity in batches...")

theta = 0.5
for i in tqdm(range(0, len(test_data_str), BATCH_SIZE)):
    batch = test_data_str[i:i+BATCH_SIZE]
    doc_ids = [doc_id for doc_id, _ in batch]
    texts = [text for _, text in batch]
    
    embeddings_batch = model_sim.encode(texts, convert_to_numpy=True, batch_size=128, device=device)

    sims_batch = cosine_similarity(embeddings, embeddings_batch)
    sims_batch = sims_batch.T  

    for idx, sims in enumerate(sims_batch):
        weighted_scores = []
        for (start, end) in symptom_ranges:
            seg = sims[start:end]
            if end - start > 5:
                score = np.sum(seg * severity_weights2) / np.sum(severity_weights2)
            else:
                score = np.sum(seg * severity_weights1) / np.sum(severity_weights2)
            weighted_scores.append(score)

        if max(weighted_scores) > theta:
            filtered_data.append((doc_ids[idx], texts[idx], weighted_scores))

logging.info(f"Filtered to {len(filtered_data)} sentences (from {len(test_data_str)})")

# ---- Test dataset ----

class DepressionDatasetTest(Dataset):
    def __init__(self, data, max_len=512):
        self.max_len = max_len
        self.data = data 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc_id, sentence, sim_vector = self.data[idx]

        encoding = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        similarity_scores = torch.tensor(sim_vector, dtype=torch.float32)

        return {
            "doc_id": doc_id,
            "text": sentence,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "similarity": similarity_scores,
        }

# Now build the dataset and dataloader
test_dataset = DepressionDatasetTest(filtered_data)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
logging.info(len(test_dataloader))


import csv

def score_sentence(dataloader, wh_model):
    """
    Perform inference on a DataLoader and return predicted scores.
    :param dataloader: DataLoader with test data
    :param model: Trained model
    :return: List of predictions for each sample
    """
    all_predictions = []
    all_ids = []
    all_sentences = []
    zenbat = len(dataloader)
    
    with torch.no_grad():  # No need to track gradients
        i = 1
        for batch in dataloader:
            logging.info(f"Batch {i} of {zenbat}")
            ids = batch["doc_id"]
            text = batch["text"]
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            similarity_b = batch['similarity'].to(device)
            # mask = batch['mask'].to(device)
            
            logits = wh_model(input_ids, attention_mask, similarity_b)  # Model output (batch_size, 21)
            probs = torch.sigmoid(logits)  # (batch_size, 21); probabilities
            
            all_predictions.extend(probs.cpu().numpy()*10) # between 0 and 10
            all_ids.extend(ids)
            all_sentences.extend(text)

            rows = zip(all_ids, all_predictions)

            i += 1
        
    return all_predictions, all_ids, all_sentences

logging.info("scoring sentences")
preds, ids, sentences = score_sentence(test_dataloader, model)

logging.info("ranking results")
ranked_results = []

# Initialize dictionary to store predictions for each symptom (0 to 20)
symp = {i: {} for i in range(21)}  # Initialize empty dictionaries for each symptom

# Assuming preds is a list of lists: each element in preds corresponds to the predicted probabilities for each symptom
# And ids is a list of sentence IDs
for out, zein, sent in zip(preds, ids, sentences):
    for i in range(21):  # Iterate through each symptom (0 to 20)
        if zein not in symp[i]:  # If sentence not yet stored for this symptom
            symp[i][zein] = out[i]  # Store the predicted probability for the symptom
        else:
            if symp[i][zein] != out[i]:  # If the predicted probability differs
                logging.info("Woops")

# Now we need to sort sentences for each symptom based on the prediction scores
for i in range(21):
    # Sort sentence predictions by their probability in descending order
    sorted_sentences = sorted(symp[i].items(), key=lambda x: x[1], reverse=True)
    
    # Generate TREC formatted results for this symptom
    for rank, (sentence_id, score) in enumerate(sorted_sentences, start=1):
        ranked_results.append(f"{i+1} Q0 {sentence_id} {rank:04d} {score:.4f} inference_run_xx")
        if rank == 1000:
            break

# Save results to TREC submission file. CHANGE NAME!!!!
with open("submission_run_xx.trec", "w") as f:
    f.write("\n".join(ranked_results))