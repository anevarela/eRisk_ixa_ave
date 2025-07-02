import os
import pandas as pd
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Start of script")

def preprocess_data(trec_folder1, trec_folder2, csv_file1, csv_file2):
    """
    Reads and processes training data.
    
    :param trec_folder: Path to folder containing .trec sentence files.
    :param csv_file: Path to CSV file with symptom labels.
    :return: List of (sentence_id, sentence_text, label_array)
    """

    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2, names = ["query","q0","docid","rel"], sep = "\t")
    # print(len(df1), len(df2)) number of data points available for training (separate instances for each symptom)
    label_dict = defaultdict(lambda: [-1] * 21)  # Default: all labels = -1 (missing)

    for _, row in df1.iterrows():
        symptom_idx = int(row["query"]) - 1  # Convert to base 0 index
        sentence_id = row["docid"]
        # print(sentence_id)
        label_dict[sentence_id][symptom_idx] = row["rel"]  # If the symptom is present for a sentence, we un-mask the label
    for _, row in df2.iterrows():
        symptom_idx = int(row["query"]) - 1  # Convert to base 0 index
        sentence_id = row["docid"]
        label_dict[sentence_id][symptom_idx] = row["rel"]  # If the symptom is present for a sentence, we un-mask the label
    # print(len(label_dict)) number of data points available for training (separate instances for each symptom)
    
    user_labeled_sentences = {}  # Store labeled sentences per user (for classification)

    for filename in os.listdir(trec_folder1):
        if filename.endswith(".trec"):
            file_path = os.path.join(trec_folder1, filename)
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
                            labels = label_dict[current_sentence_id]
                            if 1 in labels or 0 in labels:
                                user_labeled_sentences[user_id].append((current_sentence_id, current_text, labels))

                            current_sentence_id, current_text = None, None

    for filename in os.listdir(trec_folder2):
        if filename.endswith(".trec"):
            file_path = os.path.join(trec_folder2, filename)
            user_id = file_path[:-5]
            user_labeled_sentences[user_id] = []

            with open(file_path, "r", encoding="utf-8") as file:
                current_sentence_id, current_text = None, None

                for line in file:
                    line = line.strip()
                    if line.startswith("<DOCNO>"):
                        current_sentence_id = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
                        continue
                    elif line.startswith("<TEXT>"):
                        current_text = line.replace("<TEXT>", "").replace("</TEXT>", "").strip()
                        if current_sentence_id and current_text:
                            labels = label_dict[current_sentence_id]
                            if 1 in labels or 0 in labels:
                                user_labeled_sentences[user_id].append((current_sentence_id, current_text, labels))

                            current_sentence_id, current_text = None, None
    
    return user_labeled_sentences

data_maj = preprocess_data("../data/training_data/eRisk2023_T1/new_data/","../data/training_data/eRisk2024_T1/erisk 2024 - t1 - collection/","../data/training_data/eRisk2023_T1/g_qrels_majority_2.csv","../data/training_data/eRisk2024_T1/majority_erisk_2024.csv")
data_con = preprocess_data("../data/training_data/eRisk2023_T1/new_data/","../data/training_data/eRisk2024_T1/erisk 2024 - t1 - collection/","../data/training_data/eRisk2023_T1/g_rels_consenso.csv","../data/training_data/eRisk2024_T1/consensus_erisk_2024.csv")

import numpy as np
from sklearn.utils import shuffle

users = list(data_maj.keys())
users = shuffle(users, random_state = 42)

split_idx = int(len(users) * 0.8)  # 80% train, 20% val
train_users = users[:split_idx]
val_users = users[split_idx:]

train_data_maj = [sample for user in train_users for sample in data_maj.get(user, [])]
val_data_maj = [sample for user in val_users for sample in data_maj.get(user, [])]

train_data_con = [sample for user in train_users for sample in data_con.get(user, [])]
val_data_con = [sample for user in val_users for sample in data_con.get(user, [])]

import re

with open("../data/BDI_simplified.txt", "r", encoding="utf-8") as f:
    # print("reading txt file")
    raw_questions = f.read().strip().lower().split("\n\n")  # Splitting symptoms

# print("preparing symptom severity list")
def clean_text(text):
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\.\s*$", "", text)
    return text.strip()  # Remove leading numbers
questions = [list(map(clean_text, symptom.split("\n"))) for symptom in raw_questions] # Splitting by severity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    sentence_embedding = model_sim.encode([sentence], convert_to_numpy=True)  # Shape: (1, embedding_size)
    similarities = cosine_similarity(embeddings, sentence_embedding).flatten()  # Shape: (all_levels,)

    weighted_scores = []
    
    for (start, end) in symptom_ranges:
        symptom_similarities = similarities[start:end]  # Extract relevant similarities for this symptom

        if end-start > 5:
            weighted_score = np.sum(symptom_similarities * severity_weights2) / np.sum(severity_weights2)  # Normalize
            weighted_scores.append(weighted_score)
        else:
            weighted_score = np.sum(symptom_similarities * severity_weights1) / np.sum(severity_weights2)  # Normalize
            weighted_scores.append(weighted_score)
    
    return np.array(weighted_scores)  # Shape: (21,)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

class DepressionDataset(Dataset):
    def __init__(self, data_maj, data_con, max_len=512):
        """
        :param data: List of tuples (id, sentence, labels).
        :param model: Sentence embedding model for similarity calculation.
        :param max_len: Maximum token length for BERT.
        """
        self.data = {}  # List of tuples (id, sentence, labels)
        self.max_len = max_len

        for instance in data_maj:
            # print(instance)
            doc_id, sentence, labels = instance
            self.data[doc_id] = {
                "doc_id": doc_id,
                "text": sentence,
                "labels": torch.tensor(labels, dtype=torch.float32),  # 21 symptoms
                "is_consensus": torch.ones(len(labels), dtype=torch.float32)  # Default weight 1.0
            }
        # print(self.max_len)

        for (doc_id, sentence, labels) in data_con:
            if doc_id in self.data:
                # Increase the weight for consensus-labeled symptoms
                for i in range(len(labels)):
                    if labels[i] in [0, 1]:  # Valid consensus label
                        self.data[doc_id]["is_consensus"][i] = 2.0  # Consensus-weighted
            else:
                # If not in majority, add as a new entry with consensus weight
                print(sentence)
                self.data[doc_id] = {
                    "doc_id": doc_id,
                    "text": sentence,
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "is_consensus": torch.ones(len(labels), dtype=torch.float32) * 2.0  # Fully consensus-labeled
                }
                print(doc_id)

        self.data = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        
        sample = self.data[idx]
        encoding = tokenizer(sample["text"], padding="max_length", truncation=True, 
                             max_length=self.max_len, return_tensors="pt")
        similarity_scores = torch.tensor(similarity(sample["text"]), dtype=torch.float32)
        #print(idx)
        return {
            "doc_id": sample["doc_id"],  # Store doc_id for tracking
            "text": sample["text"],
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "similarity": similarity_scores,
            "labels": sample["labels"],
            "is_consensus": sample["is_consensus"]  # New consensus weighting
        }

dataset = DepressionDataset(train_data_maj, train_data_con)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn

class WeightedLoss(nn.Module):
    def __init__(self, weight_majority=1.0, weight_consensus=2.0, threshold=0.5):
        """
        :param weight_majority: Weight for majority-labeled samples.
        :param weight_consensus: Weight for consensus-labeled samples.
        :param threshold: Probability threshold for auto-labeling as 1.
        """
        super(WeightedLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.weight_majority = weight_majority
        self.weight_consensus = weight_consensus
        self.threshold = threshold

    def forward(self, logits, targets, similarity_scores, is_consensus):
        """
        :param logits: Predicted logits from the model (before activation).
        :param targets: True labels (-1 means ignore).
        :param similarity_scores: Precomputed similarity scores for thresholding.
        :param is_consensus: Boolean tensor (1 for consensus, 0 for majority).
        :return: Weighted loss, ignoring -1 labels.
        """
        probs = torch.sigmoid(logits) # Probabilities
        adjusted_probs = torch.where(probs > self.threshold, torch.tensor(1.0, device=probs.device), probs) #Threshold
        valid_mask = (targets != -1).float() # Unnecessary labels

        loss = self.loss_fn(adjusted_probs, targets)
        weights = torch.where(is_consensus, self.weight_consensus, self.weight_majority) # Applies weights
        # print(loss.shape, weights.shape, valid_mask.shape)
        loss = loss * weights.unsqueeze(1).expand(-1, 21) * valid_mask # Masks -1

        return loss.sum() / valid_mask.sum().clamp(min=1.0)

import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

class DepressionModel(nn.Module):
    def __init__(self, text_model_name="mlm_finetuned_model", hidden_size=768):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize
wh_model = DepressionModel().to(device)
optimizer = optim.AdamW(wh_model.parameters(), lr=2e-5)
beta = 0.5
criterion = WeightedLoss(weight_majority=1.0, weight_consensus=2.0, threshold=beta)

val_dataset = DepressionDataset(val_data_maj, val_data_con)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Training
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

def evaluate(wh_model, val_dataloader, loss_fn):
    """Evaluate model on validation set."""
    wh_model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            similarity = batch["similarity"].to(device)
            labels = batch["labels"].to(device)
            # mask = batch["mask"].to(device)
            is_consensus = batch["is_consensus"].to(device)

            outputs = wh_model(input_ids, attention_mask, similarity)
            
            loss = loss_fn(outputs, labels, similarity, is_consensus)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            valid_mask = labels != -1
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())

    avg_loss = val_loss / len(val_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, f1


def train(wh_model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=5):
    """Train model with validation tracking."""
    wh_model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            similarity = batch["similarity"].to(device)
            labels = batch["labels"].to(device)
            # mask = batch["mask"].to(device)
            is_consensus = batch["is_consensus"].to(device)

            outputs = wh_model(input_ids, attention_mask, similarity)

            loss = loss_fn(outputs, labels, similarity, is_consensus)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss, val_accuracy, val_f1 = evaluate(wh_model, val_dataloader, loss_fn)

        logging.info(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_dataloader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1 (macro): {val_f1:.4f}")
    
train(wh_model, dataloader, val_dataloader, optimizer, criterion)

torch.save(wh_model.state_dict(), "trained_model")