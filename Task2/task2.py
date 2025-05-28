# ==========================================================================
# Task 2: Multi-Label Emotion Recognition from Text
# ==========================================================================
# Objective: Classify multiple emotions in text using the GoEmotions dataset
# and fine-tune a BERT model for multi-label classification.
# ==========================================================================

# --------------------------------------------------------------------------
# 1. Install necessary libraries (if not already installed)
# --------------------------------------------------------------------------
# pip install datasets pandas scikit-learn torch transformers

# --------------------------------------------------------------------------
# 2. Import Libraries
# --------------------------------------------------------------------------
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------------------------------------------------------
# 3. Load GoEmotions Dataset
# --------------------------------------------------------------------------
print("[INFO] Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions", "raw")
train_data = dataset["train"]

texts = train_data["text"]     # List of Reddit comments
labels = train_data["labels"]  # List of emotion index lists

# --------------------------------------------------------------------------
# 4. MultiLabel Binarization (27 emotions + 1 neutral = 28 labels)
# --------------------------------------------------------------------------
mlb = MultiLabelBinarizer(classes=list(range(28)))
Y = mlb.fit_transform(labels)

# --------------------------------------------------------------------------
# 5. Tokenize Texts using BERT Tokenizer
# --------------------------------------------------------------------------
print("[INFO] Tokenizing texts...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
X = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

input_ids = X["input_ids"]
attention_mask = X["attention_mask"]
labels_tensor = torch.tensor(Y).float()

# --------------------------------------------------------------------------
# 6. Build BERT-based Multi-label Classification Model
# --------------------------------------------------------------------------
class BertForMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.pooler_output)
        return torch.sigmoid(self.classifier(cls_output))  # Multi-label = sigmoid

# Instantiate model
model = BertForMultiLabel(num_labels=28)

# --------------------------------------------------------------------------
# 7. Prepare DataLoader (use small subset for demo)
# --------------------------------------------------------------------------
print("[INFO] Preparing training data...")
demo_size = 1000
dataset = TensorDataset(
    input_ids[:demo_size],
    attention_mask[:demo_size],
    labels_tensor[:demo_size]
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------------------------------------------------------
# 8. Train the Model (1 epoch demo)
# --------------------------------------------------------------------------
print("[INFO] Starting training...")
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(1):
    for batch in loader:
        ids, masks, targets = batch
        outputs = model(ids, masks)
        loss = nn.BCELoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("[INFO] Training complete.")

# --------------------------------------------------------------------------
# 9. Evaluation (on first 100 samples)
# --------------------------------------------------------------------------
print("[INFO] Evaluating model...")
model.eval()
with torch.no_grad():
    preds = model(input_ids[:100], attention_mask[:100])
    preds_binary = (preds > 0.5).int().numpy()
    y_true = Y[:100]

# Evaluation metrics
print("Hamming Loss:", hamming_loss(y_true, preds_binary))
print("F1 Score (micro):", f1_score(y_true, preds_binary, average='micro'))

# --------------------------------------------------------------------------
# 10. Save model if needed
# --------------------------------------------------------------------------
# torch.save(model.state_dict(), "bert_emotion_model.pth")

print("✅ Task 2 complete.")
