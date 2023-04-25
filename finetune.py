import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

model_name = "distilbert-base-uncased"

#Reading text
df = pd.read_csv('train.csv')
train_texts = df["comment_text"].values
train_labels = df[df.columns[2:]].values
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

#Dataset class to create the labels and encode them
class TextDataset(Dataset):
  def __init__(self,texts,labels):
    self.texts = texts
    self.labels = labels

  def __getitem__(self,idx):
    encodings = tokenizer(self.texts[idx], truncation=True, padding="max_length")
    item = {key: torch.tensor(val) for key, val in encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx],dtype=torch.float32)
    del encodings
    return item

  def __len__(self):
    return len(self.labels)

#This is the tokenizer for the current model
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

#Set up the dataset
train_dataset = TextDataset(train_texts,train_labels)
val_dataset = TextDataset(val_texts, val_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Use multilabel model because there are 6 variables to fintune for
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6, problem_type="multi_label_classification")
model.to(device)
model.train()

#Use these parameters
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

#Finetune process
for epoch in range(1):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

#Upload trained model to a file
model.save_pretrained("sentiment_custom_model")

#Upload tokenizer to a file
tokenizer.save_pretrained("sentiment_tokenizer")