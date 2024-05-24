import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

class MovieDataset(Dataset):
    def __init__(self, transcriptions, labels, tokenizer, max_len):
        self.transcriptions = transcriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.transcriptions)
    
    def __getitem__(self, item):
        transcription = str(self.transcriptions[item])
        labels = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            transcription,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': transcription,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

class BERTMultiLabelClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

class Trainer:
    def __init__(self, model, device, train_data_loader, val_data_loader, optimizer, criterion):
        self.model = model
        self.device = device
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for d in self.train_data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / len(self.train_data_loader)

    def eval_model(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for d in self.val_data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.val_data_loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


        
if __name__ == '__main__':
    # Load your data
    data = pd.read_csv(r"D:\AFINITY_TEST\labeled_data.csv")  # Adjust the path accordingly
    transcriptions = data['transcription'].tolist()
    genres = data['top_genres'].apply(eval).tolist()  # Convert string of list to list

    # Initialize tokenizer and model parameters
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 512
    mlb = MultiLabelBinarizer(classes=["Drama", "Action", "Sci-fi", "Comedy", "Horror", "Adventure"])
    labels = mlb.fit_transform(genres)

    # Create dataset and dataloader
    train_texts, val_texts, train_labels, val_labels = train_test_split(transcriptions, labels, test_size=0.1)
    train_dataset = MovieDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = MovieDataset(val_texts, val_labels, tokenizer, max_len)
    train_data_loader = DataLoader(train_dataset, batch_size=4, num_workers=2)
    val_data_loader = DataLoader(val_dataset, batch_size=4, num_workers=2)

    # Set up model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTMultiLabelClassifier(n_classes=6).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, device, train_data_loader, val_data_loader, optimizer, criterion)
    for epoch in range(3):
        train_loss = trainer.train_epoch()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")
        val_loss = trainer.eval_model()
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    # Save model
    trainer.save_model('bert_multilabel_classifier.pth')
