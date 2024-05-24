import os
import pandas as pd
import whisper
import moviepy.editor as mp
import spacy
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer

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

class VideoTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")
        print("Whisper model loaded successfully")

    def transcribe_video(self, video_path):
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio_path = video_path.replace('.mp4', '.wav')
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        result = self.model.transcribe(audio_path)
        os.remove(audio_path)  # Clean up audio file
        return result['text']

class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # Load the English tokenizer, tagger, parser, NER, and word vectors

    def normalize_text(self, text):
        text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
        return text

    def remove_stop_words(self, text):
        stopwords = self.nlp.Defaults.stop_words
        return " ".join([word for word in text.split() if word not in stopwords])

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def preprocess_text(self, text):
        text = self.normalize_text(text)
        text = self.remove_stop_words(text)
        text = self.lemmatize_text(text)
        return text

class Predictor:
    def __init__(self, model_path, tokenizer, max_len, device, classes):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.mlb = MultiLabelBinarizer(classes=classes)
        self.mlb.fit([classes])  # Fit the mlb to initialize classes_ attribute
        self.model = BERTMultiLabelClassifier(n_classes=len(self.mlb.classes)).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs).cpu()
            predictions = (probabilities > 0.4).numpy().astype(int)
        
        return self.mlb.inverse_transform(predictions)[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    classes = ["Drama", "Action", "Sci-fi", "Comedy", "Horror", "Adventure"]
    model_path = "D:/AFINITY_TEST/trained_bert/bert_multilabel_classifier.pth"
    max_len = 512

    video_transcriber = VideoTranscriber()
    data_preprocessor = DataPreprocessor()
    predictor = Predictor(model_path, tokenizer, max_len, device, classes)
    
    # Specify your video path here
    video_path = r"D:\AFINITY_TEST\yt_trailers\tVdn8JH91Dg.mp4"
    transcript = video_transcriber.transcribe_video(video_path)
    processed_transcript = data_preprocessor.preprocess_text(transcript)
    print("Processed Transcript:", processed_transcript)

    predicted_labels = predictor.predict(processed_transcript)
    print(f"Predicted labels: {predicted_labels}")

if __name__ == '__main__':
    main()
