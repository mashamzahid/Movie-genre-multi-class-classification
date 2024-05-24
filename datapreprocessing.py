import pandas as pd
import spacy

class DataPreprocessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.nlp = spacy.load("en_core_web_sm")  # Load the English tokenizer, tagger, parser, NER, and word vectors
        self.stopwords = self.nlp.Defaults.stop_words  # Using spaCy's built-in stop words

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        print("Dataset loaded successfully.")
        print(f"Initial number of entries: {len(self.df)}")

    def normalize_text(self, text):
        # Normalize text by removing punctuation and converting to lowercase
        text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
        return text

    def remove_stop_words(self, text):
        # Remove stop words from the text
        return " ".join([word for word in text.split() if word not in self.stopwords])

    def lemmatize_text(self, text):
        # Lemmatize text using spaCy
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def preprocess_transcriptions(self):
        # Apply normalization, stop word removal, and lemmatization
        self.df['transcription'] = self.df['transcription'].fillna("")
        self.df['transcription'] = self.df['transcription'].apply(self.normalize_text)
        self.df['transcription'] = self.df['transcription'].apply(self.remove_stop_words)
        self.df['transcription'] = self.df['transcription'].apply(self.lemmatize_text)
        print("Preprocessing of transcriptions completed.")

    def clean_data(self):
        # Assuming this includes other cleaning logic
        self.preprocess_transcriptions()  # Preprocessing steps
        average_length = self.df['transcription'].apply(len).mean()
        min_length = average_length * 0.5
        self.df = self.df[self.df['transcription'].apply(len) >= min_length]
        print(f"Final number of entries after cleaning: {len(self.df)}")

    def save_cleaned_data(self):
        self.df.to_csv(self.output_path, index=False)
        print(f"Data saved to {self.output_path}")

# Example usage
input_path = r"D:\AFINITY_TEST\yt_trailers\updated_metadata.csv"
output_path = r"D:\AFINITY_TEST\preprocessed_metadata.csv"

# Create an instance of DataPreprocessor
preprocessor = DataPreprocessor(input_path, output_path)

# Process the data
preprocessor.load_data()
preprocessor.clean_data()
preprocessor.save_cleaned_data()
