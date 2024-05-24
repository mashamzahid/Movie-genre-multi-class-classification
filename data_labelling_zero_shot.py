import pandas as pd
from transformers import pipeline
import torch

class DataLabeler:
    def __init__(self, input_csv_path, output_csv_path):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.genre_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

    def load_data(self):
        """Load data from CSV file."""
        self.data_frame = pd.read_csv(self.input_csv_path)
        print("Dataset loaded successfully.")

    def label_movie_genres(self):
        """Label data with movie genres using zero-shot classification."""
        candidate_genres = ["Drama", "Action", "Sci-fi", "Comedy", "Horror", "Adventure"]
        
        def get_top_genres(text):
            """Classify text and extract top two genres."""
            if text:
                result = self.genre_classifier(text, candidate_genres, multi_label=True)
                top_genres = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)[:2]
                return [genre for genre, _ in top_genres]
            else:
                return []

        # Apply genre classification to each transcription
        self.data_frame['top_genres'] = self.data_frame['transcription'].apply(get_top_genres)
        print("Genre classification completed.")

    def save_labeled_data(self):
        """Save the labeled data to a CSV file."""
        self.data_frame.to_csv(self.output_csv_path, index=False)
        print(f"Labeled data saved to {self.output_csv_path}")

# Example usage
input_csv_path = r"D:\AFINITY_TEST\preprocessed_metadata.csv"
output_csv_path = r"D:\AFINITY_TEST\labeled_data.csv"

# Create an instance of DataLabeler
labeler = DataLabeler(input_csv_path, output_csv_path)
labeler.load_data()
labeler.label_movie_genres()
labeler.save_labeled_data()
