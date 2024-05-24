import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class TextClusterer:
    def __init__(self, input_csv_path):
        self.input_csv_path = input_csv_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self):
        """Load data from CSV file."""
        self.data_frame = pd.read_csv(self.input_csv_path)
        print("Dataset loaded successfully.")

    def generate_embeddings(self):
        """Generate embeddings for each transcription."""
        self.data_frame['embeddings'] = self.data_frame['transcription'].map(lambda x: self.model.encode(x, show_progress_bar=True))

    def cluster_data(self, num_clusters=6):
        """Cluster data using K-means."""
        embeddings = list(self.data_frame['embeddings'])
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(embeddings)
        self.data_frame['cluster'] = self.cluster_labels
        print("Data clustered into", num_clusters, "groups.")



    def visualize_clusters(self):
        """Visualize the clusters using t-SNE with centroids calculated in t-SNE reduced space."""
        # Convert list of embeddings to a NumPy array
        embeddings_array = np.array(self.data_frame['embeddings'].tolist())

        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)
        tsne_results = tsne.fit_transform(embeddings_array)

        # Calculate new centroids in the t-SNE reduced space
        new_centroids = []
        for i in range(self.kmeans.n_clusters):
            cluster_points = tsne_results[self.cluster_labels == i]
            centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(centroid)

        # Map clusters to colors based on their new centroids
        distances = np.linalg.norm(tsne_results - np.array(new_centroids)[:, np.newaxis], axis=2)
        closest_centroids = np.argmin(distances, axis=0)

        # Create a scatter plot with sorted labels based on proximity to new centroids
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=closest_centroids, cmap='viridis', s=100, alpha=0.6, edgecolors='w', linewidths=2)
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)

        # Label points with their index or any specific identifier if necessary
        for i, txt in enumerate(self.data_frame.index):
            ax.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=9)

        plt.title('Enhanced t-SNE visualization of text clusters')
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
        plt.grid(True)
        plt.show()


    def save_labeled_data(self, output_csv_path):
        """Save the labeled data to a CSV file."""
        self.data_frame.drop(columns=['embeddings'], inplace=True)  # Remove embeddings for saving space
        self.data_frame.to_csv(output_csv_path, index=False)
        print(f"Labeled data saved to {output_csv_path}")

# Example usage
input_csv_path = r"D:\AFINITY_TEST\preprocessed_metadata.csv"
output_csv_path = r"D:\AFINITY_TEST\labeled_data_2.csv"

# Create an instance of TextClusterer
clusterer = TextClusterer(input_csv_path)
clusterer.load_data()
clusterer.generate_embeddings()
clusterer.cluster_data()
clusterer.visualize_clusters()
clusterer.save_labeled_data(output_csv_path)
