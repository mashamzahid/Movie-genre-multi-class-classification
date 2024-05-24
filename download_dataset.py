import pandas as pd
import os
import yt_dlp

# Load only the first 100 rows from the dataset
file_path = r"D:\AFINITY_TEST\ml-youtube.csv"  # Replace with the actual file path
df = pd.read_csv(file_path, nrows=100)

# Create a folder to save the downloaded videos
download_folder = r"D:\AFINITY_TEST\yt_trailers"  # Replace with your desired download folder path
os.makedirs(download_folder, exist_ok=True)

# Function to download videos and clean DataFrame
def download_videos_and_clean(df, download_folder):
    successful_downloads = []
    failed_downloads = []

    for index, row in df.iterrows():
        youtube_id = row['youtubeId']
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(download_folder, f'{youtube_id}.mp4')
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
            successful_downloads.append(index)
        except Exception as e:
            print(f"Error downloading {youtube_id}: {e}")
            failed_downloads.append(index)

    # Remove failed downloads from DataFrame
    df_cleaned = df.drop(failed_downloads).reset_index(drop=True)
    return df_cleaned

# Extract YouTube IDs and download videos
df_cleaned = download_videos_and_clean(df, download_folder)

# Save the cleaned DataFrame to a new CSV file
cleaned_metadata_path = os.path.join(download_folder, 'cleaned_metadata.csv')
df_cleaned.to_csv(cleaned_metadata_path, index=False)

print("Data cleaning and downloading complete. Cleaned metadata saved.")
