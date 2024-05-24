import os
import pandas as pd
from tqdm import tqdm
import whisper
import subprocess
import moviepy.editor as mp




class VideoTranscriber:
    def __init__(self, metadata_path, video_folder):
        self.metadata_path = metadata_path
        self.video_folder = video_folder
        self.df = pd.read_csv(metadata_path)
        print("csv loaded")
        self.model = whisper.load_model("base")  # Load the Whisper model, use 'tiny', 'base', or 'large' as needed
        print("model loaded successfully")

    def transcribe_video(self, video_path):
        """Using the Whisper ASR model."""
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio_path = video_path.replace('.mp4', '.wav')
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        result = self.model.transcribe(audio_path, fp16=False)
        os.remove(audio_path)  # Clean up audio file
        return result['text']

    def process_videos(self):
        """Process all videos and add transcriptions to DataFrame."""
        self.df['transcription'] = ''  # Add a new column for transcriptions

        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            youtube_id = row['youtubeId']
            video_path = os.path.join(self.video_folder, f"{youtube_id}.mp4")
            
            if os.path.exists(video_path):
                transcription = self.transcribe_video(video_path)
                self.df.at[_, 'transcription'] = transcription
            else:
                print(f"Video file not found: {video_path}")

    def save_results(self, output_path):
        """Save the DataFrame with transcriptions to a new CSV file."""
        self.df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")


video_folder = r"D:\AFINITY_TEST\yt_trailers" 
metadata_path = r"D:\AFINITY_TEST\yt_trailers\cleaned_metadata.csv"
output_path = r"D:\AFINITY_TEST\yt_trailers\updated_metadata.csv"

transcriber = VideoTranscriber(metadata_path, video_folder)
print("starting transcription")
transcriber.process_videos()
print("vidoe transcription successful")
transcriber.save_results(output_path)
print("New Dataset Saved")



