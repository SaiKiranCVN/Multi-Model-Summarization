from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from summarizer import Summarizer, SingletonABCMeta
from audio_summarizer import AudioSummarizer
from text_summarizer import TextSummarizer


# Video summarizer (handles frame extraction and audio extraction)
class VideoSummarizer(Summarizer, metaclass=SingletonABCMeta):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(Summarizer.device)
            self.initialized = True
    # Extract Audio from Video using moviepy
    def extract_audio_from_video(self,video_path):
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')  # Temporary audio file, for now in the intrest in time, we are only considering mp4
        video.audio.write_audiofile(audio_path)
        return audio_path
    # Extract Frames from the Video
    def extract_frames(self,video_path, frame_rate=1):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(fps // frame_rate) == 0:
                frames.append(frame)

            frame_count += 1
        cap.release()
        return np.array(frames)

    # Convert Frames to PIL images
    def frames_to_pil(self,frames):
        pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        return pil_images

    # Summarize Video with ClipBERT (Visual)
    def summarize_visual(self,frames):
        pil_images = self.frames_to_pil(frames)

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        with torch.no_grad():
            visual_features = self.model.get_image_features(**inputs)

        # Placeholder for actual visual summary
        visual_summary = "The visual content of the video includes: " + " ".join(["scene" for _ in frames]) + "."

        return visual_summary
    # Transcribe and Summarize Audio (above code)
    # Combine Visual and Audio Summaries
    def summarize(self, video_path):
        # Extract visual frames
        frames = self.extract_frames(video_path, frame_rate=1)

        # Summarize visual content
        visual_summary = self.summarize_visual(frames)

        # Extract audio path and summarize audio content
        audio_path = self.extract_audio_from_video(video_path)
        # audio_summary = summarize_audio(audio_path)
        audio_transcription = AudioSummarizer().transcribe_audio(audio_path)


        # Combine audio transcription and visual description into one text
        combined_text = f"Audio transcription: {audio_transcription}\nVisual description: {visual_summary}"

        temp_file_path = "combined_text.txt"
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(combined_text)
        try:
            final_summary = TextSummarizer().summarize(temp_file_path)
        finally:
            # Delete the temporary text file after summarization
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

        return final_summary