from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
from summarizer import Summarizer, SingletonABCMeta
from text_summarizer import TextSummarizer


# Audio summarizer (assuming you have extracted the audio as text using Speech-to-Text)
class AudioSummarizer(Summarizer, metaclass=SingletonABCMeta):
    def __init__(self, model_name="openai/whisper-base"):
        torch.cuda.empty_cache() 
        if not hasattr(self, 'initialized'):
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(Summarizer.device)
            self.initialized = True
    # Function to split audio into chunks of a specific length (in seconds)
    def split(self,audio, chunk_length_sec, sample_rate):
        chunk_length_samples = chunk_length_sec * sample_rate
        num_chunks = int(audio.shape[1] // chunk_length_samples) + 1
        chunks = torch.split(audio, int(chunk_length_samples), dim=1)
        return chunks

    def summarize_chunk(self, audio_chunk, device):
        # Normalize audio input
        audio_chunk = (audio_chunk - audio_chunk.mean()) / torch.sqrt(audio_chunk.var() + 1e-7)

        # Process audio into input format for Whisper
        input_features = self.processor(audio_chunk.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # Decode predicted tokens into text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription


    def transcribe_audio(self, audio_path):
        # Load audio
        audio_input, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_input = resampler(audio_input)

        # Convert to mono if stereo
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)

        # Split the audio into chunks
        audio_chunks = self.split(audio_input, chunk_length_sec=30, sample_rate=16000)

        # Transcribe each chunk and accumulate results
        transcription = ""
        for chunk in audio_chunks:
            chunk_transcription = self.summarize_chunk(chunk, Summarizer.device)
            transcription += chunk_transcription + " "  # Add a space between chunks

        return transcription.strip()
    
    def summarize(self, audio_path):
        audio_text = self.transcribe_audio(audio_path)
        # Write the transcribed text to a temporary text file
        temp_file_path = "temp_audio_transcription.txt"
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(audio_text)
        try:
            summary = TextSummarizer().summarize(temp_file_path)
        finally:
            # Delete the temporary text file after summarization
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        # Return the summary
        return summary
