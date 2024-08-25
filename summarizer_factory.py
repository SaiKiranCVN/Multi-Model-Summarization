from text_summarizer import TextSummarizer
from audio_summarizer import AudioSummarizer
from video_summarizer import VideoSummarizer

class SummarizerFactory:
    @staticmethod
    def get_summarizer(file_type):
        if file_type == 'text':
            return TextSummarizer()
        elif file_type == 'audio':
            return AudioSummarizer()
        elif file_type == 'video':
            return VideoSummarizer()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
