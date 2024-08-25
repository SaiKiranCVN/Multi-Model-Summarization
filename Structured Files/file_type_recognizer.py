import os

class FileTypeRecognizer:
    @staticmethod
    def recognize(file_path):
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension in ['.txt', '.pdf', '.docx']:
            return 'text'
        elif file_extension in ['.mp3', '.wav']:
            return 'audio'
        elif file_extension in ['.mp4', '.avi']:
            return 'video'
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
