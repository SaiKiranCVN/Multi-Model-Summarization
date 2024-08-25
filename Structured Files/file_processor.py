from file_type_recognizer import FileTypeRecognizer
from summarizer_factory import SummarizerFactory

class FileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_type = FileTypeRecognizer.recognize(file_path)
        self.summarizer = SummarizerFactory.get_summarizer(self.file_type)
    
    def process(self):
        return self.summarizer.summarize(self.file_path)
