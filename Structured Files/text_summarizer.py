import os
from transformers import BartTokenizer, BartForConditionalGeneration
import docx
import PyPDF2
from summarizer import Summarizer, SingletonABCMeta


# Text summarizer for plain text, PDF, DOCX
class TextSummarizer(Summarizer, metaclass=SingletonABCMeta):
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        if not hasattr(self, 'initialized'):
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name).to(Summarizer.device)
            self.initialized = True

    # Function to read the file based on its extension
    def read_file(self,file_path):
        try:
            _, file_extension = os.path.splitext(file_path)

            if file_extension.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_extension.lower() == '.docx':
                doc = docx.Document(file_path)
                return ' '.join([paragraph.text for paragraph in doc.paragraphs])
            elif file_extension.lower() == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return ' '.join([page.extract_text() for page in pdf_reader.pages])

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return None
        except Exception as e:
            print(f"Error: There was an issue reading the file '{file_path}'. Error: {str(e)}")
            return None
    # Function to split long text into smaller chunks
    def split(self,text, chunk_size=1024):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    # Function to summarize a single chunk
    def summarize_chunk(self,text, max_length=200, min_length=50):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    def summarize(self,text_path, chunk_size=1024, max_summary_length=200, min_summary_length=50):
        text = self.read_file(text_path)
        if text is None:
            return None
        # Split the text into smaller chunks
        chunks = self.split(text, chunk_size=chunk_size)
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = self.summarize_chunk(chunk, max_length=max_summary_length, min_length=min_summary_length)
            chunk_summaries.append(chunk_summary)

        # Combine all chunk summaries into one document
        combined_summary_text = ' '.join(chunk_summaries)

        # Summarize the combined chunk summaries to get the final summary
        final_summary = self.summarize_chunk(combined_summary_text, max_length=max_summary_length, min_length=min_summary_length)

        return final_summary