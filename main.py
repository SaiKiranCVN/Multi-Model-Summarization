import argparse
from file_processor import FileProcessor

def process_file(file_path):
    try:
        file_processor = FileProcessor(file_path)
        summary = file_processor.process()
        print(f"Summary for {file_path}:\n{summary}\n")
    except ValueError as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Summarize text, audio, or video files.")
    
    # Define the expected arguments
    parser.add_argument('--text', type=str, help="Path to a text file (PDF, TXT, DOCX)")
    parser.add_argument('--audio', type=str, help="Path to an audio file (MP3)")
    parser.add_argument('--video', type=str, help="Path to a video file (MP4)")
    
    # Parse the arguments
    args = parser.parse_args()

    # Process the corresponding file based on the provided argument
    if args.text:
        print(f"Processing text file: {args.text}")
        process_file(args.text)
    elif args.audio:
        print(f"Processing audio file: {args.audio}")
        process_file(args.audio)
    elif args.video:
        print(f"Processing video file: {args.video}")
        process_file(args.video)
    else:
        print("Please provide a valid argument (--text, --audio, or --video) and the corresponding file path.")

if __name__ == "__main__":
    main()
