import os
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_pdf_documents(directory):
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file}...")
            file_path = os.path.join(directory, pdf_file)
            
            # 1. Try Standard Extraction
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            # 2. Fallback to OCR if text is empty (Scanned PDF)
            if not text.strip():
                print(f"Standard extraction failed for {pdf_file}. Starting OCR...")
                # Convert PDF pages to images
                images = convert_from_path(file_path, poppler_path=r"C:\poppler\Library\bin")
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image)
                    text += page_text + "\n"
                    print(f"Finished OCR for page {i+1}")

            if not text.strip():
                print(f"Warning: No text could be extracted from {pdf_file} even with OCR.")
                continue

            doc = Document(
                page_content=text,
                metadata={"source": pdf_file}
            )
            documents.append(doc)
            print(f"Successfully processed {pdf_file}")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return documents

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF files here.")
        return

    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")

    try:
        print("Loading embedding model (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("Loading and processing PDF documents...")
        documents = load_pdf_documents("data")
        
        if not documents:
            print("No documents were successfully processed. Check your data folder.")
            return
            
        print(f"Successfully loaded {len(documents)} documents")

        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(text_chunks)} chunks")

        print("Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )
        
        print("--- SUCCESS: Brain built and stored in 'vector_db_dir' ---")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()