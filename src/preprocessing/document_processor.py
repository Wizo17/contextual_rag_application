import os
import fitz
import PyPDF2
import docx
import win32com.client as win32
from utils.logger import logger

def load_document_PyPDF2(file_path: str):
    # TODO Write docstring

    try:
        output_text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page in pdf_reader.pages:
                output_text += page.extract_text() + "\n"

        logger.info(f"Reading pdf file: {file_path}")
        return output_text.strip()
    except FileNotFoundError as e:
        logger.error(f"File {file_path} not found: {e}")
        return ""
    except Exception as e:
        logger.error(f"An error occured during pdf load file: {e}")
        return ""

def load_document_PyMuPDF(file_path: str):
    # TODO Write docstring

    try:
        output_text = ""
        with fitz.open(file_path) as file:
            for page in file:
                output_text += page.get_text() + "\n"

        logger.info(f"Reading pdf file: {file_path}")
        return output_text.strip()
    except FileNotFoundError as e:
        logger.error(f"File {file_path} not found: {e}")
        return ""
    except Exception as e:
        logger.error(f"An error occured during pdf load file: {e}")
        return ""

def load_document_doc(file_path: str):
    # TODO Write docstring

    try:
        output_text = ""

        if file_path.lower().endswith('.docx'):
            doc = docx.Document(file_path)
            output_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file_path.lower().endswith('.doc'):
            word = win32.Dispatch("Word.Application")
            doc = word.Documents.Open(file_path)
            output_text = doc.Content.Text
            doc.Close()
            word.Quit()
        else:
            logger.error(f"File format not supported: {file_path}")
            return ""

        logger.info(f"Reading doc(x) file: {file_path}")
        return output_text.strip()
    except FileNotFoundError as e:
        logger.error(f"File {file_path} not found: {e}")
        return ""
    except Exception as e:
        logger.error(f"An error occured during doc load file: {e}")
        return ""

def load_document_text(file_path: str):
    # TODO Write docstring

    try:
        output_text = ""

        with open(file_path, 'r', encoding='utf-8') as file:
            output_text = file.read()

        logger.info(f"Reading text file: {file_path}")
        return output_text.strip()
    except FileNotFoundError as e:
        logger.error(f"File {file_path} not found: {e}")
        return ""
    except Exception as e:
        logger.error(f"An error occured during txt load file: {e}")
        return ""

def list_files_in_directory(dir_path: str, complete=False):
    # TODO Write docstring

    try:
        if not os.path.exists(dir_path):
            print(f"Path '{dir_path}' doesn't exist.")
            return []
            
        if not os.path.isdir(dir_path):
            print(f"'{dir_path}' isn't folder.")
            return []
        
        output_list = []
        all_items = os.listdir(dir_path)
        for item in all_items:
            if os.path.isfile(os.path.join(dir_path, item)):
                if complete:
                    file_path = os.path.abspath(os.path.join(dir_path, item))
                else:
                    file_path = item
                output_list.append(file_path)

        logger.info(f"Listing files in: {dir_path}")
        return output_list
    except Exception as e:
        logger.error(f"An error occured during list files in directory {dir_path}: {e}")
        return []

def load_documents(dir_path: str, limit: int = -1):
    # TODO Write docstring
    processed_docs = []

    count = 0
    for path in list_files_in_directory(dir_path, True):
        count += 1
        if limit != -1 and count > limit:
            break

        content = ""
        if path.lower().endswith('.pdf'):
            content = load_document_PyMuPDF(path)
        elif path.lower().endswith('.doc'):
            content = load_document_doc(path)
        elif path.lower().endswith('.docx'):
            content = load_document_doc(path)
        elif path.lower().endswith('.txt'):
            content = load_document_text(path)
        else:
            logger.error(f"File type not supported: {path}")
        
        processed_docs.append(content)
    
    return processed_docs
