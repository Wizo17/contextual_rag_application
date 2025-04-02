import os
import fitz
import PyPDF2
import docx
import win32com.client as win32
from utils.logger import logger

def load_document_PyPDF2(file_path: str):
    """Load and extract text from a PDF file using PyPDF2.

    This function opens a PDF file using PyPDF2 and extracts the text content from all pages,
    concatenating them with newlines between pages.

    Args:
        file_path (str): Path to the PDF file to be processed.

    Returns:
        str: The extracted text content from the PDF file.
             Returns an empty string if the file is not found or an error occurs.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If any other error occurs during PDF processing.
    """

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
    """Load and extract text from a PDF file using PyMuPDF (fitz).

    This function opens a PDF file using PyMuPDF and extracts the text content from all pages,
    concatenating them with newlines between pages. PyMuPDF generally provides better text 
    extraction quality compared to PyPDF2, especially for complex PDF layouts.

    Args:
        file_path (str): Path to the PDF file to be processed.

    Returns:
        str: The extracted text content from the PDF file.
             Returns an empty string if the file is not found or an error occurs.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If any other error occurs during PDF processing.
    """

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
    """Load and extract text from a Word document (.doc or .docx).

    This function opens a Word document and extracts its text content. It supports both
    modern .docx format (using python-docx) and legacy .doc format (using win32com).
    For .docx files, it concatenates the text from all paragraphs with newlines.
    For .doc files, it extracts the full document content using Word automation.

    Args:
        file_path (str): Path to the Word document to be processed.

    Returns:
        str: The extracted text content from the document.
             Returns an empty string if the file is not found, format not supported,
             or an error occurs.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If any other error occurs during document processing.
    """

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
    """Load and extract text from a plain text file (.txt).

    This function opens a text file and reads its content. It handles UTF-8 encoded text files
    and returns the content as a single string with leading/trailing whitespace removed.

    Args:
        file_path (str): Path to the text file to be processed.

    Returns:
        str: The text content from the file.
             Returns an empty string if the file is not found or an error occurs.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If any other error occurs during file processing.
    """

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
    """List all files in a directory.

    This function retrieves a list of files from the specified directory. It can return either
    just the filenames or the complete absolute paths to the files.

    Args:
        dir_path (str): Path to the directory to list files from.
        complete (bool, optional): If True, returns absolute file paths. If False, returns only filenames.
            Defaults to False.

    Returns:
        list[str]: List of files in the directory.
                  Returns an empty list if the directory doesn't exist or an error occurs.

    Raises:
        Exception: If any error occurs while listing directory contents.
    """

    try:
        if not os.path.exists(dir_path):
            logger.error(f"Path '{dir_path}' doesn't exist.")
            return []
            
        if not os.path.isdir(dir_path):
            logger.error(f"'{dir_path}' isn't folder.")
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
    """Load and process documents from a directory.

    This function loads documents from the specified directory, supporting multiple file formats
    (PDF, DOC, DOCX, TXT). It processes each file and extracts its content.

    Args:
        dir_path (str): Path to the directory containing documents to process.
        limit (int, optional): Maximum number of documents to process. If -1, processes all documents.
            Defaults to -1.

    Returns:
        list[dict]: List of dictionaries containing processed documents.
                   Each dictionary has:
                   - 'file_path': Absolute path to the document
                   - 'content': Extracted text content from the document
                   Returns an empty list if an error occurs.
    """
    
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
        
        processed_docs.append({
            "file_path": path,
            "content": content
        })
    
    return processed_docs
