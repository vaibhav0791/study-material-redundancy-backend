# main_debug.py

# Comprehensive Debugging for PDF Extraction

import logging
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF file.
    """
    try:
        logger.debug(f"Attempting to open PDF file: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = """
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            logger.debug("Successfully extracted text from the PDF.")
            return text
    except Exception as e:
        logger.error(f"Error occurred while extracting text from PDF: {e}")
        return None


if __name__ == '__main__':
    pdf_file_path = 'example.pdf'  # Example PDF path
    extracted_text = extract_text_from_pdf(pdf_file_path)
    if extracted_text:
        logger.info("Text extracted successfully.")
    else:
        logger.info("Failed to extract text.")