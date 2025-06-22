# modules/rag_system/rag_file_utils.py

import logging
import re
import chardet
from pathlib import Path
import pandas as pd
import PyPDF2
import docx
from bs4 import BeautifulSoup

class FileProcessor:
    def __init__(self):
        self.logger = logging.getLogger("FileProcessor")

    def extract_text_from_file(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        try:
            if ext == ".txt":
                return self.process_txt(file_path)
            elif ext == ".csv":
                return self.process_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                return self.process_xlsx(file_path)
            elif ext == ".pdf":
                return self.process_pdf(file_path)
            elif ext == ".docx":
                return self.process_docx(file_path)
            elif ext == ".html":
                return self.process_html(file_path)
            else:
                self.logger.warning(f"Unsupported format: {file_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Extraction failed: {file_path}", exc_info=True)
            return ""

    def detect_encoding(self, file_path: str) -> str:
        raw = Path(file_path).read_bytes()
        result = chardet.detect(raw)
        return result['encoding'] or 'utf-8'

    def process_txt(self, file_path: str) -> str:
        text = Path(file_path).read_text(encoding=self.detect_encoding(file_path))
        return self.clean_text(text)

    def process_csv(self, file_path: str) -> str:
        df = pd.read_csv(file_path, encoding=self.detect_encoding(file_path))
        df = df.dropna(how='all')
        return self.clean_text(df.to_string())

    def process_xlsx(self, file_path: str) -> str:
        df = pd.read_excel(file_path, engine="openpyxl")
        df = df.dropna(how='all')
        return self.clean_text(df.to_string())

    def process_pdf(self, file_path: str) -> str:
        reader = PyPDF2.PdfReader(file_path)
        text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        return self.clean_text(text)

    def process_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return self.clean_text(text)

    def process_html(self, file_path: str) -> str:
        soup = BeautifulSoup(Path(file_path).read_text(self.detect_encoding(file_path)), "lxml")
        for t in soup(["script","style"]):
            t.decompose()
        return self.clean_text(soup.get_text(separator="\n"))

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_encoding(self, text: str) -> str:
        return text.encode('utf-8', errors='ignore').decode('utf-8')

    def remove_html_tags(self, text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)

    def filter_empty_cells(self, data: list) -> list:
        return [row for row in data if any(cell.strip() for cell in row)]

    def get_supported_formats(self) -> list:
        return [".txt", ".csv", ".xlsx", ".pdf", ".docx", ".html"]

    def validate_file(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.get_supported_formats() and Path(file_path).exists()
