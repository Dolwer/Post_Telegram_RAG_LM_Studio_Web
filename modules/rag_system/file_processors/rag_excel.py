import logging
import pandas as pd
from pathlib import Path

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "excel",
            "parser": "rag_excel",
            "lines": 0,
            "chars": 0,
            "percent_empty": 0.0,
        }
    }
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        orig_rows = len(df)
        df = df.dropna(how='all')
        def is_useless_row(row):
            return all((str(cell).strip() == "" or str(cell).strip() == "nan") for cell in row)
        cleaned_df = df[~df.apply(is_useless_row, axis=1)]
        percent_empty = 100.0 * (orig_rows - len(cleaned_df)) / (orig_rows or 1)
        # Clean HTML tags in each cell
        import re
        def clean_cell(cell):
            return re.sub(r'<[^>]+>', '', str(cell)).strip()
        cleaned_df = cleaned_df.applymap(clean_cell)
        text = cleaned_df.to_string(index=False)
        result["text"] = text.strip()
        result["success"] = True
        result["cleaned"] = (len(cleaned_df) != orig_rows)
        result["meta"]["lines"] = len(cleaned_df)
        result["meta"]["chars"] = len(result["text"])
        result["meta"]["percent_empty"] = percent_empty
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_excel").warning(f"EXCEL extraction failed for {file_path}: {e}")
    return result
