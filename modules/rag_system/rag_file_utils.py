import logging
from pathlib import Path
from glob import glob

from .file_processor_manager import FileProcessorManager
from .hook_manager import HookManager
from .hooks import ALL_HOOKS

# Импортируем все парсеры
from .file_processors.rag_txt import extract_text as txt_parser
from .file_processors.rag_csv import extract_text as csv_parser
from .file_processors.rag_excel import extract_text as excel_parser
from .file_processors.rag_pdf import extract_text as pdf_parser
from .file_processors.rag_docx import extract_text as docx_parser
from .file_processors.rag_html import extract_text as html_parser
from .file_processors.rag_markdown import extract_text as markdown_parser
from .file_processors.rag_json import extract_text as json_parser
from .file_processors.rag_pptx import extract_text as pptx_parser
from .file_processors.rag_audio import extract_text as audio_parser
from .file_processors.rag_video import extract_text as video_parser
from .file_processors.rag_fallback_textract import extract_text as textract_fallback

class RAGFileUtils:
    """
    Центральная обёртка для обработки файлов с поддержкой хуков, batch-режимов и статистики.
    """
    def __init__(self, logger=None, hooks_config=None):
        self.logger = logger or logging.getLogger("RAGFileUtils")
        self.hook_manager = HookManager(self.logger)
        self.manager = FileProcessorManager(self.logger, self.hook_manager)
        # Регистрируем парсеры по расширениям
        self.manager.register_parser([".txt"], txt_parser)
        self.manager.register_parser([".csv"], csv_parser)
        self.manager.register_parser([".xlsx", ".xls"], excel_parser)
        self.manager.register_parser([".pdf"], pdf_parser)
        self.manager.register_parser([".docx"], docx_parser)
        self.manager.register_parser([".html", ".htm"], html_parser)
        self.manager.register_parser([".md", ".markdown"], markdown_parser)
        self.manager.register_parser([".json"], json_parser)
        self.manager.register_parser([".pptx"], pptx_parser)
        self.manager.register_parser([".mp3", ".wav", ".flac"], audio_parser)
        self.manager.register_parser([".mp4", ".avi", ".mov"], video_parser)
        self.manager.register_fallback(textract_fallback)
        # Регистрируем хуки из конфига (если передан)
        if hooks_config:
            for hook_name, params in hooks_config.get("pre", {}).items():
                hook_cls = ALL_HOOKS[hook_name]
                self.hook_manager.register_pre_hook(hook_cls(**params))
            for hook_name, params in hooks_config.get("post", {}).items():
                hook_cls = ALL_HOOKS[hook_name]
                self.hook_manager.register_post_hook(hook_cls(**params))

    def extract_text(self, file_path: str, **kwargs) -> dict:
        """Извлекает текст и метаданные из любого поддерживаемого файла."""
        return self.manager.extract_text(file_path, **kwargs)

    def extract_text_batch(self, dir_path: str, pattern="**/*", recursive=True, skip_partial=True, **kwargs) -> list:
        """
        Batch-обработка всех файлов в директории (или по паттерну).
        Возвращает список dict-результатов и агрегированную статистику.
        """
        # Генерируем список файлов
        search_path = str(Path(dir_path) / pattern)
        files = glob(search_path, recursive=recursive)
        results = self.manager.extract_text_batch(files, skip_partial=skip_partial, **kwargs)
        return results

    def get_supported_extensions(self):
        return self.manager.get_supported_extensions()

    def get_stats_from_results(self, results: list) -> dict:
        """
        Агрегирует статистику по batch-результатам: success_rate, avg_length, lang_distribution и др.
        """
        total = len(results)
        success = sum(1 for r in results if r.get("success"))
        partial = sum(1 for r in results if r["meta"].get("partial_success"))
        errors = [r.get("error") for r in results if not r.get("success")]
        lengths = [r["meta"].get("chars", 0) for r in results if r.get("success")]
        langs = {}
        for r in results:
            lang = r["meta"].get("lang")
            if lang:
                langs[lang] = langs.get(lang, 0) + 1
        stats = {
            "total": total,
            "success": success,
            "partial": partial,
            "success_rate": round(success/total, 3) if total else 0,
            "avg_length": round(sum(lengths)/success, 1) if success else 0,
            "lang_distribution": langs,
            "errors": errors,
        }
        return stats

    def get_session_errors(self, results: list) -> list:
        """Возвращает список ошибок по batch-результатам."""
        return [
            {
                "file": r["meta"].get("file_path"),
                "reason": r.get("error"),
                "partial": r["meta"].get("partial_success", False)
            }
            for r in results if not r.get("success")
        ]
