import re
import logging
from typing import Dict, List, Tuple

class ContentValidator:
    def __init__(self, config: dict):
        self.logger = logging.getLogger("ContentValidator")
        self.max_length_text = config["telegram"].get("max_text_length", 4096)
        self.max_length_caption = config["telegram"].get("max_caption_length", 1024)
        self.min_length_text = config["telegram"].get("min_text_length", 100)  # можно добавить в конфиг
        self.min_paragraphs = config["telegram"].get("min_paragraphs", 1)

    def validate_content(self, text: str, has_media: bool = False) -> str:
        """
        Полная валидация текста: чистка, удаление таблиц, ссылок, think, html, спецсимволов, проверка длины и структуры.
        Если невалидно — ValueError с подробным описанием причин.
        """
        original = text
        text, clean_stats = self.full_clean(text)
        reasons = self._reasons_invalid(text, has_media)
        if reasons:
            self.logger.warning(f"Content rejected for: {', '.join(reasons)}")
            raise ValueError(f"Content not valid: {', '.join(reasons)}")

        self.logger.debug(f"Content passed validation: {self.get_content_stats(text)}; Clean stats: {clean_stats}")
        return text.strip()

    def full_clean(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Очистка текста: think, таблицы, ссылки, html, спецсимволы, дубли, пустые строки.
        Возвращает очищенный текст и статистику по чистке.
        """
        stats = dict(removed_think=0, removed_tables=0, removed_links=0, removed_html=0, removed_special=0, removed_dupes=0)
        orig_len = len(text)
        # Remove <think>...</think>
        n_think = len(re.findall(r"<think>.*?</think>", text, flags=re.DOTALL))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        stats["removed_think"] = n_think

        # Remove markdown tables (|...| and ---|---)
        n_tables_md = len(re.findall(r"^\s*\|.+\|", text, flags=re.MULTILINE))
        text = re.sub(r"(?m)^\s*\|.+\|\s*$", "", text)
        n_tables_md2 = len(re.findall(r"^:?-{3,}:?\|", text, flags=re.MULTILINE))
        text = re.sub(r"(?m)^:?-{3,}:?\|.*$", "", text)
        stats["removed_tables"] = n_tables_md + n_tables_md2

        # Remove HTML tables
        n_tables_html = len(re.findall(r"<table.*?>.*?</table>", text, flags=re.DOTALL | re.IGNORECASE))
        text = re.sub(r"<table.*?>.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)
        stats["removed_tables"] += n_tables_html

        # Remove markdown and HTML links
        n_links_md = len(re.findall(r"\[(.*?)\]\(.*?\)", text))
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        n_links_html = len(re.findall(r"<a .*?href=.*?>(.*?)</a>", text, flags=re.IGNORECASE))
        text = re.sub(r"<a .*?href=.*?>(.*?)</a>", r"\1", text, flags=re.IGNORECASE)
        stats["removed_links"] = n_links_md + n_links_html

        # Remove HTML/Markdown tags
        n_html = len(re.findall(r"</?(div|span|p|b|i|u|strong|em|code|pre|br|a|img)[^>]*>", text, flags=re.IGNORECASE))
        text = re.sub(r"</?(div|span|p|b|i|u|strong|em|code|pre|br|a|img)[^>]*>", "", text, flags=re.IGNORECASE)
        stats["removed_html"] = n_html

        # Remove special/unicode control characters, invisible chars, html entities
        n_special = len(re.findall(r"[\x00-\x08\x0B-\x1F\x7F\u200B&[a-z]+;]+", text))
        text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F\u200B]+", "", text)
        text = re.sub(r"&[a-z]+;", "", text)  # HTML entities
        stats["removed_special"] = n_special

        # Remove duplicate lines
        lines = text.splitlines()
        seen = set()
        deduped = []
        for line in lines:
            l = line.strip()
            if l and l not in seen:
                deduped.append(line)
                seen.add(l)
        stats["removed_dupes"] = len(lines) - len(deduped)
        text = "\n".join(deduped)

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip(), stats

    def validate_length(self, text: str, has_media: bool) -> bool:
        limit = self.max_length_caption if has_media else self.max_length_text
        return len(text) <= limit

    def _reasons_invalid(self, text: str, has_media: bool) -> List[str]:
        reasons = []
        limit = self.max_length_caption if has_media else self.max_length_text
        if len(text) > limit:
            reasons.append(f"exceeds max length ({len(text)} > {limit})")
        if len(text.strip()) < self.min_length_text:
            reasons.append(f"too short ({len(text.strip())} < {self.min_length_text})")
        if text.count("\n") < self.min_paragraphs:
            reasons.append("too few paragraphs")
        # Optional: add more rules here
        if re.search(r"nan", text, re.IGNORECASE):
            reasons.append("contains 'nan'")
        if re.search(r"data too large for file format", text, re.IGNORECASE):
            reasons.append("contains 'data too large for file format'")
        return reasons

    def get_content_stats(self, text: str) -> Dict[str, int]:
        return {
            "length": len(text),
            "lines": text.count("\n"),
            "words": len(text.split()),
            "paragraphs": len([l for l in text.splitlines() if l.strip()])
        }

    def detect_thinking_patterns(self, text: str) -> List[str]:
        return re.findall(r"<think>.*?</think>", text, flags=re.DOTALL)

    def format_for_telegram(self, text: str) -> str:
        return text.strip()
