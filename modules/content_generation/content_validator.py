# modules/content_generation/content_validator.py

import re
import logging
from typing import Dict

class ContentValidator:
    def __init__(self, config: dict):
        self.logger = logging.getLogger("ContentValidator")
        self.max_length_text = config["telegram"].get("max_text_length", 4096)
        self.max_length_caption = config["telegram"].get("max_caption_length", 1024)

    def validate_content(self, text: str, has_media: bool = False) -> str:
        original = text
        text = self.remove_thinking_blocks(text)
        text = self.remove_tables(text)
        text = self.remove_links(text)
        text = self.clean_html_markdown(text)
        text = self.clean_special_characters(text)

        if not self.validate_length(text, has_media):
            self.logger.warning("Text too long for Telegram.")
            raise ValueError("Text exceeds Telegram limits.")

        if not self.validate_content_quality(text):
            self.logger.warning("Content quality is too low.")
            raise ValueError("Text does not meet quality criteria.")

        return text.strip()

    def validate_length(self, text: str, has_media: bool) -> bool:
        limit = self.max_length_caption if has_media else self.max_length_text
        return len(text) <= limit

    def remove_thinking_blocks(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def remove_tables(self, text: str) -> str:
        text = self.remove_markdown_tables(text)
        text = self.remove_html_tables(text)
        return text

    def remove_markdown_tables(self, text: str) -> str:
        # Удаляет и строки с '|' и разделители ---|---
        return re.sub(r"(?m)^\s*(\|.+\|)|(:?-{3,}:?\|)+.*$", "", text)

    def remove_html_tables(self, text: str) -> str:
        return re.sub(r"<table.*?>.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)

    def remove_links(self, text: str) -> str:
        # Markdown [text](url) → text
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        # HTML <a href="url">text</a> → text
        text = re.sub(r"<a .*?href=.*?>(.*?)</a>", r"\1", text, flags=re.IGNORECASE)
        return text

    def clean_html_markdown(self, text: str) -> str:
        return re.sub(r"</?(div|span|p|b|i|u|strong|em|code|pre|br|a|img)[^>]*>", "", text, flags=re.IGNORECASE)

    def clean_special_characters(self, text: str) -> str:
        return re.sub(r"[\x00-\x08\x0B-\x1F\x7F\u200B]+", "", text)

    def validate_content_quality(self, text: str) -> bool:
        if len(text.strip()) < 100:
            self.logger.warning("Text is too short.")
            return False
        if text.count("\n") < 1:
            self.logger.warning("Too few paragraphs.")
            return False
        return True

    def get_content_stats(self, text: str) -> Dict[str, int]:
        return {
            "length": len(text),
            "lines": text.count("\n"),
            "words": len(text.split())
        }

    def detect_thinking_patterns(self, text: str) -> list:
        return re.findall(r"<think>.*?</think>", text, flags=re.DOTALL)

    def format_for_telegram(self, text: str) -> str:
        return text.strip()
