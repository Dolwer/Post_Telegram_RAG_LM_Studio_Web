import re
import logging
from typing import Dict, List, Tuple, Optional

class ContentValidator:
    """
    Класс проверки и очистки текста перед публикацией в Telegram. Учитывает лимиты Telegram
    (code units), чистит мусор, размышления, таблицы, html, markdown, спецсимволы, поддерживает
    подготовку разметки для Telegram (HTML/Markdown).
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger("ContentValidator")
        self.max_length_text = config["telegram"].get("max_text_length", 4096)
        self.max_length_caption = config["telegram"].get("max_caption_length", 1024)
        self.min_length_text = config["telegram"].get("min_text_length", 100)
        self.min_paragraphs = config["telegram"].get("min_paragraphs", 1)
        self.format_mode = config["telegram"].get("parse_mode", "HTML")

    @staticmethod
    def _telegram_code_units(text: str) -> int:
        """Подсчёт code units, как на сервере Telegram (UTF-16)."""
        return len(text.encode('utf-16-le')) // 2

    @classmethod
    def _static_full_clean(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Статическая версия очистки (для prompt_builder и др). Возвращает очищенный текст и стату.
        """
        stats = dict(removed_think=0, removed_tables=0, removed_links=0, removed_html=0, removed_special=0, removed_dupes=0)
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

        # Remove HTML/Markdown tags (оставляем только разрешённые Telegram)
        allowed_tags = {'b', 'i', 'u', 's', 'code', 'pre', 'a', 'span'}
        n_html = len(re.findall(r"</?([a-zA-Z0-9]+)[^>]*>", text))
        text = re.sub(
            r"</?([a-zA-Z0-9]+)[^>]*>",
            lambda m: m.group(0) if m.group(1).lower() in allowed_tags else "",
            text
        )
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

    def full_clean(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Очищает текст (см. _static_full_clean)."""
        return self._static_full_clean(text)

    def validate_content(self, text: str, has_media: bool = False) -> str:
        """
        Полная валидация текста: чистка, удаление таблиц, ссылок, think, html, спецсимволов,
        проверка длины и структуры. Если невалидно — ValueError.
        """
        original = text
        # Весь pipeline: сначала очистить, потом проверить длину!
        text, clean_stats = self.full_clean(text)
        reasons = self._reasons_invalid(text, has_media)
        if reasons:
            self.logger.warning(f"Content rejected for: {', '.join(reasons)}")
            raise ValueError(f"Content not valid: {', '.join(reasons)}")

        self.logger.debug(f"Content passed validation: {self.get_content_stats(text)}; Clean stats: {clean_stats}")
        return text.strip()

    def validate_length(self, text: str, has_media: bool) -> bool:
        limit = self.max_length_caption if has_media else self.max_length_text
        return self._telegram_code_units(text) <= limit

    def _reasons_invalid(self, text: str, has_media: bool) -> List[str]:
        reasons = []
        limit = self.max_length_caption if has_media else self.max_length_text
        codeunits = self._telegram_code_units(text)
        if codeunits > limit:
            reasons.append(f"exceeds max code units ({codeunits} > {limit})")
        if len(text.strip()) < self.min_length_text:
            reasons.append(f"too short ({len(text.strip())} < {self.min_length_text})")
        if text.count("\n") < self.min_paragraphs:
            reasons.append("too few paragraphs")
        if re.search(r"nan", text, re.IGNORECASE):
            reasons.append("contains 'nan'")
        if re.search(r"data too large for file format", text, re.IGNORECASE):
            reasons.append("contains 'data too large for file format'")
        return reasons

    def get_content_stats(self, text: str) -> Dict[str, int]:
        return {
            "length": len(text),
            "codeunits": self._telegram_code_units(text),
            "lines": text.count("\n"),
            "words": len(text.split()),
            "paragraphs": len([l for l in text.splitlines() if l.strip()])
        }

    def detect_thinking_patterns(self, text: str) -> List[str]:
        return re.findall(r"<think>.*?</think>", text, flags=re.DOTALL)

    # --- Форматирование для Telegram (HTML/Markdown) ---
    @staticmethod
    def markdown_to_telegram_html(text: str) -> str:
        """Конвертирует markdown-разметку в HTML для Telegram."""
        text = re.sub(r'```(.*?)```', lambda m: f"<pre>{m.group(1)}</pre>", text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'___(.+?)___', r'<u><i>\1</i></u>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
        text = re.sub(r'^---$', r'\n', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def normalize_html_for_telegram(text: str) -> str:
        """Оставляет только поддерживаемые Telegram HTML-теги."""
        tag_map = [
            ('strong', 'b'), ('em', 'i'),
            ('ins', 'u'), ('strike', 's'), ('del', 's')
        ]
        for src, dst in tag_map:
            text = re.sub(rf'<{src}(\s*?)>', f'<{dst}>', text, flags=re.IGNORECASE)
            text = re.sub(rf'</{src}>', f'</{dst}>', text, flags=re.IGNORECASE)
        def filter_span(m):
            tag = m.group(0)
            if 'class="tg-spoiler"' in tag or "class='tg-spoiler'" in tag:
                return tag
            return ''
        text = re.sub(r'<span(?:\s+[^>]*)?>', filter_span, text, flags=re.IGNORECASE)
        text = re.sub(r'</span>', lambda m: m.group(0) if 'tg-spoiler' in text[max(0, m.start()-30):m.start()] else '', text, flags=re.IGNORECASE)
        def remove_unsupported_tags(match):
            tag = match.group(1)
            allowed = {'b', 'i', 'u', 's', 'code', 'pre', 'a', 'span'}
            if tag.lower() in allowed:
                return match.group(0)
            return ''
        text = re.sub(r'</?([a-zA-Z0-9]+)(\s+[^>]*)?>', remove_unsupported_tags, text)
        return text

    @staticmethod
    def html_escape_telegram(s: str) -> str:
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def format_for_telegram(self, text: str, max_len: Optional[int] = None) -> str:
        """
        Универсальный pre/post-процессор для Telegram. Markdown→HTML, очистка, экранирование, лимит длины.
        """
        # Markdown → HTML
        html_text = self.markdown_to_telegram_html(text)
        html_text2 = self.normalize_html_for_telegram(html_text)
        # selective escape вне тегов
        TAG_RE = re.compile(r'</?([a-zA-Z0-9]+)(\s+[^>]*)?>')
        parts = []
        last = 0
        for m in TAG_RE.finditer(html_text2):
            start, end = m.span()
            parts.append(self.html_escape_telegram(html_text2[last:start]))
            parts.append(html_text2[start:end])
            last = end
        parts.append(self.html_escape_telegram(html_text2[last:]))
        result = ''.join(parts)
        # Ограничиваем длину
        max_len = max_len or self.max_length_text
        if self._telegram_code_units(result) > max_len:
            # Усечь аккуратно
            chars = list(result)
            cu = 0
            idx = 0
            while idx < len(chars) and cu < max_len:
                ch = chars[idx]
                cu += len(ch.encode('utf-16-le')) // 2
                if cu > max_len:
                    break
                idx += 1
            result = ''.join(chars[:idx])
            result = result.rstrip() + "…"
        return result.strip()
