import re
import logging
from typing import Dict, List, Tuple, Optional

class ContentValidator:
    """
    Класс для глубокой очистки и валидации текста перед публикацией в Telegram.
    Поддерживает:
    - все лимиты Telegram (по code units UTF-16)
    - очистку мусора, html, markdown, ссылок, таблиц, спецсимволов
    - преобразование markdown в поддерживаемый HTML для Telegram
    - анализ и логирование качества и структуры текста
    - адаптацию под caption/media
    """

    # Какие теги Telegram реально поддерживает в HTML-режиме
    TELEGRAM_HTML_TAGS = {'b', 'i', 'u', 's', 'code', 'pre', 'a', 'span'}

    def __init__(self, config: dict):
        self.logger = logging.getLogger("ContentValidator")
        telegram_cfg = config.get("telegram", {})
        self.max_length_text = telegram_cfg.get("max_text_length", 4096)
        self.max_length_caption = telegram_cfg.get("max_caption_length", 1024)
        self.min_length_text = telegram_cfg.get("min_text_length", 100)
        self.min_paragraphs = telegram_cfg.get("min_paragraphs", 1)
        self.parse_mode = telegram_cfg.get("parse_mode", "HTML")

    @staticmethod
    def telegram_code_units(text: str) -> int:
        """Вычисляет длину текста в code units (UTF-16), как делает Telegram."""
        return len(text.encode("utf-16-le")) // 2

    def full_clean(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Глубокая очистка текста перед публикацией:
        - убирает <think>...</think>
        - убирает markdown/html-таблицы
        - убирает html и markdown-ссылки
        - убирает неразрешённые html/markdown-теги (оставляет только допустимые)
        - убирает спецсимволы, невидимые символы, управляющие unicode
        - убирает дубликаты строк
        - убирает избыточные пустые строки (>2 подряд)
        Возвращает: очищенный текст, статистику по чистке.
        """
        stats = dict(
            removed_think=0,
            removed_tables=0,
            removed_links=0,
            removed_html=0,
            removed_special=0,
            removed_dupes=0
        )

        # 1. Remove <think>...</think>
        n_think = len(re.findall(r"<think>.*?</think>", text, flags=re.DOTALL))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        stats["removed_think"] = n_think

        # 2. Remove markdown tables (|...| and ---|---)
        n_tables_md = len(re.findall(r"^\s*\|.+\|", text, flags=re.MULTILINE))
        text = re.sub(r"(?m)^\s*\|.+\|\s*$", "", text)
        n_tables_md2 = len(re.findall(r"^:?-{3,}:?\|", text, flags=re.MULTILINE))
        text = re.sub(r"(?m)^:?-{3,}:?\|.*$", "", text)
        stats["removed_tables"] = n_tables_md + n_tables_md2

        # 3. Remove HTML tables
        n_tables_html = len(re.findall(r"<table.*?>.*?</table>", text, flags=re.DOTALL | re.IGNORECASE))
        text = re.sub(r"<table.*?>.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)
        stats["removed_tables"] += n_tables_html

        # 4. Remove markdown and HTML links (оставить только текст)
        n_links_md = len(re.findall(r"\[(.*?)\]\(.*?\)", text))
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        n_links_html = len(re.findall(r"<a .*?href=.*?>(.*?)</a>", text, flags=re.IGNORECASE))
        text = re.sub(r"<a .*?href=.*?>(.*?)</a>", r"\1", text, flags=re.IGNORECASE)
        stats["removed_links"] = n_links_md + n_links_html

        # 5. Remove html/markdown tags (кроме разрешённых Telegram)
        def remove_unsupported_tags(match):
            tag = match.group(1)
            if tag.lower() in ContentValidator.TELEGRAM_HTML_TAGS:
                return match.group(0)
            return ""
        n_html = len(re.findall(r"</?([a-zA-Z0-9]+)[^>]*>", text))
        text = re.sub(r"</?([a-zA-Z0-9]+)[^>]*>", remove_unsupported_tags, text)
        stats["removed_html"] = n_html

        # 6. Remove special/unicode control characters, invisible chars, html entities
        n_special = len(re.findall(r"[\x00-\x08\x0B-\x1F\x7F\u200B&[a-z]+;]+", text))
        text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F\u200B]+", "", text)
        text = re.sub(r"&[a-z]+;", "", text)  # HTML entities
        stats["removed_special"] = n_special

        # 7. Remove duplicate lines (оставляет только уникальные непустые строки)
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

        # 8. Remove excessive blank lines (оставляет не больше двух подряд)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip(), stats

    def validate_content(self, text: str, has_media: bool = False) -> str:
        """
        Полная валидация текста:
        - очистка (full_clean)
        - проверка лимитов (длина, абзацы, спецслов)
        - логирование причин отказа
        Если невалидно — ValueError с описанием.
        """
        original = text
        text, clean_stats = self.full_clean(text)
        reasons = self._reasons_invalid(text, has_media)
        if reasons:
            self.logger.warning(f"Content rejected for: {', '.join(reasons)}")
            raise ValueError(f"Content not valid: {', '.join(reasons)}")
        self.logger.debug(f"Content passed validation: {self.get_content_stats(text)}; Clean stats: {clean_stats}")
        return text.strip()

    def validate_length(self, text: str, has_media: bool) -> bool:
        limit = self.max_length_caption if has_media else self.max_length_text
        return self.telegram_code_units(text) <= limit

    def _reasons_invalid(self, text: str, has_media: bool) -> List[str]:
        reasons = []
        limit = self.max_length_caption if has_media else self.max_length_text
        codeunits = self.telegram_code_units(text)
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
            "codeunits": self.telegram_code_units(text),
            "lines": text.count("\n"),
            "words": len(text.split()),
            "paragraphs": len([l for l in text.splitlines() if l.strip()])
        }

    def detect_thinking_patterns(self, text: str) -> List[str]:
        """
        Возвращает все <think>...</think> блоки (если есть).
        """
        return re.findall(r"<think>.*?</think>", text, flags=re.DOTALL)

    # ========== Markdown → Telegram HTML ==========

    @staticmethod
    def markdown_to_telegram_html(text: str) -> str:
        """
        Преобразует markdown-разметку в HTML-теги, поддерживаемые Telegram.
        - **bold** → <b>
        - *italic* → <i>
        - __underline__ → <u>
        - ~~strikethrough~~ → <s>
        - `code` → <code>
        - ```pre``` → <pre>
        - [link](url) → <a href="url">
        """
        # Многострочные блоки кода (```code```)
        text = re.sub(r'```([\s\S]*?)```', lambda m: f"<pre>{m.group(1)}</pre>", text)
        # Инлайн-код
        text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
        # Жирный + курсив (***) или (___)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'___(.+?)___', r'<u><i>\1</i></u>', text)
        # Жирный (**)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Подчёркнутый (__)
        text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
        # Курсив (*...*)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        # Курсив (_..._)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)
        # Зачёркнутый (~~...~~)
        text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
        # Ссылки [text](url)
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
        # Горизонтальная линия
        text = re.sub(r'^---$', r'\n', text, flags=re.MULTILINE)
        # Удалить лишние маркеры списков (оставить только текст)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def normalize_html_for_telegram(text: str) -> str:
        """
        Оставляет только разрешённые Telegram HTML-теги.
        Остальные убирает полностью. Поддерживает <span class="tg-spoiler">.
        """
        # Замена распространённых тегов-синонимов на поддерживаемые
        tag_map = [
            ('strong', 'b'), ('em', 'i'), ('ins', 'u'), ('strike', 's'), ('del', 's')
        ]
        for src, dst in tag_map:
            text = re.sub(rf'<{src}(\s*?)>', f'<{dst}>', text, flags=re.IGNORECASE)
            text = re.sub(rf'</{src}>', f'</{dst}>', text, flags=re.IGNORECASE)
        # Разрешить только поддерживаемые теги (и <span class="tg-spoiler">)
        def allowed_tag(match):
            tag = match.group(1).lower()
            attrs = match.group(2) or ""
            if tag in ContentValidator.TELEGRAM_HTML_TAGS:
                # Спойлеры <span class="tg-spoiler"> - оставить с атрибутом
                if tag == "span" and "tg-spoiler" in attrs:
                    return match.group(0)
                elif tag != "span":
                    return f"<{tag}>"
            return ""
        text = re.sub(r'<([a-zA-Z0-9]+)(\s+[^>]*)?>', allowed_tag, text)
        # Закрывающие теги
        def allowed_close_tag(match):
            tag = match.group(1).lower()
            if tag in ContentValidator.TELEGRAM_HTML_TAGS:
                return match.group(0)
            return ""
        text = re.sub(r'</([a-zA-Z0-9]+)>', allowed_close_tag, text)
        return text

    @staticmethod
    def html_escape_telegram(s: str) -> str:
        """
        Экранирует &, <, > вне тегов для безопасности (Telegram HTML).
        """
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def format_for_telegram(self, text: str, max_len: Optional[int] = None) -> str:
        """
        Преобразует текст для публикации в Telegram:
        - Markdown → HTML
        - очистка от неподдерживаемых тегов
        - экранирование спецсимволов вне тегов
        - ограничение длины по code units (аккуратно по символам)
        """
        html_text = self.markdown_to_telegram_html(text)
        html_text2 = self.normalize_html_for_telegram(html_text)
        # Экранируем только вне тегов!
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
        # Ограничиваем длину (по code units)
        max_len = max_len or self.max_length_text
        if self.telegram_code_units(result) > max_len:
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
