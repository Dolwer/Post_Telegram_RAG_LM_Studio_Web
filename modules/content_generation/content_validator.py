import logging
import re

class ContentValidator:
    """
    Валидатор и корректор текста для Telegram-постинга.
    - Контролирует лимиты Telegram (4096/1024)
    - Удаляет таблицы (markdown/html), размышления (<think>...</think>), спецсимволы
    - Экранирует markdown V2, защищает от Telegram-банов
    - Проверяет смысловую и структурную пригодность результата
    """

    TELEGRAM_TEXT_LIMIT = 4096
    TELEGRAM_CAPTION_LIMIT = 1024
    FORBIDDEN_SYMBOLS = ["\u202e", "\u202d", "\u202c"]  # RLO/LRO, опасные для Telegram
    MARKDOWN_SPECIAL = r'_*[]()~`>#+-=|{}.!'
    # Для удаления таблиц
    MARKDOWN_TABLE_PATTERN = re.compile(
        r'(?:\|[^\n|]+\|[^\n]*\n)+(?:\|[-:| ]+\|[^\n]*\n)+(?:\|[^\n|]+\|[^\n]*\n?)+', re.MULTILINE)
    HTML_TABLE_PATTERN = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    # Для удаления <think>...</think> и вариаций
    THINK_PATTERN = re.compile(r'<\s*think[^>]*>.*?<\s*/\s*think\s*>', re.IGNORECASE | re.DOTALL)

    def __init__(self, config=None):
        self.logger = logging.getLogger("ContentValidator")
        self.config = config or {}

    def validate_content(self, text: str, has_media: bool = False) -> str:
        """
        Главная функция: полная очистка и валидация текста перед Telegram.
        Если не проходит лимит — вернуть пустую строку, выше обработать повторный запрос к LLM.
        """
        if not isinstance(text, str):
            self.logger.error("Content for validation is not a string!")
            return ""

        text = text.strip()
        if not text:
            self.logger.warning("Empty content on validation entrance.")
            return ""

        text = self._basic_cleanup(text)
        text = self.remove_thinking_blocks(text)
        text = self.remove_tables(text)
        text = self.clean_html_markdown(text)
        text = self._remove_forbidden_symbols(text)
        text = self._deduplicate_empty_lines(text)
        text = self._fix_markdown(text)

        # Лимит Telegram
        limit = self.TELEGRAM_CAPTION_LIMIT if has_media else self.TELEGRAM_TEXT_LIMIT
        if len(text) > limit:
            self.logger.warning(f"Content too long for Telegram ({len(text)} > {limit}), request shorter version in LLM.")
            return ""  # Нужно повторно запросить у LLM — не резать здесь!

        text = self._final_antiartifacts(text)
        text = text.strip()

        if not self.validate_content_quality(text):
            self.logger.warning("Content failed quality check: too short, nonsensical, or spammy.")
            return ""

        return text

    def _basic_cleanup(self, text: str) -> str:
        # Удаляем мусор: nan, None, NULL, невидимые символы, спецартефакты
        text = re.sub(r'\b(nan|None|null|NULL)\b', '', text, flags=re.I)
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e]+', '', text)
        text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
        text = re.sub(r'_x[0-9A-Fa-f]{4}_', '', text)
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        text = re.sub(r' {3,}', '  ', text)
        return text

    def _remove_forbidden_symbols(self, text: str) -> str:
        for sym in self.FORBIDDEN_SYMBOLS:
            text = text.replace(sym, '')
        # Можно расширить список запрещённых символов/emoji при необходимости
        return text

    def _deduplicate_empty_lines(self, text: str) -> str:
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip('\n')

    def _fix_markdown(self, text: str) -> str:
        """
        Экранирует спецсимволы markdown V2 вне кода/ссылок, не трогает code blocks.
        """
        def escape_md(match):
            part = match.group(0)
            for c in self.MARKDOWN_SPECIAL:
                part = part.replace(c, '\\' + c)
            return part

        # Не экранируем в inline code и code blocks
        segments = []
        last_end = 0
        for m in re.finditer(r'(```.*?```|`[^`]*`)', text, flags=re.DOTALL):
            # До кода
            if m.start() > last_end:
                seg = text[last_end:m.start()]
                segments.append(re.sub(r'([_*[\]()~`>#+\-=|{}.!])', r'\\\1', seg))
            segments.append(m.group(0))
            last_end = m.end()
        if last_end < len(text):
            seg = text[last_end:]
            segments.append(re.sub(r'([_*[\]()~`>#+\-=|{}.!])', r'\\\1', seg))
        fixed = ''.join(segments)
        fixed = re.sub(r'\\+$', '', fixed)  # Telegram не любит обратные слэши в конце
        return fixed

    def _final_antiartifacts(self, text: str) -> str:
        # Удаляем опасные невидимые символы (кроме \n и базовых)
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7Eа-яА-ЯёЁa-zA-Z0-9.,:;!?()\[\]{}<>@#%^&*_+=/\\|\'\"`~$№-]', '', text)
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r',,+', ',', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\[([^\]]+)\]\((javascript|data):[^\)]+\)', r'\1', text, flags=re.I)
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', text)
        return text.strip()

    def remove_tables(self, text: str) -> str:
        """
        Удаляет markdown и html таблицы.
        """
        text = self.MARKDOWN_TABLE_PATTERN.sub('', text)
        text = self.HTML_TABLE_PATTERN.sub('', text)
        return text

    def remove_thinking_blocks(self, text: str) -> str:
        """
        Удаляет все блоки размышлений <think>...</think> (и вариации).
        """
        text = self.THINK_PATTERN.sub('', text)
        # Иногда размышления могут быть выделены псевдотегами или markdown — добавь по необходимости
        # Например, <размышление>...</размышление>, [think]...[/think], и т.п.
        return text

    def clean_html_markdown(self, text: str) -> str:
        """
        Убирает html/markdown теги, кроме безопасных для Telegram.
        """
        # Удаляем все html-теги кроме <b>, <i>, <u>, <s>, <code>, <pre>, <a>
        allowed_tags = ['b','i','u','s','code','pre','a']
        text = re.sub(r'<(?!\/?(?:' + '|'.join(allowed_tags) + r')\b)[^>]+>', '', text, flags=re.IGNORECASE)
        # Удаляем markdown заголовки, списки, лишние символы разметки
        text = re.sub(r'^\s*#.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        # Удаляем лишние горизонтальные линии
        text = re.sub(r'^[-=]{3,}$', '', text, flags=re.MULTILINE)
        return text

    def validate_content_quality(self, text: str) -> bool:
        """
        Проверка на бессмысленный, пустой или "спамный" результат.
        """
        if not text or not isinstance(text, str):
            return False
        # Слишком коротко
        if len(text) < 15:
            return False
        # Только эмодзи или спецсимволы
        if re.fullmatch(r'[\s.,:;!?()\[\]{}<>@#%^&*_+=/\\|\'\"`~$№0-9a-zA-Zа-яА-ЯёЁ-]*', text):
            return False
        # Повтор одного и того же предложения/слова
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(set(lines)) <= 2 and len(lines) > 1:
            return False
        # Слишком много одинаковых символов подряд (спам)
        if re.search(r'(.)\1{10,}', text):
            return False
        return True

    # Для тестов и отладки
    def validate_plain(self, text: str, limit: int = 4096) -> str:
        text = self._basic_cleanup(text)
        text = self._deduplicate_empty_lines(text)
        text = self._remove_forbidden_symbols(text)
        text = text[:limit]
        return text.strip()
