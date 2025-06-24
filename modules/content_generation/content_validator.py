import logging
import re
from typing import Dict, Optional
import emoji

class ContentValidator:
    TELEGRAM_TEXT_LIMIT = 4096
    TELEGRAM_SAFE_LIMIT = 4000
    MIN_CONTENT_LENGTH = 15
    MAX_EMOJI_FRACTION = 0.5
    MAX_EMOJI_RUN = 5

    ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre"}

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self._init_patterns()

    def _init_patterns(self):
        self.re_tag = re.compile(r'</?([a-zA-Z0-9]+)[^>]*>')
        self.re_table_md = re.compile(
            r'(?:\|[^\n|]+\|[^\n]*\n)+(?:\|[-:| ]+\|[^\n]*\n)+(?:\|[^\n|]+\|[^\n]*\n?)+', re.MULTILINE)
        self.re_table_html = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
        self.re_think = re.compile(r'<\s*think[^>]*>.*?<\s*/\s*think\s*>', re.IGNORECASE | re.DOTALL)
        self.re_null = re.compile(r'\b(nan|None|null|NULL)\b', re.I)
        self.re_unicode = re.compile(r'[\u200b-\u200f\u202a-\u202e]+')
        self.re_hex = re.compile(r'\\x[0-9a-fA-F]{2}')
        self.re_unicode_hex = re.compile(r'_x[0-9A-Fa-f]{4}_')
        self.re_html_entity = re.compile(r'&[a-zA-Z0-9#]+;')
        self.re_spaces = re.compile(r' {3,}')
        self.re_invalid = re.compile(r'[^\x09\x0A\x0D\x20-\x7Eа-яА-ЯёЁa-zA-Z0-9.,:;!?()\[\]{}<>@#%^&*_+=/\\|\'\"`~$№-]')
        self.re_dots = re.compile(r'\.{3,}')
        self.re_commas = re.compile(r',,+')
        self.re_js_links = re.compile(r'\[([^\]]+)\]\((javascript|data):[^\)]+\)', re.I)
        self.re_multi_spaces = re.compile(r' {2,}')
        self.re_multi_newline = re.compile(r'\n{3,}', re.MULTILINE)
        self.re_repeated_chars = re.compile(r'(.)\1{10,}')

    def validate_content(self, text: str) -> str:
        if not isinstance(text, str):
            self.logger.error("Content validation input is not a string")
            return ""
        text = text.strip()
        if not text:
            self.logger.warning("Empty content provided for validation")
            return ""

        text = self._remove_forbidden_html_tags(text)
        text = self._remove_tables_and_thinking(text)
        text = self._clean_junk(text)
        text = self._filter_emoji_spam(text)
        text = self._ensure_telegram_limits(text)

        if not self._content_quality_check(text):
            self.logger.warning("Content failed quality validation")
            return ""
        return text.strip()

    def _remove_forbidden_html_tags(self, text: str) -> str:
        def _strip_tag(m):
            tag = m.group(1).lower()
            return m.group(0) if tag in self.ALLOWED_TAGS else ''
        return self.re_tag.sub(_strip_tag, text)

    def _remove_tables_and_thinking(self, text: str) -> str:
        text = self.re_table_md.sub('', text)
        text = self.re_table_html.sub('', text)
        text = self.re_think.sub('', text)
        return text

    def _clean_junk(self, text: str) -> str:
        text = self.re_null.sub('', text)
        text = self.re_unicode.sub('', text)
        text = self.re_hex.sub('', text)
        text = self.re_unicode_hex.sub('', text)
        text = self.re_html_entity.sub('', text)
        text = self.re_spaces.sub('  ', text)
        text = self.re_invalid.sub('', text)
        text = self.re_dots.sub('…', text)
        text = self.re_commas.sub(',', text)
        text = self.re_js_links.sub(r'\1', text)
        text = self.re_multi_spaces.sub(' ', text)
        text = self.re_multi_newline.sub('\n\n', text)
        return text.strip()

    def _filter_emoji_spam(self, text: str) -> str:
        # Оставляет эмодзи, но блокирует спам (>50% всего текста — эмодзи, >5 подряд одинаковых)
        chars = list(text)
        emojis = [c for c in chars if self._is_emoji(c)]
        if not text:
            return ""
        emoji_fraction = len(emojis) / max(len(chars), 1)
        if emoji_fraction > self.MAX_EMOJI_FRACTION:
            self.logger.warning("Too many emojis in text, likely spam")
            return ""
        # Блокировать длинные серии одинаковых эмодзи
        if self._has_long_emoji_run(chars):
            self.logger.warning("Emoji spam detected (long run)")
            return ""
        return text

    def _is_emoji(self, char: str) -> bool:
        # emoji.is_emoji поддерживает одиночные и сложные эмодзи (emoji>=2.0.0)
        try:
            return emoji.is_emoji(char)
        except Exception:
            # fallback на стандартные emoji unicode диапазоны
            return bool(re.match(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', char))

    def _has_long_emoji_run(self, chars) -> bool:
        if not chars:
            return False
        last = None
        run = 0
        for c in chars:
            if self._is_emoji(c):
                if c == last:
                    run += 1
                    if run >= self.MAX_EMOJI_RUN:
                        return True
                else:
                    last = c
                    run = 1
            else:
                last = None
                run = 0
        return False

    def _ensure_telegram_limits(self, text: str) -> str:
        if len(text) <= self.TELEGRAM_TEXT_LIMIT:
            return text
        cut = self.TELEGRAM_SAFE_LIMIT
        for i in range(cut - 100, cut):
            if i < len(text) and text[i] in [".", "!", "?", "\n\n"]:
                cut = i + 1
                break
        truncated = text[:cut].rstrip()
        if not truncated.endswith(('...', '…')):
            truncated += '…'
        return truncated

    def _content_quality_check(self, text: str) -> bool:
        if not text or len(text) < self.MIN_CONTENT_LENGTH:
            return False
        word_count = len(re.findall(r'\w+', text))
        if word_count < 3:
            return False
        if self.re_repeated_chars.search(text):
            return False
        return True
