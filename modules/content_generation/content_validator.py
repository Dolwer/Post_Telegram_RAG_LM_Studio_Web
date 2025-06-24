import logging
import re
from typing import Dict, Optional, Set
from functools import lru_cache

class ContentValidator:
    TELEGRAM_TEXT_LIMIT = 4096
    TELEGRAM_SAFE_LIMIT = 4000
    MIN_CONTENT_LENGTH = 15
    MAX_REPEATED_CHARS = 10
    
    FORBIDDEN_SYMBOLS = frozenset(["\u202e", "\u202d", "\u202c"])
    MARKDOWN_SPECIAL_CHARS = frozenset('_*[]()~`>#+-=|{}.!')
    
    _compiled_patterns = {}
    _validation_cache = {}
    
    @classmethod
    def _get_compiled_pattern(cls, pattern_name: str, pattern: str, flags: int = 0) -> re.Pattern:
        if pattern_name not in cls._compiled_patterns:
            cls._compiled_patterns[pattern_name] = re.compile(pattern, flags)
        return cls._compiled_patterns[pattern_name]
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self._init_patterns()
        
    def _init_patterns(self):
        self.markdown_table_pattern = self._get_compiled_pattern(
            'markdown_table',
            r'(?:\|[^\n|]+\|[^\n]*\n)+(?:\|[-:| ]+\|[^\n]*\n)+(?:\|[^\n|]+\|[^\n]*\n?)+',
            re.MULTILINE
        )
        
        self.html_table_pattern = self._get_compiled_pattern(
            'html_table',
            r'<table[\s\S]*?</table>',
            re.IGNORECASE
        )
        
        self.think_pattern = self._get_compiled_pattern(
            'think_block',
            r'<\s*think[^>]*>.*?<\s*/\s*think\s*>',
            re.IGNORECASE | re.DOTALL
        )
        
        self.basic_cleanup_patterns = {
            'null_values': self._get_compiled_pattern('null_values', r'\b(nan|None|null|NULL)\b', re.I),
            'unicode_formatting': self._get_compiled_pattern('unicode_formatting', r'[\u200b-\u200f\u202a-\u202e]+'),
            'hex_escape': self._get_compiled_pattern('hex_escape', r'\\x[0-9a-fA-F]{2}'),
            'unicode_escape': self._get_compiled_pattern('unicode_escape', r'_x[0-9A-Fa-f]{4}_'),
            'html_entities': self._get_compiled_pattern('html_entities', r'&[a-zA-Z0-9#]+;'),
            'multiple_spaces': self._get_compiled_pattern('multiple_spaces', r' {3,}')
        }
        
        self.final_cleanup_patterns = {
            'invalid_chars': self._get_compiled_pattern(
                'invalid_chars',
                r'[^\x09\x0A\x0D\x20-\x7Eа-яА-ЯёЁa-zA-Z0-9.,:;!?()\[\]{}<>@#%^&*_+=/\\|\'\"`~$№-]'
            ),
            'multiple_dots': self._get_compiled_pattern('multiple_dots', r'\.{3,}'),
            'multiple_commas': self._get_compiled_pattern('multiple_commas', r',,+'),
            'js_links': self._get_compiled_pattern(
                'js_links',
                r'\[([^\]]+)\]\((javascript|data):[^\)]+\)',
                re.I
            ),
            'repeated_chars': self._get_compiled_pattern('repeated_chars', r'(.)\1{10,}')
        }
        
        self.html_cleanup_patterns = {
            'disallowed_tags': self._get_compiled_pattern(
                'disallowed_tags',
                r'<(?!\/?(?:b|i|u|s|code|pre|a)\b)[^>]+>',
                re.IGNORECASE
            ),
            'markdown_headers': self._get_compiled_pattern('markdown_headers', r'^\s*#.*$', re.MULTILINE),
            'markdown_lists': self._get_compiled_pattern('markdown_lists', r'^\s*[-*]\s+', re.MULTILINE),
            'markdown_dividers': self._get_compiled_pattern('markdown_dividers', r'^[-=]{3,}$', re.MULTILINE)
        }

    @lru_cache(maxsize=1024)
    def _cached_basic_cleanup(self, text: str) -> str:
        for pattern in self.basic_cleanup_patterns.values():
            text = pattern.sub('', text)
        text = self.basic_cleanup_patterns['multiple_spaces'].sub('  ', text)
        return text

    def validate_content(self, text: str) -> str:
        if not isinstance(text, str):
            self.logger.error("Content validation input is not a string")
            return ""
            
        text = text.strip()
        if not text:
            self.logger.warning("Empty content provided for validation")
            return ""

        cache_key = hash(text[:500])
        if cache_key in self._validation_cache:
            cached_result = self._validation_cache[cache_key]
            if len(cached_result) <= self.TELEGRAM_TEXT_LIMIT:
                return cached_result

        original_length = len(text)
        
        text = self._perform_cleanup_pipeline(text)
        
        if not self._validate_content_quality(text):
            self.logger.warning("Content failed quality validation")
            return ""

        text = self._ensure_telegram_limits(text)
        
        final_text = text.strip()
        
        if len(self._validation_cache) < 1000:
            self._validation_cache[cache_key] = final_text
            
        self.logger.debug(f"Content validation: {original_length} -> {len(final_text)} chars")
        return final_text

    def _perform_cleanup_pipeline(self, text: str) -> str:
        text = self._cached_basic_cleanup(text)
        text = self._remove_forbidden_symbols(text)
        text = self.remove_thinking_blocks(text)
        text = self.remove_tables(text)
        text = self._clean_html_markdown(text)
        text = self._fix_markdown_escaping(text)
        text = self._final_cleanup(text)
        return text

    def _remove_forbidden_symbols(self, text: str) -> str:
        for symbol in self.FORBIDDEN_SYMBOLS:
            text = text.replace(symbol, '')
        return text

    def _fix_markdown_escaping(self, text: str) -> str:
        segments = []
        last_end = 0
        
        for match in re.finditer(r'(```.*?```|`[^`]*`)', text, re.DOTALL):
            if match.start() > last_end:
                segment = text[last_end:match.start()]
                segments.append(self._escape_markdown_segment(segment))
            segments.append(match.group(0))
            last_end = match.end()
            
        if last_end < len(text):
            segment = text[last_end:]
            segments.append(self._escape_markdown_segment(segment))
            
        return ''.join(segments)

    def _escape_markdown_segment(self, text: str) -> str:
        for char in self.MARKDOWN_SPECIAL_CHARS:
            text = text.replace(char, f'\\{char}')
        return text

    def _final_cleanup(self, text: str) -> str:
        text = self.final_cleanup_patterns['invalid_chars'].sub('', text)
        text = self.final_cleanup_patterns['multiple_dots'].sub('…', text)
        text = self.final_cleanup_patterns['multiple_commas'].sub(',', text)
        text = self.final_cleanup_patterns['js_links'].sub(r'\1', text)
        
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def remove_tables(self, text: str) -> str:
        text = self.markdown_table_pattern.sub('', text)
        text = self.html_table_pattern.sub('', text)
        return text

    def remove_thinking_blocks(self, text: str) -> str:
        return self.think_pattern.sub('', text)

    def _clean_html_markdown(self, text: str) -> str:
        for pattern in self.html_cleanup_patterns.values():
            text = pattern.sub('', text)
        return text

    def _validate_content_quality(self, text: str) -> bool:
        if not text or len(text) < self.MIN_CONTENT_LENGTH:
            return False
            
        if self.final_cleanup_patterns['repeated_chars'].search(text):
            return False
            
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 1 and len(set(lines)) <= 2:
            return False
            
        word_count = len(text.split())
        if word_count < 3:
            return False
            
        return True

    def _ensure_telegram_limits(self, text: str) -> str:
        if len(text) <= self.TELEGRAM_TEXT_LIMIT:
            return text
            
        self.logger.warning(f"Content exceeds Telegram limit ({len(text)} > {self.TELEGRAM_TEXT_LIMIT})")
        
        truncation_point = self.TELEGRAM_SAFE_LIMIT
        
        sentence_endings = ['.', '!', '?', '\n\n']
        best_cut = truncation_point
        
        for i in range(truncation_point - 100, truncation_point):
            if i < len(text) and text[i] in sentence_endings:
                best_cut = i + 1
                break
                
        truncated = text[:best_cut].rstrip()
        if not truncated.endswith(('...', '…')):
            truncated += '…'
            
        return truncated

    @classmethod
    def clear_cache(cls):
        cls._validation_cache.clear()
        
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            'cache_size': len(self._validation_cache),
            'pattern_cache_size': len(self._compiled_patterns)
        }
