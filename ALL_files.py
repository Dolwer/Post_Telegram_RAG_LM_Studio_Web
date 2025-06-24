# данный код относится к файлу main.py

import os
import sys
import time
import signal
import logging
import traceback
from datetime import datetime

from modules.utils.logs import get_logger, log_system_info
from modules.utils.config_manager import ConfigManager
from modules.utils.state_manager import StateManager
from modules.utils.media_handler import MediaHandler
from modules.rag_system.rag_retriever import RAGRetriever
from modules.external_apis.web_search import WebSearchClient
from modules.external_apis.telegram_client import TelegramClient
from modules.content_generation.prompt_builder import PromptBuilder
from modules.content_generation.content_validator import ContentValidator
from modules.content_generation.lm_client import LMStudioClient

def slugify(text: str, maxlen=64) -> str:
    """Преобразует строку в slug (например, для имени файла/лога)"""
    import re
    slug = re.sub(r'[^\w\s-]', '', text).strip().lower()
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug[:maxlen]

def clean_context(text: str) -> str:
    """Удаляет мусорные строки из контекста, очищает от 'nan', 'Data too large for file format' и пустых строк."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if line.strip() and
           "nan" not in line.lower() and
           "data too large for file format" not in line.lower()
    ]
    return "\n".join(cleaned)

class TelegramRAGSystem:
    def __init__(self, config_path: str = "config/config.json"):
        self.logger = get_logger("Main")
        self.shutdown_requested = False
        self.stats = {
            "total": 0,
            "success": 0,
            "fail": 0,
            "skipped": 0,
            "topics": []
        }
        try:
            log_system_info(self.logger)
            self.logger.info("🚀 Initializing TelegramRAGSystem...")
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
            self._validate_config()
            self._init_services()
            self._load_topics()
            # Health check LM Studio
            if not self.lm_client.check_connection():
                self.logger.critical("LM Studio not available or model not loaded, aborting.")
                sys.exit(1)
            self.logger.info("Configuration validated successfully.")
        except Exception as e:
            self.logger.critical("Failed during initialization", exc_info=True)
            sys.exit(1)

        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

    def _validate_config(self):
        if not self.config_manager.validate_config():
            self.logger.critical("Invalid config.json structure or missing values")
            sys.exit(1)

    def _init_services(self):
        paths = self.config["paths"]
        self.state_manager = StateManager(paths["processed_topics_file"])
        self.rag_retriever = RAGRetriever(self.config["rag"])
        self.media_handler = MediaHandler(paths["media_dir"], self.config)
        self.lm_client = LMStudioClient(
            base_url=self.config["lm_studio"]["base_url"],
            model=self.config["lm_studio"]["model"],
            config=self.config["lm_studio"]
        )
        self.prompt_builder = PromptBuilder(paths["prompt_folders"])
        self.content_validator = ContentValidator(self.config)
        self.web_search = WebSearchClient(
            api_key=self.config_manager.get_serper_api_key(),
            endpoint=self.config_manager.get_config_value("serper.endpoint"),
            results_limit=self.config_manager.get_config_value("serper.results_limit")
        )
        self.telegram_client = TelegramClient(
            token=self.config_manager.get_telegram_token(),
            channel_id=self.config_manager.get_telegram_channel_id(),
            config=self.config["telegram"]
        )

    def _load_topics(self):
        topics_file = os.path.join(self.config["paths"]["data_dir"], "topics.txt")
        if os.path.isfile(topics_file):
            with open(topics_file, "r", encoding="utf-8") as f:
                topics = [line.strip() for line in f if line.strip()]
            self.state_manager.add_topics(topics)
            self.logger.info(f"{len(topics)} topics loaded from file.")
        else:
            self.logger.warning("No topics file found.")

    def _graceful_shutdown(self, *_):
        self.logger.warning("Received shutdown signal.")
        self.shutdown_requested = True

    def _combine_context(self, topic: str, topic_logger=None) -> str:
        """Собирает и очищает объединённый контекст для промпта."""
        rag, web = "", ""
        try:
            rag = self.rag_retriever.retrieve_context(topic)
            if topic_logger:
                topic_logger.info(f"RAG context:\n{rag[:2000]}")
        except Exception as e:
            if topic_logger:
                topic_logger.error(f"Failed to retrieve RAG context: {e}\n{traceback.format_exc()}")
            rag = ""
        try:
            web_results = self.web_search.search(topic)
            web = self.web_search.extract_content(web_results)
            if topic_logger:
                topic_logger.info(f"Web search context:\n{web[:2000]}")
        except Exception as e:
            if topic_logger:
                topic_logger.error(f"Failed to retrieve web context: {e}\n{traceback.format_exc()}")
            web = ""
        full_context = f"{rag}\n\n[WEB]\n{web}"
        full_context = clean_context(full_context)
        if topic_logger:
            topic_logger.debug(f"Combined context (truncated): {full_context[:1000]}")
        return full_context

    def _shorten_if_needed(self, text: str, prompt: str, has_media: bool, topic_logger=None) -> str:
        """Если контент слишком длинный — автоматически просит нейросеть сократить его."""
        try:
            self.content_validator.validate_content(text, has_media=has_media)
            return text
        except ValueError as e:
            if topic_logger:
                topic_logger.warning(f"Content too long, requesting shortened version... {e}")
            return self.lm_client.request_shorter_version(prompt, current_length=len(text), target_length=1024 if has_media else 4096)

    def _process_topic(self, topic: str) -> bool:
        topic_slug = slugify(topic)
        topic_logger = logging.getLogger(f"TopicLogger.{topic_slug}")
        topic_logger.setLevel(logging.DEBUG)
        topic_logfile = os.path.join("logs/topics", f"{topic_slug}.log")
        os.makedirs(os.path.dirname(topic_logfile), exist_ok=True)
        if not topic_logger.handlers:
            fh = logging.FileHandler(topic_logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            fh.setLevel(logging.DEBUG)
            topic_logger.addHandler(fh)
        topic_logger.info(f"=== Processing topic: {topic} ===")

        self.stats["total"] += 1
        topic_result = {"topic": topic, "slug": topic_slug, "status": "unknown", "error": "", "timestamp": datetime.now().isoformat()}

        try:
            context = self._combine_context(topic, topic_logger)
            if not context or not context.strip():
                msg = "Empty context after RAG/web. Skipping topic."
                topic_logger.error(msg)
                topic_result["status"] = "skipped"
                topic_result["error"] = msg
                self.stats["skipped"] += 1
                self.stats["topics"].append(topic_result)
                return False

            media_file = self.media_handler.get_random_media_file()
            topic_logger.info(f"Media file selected: {media_file}")

            try:
                prompt = self.prompt_builder.build_prompt(topic, context, media_file)
                topic_logger.debug(f"Prompt built (truncated): {prompt[:1500]}")
            except Exception as e:
                msg = f"Invalid prompt: {e}"
                topic_logger.error(msg)
                topic_result["status"] = "skipped"
                topic_result["error"] = msg
                self.stats["skipped"] += 1
                self.stats["topics"].append(topic_result)
                return False

            # Определяем лимиты Telegram для caption/text
            has_uploadfile = "{UPLOADFILE}" in prompt
            has_media = bool(media_file) and has_uploadfile

            min_context_length = 512
            max_attempts = 5
            attempt = 0
            orig_context = context
            while attempt < max_attempts:
                try:
                    topic_logger.info(f"Generation attempt {attempt+1}: prompt len={len(prompt)}, context len={len(context)}")
                    response = self.lm_client.generate_with_retry(prompt)
                    if not response or not response.strip():
                        raise ValueError("Empty response from LM Studio")
                    topic_logger.info(f"LM Studio response received, len={len(response)}")
                    break  # успех
                except ValueError as e:
                    msg = str(e)
                    if "Prompt too long" in msg or "prompt/messages too long" in msg.lower():
                        # Сократить context и перестроить prompt
                        if len(context) > min_context_length:
                            new_len = max(min_context_length, int(len(context) * 0.7))
                            topic_logger.warning(f"Prompt too long. Shortening context from {len(context)} to {new_len} chars and retrying...")
                            context = context[:new_len]
                            prompt = self.prompt_builder.build_prompt(topic, context, media_file)
                            attempt += 1
                            continue
                        else:
                            topic_logger.error(f"Context cannot be shortened further, giving up.")
                            topic_result["status"] = "fail"
                            topic_result["error"] = msg
                            self.stats["fail"] += 1
                            self.stats["topics"].append(topic_result)
                            return False
                    else:
                        topic_logger.error(f"ValueError: {msg}")
                        topic_result["status"] = "fail"
                        topic_result["error"] = msg
                        self.stats["fail"] += 1
                        self.stats["topics"].append(topic_result)
                        return False
                except Exception as e:
                    msg = str(e)
                    topic_logger.error(f"LM Studio generation error: {msg}")
                    topic_result["status"] = "fail"
                    topic_result["error"] = msg
                    self.stats["fail"] += 1
                    self.stats["topics"].append(topic_result)
                    return False
                attempt += 1
            else:
                topic_logger.error(f"All attempts failed due to prompt length. Skipping.")
                topic_result["status"] = "fail"
                topic_result["error"] = "Max prompt shortening attempts reached."
                self.stats["fail"] += 1
                self.stats["topics"].append(topic_result)
                return False

            topic_logger.debug(f"LM Studio response (truncated): {response[:1500]}")
            response = self._shorten_if_needed(response, prompt, has_media=has_media, topic_logger=topic_logger)
            try:
                validated = self.content_validator.validate_content(response, has_media=has_media)
            except Exception as e:
                msg = f"Content validation error: {e}"
                topic_logger.error(msg)
                topic_result["status"] = "fail"
                topic_result["error"] = msg
                self.stats["fail"] += 1
                self.stats["topics"].append(topic_result)
                return False

            formatted = self.content_validator.format_for_telegram(validated, max_len=(1024 if has_media else 4096))
            topic_logger.debug(f"Formatted content for Telegram (truncated): {formatted[:1000]}")

            if has_media:
                result = self.telegram_client.send_media_message(formatted, media_file)
            else:
                result = self.telegram_client.send_text_message(formatted)

            if result:
                topic_logger.info("Topic successfully posted to Telegram.")
                topic_result["status"] = "success"
                self.stats["success"] += 1
            else:
                msg = "Failed to publish topic to Telegram (API error or content too long)."
                topic_logger.error(msg)
                topic_result["status"] = "fail"
                topic_result["error"] = msg
                self.stats["fail"] += 1

            self.stats["topics"].append(topic_result)
            return result

        except Exception as e:
            msg = f"Unhandled error on topic: {e}\n{traceback.format_exc()}"
            topic_logger.error(msg)
            topic_result["status"] = "fail"
            topic_result["error"] = msg
            self.stats["fail"] += 1
            self.stats["topics"].append(topic_result)
            return False

    def _heartbeat(self, heartbeat_counter):
        self.logger.info(f"Heartbeat: {heartbeat_counter} topics processed. "
                         f"Success: {self.stats['success']}, Fail: {self.stats['fail']}, Skipped: {self.stats['skipped']}.")

    def _log_summary(self):
        self.logger.info("===== PROCESSING SUMMARY =====")
        self.logger.info(f"Total topics: {self.stats['total']}")
        self.logger.info(f"Success: {self.stats['success']}")
        self.logger.info(f"Fail: {self.stats['fail']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        # Сохраним подробную статистику
        os.makedirs("logs", exist_ok=True)
        with open("logs/summary_stats.json", "w", encoding="utf-8") as f:
            import json
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

    def run(self):
        self.logger.info("Bot started.")
        heartbeat_counter = 0
        while not self.shutdown_requested:
            topic = self.state_manager.get_next_unprocessed_topic()
            if not topic:
                self.logger.info("No more topics in queue.")
                break

            success = self._process_topic(topic)
            self.state_manager.mark_topic_processed(topic, success=success)

            heartbeat_counter += 1
            if heartbeat_counter % 5 == 0:
                self._heartbeat(heartbeat_counter)
                self.state_manager.save_state()

            interval = self.config["telegram"].get("post_interval", 900)
            self.logger.debug(f"Sleeping for {interval}s before next topic...")
            time.sleep(interval)

        self._log_summary()
        self.logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    system = TelegramRAGSystem()
    system.run()


# данный код относится к файлу content_validator.py

import re
import logging
from typing import Dict, List, Tuple, Optional

class ContentValidator:
    """
    Глубокая очистка, строгая валидация, Markdown→Telegram и защита от edge-cases.
    Не выбрасывает ни одной функции, даже если кажется "дополнительной".
    """

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
        """Длина текста в code units (UTF-16), как у Telegram."""
        return len(text.encode("utf-16-le")) // 2

    def full_clean(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Удаляет <think>, таблицы, ссылки, неразрешённые теги, спецсимволы, дубли строк,
        сохраняет максимум 2 подряд идущих пустых строки (параграфы остаются!).
        """
        stats = dict(
            removed_think=0, removed_tables=0, removed_links=0,
            removed_html=0, removed_special=0, removed_dupes=0
        )
        # 1. <think>
        n_think = len(re.findall(r"<think>.*?</think>", text, flags=re.DOTALL))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        stats["removed_think"] = n_think

        # 2. Markdown и HTML таблицы
        n_tables_md = len(re.findall(r"^\s*\|.+\|", text, flags=re.MULTILINE))
        text = re.sub(r"(?m)^\s*\|.+\|\s*$", "", text)
        n_tables_md2 = len(re.findall(r"^:?-{3,}:?\|", text, flags=re.MULTILINE))
        text = re.sub(r"(?m)^:?-{3,}:?\|.*$", "", text)
        n_tables_html = len(re.findall(r"<table.*?>.*?</table>", text, flags=re.DOTALL | re.IGNORECASE))
        text = re.sub(r"<table.*?>.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)
        stats["removed_tables"] = n_tables_md + n_tables_md2 + n_tables_html

        # 3. Markdown/HTML ссылки
        n_links_md = len(re.findall(r"\[(.*?)\]\(.*?\)", text))
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        n_links_html = len(re.findall(r"<a .*?href=.*?>(.*?)</a>", text, flags=re.IGNORECASE))
        text = re.sub(r"<a .*?href=.*?>(.*?)</a>", r"\1", text, flags=re.IGNORECASE)
        stats["removed_links"] = n_links_md + n_links_html

        # 4. HTML/markdown теги (оставляем только разрешённые)
        def keep_only_telegram_tags(match):
            tag = match.group(1).lower()
            attrs = match.group(2) or ""
            if tag in self.TELEGRAM_HTML_TAGS:
                if tag == "span" and "tg-spoiler" in attrs:
                    return match.group(0)
                return f"<{tag}>"
            return ""
        n_html = len(re.findall(r'</?([a-zA-Z0-9]+)(\s[^>]*)?>', text))
        text = re.sub(r'</?([a-zA-Z0-9]+)(\s[^>]*)?>', keep_only_telegram_tags, text)
        stats["removed_html"] = n_html

        # 5. Спецсимволы, управляющие unicode
        n_special = len(re.findall(r"[\x00-\x08\x0B-\x1F\x7F\u200B&[a-z]+;]+", text))
        text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F\u200B]+", "", text)
        text = re.sub(r"&[a-z]+;", "", text)
        stats["removed_special"] = n_special

        # 6. Дубликаты строк
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

        # 7. Избыточные пустые строки (3+ → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip(), stats

    def validate_content(self, text: str, has_media: bool = False) -> str:
        """Чистит, валидирует по длине, количеству параграфов, мусору; логирует все причины отказа."""
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
        # "Параграф" — блок между двумя пустыми строками
        paragraphs = len([p for p in re.split(r'\n\s*\n', text) if p.strip()])
        if paragraphs < self.min_paragraphs:
            reasons.append(f"too few paragraphs ({paragraphs} < {self.min_paragraphs})")
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
            "paragraphs": len([p for p in re.split(r'\n\s*\n', text) if p.strip()])
        }

    def detect_thinking_patterns(self, text: str) -> List[str]:
        return re.findall(r"<think>.*?</think>", text, flags=re.DOTALL)

    @staticmethod
    def markdown_to_telegram_html(text: str) -> str:
        """Markdown → HTML для Telegram (максимально строгий разбор)."""
        text = re.sub(r'```([\s\S]*?)```', lambda m: f"<pre>{m.group(1)}</pre>", text)
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
        """Оставляет только разрешённые Telegram HTML-теги."""
        tag_map = [
            ('strong', 'b'), ('em', 'i'), ('ins', 'u'), ('strike', 's'), ('del', 's')
        ]
        for src, dst in tag_map:
            text = re.sub(rf'<{src}(\s*?)>', f'<{dst}>', text, flags=re.IGNORECASE)
            text = re.sub(rf'</{src}>', f'</{dst}>', text, flags=re.IGNORECASE)
        def allowed_tag(match):
            tag = match.group(1).lower()
            attrs = match.group(2) or ""
            if tag in ContentValidator.TELEGRAM_HTML_TAGS:
                if tag == "span" and "tg-spoiler" in attrs:
                    return match.group(0)
                elif tag != "span":
                    return f"<{tag}>"
            return ""
        text = re.sub(r'<([a-zA-Z0-9]+)(\s+[^>]*)?>', allowed_tag, text)
        def allowed_close_tag(match):
            tag = match.group(1).lower()
            if tag in ContentValidator.TELEGRAM_HTML_TAGS:
                return match.group(0)
            return ""
        text = re.sub(r'</([a-zA-Z0-9]+)>', allowed_close_tag, text)
        return text

    @staticmethod
    def html_escape_telegram(s: str) -> str:
        """Экранирует &, <, > вне тегов для безопасности (Telegram HTML)."""
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def format_for_telegram(self, text: str, max_len: Optional[int] = None) -> str:
        """
        Markdown→HTML, очистка тегов, экранирование спецсимволов вне тегов,
        обрезка по code units (UTF-16) — всё адаптировано для Telegram.
        """
        html_text = self.markdown_to_telegram_html(text)
        html_text2 = self.normalize_html_for_telegram(html_text)
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


# данный код относится к файлу lm_client.py:

import logging
import requests
import time
from typing import List, Optional, Dict, Any

class LMStudioClient:
    def __init__(self, base_url: str, model: str, config: dict):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.config = config
        self.logger = logging.getLogger("LMStudioClient")

        # Параметры генерации
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self.top_k = config.get("top_k", 40)
        self.top_p = config.get("top_p", 0.95)
        self.system_message = config.get("system_message", "You are a helpful assistant.")

        # История диалога (как список пар user/assistant)
        self.history: List[Dict[str, str]] = []
        self.history_limit = config.get("history_limit", 5)

    def check_connection(self) -> bool:
        try:
            url = f"{self.base_url}/v1/models"
            self.logger.debug(f"Checking LM Studio at: {url}")
            response = requests.get(url, timeout=10)
            self.logger.debug(f"Model info raw response: {response.text[:1000]}")
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [m.get("id") for m in models]
                if self.model in model_ids:
                    self.logger.info("Connected to LM Studio. Model OK.")
                    return True
                else:
                    self.logger.error(f"Model '{self.model}' not found in LM Studio. Available: {model_ids}")
                    return False
            self.logger.warning(f"LM Studio connection failed: {response.status_code}")
            return False
        except requests.RequestException as e:
            self.logger.error("LM Studio not reachable.", exc_info=True)
            return False

    def clear_conversation_history(self):
        self.history.clear()
        self.logger.info("LM history cleared.")

    def estimate_tokens(self, text: str) -> int:
        # Эвристика: 1 токен ≈ 4 символа
        return max(1, int(len(text) / 4))

    def _build_payload_chat(self, prompt: str) -> dict:
        # История — последние self.history_limit*2 сообщений (user/assistant)
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        if self.history_limit > 0:
            history_pairs = self.history[-self.history_limit*2:]
            for m in history_pairs:
                messages.append(m)
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def _build_payload_completion(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": None,
        }

    def _parse_response(self, response: requests.Response, mode: str) -> Optional[str]:
        try:
            self.logger.debug(f"Raw LMStudio response: {response.text[:1000]}")
            result = response.json()
        except ValueError:
            self.logger.error("Failed to decode JSON from LM response.", exc_info=True)
            return None

        if "error" in result:
            self.logger.error(f"LMStudio error: {result['error']}")
            raise ValueError("LMStudio error: " + str(result['error']))

        try:
            if mode == "chat":
                return result["choices"][0]["message"]["content"]
            else:
                return result["choices"][0]["text"]
        except (KeyError, IndexError):
            self.logger.error("LMStudio response missing required fields.", exc_info=True)
            return None

    def _validate_prompt_length(self, messages: List[Dict[str, str]]) -> None:
        # Лимит токенов для всей истории и запроса
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in messages)
        if total_tokens > self.max_tokens:
            self.logger.warning(f"Prompt too long for LM Studio ({total_tokens} > {self.max_tokens} tokens)")
            raise ValueError(f"Prompt too long: {total_tokens} > {self.max_tokens} tokens")

    def generate_content(self, prompt: str) -> Optional[str]:
        mode = "chat"
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload_chat(prompt)
        self.logger.debug(f"LMStudio chat payload (truncated): {str(payload)[:1200]}")
        # Проверка длины prompt/messages (теперь кидает Exception)
        self._validate_prompt_length(payload["messages"])

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            self.logger.info(f"[LM] POST {url} [{response.status_code}] in {response.elapsed.total_seconds():.2f}s")
            content = self._parse_response(response, mode)
            if content:
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": content})
                if len(self.history) > self.history_limit * 2:
                    self.history = self.history[-self.history_limit*2:]
                return content
        except requests.RequestException as e:
            self.logger.error(f"Request failed to LMStudio at {url}", exc_info=True)
            raise

        # fallback to completions
        self.logger.warning("Falling back to /v1/completions")
        fallback_url = f"{self.base_url}/v1/completions"
        fallback_payload = self._build_payload_completion(prompt)
        self.logger.debug(f"LMStudio completion payload (truncated): {str(fallback_payload)[:1200]}")
        self._validate_prompt_length([{"role": "user", "content": prompt}])
        try:
            response = requests.post(fallback_url, json=fallback_payload, timeout=self.timeout)
            self.logger.info(f"[LM] Fallback POST {fallback_url} [{response.status_code}] in {response.elapsed.total_seconds():.2f}s")
            return self._parse_response(response, "completion")
        except requests.RequestException as e:
            self.logger.error(f"Fallback request also failed.", exc_info=True)
            raise

    def generate_with_retry(self, prompt: str, max_retries: int = 3, delay: float = 3.0) -> str:
        last_exception = None
        for attempt in range(max_retries):
            try:
                result = self.generate_content(prompt)
                if result:
                    return result
                else:
                    self.logger.warning(f"LM generation failed (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
            except ValueError as ve:
                # Если ошибка - prompt слишком длинный, пробрасываем дальше
                if "Prompt too long" in str(ve):
                    raise
                last_exception = ve
                self.logger.warning(f"LMStudio value exception: {ve}. Retrying in {delay}s...")
            except Exception as e:
                last_exception = e
                self.logger.warning(f"LMStudio internal exception: {e}. Retrying in {delay}s...")
            time.sleep(delay)

        self.logger.critical("All LMStudio attempts failed. Raising exception.")
        raise RuntimeError(f"LMStudio generation failed after {max_retries} retries. Last error: {last_exception}")

    def request_shorter_version(self, original_prompt: str, current_length: int, target_length: int) -> str:
        shorten_instruction = (
            f"{original_prompt}\n\nPlease shorten the response to less than {target_length} characters."
        )
        try:
            result = self.generate_with_retry(shorten_instruction)
            return result if result else ""
        except Exception as e:
            self.logger.error("Failed to generate shorter version.", exc_info=True)
            return ""

    def get_generation_stats(self) -> dict:
        return {
            "history_length": len(self.history),
            "history_tokens": sum(self.estimate_tokens(m['content']) for m in self.history)
        }

    def health_check(self) -> dict:
        return {
            "connected": self.check_connection(),
            "model": self.model,
            "stats": self.get_generation_stats()
        }

# данный код относится к файлу prompt_builder.py

import os
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

class PromptBuilder:
    """
    Модуль для сборки промпта из шаблонов, строгого контроля структуры, адаптации под RAG/LM.
    Не выбрасывает ни одной функции, не упрощает интерфейс, сохраняет гибкость для масштабирования.
    """

    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]

    def __init__(self, prompt_folders: List[str]):
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        """Загружает шаблоны из всех указанных папок, сохраняет полные пути."""
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                self.templates[str(folder)] = []
                continue
            self.templates[str(folder)] = self.scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def scan_prompt_folder(self, folder_path: Path) -> List[str]:
        """Сканирует папку и возвращает список путей к .txt шаблонам."""
        return [str(p) for p in folder_path.glob("*.txt")]

    def select_random_templates(self) -> List[str]:
        """Из каждой папки выбирает случайный шаблон (или пустую строку, если папка пуста)."""
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            if templates:
                selected.append(random.choice(templates))
            else:
                selected.append("")
        return selected

    def read_template_file(self, file_path: str) -> str:
        """Читает шаблон из файла. Возвращает пустую строку при ошибке."""
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content
        except Exception as e:
            self.logger.error(f"Failed to read prompt file: {file_path}", exc_info=True)
            return ""

    def validate_prompt_structure(self, template: str) -> bool:
        """Проверяет наличие всех обязательных плейсхолдеров."""
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            self.logger.error(f"Prompt missing required placeholders: {missing}")
            return False
        return True

    def replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        """Подставляет значения во все плейсхолдеры."""
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def build_prompt(self, topic: str, context: str, media_file: Optional[str] = None) -> str:
        """
        Собирает итоговый prompt:
        - Выбирает по одному шаблону из каждой папки (можно гибко расширять).
        - Объединяет шаблоны через двойной перенос.
        - Подставляет все значения.
        - Логирует неиспользованные плейсхолдеры.
        - Обрезает избыточные пустые строки (3+ → 2), но сохраняет параграфы.
        """
        template_paths = self.select_random_templates()
        template_texts = [self.read_template_file(path) for path in template_paths]
        content = "\n\n".join(template_texts).strip()

        if not self.validate_prompt_structure(content):
            raise ValueError("Prompt structure validation failed. Missing required placeholders in templates.")

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
            "{UPLOADFILE}": media_file.strip() if media_file else "",
        }
        prompt = self.replace_placeholders(content, replacements)

        # Логируем неиспользованные плейсхолдеры (чтобы не пропустить баг с шаблоном!)
        unused = [ph for ph in self.REQUIRED_PLACEHOLDERS + ["{UPLOADFILE}"] if ph in prompt]
        if unused:
            self.logger.warning(f"Prompt still contains unused placeholders: {unused}")

        # Только избыточные пустые строки (3+ → 2), параграфы не ломаются
        prompt = self._compact_whitespace(prompt)
        self.logger.debug(f"Final prompt (truncated): {prompt[:1000]}")
        return prompt

    def _compact_whitespace(self, text: str) -> str:
        """Заменяет три и более подряд идущих пустых строки на две (сохраняет структуру абзацев)."""
        return re.sub(r"\n{3,}", "\n\n", text)

    def check_placeholder_presence(self, template: str) -> Dict[str, bool]:
        """Проверяет, есть ли все ключевые плейсхолдеры в шаблоне (для тестов и дебага)."""
        all_ph = self.REQUIRED_PLACEHOLDERS + ["{UPLOADFILE}"]
        return {ph: (ph in template) for ph in all_ph}

    def get_template_stats(self) -> Dict[str, int]:
        """Статистика: сколько шаблонов в каждой папке."""
        return {folder: len(templates) for folder, templates in self.templates.items()}

    def reload_templates(self) -> None:
        """Перезагружает шаблоны (например, после обновления файлов на диске)."""
        self.logger.info("Reloading prompt templates...")
        self.load_prompt_templates()

    def test_template_combination(self, topic: str, context: str) -> str:
        """Для тестов: собирает prompt со случайным шаблоном и тестовой картинкой."""
        return self.build_prompt(topic, context, media_file="media/sample.jpg")

# данный код относится к файлу config_manager.py

import json
import os
import logging
from typing import Optional, Any, Dict

class ConfigManager:
    """
    Менеджер конфигураций для всей системы автопостинга с RAG и LM Studio.
    Гарантирует: корректную загрузку, подробную валидацию и безопасный доступ к параметрам.
    """

    def __init__(self, config_path: str = "config/config.json"):
        self.logger = logging.getLogger("ConfigManager")
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Загружает конфиг из JSON-файла. В случае ошибки (отсутствие, синтаксис) — критический лог и exit.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.logger.info(f"Loaded config from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.critical(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.critical(f"Failed to parse config.json: {e}")
            raise

    def validate_config(self) -> bool:
        """
        Проверяет наличие всех обязательных секций и ключей. Логирует каждую причину.
        Возвращает: True — если всё ок, иначе False.
        """
        errors = []
        config = self.config

        # Обязательные секции
        required_sections = ["lm_studio", "rag", "telegram", "serper", "processing", "paths"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")

        # Проверка ключей для lm_studio
        if "lm_studio" in config:
            for key in ["base_url", "model", "max_tokens", "temperature", "timeout"]:
                if key not in config["lm_studio"]:
                    errors.append(f"Missing key '{key}' in section 'lm_studio'")

        # Проверка ключей для rag
        if "rag" in config:
            for key in ["embedding_model", "chunk_size", "chunk_overlap", "max_context_length", "media_context_length", "similarity_threshold"]:
                if key not in config["rag"]:
                    errors.append(f"Missing key '{key}' in section 'rag'")

        # Проверка ключей telegram
        if "telegram" in config:
            for key in ["post_interval", "max_retries"]:
                if key not in config["telegram"]:
                    errors.append(f"Missing key '{key}' in section 'telegram'")

        # serper
        if "serper" in config:
            for key in ["results_limit"]:
                if key not in config["serper"]:
                    errors.append(f"Missing key '{key}' in section 'serper'")

        # processing
        if "processing" in config:
            for key in ["batch_size", "max_file_size_mb"]:
                if key not in config["processing"]:
                    errors.append(f"Missing key '{key}' in section 'processing'")

        # paths
        if "paths" in config:
            for key in ["media_dir", "prompt_folders", "data_dir", "processed_topics_file"]:
                if key not in config["paths"]:
                    errors.append(f"Missing key '{key}' in section 'paths'")

        if errors:
            for err in errors:
                self.logger.critical(f"Config validation error: {err}")
            return False
        return True

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Позволяет получать значение по "пути" через точку, например: 'lm_studio.base_url'
        Если не найдено — возвращает default.
        """
        keys = key_path.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"Config key not found: {key_path}, using default: {default}")
            return default

    def get_telegram_token(self) -> str:
        """Читает токен Telegram-бота из config/telegram_token.txt"""
        token_file = os.path.join("config", "telegram_token.txt")
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read().strip()
            if not token:
                self.logger.critical("Telegram token file is empty!")
                raise ValueError("Empty Telegram token")
            return token
        except Exception as e:
            self.logger.critical(f"Failed to read Telegram token: {e}")
            raise

    def get_telegram_channel_id(self) -> str:
        """Читает ID канала Telegram из config/telegram_channel.txt"""
        channel_file = os.path.join("config", "telegram_channel.txt")
        try:
            with open(channel_file, "r", encoding="utf-8") as f:
                channel_id = f.read().strip()
            if not channel_id:
                self.logger.critical("Telegram channel ID file is empty!")
                raise ValueError("Empty Telegram channel ID")
            return channel_id
        except Exception as e:
            self.logger.critical(f"Failed to read Telegram channel ID: {e}")
            raise

    def get_lm_studio_config(self) -> dict:
        return self.config.get("lm_studio", {})

    def get_rag_config(self) -> dict:
        return self.config.get("rag", {})

    def get_serper_api_key(self) -> Optional[str]:
        """Пробует взять serper API ключ из переменной окружения или файла."""
        api_key = os.environ.get("SERPER_API_KEY")
        if api_key:
            return api_key
        # Можно добавить вариант с файлом
        key_file = os.path.join("config", "serper_api_key.txt")
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            if api_key:
                return api_key
        # Fallback: попробовать из конфига
        return self.config.get("serper", {}).get("api_key")

    def get_all_config(self) -> dict:
        """Возвращает полный конфиг (для отладки, без секретных полей)."""
        safe_config = self.config.copy()
        # Можно тут удалить/заменить чувствительные данные, если они есть
        return safe_config

    def save_config(self) -> None:
        """Сохраняет текущий конфиг обратно в файл."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            self.logger.info("Config saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def reload_config(self) -> None:
        """Перечитывает конфиг с диска."""
        self.config = self._load_config()
        self.logger.info("Config reloaded.")

# данный текст относится к файлу config.json

{
    "version": "1.0.0",
    "environment": "production",
    "telegram": {
        "bot_token_file": "config/telegram_token.txt",
        "channel_id_file": "config/telegram_channel.txt",
        "retry_attempts": 3,
        "retry_delay": 3.0,
        "enable_preview": true,
        "max_caption_length": 1024,
        "post_interval": 10,
        "max_retries": 3
    },
    "language_model": {
        "url": "http://localhost:1234/v1/chat/completions",
        "model_name": "qwen3-14b",
        "max_tokens": 4096,
        "max_chars": 20000,
        "temperature": 0.7,
        "timeout": 1500,
        "history_limit": 3,
        "system_message": "Вы — эксперт по бровям и ресницам.",
        "max_chars_with_media": 4096
    },
    "lm_studio": {
        "base_url": "http://localhost:1234",
        "model": "qwen/qwen3-14b",
        "max_tokens": 4096,
        "max_chars": 20000,
        "temperature": 0.7,
        "timeout": 1500,
        "history_limit": 3,
        "system_message": "Вы — эксперт по бровям и ресницам.",
        "max_chars_with_media": 4096
    },
    "retrieval": {
        "chunk_size": 500,
        "overlap": 100,
        "top_k_title": 2,
        "top_k_faiss": 8,
        "top_k_final": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        "cross_encoder": "cross-encoder/stsb-roberta-large"
    },
    "rag": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k_title": 2,
        "top_k_faiss": 8,
        "top_k_final": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        "cross_encoder": "cross-encoder/stsb-roberta-large",
        "max_context_length": 4096,
        "media_context_length": 1024,
        "similarity_threshold": 0.7
    },
    "system": {
        "chunk_usage_limit": 10,
        "usage_reset_days": 7,
        "diversity_boost": 0.3,
        "max_retries": 3,
        "backoff_factor": 1.5
    },
    "paths": {
        "data_dir": "data",
        "log_dir": "logs",
        "inform_dir": "inform",
        "media_dir": "media",
        "index_file": "data/faiss_index.idx",
        "context_file": "data/faiss_contexts.json",
        "usage_stats_file": "data/usage_statistics.json",
        "processed_topics_file": "data/state.json",
        "prompt_folders": [
            "data/prompt_1",
            "data/prompt_2",
            "data/prompt_3"
        ]
    },
    "temp_files": {
        "cleanup_interval_hours": 24,
        "max_size_mb": 1000,
        "min_free_space_mb": 500
    },
    "logging": {
        "level": "INFO",
        "file_max_mb": 5,
        "backup_count": 3
    },
    "content_validator": {
        "remove_tables": true,
        "max_length_no_media": 4096,
        "max_length_with_media": 1024
    },
    "schedule": {
        "interval_seconds": 900
    },
    "external_apis": {
        "serper_api_key_file": "config/serper_api_key.txt",
        "serper_endpoint": "https://google.serper.dev/search",
        "serper_results_limit": 10
    },
    "serper": {
        "api_key_file": "config/serper_api_key.txt",
        "endpoint": "https://google.serper.dev/search",
        "results_limit": 10
    },
    "processing": {
        "max_tasks_per_run": 1,
        "max_errors": 5,
        "error_backoff_sec": 30,
        "max_processing_time_sec": 300,
        "shutdown_on_critical": true,
        "batch_size": 1,
        "max_file_size_mb": 100
    }
}
