# участок кода файла main.py
import sys
import signal
import time
import os

from modules.utils.config_manager import ConfigManager
from modules.utils.logs import get_logger, log_system_info
from modules.utils.state_manager import StateManager
from modules.rag_system.rag_retriever import RAGRetriever
from modules.external_apis.telegram_client import TelegramClient
from modules.external_apis.web_search import WebSearchClient
from modules.content_generation.lm_client import LMStudioClient
from modules.content_generation.prompt_builder import PromptBuilder
from modules.content_generation.content_validator import ContentValidator

class MonitoringService:
    def __init__(self, logger):
        self.topics_processed = 0
        self.topics_failed = 0
        self.logger = logger

    def log_success(self, topic):
        self.topics_processed += 1
        self.logger.info(f"[MONITOR] Topic processed: {topic}")

    def log_failure(self, topic, error):
        self.topics_failed += 1
        self.logger.error(f"[MONITOR] Topic failed: {topic}, error: {error}")

    def report(self):
        self.logger.info(f"[MONITOR] Stats: Success: {self.topics_processed}, Failed: {self.topics_failed}")

class TelegramRAGSystem:
    def __init__(self, config_path: str = "config/config.json"):
        self.logger = get_logger("Main")
        self.logger.info("🚀 Initializing TelegramRAGSystem...")
        self.shutdown_requested = False

        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        except Exception as e:
            self.logger.critical("Config initialization failed", exc_info=True)
            sys.exit(1)

        self.setup_logging()
        self.validate_configuration()
        self.initialize_services()
        self.autoload_topics()

    def setup_logging(self):
        log_system_info(self.logger)

    def validate_configuration(self):
        if not self.config_manager.validate_config():
            self.logger.critical("Configuration validation failed.")
            sys.exit(1)
        self.logger.info("Configuration validated successfully.")

    def initialize_services(self):
        try:
            self.rag_retriever = RAGRetriever(config=self.config["rag"])
            self.state_manager = StateManager(state_file="data/state.json")
            self.monitoring = MonitoringService(self.logger)

            self.lm_client = LMStudioClient(
                base_url=self.config["lm_studio"]["base_url"],
                model=self.config["lm_studio"]["model"],
                config=self.config["lm_studio"]
            )
            self.prompt_builder = PromptBuilder(
                prompt_folders=self.config["paths"].get("prompt_folders", [
                    "data/prompt_1", "data/prompt_2", "data/prompt_3"
                ])
            )
            self.content_validator = ContentValidator(config=self.config)

            serper_api_key = self.config_manager.get_serper_api_key()
            serper_endpoint = self.config_manager.get_config_value("serper.endpoint", "https://google.serper.dev/search")
            serper_results_limit = self.config_manager.get_config_value("serper.results_limit", 10)
            self.web_search = WebSearchClient(
                api_key=serper_api_key,
                endpoint=serper_endpoint,
                results_limit=serper_results_limit
            )

            token = self.config_manager.get_telegram_token()
            channel_id = self.config_manager.get_telegram_channel_id()
            self.telegram_client = TelegramClient(
                token=token,
                channel_id=channel_id,
                config=self.config["telegram"]
            )
        except Exception as e:
            self.logger.critical("Component initialization failed", exc_info=True)
            sys.exit(1)

    def autoload_topics(self):
        topics_file = "data/topics.txt"
        if not os.path.isfile(topics_file):
            self.logger.warning(f"Topics file not found: {topics_file}")
            return

        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                topics = [line.strip() for line in f if line.strip()]
            existing = set(self.state_manager.get_unprocessed_topics() +
                           self.state_manager.get_processed_topics() +
                           self.state_manager.get_failed_topics())
            new_topics = [t for t in topics if t not in existing]
            if new_topics:
                self.logger.info(f"Autoloading {len(new_topics)} new topics into queue")
                self.state_manager.add_topics(new_topics)
            else:
                self.logger.info("No new topics found to autoload")
        except Exception as e:
            self.logger.error("Failed to autoload topics", exc_info=True)

    def graceful_shutdown(self, *_):
        self.shutdown_requested = True
        self.logger.warning("Shutdown signal received. Exiting loop...")

    def get_next_topic(self) -> str:
        topic = self.state_manager.get_next_unprocessed_topic()
        if topic:
            self.logger.info(f"Next topic selected: {topic}")
        else:
            self.logger.info("No more topics to process.")
        return topic

    def truncate_rag_context(self, rag_context: str, limit: int = 10000) -> str:
        if rag_context and len(rag_context) > limit:
            self.logger.warning(f"RAG context too long ({len(rag_context)} > {limit}), truncating.")
            return rag_context[:limit] + "\n... [RAG контекст обрезан до 10 000 символов]"
        return rag_context

    def combine_contexts(self, rag_context: str, web_context: str) -> str:
        if not rag_context and not web_context:
            return ""
        elif not rag_context:
            return f"[Web context only]\n\n{web_context}"
        elif not web_context:
            return f"{rag_context}\n\n[Нет web-контекста]"
        return f"{rag_context}\n\n[Доп. контекст из поиска]\n\n{web_context}"

    def update_processing_state(self, topic: str, success: bool):
        try:
            self.state_manager.mark_topic_processed(topic, success)
            self.logger.info(f"Topic '{topic}' marked as {'processed' if success else 'failed'}.")
        except Exception as e:
            self.logger.error(f"Failed to update state for topic '{topic}': {str(e)}", exc_info=True)

    def handle_error(self, topic: str, error: Exception):
        try:
            self.logger.error(f"Error processing topic '{topic}': {str(error)}", exc_info=True)
            self.update_processing_state(topic, success=False)
            self.monitoring.log_failure(topic, error)
        except Exception as e:
            self.logger.critical("Failed during error handling!", exc_info=True)

    def main_processing_loop(self):
        while not self.shutdown_requested:
            topic = self.get_next_topic()
            if not topic:
                break

            try:
                rag_context = self.rag_retriever.retrieve_context(topic)
                rag_context = self.truncate_rag_context(rag_context, 10000)  # Ограничение RAG контекста до 10k

                if not isinstance(rag_context, str) or not rag_context.strip():
                    self.logger.error(f"RAG context is empty for topic: {topic}")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "RAG context is empty")
                    continue

                web_results = self.web_search.search(topic)
                web_context = self.web_search.extract_content(web_results) if web_results else ""
                if not isinstance(web_context, str):
                    web_context = ""

                full_context = self.combine_contexts(rag_context, web_context)
                self.logger.debug(f"[{topic}] full_context length: {len(full_context)}, preview: {full_context[:300]}")

                prompt, prompt_template = self.prompt_builder.build_prompt(
                    topic=topic,
                    context=full_context
                )

                if not prompt or not prompt.strip():
                    self.logger.error(f"Prompt building failed (empty) for topic '{topic}'.")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "Prompt building failed (empty prompt)")
                    continue

                self.logger.debug(f"Prompt to LM Studio for topic '{topic}':\n{prompt[:1000]}")

                max_lm_retries = self.config["lm_studio"].get("max_retries", 3)
                try:
                    content = self.lm_client.generate_with_retry(
                        prompt_template,
                        topic,
                        full_context,
                        max_retries=max_lm_retries
                    )
                except Exception as e:
                    self.logger.error(f"LM Studio generation failed after retries for topic '{topic}': {e}")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, f"LM Studio generation failed: {e}")
                    continue

                if not content or not content.strip():
                    self.logger.error(f"Generated content is empty for topic '{topic}'.")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "Generated content is empty")
                    continue

                validated_content = self.content_validator.validate_content(content)
                if not validated_content or not validated_content.strip():
                    self.logger.error(f"Validated content is empty for topic '{topic}'.")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "Validated content is empty")
                    continue

                success = False
                max_retries = self.config["telegram"].get("max_retries", 3)
                for attempt in range(1, max_retries + 1):
                    try:
                        success = self.telegram_client.send_text_message(validated_content)
                        if success:
                            break
                    except Exception as te:
                        self.logger.error(f"Telegram send failed (attempt {attempt}): {te}")
                        time.sleep(2)
                
                self.update_processing_state(topic, success)
                if success:
                    self.monitoring.log_success(topic)
                else:
                    self.monitoring.log_failure(topic, "Telegram send failed")

                self.monitoring.report()
                time.sleep(self.config["telegram"].get("post_interval", 15))

            except Exception as e:
                self.handle_error(topic, e)
                continue

    def run(self):
        self.logger.info("System starting up...")
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        try:
            inform_folder = self.config["rag"].get("inform_folder", "inform/")
            self.rag_retriever.process_inform_folder(inform_folder)
            self.rag_retriever.build_knowledge_base()
        except Exception as e:
            self.logger.critical(f"Failed to build RAG knowledge base: {e}", exc_info=True)
            sys.exit(1)
        self.main_processing_loop()
        self.logger.info("System shut down gracefully.")

if __name__ == "__main__":
    system = TelegramRAGSystem()
    system.run()




# участок кода файла content_validator.py
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

    # Только теги, поддерживаемые Telegram: https://core.telegram.org/bots/api#formatting-options
    ALLOWED_TAGS = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "code", "pre", "a"}

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
        # Markdown to Telegram HTML patterns
        self.re_md_code_block = re.compile(r'```(.*?)```', re.DOTALL)
        self.re_md_inline_code = re.compile(r'`([^`\n]+)`')
        self.re_md_bold1 = re.compile(r'(?<!\*)\*\*([^\*]+)\*\*(?!\*)')
        self.re_md_bold2 = re.compile(r'__([^_]+)__')
        self.re_md_italic1 = re.compile(r'(?<!\*)\*([^\*]+)\*(?!\*)')
        self.re_md_italic2 = re.compile(r'(?<!_)_([^_]+)_(?!_)')
        self.re_md_strike = re.compile(r'~~([^~]+)~~')
        self.re_md_url = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')

    def validate_content(self, text: str) -> str:
        if not isinstance(text, str):
            self.logger.error("Content validation input is not a string")
            return ""
        text = text.strip()
        if not text:
            self.logger.warning("Empty content provided for validation")
            return ""

        # 1. Удалить размышления (think)
        text = self.remove_thinking_blocks(text)
        # 2. Преобразовать markdown-разметку в Telegram HTML
        text = self.convert_markdown_to_telegram_html(text)
        # 3. Оставить только разрешённые html-теги Telegram
        text = self._remove_forbidden_html_tags(text)
        # 4. Удалить таблицы, если вдруг остались
        text = self._remove_tables_and_thinking(text)
        # 5. Прочая чистка
        text = self._clean_junk(text)
        # 6. Ограничить emoji-спам
        text = self._filter_emoji_spam(text)
        # 7. Ограничить длину для Telegram
        text = self._ensure_telegram_limits(text)

        if not self._content_quality_check(text):
            self.logger.warning("Content failed quality validation")
            return ""
        return text.strip()

    def remove_thinking_blocks(self, text: str) -> str:
        return self.re_think.sub('', text)

    def convert_markdown_to_telegram_html(self, text: str) -> str:
        # Ссылки [текст](url)
        text = self.re_md_url.sub(r'<a href="\2">\1</a>', text)

        # Многострочный код (Telegram поддерживает <pre>)
        text = self.re_md_code_block.sub(lambda m: f"<pre>{self.escape_html(m.group(1).strip())}</pre>", text)
        # Инлайн-код
        text = self.re_md_inline_code.sub(lambda m: f"<code>{self.escape_html(m.group(1).strip())}</code>", text)
        # Жирный
        text = self.re_md_bold1.sub(r'<b>\1</b>', text)
        text = self.re_md_bold2.sub(r'<b>\1</b>', text)
        # Курсив
        text = self.re_md_italic1.sub(r'<i>\1</i>', text)
        text = self.re_md_italic2.sub(r'<i>\1</i>', text)
        # Зачёркнутый
        text = self.re_md_strike.sub(r'<s>\1</s>', text)
        return text

    def escape_html(self, text: str) -> str:
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

    def _remove_forbidden_html_tags(self, text: str) -> str:
        # Удаляем все теги, кроме разрешённых Telegram
        def _strip_tag(m):
            tag = m.group(1).lower()
            if tag in self.ALLOWED_TAGS:
                return m.group(0)
            return ''
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
        if self._has_long_emoji_run(chars):
            self.logger.warning("Emoji spam detected (long run)")
            return ""
        return text

    def _is_emoji(self, char: str) -> bool:
        try:
            return emoji.is_emoji(char)
        except Exception:
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




# участок кода файла lm_client.py
import logging
import requests
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class LMStudioClient:
    LM_MAX_TOTAL_CHARS = 20000
    TELEGRAM_LIMIT = 4096

    def __init__(self, base_url: str, model: str, config: Dict[str, Any]):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        self.history_limit = config.get("history_limit", 3)
        self.system_message = config.get("system_message", None)
        self.top_p = config.get("top_p", None)
        self.top_k = config.get("top_k", None)
        self.logger = logging.getLogger("LMStudioClient")
        self.history: List[Dict[str, str]] = []
        self._validate_config()
        self._check_health_on_init()
        self.log_dir_success = "logs/lmstudio/success"
        self.log_dir_failed = "logs/lmstudio/failed"
        self.log_dir_prompts = "logs/lmstudio/prompts"
        os.makedirs(self.log_dir_success, exist_ok=True)
        os.makedirs(self.log_dir_failed, exist_ok=True)
        os.makedirs(self.log_dir_prompts, exist_ok=True)

    def _validate_config(self):
        assert isinstance(self.max_tokens, int) and self.max_tokens > 0, "max_tokens must be positive integer"
        assert isinstance(self.LM_MAX_TOTAL_CHARS, int) and self.LM_MAX_TOTAL_CHARS > 1000, "LM_MAX_TOTAL_CHARS must be > 1000"
        assert isinstance(self.temperature, (float, int)), "temperature must be float"
        if self.top_p is not None:
            assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0,1]"
        if self.top_k is not None:
            assert isinstance(self.top_k, int) and self.top_k >= 0, "top_k must be non-negative int"

    def _check_health_on_init(self):
        status = self.health_check()
        if status.get("status") != "ok":
            self.logger.warning(f"LM Studio health check: {status}")
        else:
            self.logger.info(f"LMStudioClient connected to model '{self.model}'. Health OK.")

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and any(self.model in m.get("id", "") for m in data["data"]):
                return {"status": "ok"}
            return {"status": "model_not_found"}
        except Exception as e:
            self.logger.warning(f"Health check fallback: /v1/models endpoint not found, trying /v1/chat/completions dummy call.")
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=10)
                if resp.status_code in (200, 400):
                    return {"status": "ok"}
            except Exception as e2:
                self.logger.error(f"Health check failed: {e2}")
            return {"status": "unreachable"}

    def clear_conversation_history(self):
        self.history = []
        self.logger.debug("LMStudioClient: conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        if user_message and isinstance(user_message, str) and user_message.strip():
            self.history.append({"role": "user", "content": user_message})
        if bot_message and isinstance(bot_message, str) and bot_message.strip():
            self.history.append({"role": "assistant", "content": bot_message})
        if self.history_limit > 0 and len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def _clean_history(self) -> List[Dict[str, str]]:
        clean = []
        for m in self.history[-self.history_limit * 2:]:
            if (
                isinstance(m, dict)
                and m.get("role") in {"user", "assistant", "system"}
                and isinstance(m.get("content"), str)
                and m["content"].strip()
                and "nan" not in m["content"]
            ):
                clean.append(m)
            else:
                self.logger.warning(f"Skipping invalid message in LLM history: {m}")
        return clean

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str) -> str:
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": "CONTEXT_PLACEHOLDER",
        }
        prompt_wo_context = prompt_template
        for key, value in replacements.items():
            prompt_wo_context = prompt_wo_context.replace(key, value)
        prompt_wo_context_len = len(prompt_wo_context.replace("CONTEXT_PLACEHOLDER", ""))

        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        static_chars = sum(len(m["content"]) for m in messages) + prompt_wo_context_len
        available = self.LM_MAX_TOTAL_CHARS - static_chars
        if available <= 0:
            self.logger.warning(f"No room for context: static_chars={static_chars} > limit={self.LM_MAX_TOTAL_CHARS}")
            return ""
        context = context.strip()
        if len(context) > available:
            self.logger.warning(f"Context too long for LLM ({len(context)} > {available}), truncating context")
            context = context[:available]
        return context

    def _build_messages(self, prompt_template: str, topic: str, context: str) -> List[Dict[str, str]]:
        context = self._truncate_context_for_llm(prompt_template, topic, context)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context,
        }
        prompt = prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        prompt = prompt.replace("nan", "").strip()
        
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        messages.append({"role": "user", "content": prompt})
        
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > self.LM_MAX_TOTAL_CHARS:
            self.logger.warning(f"Total LLM payload too long ({total_chars} > {self.LM_MAX_TOTAL_CHARS}), trimming prompt/history")
            excess = total_chars - self.LM_MAX_TOTAL_CHARS
            if len(messages[-1]["content"]) > excess:
                messages[-1]["content"] = messages[-1]["content"][:len(messages[-1]["content"]) - excess]
            else:
                while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                    removed = messages.pop(1)
                    self.logger.warning(f"Removed old history message to fit LM payload: {removed}")
                    total_chars = sum(len(m["content"]) for m in messages)
                if total_chars > self.LM_MAX_TOTAL_CHARS:
                    last = messages[-1]
                    last["content"] = last["content"][:max(0, len(last["content"]) - (total_chars - self.LM_MAX_TOTAL_CHARS))]
        return messages

    def _request_shorter_content(self, prompt_template: str, topic: str, context: str, current_length: int) -> str:
        shorter_prompt = f"{prompt_template}\n\nВАЖНО: Твой предыдущий ответ был слишком длинный ({current_length} символов). Напиши более коротко, сохрани смысл и структуру."
        messages = self._build_messages(shorter_prompt, topic, context)
        return self._make_request(messages)

    def _make_request(self, messages: List[Dict[str, str]]) -> str:
        chat_url = f"{self.base_url}/v1/chat/completions"
        payload_chat = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if self.top_p is not None:
            payload_chat["top_p"] = self.top_p
        if self.top_k is not None:
            payload_chat["top_k"] = self.top_k

        self.logger.debug(f"LMStudioClient: Sending chat payload to {chat_url}: {str(payload_chat)[:800]}")

        try:
            response = requests.post(chat_url, json=payload_chat, timeout=self.timeout)
        except Exception as e:
            self.logger.error(f"Error during POST to LM Studio: {e}")
            return ""
        if not response.ok:
            self.logger.error(f"LM Studio response HTTP error: {response.status_code} {response.text[:200]}")
            return ""
        self.logger.debug(f"LMStudioClient: raw response: {response.text[:1000]}")
        try:
            result = response.json()
        except Exception as e:
            self.logger.error("Failed to decode LM Studio response as JSON", exc_info=True)
            result = {}
        
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not text:
            self.logger.warning("Empty chat response, fallback to completions endpoint.")
            comp_url = f"{self.base_url}/v1/completions"
            payload = {
                "model": self.model,
                "prompt": messages[-1]['content'],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            if self.top_k is not None:
                payload["top_k"] = self.top_k
            try:
                comp_resp = requests.post(comp_url, json=payload, timeout=self.timeout)
                comp_resp.raise_for_status()
                comp_result = comp_resp.json()
                text = comp_result.get("choices", [{}])[0].get("text", "")
            except Exception as e:
                self.logger.error(f"Failed to get completion from fallback endpoint: {e}")
                text = ""
        if not isinstance(text, str):
            raise ValueError("LM Studio returned non-string result.")
        return text.strip()

    def _save_lm_log(self, text: str, topic: str, success: bool, prompt: Optional[str] = None, attempt: int = 0):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        folder = self.log_dir_success if success else self.log_dir_failed
        filename = f"{date_str}_attempt{attempt}_{safe_topic[:40]}.txt"
        try:
            with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            self.logger.error(f"Failed to save LM log: {e}")
        if prompt:
            try:
                with open(os.path.join(self.log_dir_prompts, f"{date_str}_attempt{attempt}_{safe_topic[:40]}_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception as e:
                self.logger.error(f"Failed to save LM prompt log: {e}")

    def generate_content(self, prompt_template: str, topic: str, context: str, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        text = ""
        last_prompt = ""
        try:
            messages = self._build_messages(prompt_template, topic, context)
            last_prompt = messages[-1]['content'] if messages else ""
            text = self._make_request(messages)
            self._save_lm_log(text, topic, False, last_prompt, attempt=0)
            if not text or not text.strip():
                self.logger.warning("LM Studio returned empty text from both endpoints.")
                return ""

            if len(text) > self.TELEGRAM_LIMIT:
                self.logger.warning(f"Generated content too long ({len(text)} chars), requesting shorter version")
                attempts = 0
                max_attempts = 3
                while len(text) > self.TELEGRAM_LIMIT and attempts < max_attempts:
                    attempts += 1
                    self.logger.info(f"Attempt {attempts} to get shorter content")
                    # Очищаем историю для предотвращения накопления длинных и коротких версий
                    self.clear_conversation_history()
                    try:
                        shorter_prompt = f"{prompt_template}\n\nВАЖНО: Твой предыдущий ответ был слишком длинный ({len(text)} символов). Напиши более коротко, сохрани смысл и структуру."
                        messages = self._build_messages(shorter_prompt, topic, context)
                        last_prompt = messages[-1]['content'] if messages else ""
                        text = self._make_request(messages)
                        self._save_lm_log(text, topic, False, last_prompt, attempt=attempts)
                    except Exception as e:
                        self.logger.warning(f"Failed to get shorter content on attempt {attempts}: {e}")
                        if attempts == max_attempts:
                            self.logger.warning("Max attempts reached, truncating content")
                            text = text[:self.TELEGRAM_LIMIT-10] + "..."
                            break
                if len(text) <= self.TELEGRAM_LIMIT:
                    self.logger.info(f"Successfully got shorter content ({len(text)} chars)")

            # Логируем успешный результат в success, только если текст валидный и короткий
            if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                self._save_lm_log(text, topic, True, last_prompt, attempt=99)
                # В историю только финальный вариант
                self.add_to_history(last_prompt, text)
            else:
                self.logger.warning("Generated content is too long or empty, not saving to success log.")

            return text.strip() if text else ""
        finally:
            self.max_tokens = original_max_tokens

    def generate_with_retry(self, prompt_template: str, topic: str, context: str, max_retries: int = 3) -> str:
        last_err = None
        original_context = context
        final_text = ""
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"LMStudioClient: generation attempt {attempt}/{max_retries}")
                text = self.generate_content(prompt_template, topic, context)
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
                    break
                self.logger.warning(f"LM Studio generation returned empty or too long text (attempt {attempt})")
            except Exception as e:
                last_err = e
                msg = str(e)
                self.logger.warning(f"LMStudioClient: error on attempt {attempt}: {msg}")
                if "413" in msg or "400" in msg or "payload" in msg:
                    context = context[:max(100, len(context) // 2)]
                    self.logger.warning("Reducing context and retrying...")
                if attempt == max_retries or (self.history_limit and len(self.history) > self.history_limit * 4):
                    self.logger.warning("Clearing conversation history due to repeated failures.")
                    self.clear_conversation_history()
        # Если ни одна попытка не прошла, делаем финальный fallback
        if not final_text:
            try:
                text = self.generate_content(prompt_template, topic, original_context[:256])
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
            except Exception as e:
                self.logger.error("Final fallback attempt failed", exc_info=True)
                raise ValueError(f"LM Studio did not generate content after {max_retries} attempts: {last_err}")
        # В любом случае, возвращаем только последний валидный текст
        return final_text.strip() if final_text else ""




# участок кода файла prompt_builder.py
import os
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class PromptBuilder:
    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]
    PLACEHOLDER_PATTERN = re.compile(r"\{[A-Z_]+\}")

    def __init__(self, prompt_folders: List[str]):
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self._last_prompt_template: Optional[str] = None
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                self.templates[str(folder)] = []
                continue
            self.templates[str(folder)] = self._scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def _scan_prompt_folder(self, folder_path: Path) -> List[str]:
        return [str(p) for p in folder_path.glob("*.txt")]

    def _select_random_templates(self) -> List[str]:
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            selected.append(random.choice(templates) if templates else None)
        return selected

    def _read_template_file(self, file_path: Optional[str]) -> str:
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            msg = f"Failed to read prompt file: {file_path}"
            self.logger.error(msg, exc_info=True)
            return ""

    def _validate_prompt_structure(self, template: str) -> None:
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            raise ValueError(f"Prompt missing required placeholders: {missing}")
        for ph in self.REQUIRED_PLACEHOLDERS:
            if ph * 2 in template:
                self.logger.warning(f"Prompt contains duplicated placeholder: {ph}{ph}")
        all_placeholders = set(self.PLACEHOLDER_PATTERN.findall(template))
        supported = set(self.REQUIRED_PLACEHOLDERS)
        unsupported = [ph for ph in all_placeholders if ph not in supported]
        if unsupported:
            self.logger.warning(f"Prompt contains unsupported placeholders: {unsupported}")

    def _find_unresolved_placeholders(self, text: str) -> List[str]:
        return list(set(self.PLACEHOLDER_PATTERN.findall(text)))

    def _replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def build_prompt(self, topic: str, context: str) -> Tuple[str, str]:
        if not topic or not isinstance(topic, str):
            raise ValueError("Topic for prompt_builder is empty or not a string.")
        if not context or not isinstance(context, str):
            raise ValueError("Context for prompt_builder is empty or not a string.")

        template_paths = self._select_random_templates()
        template_texts = [self._read_template_file(path) for path in template_paths if path]
        if not any(template_texts):
            prompt_template = self._default_template()
            self.logger.warning("No prompt templates found, using default.")
        else:
            prompt_template = "\n\n".join(filter(None, template_texts)).strip()

        self._last_prompt_template = prompt_template
        self._validate_prompt_structure(prompt_template)

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
        }
        prompt = self._replace_placeholders(prompt_template, replacements)

        unresolved = self._find_unresolved_placeholders(prompt)
        critical_unresolved = [ph for ph in unresolved if ph in self.REQUIRED_PLACEHOLDERS]
        if critical_unresolved:
            raise ValueError(f"Prompt contains unresolved placeholders after replacement: {critical_unresolved}")

        return prompt, prompt_template

    def _default_template(self) -> str:
        return (
            "Ты опытный механик грузовой техники с 15-летним стажем работы.\n"
            "Работал в крупных автопарках, ремонтировал краны, фургоны, бортовые машины. "
            "Знаешь все подводные камни эксплуатации. Говоришь простым языком, приводишь примеры из практики.\n\n"
            "Тема: {TOPIC}\n\n"
            "Контекст для анализа: {CONTEXT}"
        )




# участок файла config.json

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
