# участок кода файла main.py

import sys
import signal
import time
import logging
import os

from modules.utils.config_manager import ConfigManager
from modules.utils.logs import get_logger, log_system_info
from modules.utils.state_manager import StateManager
from modules.utils.media_handler import MediaHandler
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

class RAGIngestionService:
    def __init__(self, rag_retriever, logger):
        self.rag_retriever = rag_retriever
        self.logger = logger

    def build_knowledge_base(self, folder):
        try:
            self.rag_retriever.process_inform_folder(folder)
            self.rag_retriever.build_knowledge_base()
            self.logger.info("Initial RAG knowledge base built.")
        except Exception as e:
            self.logger.error("Knowledge base build failed", exc_info=True)
            raise

class TelegramRAGSystem:
    TELEGRAM_TEXT_LIMIT = 4096
    TELEGRAM_CAPTION_LIMIT = 1024

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
            self.ingestion = RAGIngestionService(self.rag_retriever, self.logger)

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
            self.media_handler = MediaHandler(
                media_folder=self.config["paths"].get("media_dir", "media"),
                config=self.config
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
                # 1. Поиск контекста в RAG
                rag_context = self.rag_retriever.retrieve_context(topic)
                if rag_context is None:
                    self.logger.warning(f"RAG context retrieval returned None for topic: {topic}")
                    raise ValueError("RAG context retrieval returned None")
                if not isinstance(rag_context, str) or not rag_context.strip():
                    self.logger.warning(f"RAG context is empty for topic: {topic}")

                # 2. Web-поиск дополнительной информации
                web_results = self.web_search.search(topic)
                if web_results is None:
                    self.logger.warning(f"Web search returned None for topic: {topic}")
                    raise ValueError("Web search returned None")
                web_context = self.web_search.extract_content(web_results)
                if not isinstance(web_context, str) or not web_context.strip():
                    self.logger.warning(f"Web context is empty for topic: {topic}")

                # 3. Объединение контекстов
                full_context = self.combine_contexts(rag_context, web_context)
                self.logger.debug(f"[{topic}] full_context length: {len(full_context)}, preview: {full_context[:300]}")

                # 4. Выбор медиафайла (опционально)
                media_file = None
                try:
                    media_file = self.media_handler.get_random_media_file()
                    if media_file and not self.media_handler.validate_media_file(media_file):
                        self.logger.warning(f"Media file {media_file} is not valid. Skipping media.")
                        media_file = None
                except Exception as e:
                    self.logger.warning(f"Media handler error: {str(e)}")

                # 5. Сборка промпта с распаковкой кортежа (prompt, has_uploadfile, prompt_template)
                # PromptBuilder должен возвращать prompt, has_uploadfile, prompt_template
                prompt, has_uploadfile, prompt_template = self.prompt_builder.build_prompt(
                    topic=topic,
                    context=full_context,
                    media_file=media_file
                )

                # 6. Обрезка context для Telegram (отдельно для Telegram, не для LM Studio!)
                max_context_length = self.TELEGRAM_CAPTION_LIMIT if has_uploadfile else self.TELEGRAM_TEXT_LIMIT
                if len(full_context) > max_context_length:
                    self.logger.warning(
                        f"Context too long ({len(full_context)} > {max_context_length}) for topic: {topic}. Truncating."
                    )
                    full_context = full_context[:max_context_length]

                if not prompt or not prompt.strip():
                    self.logger.error(f"Prompt building failed (empty) for topic '{topic}'.")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "Prompt building failed (empty prompt)")
                    continue

                self.logger.debug(f"Prompt to LM Studio for topic '{topic}':\n{prompt[:1000]}")

                # 7. Генерация контента LM Studio (теперь передаем prompt_template, topic, context, media_file)
                max_lm_retries = self.config["lm_studio"].get("max_retries", 3)
                try:
                    content = self.lm_client.generate_with_retry(
                        prompt_template,
                        topic,
                        full_context,
                        media_file,
                        max_retries=max_lm_retries,
                        with_media=has_uploadfile
                    )
                except Exception as e:
                    self.logger.error(f"LM Studio generation failed after retries for topic '{topic}': {e}")
                    self.logger.error(f"Prompt (truncated): {prompt[:1000]}")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, f"LM Studio generation failed: {e}")
                    continue

                self.logger.debug(f"LM Studio response for topic '{topic}': {content[:1000]}")
                if not content or not content.strip():
                    self.logger.error(f"Generated content is empty for topic '{topic}'.")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "Generated content is empty")
                    continue

                # 8. Валидация и подготовка к Telegram
                try:
                    validated_content = self.content_validator.validate_content(
                        content,
                        has_media=has_uploadfile
                    )
                except Exception as e:
                    self.logger.error(f"Content validation failed for topic '{topic}': {e}")
                    self.logger.error(f"Raw content: {content[:1000]}")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, f"Content validation failed: {e}")
                    continue

                if not validated_content or not validated_content.strip():
                    self.logger.error(f"Validated content is empty for topic '{topic}'.")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "Validated content is empty")
                    continue

                # 9. Публикация в Telegram
                success = False
                max_retries = self.config["telegram"].get("max_retries", 3)
                for attempt in range(1, max_retries + 1):
                    try:
                        if has_uploadfile and media_file:
                            success = self.telegram_client.send_media_message(validated_content, media_file)
                        else:
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
            self.ingestion.build_knowledge_base(self.config["rag"].get("inform_folder", "inform/"))
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


# участок кода файла lm_client.py

import logging
import requests
from typing import Dict, Any, List, Optional

class LMStudioClient:
    """
    Клиент для взаимодействия с LM Studio (локальная LLM).
    Контролирует лимит payload (prompt + history + system_message) для LM Studio.
    Не занимается Telegram-валидированием и не режет по лимитам Telegram — только лимит самой LLM!
    """

    LM_MAX_TOTAL_CHARS = 20000  # Лимит для LM Studio (под свою модель!)

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

    def _validate_config(self):
        assert isinstance(self.max_tokens, int) and self.max_tokens > 0, "max_tokens must be positive integer"
        assert isinstance(self.LM_MAX_TOTAL_CHARS, int) and self.LM_MAX_TOTAL_CHARS > 1000, "LM_MAX_TOTAL_CHARS must be > 1000"
        assert isinstance(self.temperature, (float, int)), "temperature must be float"
        if self.top_p is not None:
            assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0,1]"
        if self.top_k is not None:
            assert isinstance(self.top_k, int) and self.top_k >= 0, "top_k must be non-negative int"

    def _check_health_on_init(self):
        try:
            status = self.health_check()
            if status.get("status") != "ok":
                raise RuntimeError(f"LM Studio health check failed: {status}")
            self.logger.info(f"LMStudioClient connected to model '{self.model}'. Health OK.")
        except Exception as e:
            self.logger.error(f"LM Studio health check failed: {e}")
            raise

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unreachable"}

    def clear_conversation_history(self):
        self.history = []
        self.logger.debug("LMStudioClient: conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        if user_message and isinstance(user_message, str) and user_message.strip():
            self.history.append({"role": "user", "content": user_message})
        if bot_message and isinstance(bot_message, str) and bot_message.strip():
            self.history.append({"role": "assistant", "content": bot_message})
        # Trim to the last N exchanges
        if self.history_limit > 0 and len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def _clean_history(self) -> List[Dict[str, str]]:
        """
        Возвращает только валидные последние сообщения, проверяет структуру.
        """
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

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str, media_file: Optional[str]) -> str:
        """
        Подбирает длину context так, чтобы prompt + history + system_message влезли в LM Studio лимит.
        Обрезает context, если надо.
        """
        # Подставляем все плейсхолдеры кроме {CONTEXT}
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": "CONTEXT_PLACEHOLDER",
            "{UPLOADFILE}": media_file.strip() if media_file else "",
        }
        prompt_wo_context = prompt_template
        for key, value in replacements.items():
            prompt_wo_context = prompt_wo_context.replace(key, value)
        prompt_wo_context_len = len(prompt_wo_context.replace("CONTEXT_PLACEHOLDER", ""))

        # Считаем размер истории и system_message
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

    def _build_messages(self, prompt_template: str, topic: str, context: str, media_file: Optional[str]) -> List[Dict[str, str]]:
        """
        Собирает сообщения для LLM API, применяет лимиты.
        """
        context = self._truncate_context_for_llm(prompt_template, topic, context, media_file)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context,
            "{UPLOADFILE}": media_file.strip() if media_file else "",
        }
        prompt = prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        prompt = prompt.replace("nan", "").strip()
        # Формируем историю
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        messages.append({"role": "user", "content": prompt})
        # Ещё раз проверяем лимит
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > self.LM_MAX_TOTAL_CHARS:
            self.logger.warning(f"Total LLM payload too long ({total_chars} > {self.LM_MAX_TOTAL_CHARS}), trimming prompt/history")
            # Обрезаем prompt
            excess = total_chars - self.LM_MAX_TOTAL_CHARS
            if len(messages[-1]["content"]) > excess:
                messages[-1]["content"] = messages[-1]["content"][:len(messages[-1]["content"]) - excess]
            else:
                # Убираем старые истории
                while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                    removed = messages.pop(1)  # после system_message
                    self.logger.warning(f"Removed old history message to fit LM payload: {removed}")
                    total_chars = sum(len(m["content"]) for m in messages)
                if total_chars > self.LM_MAX_TOTAL_CHARS:
                    last = messages[-1]
                    last["content"] = last["content"][:max(0, len(last["content"]) - (total_chars - self.LM_MAX_TOTAL_CHARS))]
        return messages

    def generate_content(
        self,
        prompt_template: str,
        topic: str,
        context: str,
        media_file: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Генерирует текст с помощью LM Studio, ограничивает весь payload до LM_MAX_TOTAL_CHARS.
        Возвращает строку или бросает ValueError.
        """
        max_tokens = max_tokens or self.max_tokens
        messages = self._build_messages(prompt_template, topic, context, media_file)
        chat_url = f"{self.base_url}/v1/chat/completions"
        payload_chat = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        if self.top_p is not None:
            payload_chat["top_p"] = self.top_p
        if self.top_k is not None:
            payload_chat["top_k"] = self.top_k

        self.logger.debug(f"LMStudioClient: Sending chat payload to {chat_url}: {str(payload_chat)[:800]}")

        try:
            response = requests.post(chat_url, json=payload_chat, timeout=self.timeout)
            self.logger.debug(f"LMStudioClient: raw response: {response.text[:1000]}")
            response.raise_for_status()
            try:
                result = response.json()
            except Exception as e:
                self.logger.error("Failed to decode LM Studio response as JSON", exc_info=True)
                result = {}
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Fallback если пусто или endpoint не поддержан
            if not text:
                self.logger.warning("Empty chat response, fallback to completions endpoint.")
                comp_url = f"{self.base_url}/v1/completions"
                payload = {
                    "model": self.model,
                    "prompt": messages[-1]['content'],
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "stream": False
                }
                if self.top_p is not None:
                    payload["top_p"] = self.top_p
                if self.top_k is not None:
                    payload["top_k"] = self.top_k
                comp_resp = requests.post(comp_url, json=payload, timeout=self.timeout)
                comp_resp.raise_for_status()
                try:
                    comp_result = comp_resp.json()
                except Exception as e:
                    self.logger.error("Failed to decode completions response as JSON", exc_info=True)
                    comp_result = {}
                text = comp_result.get("choices", [{}])[0].get("text", "")

            if text and text.strip():
                self.add_to_history(messages[-1]['content'], text)
            else:
                self.logger.warning("LM Studio returned empty text from both endpoints.")
            if not isinstance(text, str):
                raise ValueError("LM Studio returned non-string result.")
            return text.strip()
        except Exception as e:
            self.logger.error("Content generation failed on both endpoints", exc_info=True)
            self.clear_conversation_history()
            raise ValueError(f"LM Studio content generation failed: {e}")

    def generate_with_retry(
        self,
        prompt_template: str,
        topic: str,
        context: str,
        media_file: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Автоматический повтор генерации при ошибках, с учётом лимита на payload.
        При ошибке 400 — уменьшает context и очищает историю.
        """
        last_err = None
        original_context = context
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"LMStudioClient: generation attempt {attempt}/{max_retries}")
                text = self.generate_content(prompt_template, topic, context, media_file)
                if text and text.strip():
                    return text
                self.logger.warning(f"LM Studio generation returned empty text (attempt {attempt})")
            except Exception as e:
                last_err = e
                msg = str(e)
                self.logger.warning(f"LMStudioClient: error on attempt {attempt}: {msg}")
                # Если payload слишком большой — сокращаем context
                if "413" in msg or "400" in msg or "payload" in msg:
                    context = context[:max(100, len(context) // 2)]
                    self.logger.warning("Reducing context and retrying...")
                # При повторяющейся ошибке — сбрасываем историю
                if attempt == max_retries or (self.history_limit and len(self.history) > self.history_limit * 4):
                    self.logger.warning("Clearing conversation history due to repeated failures.")
                    self.clear_conversation_history()
        # Последняя попытка — с минимальным context и чистой историей
        try:
            return self.generate_content(prompt_template, topic, original_context[:256], media_file)
        except Exception as e:
            self.logger.error("Final fallback attempt failed", exc_info=True)
            raise ValueError(f"LM Studio did not generate content after {max_retries} attempts: {last_err}")

    def set_generation_parameters(self, temperature: float, top_p: Optional[float] = None, top_k: Optional[int] = None):
        assert 0.0 <= temperature <= 2.0, "temperature must be in [0,2]"
        self.temperature = temperature
        if top_p is not None:
            assert 0.0 <= top_p <= 1.0, "top_p must be in [0,1]"
            self.top_p = top_p
        if top_k is not None:
            assert isinstance(top_k, int) and top_k >= 0, "top_k must be non-negative int"
            self.top_k = top_k
        self.logger.info(f"Set generation parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}")

    def get_generation_stats(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "history_limit": self.history_limit,
            "system_message": self.system_message,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

# участок кода файла prompt_builder.py

import os
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class PromptBuilder:
    """
    Сборка промпта из шаблонов с жестким контролем структуры, подстановки плейсхолдеров,
    и надежным возвратом данных для всей цепочки генерации.
    """

    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]
    OPTIONAL_PLACEHOLDERS = ["{UPLOADFILE}"]
    PLACEHOLDER_PATTERN = re.compile(r"\{[A-Z_]+\}")

    def __init__(self, prompt_folders: List[str]):
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self.error_history: List[Dict] = []
        self._last_prompt_template: Optional[str] = None
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        """Загружает шаблоны из всех указанных папок, сохраняет пути."""
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
        """Из каждой папки выбирает случайный шаблон. Если нет файлов — None."""
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
            self._log_error(msg, exc=e)
            return ""

    def _validate_prompt_structure(self, template: str) -> None:
        """Проверяет наличие всех обязательных плейсхолдеров. Кидает ошибку если невалидно."""
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            msg = f"Prompt missing required placeholders: {missing}"
            self._log_error(msg)
            self.logger.error(msg)
            raise ValueError(msg)
        # Логируем дубли, не валим работу
        for ph in self.REQUIRED_PLACEHOLDERS + self.OPTIONAL_PLACEHOLDERS:
            if ph * 2 in template:
                self.logger.warning(f"Prompt contains duplicated placeholder: {ph}{ph}")
        # Проверка на неизвестные плейсхолдеры
        all_placeholders = set(self.PLACEHOLDER_PATTERN.findall(template))
        supported = set(self.REQUIRED_PLACEHOLDERS + self.OPTIONAL_PLACEHOLDERS)
        unsupported = [ph for ph in all_placeholders if ph not in supported]
        if unsupported:
            self.logger.warning(f"Prompt contains unsupported placeholders: {unsupported}")

    def _find_unresolved_placeholders(self, text: str) -> List[str]:
        return list(set(self.PLACEHOLDER_PATTERN.findall(text)))

    def _compact_whitespace(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)

    def _replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def _has_uploadfile_placeholder(self, template: str) -> bool:
        return "{UPLOADFILE}" in template

    def build_prompt(
        self,
        topic: str,
        context: str,
        media_file: Optional[str] = None
    ) -> Tuple[str, bool, str]:
        """
        Возвращает:
            - prompt (str): итоговый текст промпта без неразрешённых плейсхолдеров
            - has_uploadfile (bool): True, если в шаблоне был {UPLOADFILE} и он был заменён
            - prompt_template (str): оригинальный текст шаблона (до подстановки)
        """
        # 1. Проверка входных данных
        if not topic or not isinstance(topic, str):
            msg = "Topic for prompt_builder is empty or not a string."
            self._log_error(msg, context={"topic": topic})
            self.logger.error(msg)
            raise ValueError(msg)
        if not context or not isinstance(context, str):
            msg = "Context for prompt_builder is empty or not a string."
            self._log_error(msg, context={"context": context})
            self.logger.error(msg)
            raise ValueError(msg)

        # 2. Выбор шаблонов
        template_paths = self._select_random_templates()
        template_texts = [self._read_template_file(path) for path in template_paths if path]
        # Если все шаблоны пусты — дефолт
        if not any(template_texts):
            prompt_template = self._default_template()
            self.logger.warning("No prompt templates found, using default.")
        else:
            prompt_template = "\n\n".join(filter(None, template_texts)).strip()

        self._last_prompt_template = prompt_template

        # 3. Проверка структуры ДО подстановки
        self._validate_prompt_structure(prompt_template)

        # 4. Определяем, нужен ли uploadfile
        has_uploadfile = self._has_uploadfile_placeholder(prompt_template)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
            "{UPLOADFILE}": media_file.strip() if (has_uploadfile and media_file) else "",
        }
        prompt = self._replace_placeholders(prompt_template, replacements)

        # 5. Проверка после подстановки: не осталось ли плейсхолдеров
        unresolved = self._find_unresolved_placeholders(prompt)
        critical_unresolved = [
            ph for ph in unresolved if
            ph in self.REQUIRED_PLACEHOLDERS or
            (ph == "{UPLOADFILE}" and has_uploadfile)
        ]
        if critical_unresolved:
            msg = f"Prompt contains unresolved placeholders after replacement: {critical_unresolved}"
            self._log_error(msg, context={"prompt": prompt, "replacements": replacements})
            self.logger.error(msg)
            raise ValueError(msg)

        # 6. Очистка whitespace
        prompt = self._compact_whitespace(prompt)

        self.logger.debug(
            f"PromptBuilder: topic='{topic[:100]}', context_len={len(context)}, media_file='{media_file}', has_uploadfile={has_uploadfile}")
        self.logger.debug(f"PromptBuilder: template_paths={template_paths}")
        self.logger.debug(f"PromptBuilder: prompt after replacement (truncated): {prompt[:1000]}")

        return prompt, has_uploadfile, prompt_template

    def _log_error(self, msg: str, context: Optional[dict] = None, exc: Exception = None):
        entry = {
            "message": msg,
            "context": context,
            "exception": repr(exc) if exc else None
        }
        self.error_history.append(entry)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

    def get_error_history(self) -> List[Dict]:
        return self.error_history.copy()

    def check_placeholder_presence(self, template: str) -> Dict[str, bool]:
        all_ph = self.REQUIRED_PLACEHOLDERS + self.OPTIONAL_PLACEHOLDERS
        return {ph: (ph in template) for ph in all_ph}

    def get_template_stats(self) -> Dict[str, int]:
        return {folder: len(templates) for folder, templates in self.templates.items()}

    def reload_templates(self) -> None:
        self.logger.info("Reloading prompt templates...")
        self.load_prompt_templates()

    def test_template_combination(self, topic: str, context: str) -> Tuple[str, bool, str]:
        return self.build_prompt(topic, context, media_file="media/sample.jpg")

    def _default_template(self) -> str:
        return (
            "Ты опытный механик грузовой техники с 15-летним стажем работы.\n"
            "Работал в крупных автопарках, ремонтировал краны, фургоны, бортовые машины. "
            "Знаешь все подводные камни эксплуатации. Говоришь простым языком, приводишь примеры из практики.\n\n"
            "Тема: {TOPIC}\n\n"
            "Контекст для анализа: {CONTEXT}\n"
            "{UPLOADFILE}"
        )

    @property
    def last_prompt_template(self) -> Optional[str]:
        return self._last_prompt_template
