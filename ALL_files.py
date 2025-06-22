# main.py

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
    """
    Сервис отслеживания статистики обработки тем, публикаций и ошибок.
    """
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
    """
    Сервис построения и обновления базы знаний RAG.
    """
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
    """
    Главный класс системы автопостинга с RAG и LM Studio.
    Управляет жизненным циклом, координирует обработку, отвечает за интеграцию компонентов.
    """
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
        self.autoload_topics()  # --- Добавлено для автозагрузки тем ---

    def setup_logging(self):
        log_system_info(self.logger)

    def validate_configuration(self):
        if not self.config_manager.validate_config():
            self.logger.critical("Configuration validation failed.")
            sys.exit(1)
        self.logger.info("Configuration validated successfully.")

    def initialize_services(self):
        try:
            # RAG и состояние
            self.rag_retriever = RAGRetriever(config=self.config["rag"])
            self.state_manager = StateManager(state_file="data/state.json")
            self.monitoring = MonitoringService(self.logger)
            self.ingestion = RAGIngestionService(self.rag_retriever, self.logger)

            # Генерация контента
            self.lm_client = LMStudioClient(
                base_url=self.config["lm_studio"]["base_url"],
                model=self.config["lm_studio"]["model"],
                config=self.config["lm_studio"]
            )
            self.prompt_builder = PromptBuilder(prompt_folders=self.config["paths"].get("prompt_folders", [
                "data/prompt_1", "data/prompt_2", "data/prompt_3"
            ]))
            self.content_validator = ContentValidator(config=self.config)

            # Веб-поиск и медиа
            serper_api_key = self.config_manager.get_serper_api_key()
            serper_endpoint = self.config_manager.get_config_value("serper.endpoint", "https://google.serper.dev/search")
            serper_results_limit = self.config_manager.get_config_value("serper.results_limit", 10)
            self.web_search = WebSearchClient(
                api_key=serper_api_key,
                endpoint=serper_endpoint,
                results_limit=serper_results_limit
            )
            self.media_handler = MediaHandler(media_folder=self.config["paths"].get("media_dir", "media"),
                                             config=self.config)

            # Telegram клиент
            token = self.config_manager.get_telegram_token()
            channel_id = self.config_manager.get_telegram_channel_id()
            self.telegram_client = TelegramClient(token=token, channel_id=channel_id, config=self.config["telegram"])

        except Exception as e:
            self.logger.critical("Component initialization failed", exc_info=True)
            sys.exit(1)

    def autoload_topics(self):
        """
        Автоматически загружает темы из data/topics.txt и добавляет их в StateManager,
        если они ещё не обработаны или не находятся в ошибочных.
        """
        topics_file = "data/topics.txt"
        if not os.path.isfile(topics_file):
            self.logger.warning(f"Topics file not found: {topics_file}")
            return

        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                topics = [line.strip() for line in f if line.strip()]
            # Собираем все уже известные темы
            existing = set(self.state_manager.get_unprocessed_topics() +
                           self.state_manager.get_processed_topics() +
                           self.state_manager.get_failed_topics())
            # Фильтруем только новые темы
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
                    raise ValueError("RAG context retrieval returned None")
                if not rag_context.strip():
                    self.logger.warning(f"RAG context is empty for topic: {topic}")

                # 2. Web-поиск дополнительной информации
                web_results = self.web_search.search(topic)
                if web_results is None:
                    raise ValueError("Web search returned None")
                web_context = self.web_search.extract_content(web_results)
                if not web_context.strip():
                    self.logger.warning(f"Web context is empty for topic: {topic}")

                # 3. Объединение контекстов
                full_context = self.combine_contexts(rag_context, web_context)
                if not full_context.strip():
                    raise ValueError("Combined context is empty")

                # 4. Выбор медиафайла (опционально)
                media_file = None
                try:
                    media_file = self.media_handler.get_random_media_file()
                    if media_file and not self.media_handler.validate_media_file(media_file):
                        self.logger.warning(f"Media file {media_file} is not valid. Skipping media.")
                        media_file = None
                except Exception as e:
                    self.logger.warning(f"Media handler error: {str(e)}")

                # 5. Сборка промпта
                prompt = self.prompt_builder.build_prompt(
                    topic=topic,
                    context=full_context,
                    media_file=media_file
                )
                if not prompt or not prompt.strip():
                    raise ValueError("Prompt building failed (empty prompt)")

                # 6. Генерация контента
                content = self.lm_client.generate_content(prompt)
                if not content or not content.strip():
                    raise ValueError("Generated content is empty")

                # 7. Валидация контента
                validated_content = self.content_validator.validate_content(content, has_media=bool(media_file))
                if not validated_content or not validated_content.strip():
                    raise ValueError("Validated content is empty")

                # 8. Публикация в Telegram с повтором (retry)
                success = False
                max_retries = self.config["telegram"].get("max_retries", 3)
                for attempt in range(1, max_retries + 1):
                    try:
                        if media_file:
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
                time.sleep(self.config["telegram"]["post_interval"])

            except Exception as e:
                self.handle_error(topic, e)
                continue

    def run(self):
        self.logger.info("System starting up...")
        # Корректная обработка shutdown-сигналов
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

        # Построение базы знаний RAG
        try:
            self.ingestion.build_knowledge_base(self.config["rag"].get("inform_folder", "inform/"))
        except Exception as e:
            self.logger.critical(f"Failed to build RAG knowledge base: {e}", exc_info=True)
            sys.exit(1)
        # Основной цикл
        self.main_processing_loop()
        self.logger.info("System shut down gracefully.")

if __name__ == "__main__":
    system = TelegramRAGSystem()
    system.run()

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

# lm_client.py

import logging
import requests
from typing import Dict, Any, List, Optional

class LMStudioClient:
    def __init__(self, base_url: str, model: str, config: Dict[str, Any]):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_chars = config.get("max_chars", 20000)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        self.history_limit = config.get("history_limit", 3)
        self.system_message = config.get("system_message", None)
        self.max_chars_with_media = config.get("max_chars_with_media", 4096)
        self.logger = logging.getLogger("LMStudioClient")
        self.history: List[Dict[str, str]] = []

        if self.system_message and len(self.system_message) > 1000:
            self.logger.warning("System message unusually long.")

        self.logger.info(f"LMStudioClient initialized with model '{model}' and config: {config}")

    def check_connection(self) -> bool:
        """
        Проверка доступности LM Studio и наличия нужной модели.
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            models = response.json().get("data", [])
            is_ok = any(m["id"] == self.model for m in models)
            self.logger.info("LM Studio connection OK" if is_ok else f"Model '{self.model}' not found in LM Studio")
            return is_ok
        except Exception as e:
            self.logger.error("Failed to connect to LM Studio", exc_info=True)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получить json-информацию о моделях LM Studio.
        """
        try:
            resp = requests.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error("Model info retrieval failed", exc_info=True)
            return {}

    def generate_content(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        with_media: bool = False
    ) -> str:
        """
        Основной метод генерации текста.
        Сначала пытается chat endpoint, если неудача - fallback на обычный completions endpoint.
        """
        max_tokens = max_tokens or self.max_tokens

        # Compose prompt with system message
        full_prompt = ""
        if self.system_message:
            full_prompt += f"{self.system_message}\n"
        full_prompt += prompt

        # Попробовать chat endpoint первым
        url_chat = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload_chat = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url_chat, json=payload_chat, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            result = response.json()
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Fallback если пусто или chat endpoint не поддерживается
            if not text:
                self.logger.warning("Empty response from chat endpoint, fallback to completions endpoint")
                url_comp = f"{self.base_url}/v1/completions"
                payload_comp = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "stream": False
                }
                response = requests.post(url_comp, json=payload_comp, timeout=self.timeout, headers=headers)
                response.raise_for_status()
                result = response.json()
                text = result.get("choices", [{}])[0].get("text", "")

            limit = self.max_chars_with_media if with_media else self.max_chars
            if not self.validate_response_length(text, limit):
                self.logger.warning(f"Truncating response to {limit} chars")
                text = text[:limit]
            return text.strip()
        except Exception as e:
            self.logger.error("Content generation failed", exc_info=True)
            raise

    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        with_media: bool = False
    ) -> str:
        """
        Повторяет генерацию несколько раз в случае ошибки.
        """
        for attempt in range(1, max_retries + 1):
            try:
                return self.generate_content(prompt, with_media=with_media)
            except Exception as e:
                self.logger.warning(f"Retry {attempt}/{max_retries} for prompt generation")
                if attempt == max_retries:
                    raise

    def validate_response_length(self, text: str, max_length: int = None) -> bool:
        """
        Проверяет, не превышает ли текст максимальную длину.
        """
        if max_length is None:
            max_length = self.max_chars
        if len(text) <= max_length:
            return True
        self.logger.warning(f"Response exceeds max length ({len(text)} > {max_length})")
        return False

    def request_shorter_version(self, original_prompt: str, current_length: int, target_length: int) -> str:
        """
        Просит LM Studio сделать текст короче, если превышен лимит.
        """
        instruction = (
            f"\n\n[ИНСТРУКЦИЯ]: Сократи предыдущий текст до {target_length} символов. "
            "Сохрани ключевые идеи и структуру, но сделай более компактным."
        )
        new_prompt = original_prompt + instruction
        return self.generate_with_retry(new_prompt)

    def estimate_tokens(self, text: str) -> int:
        """
        Грубая оценка числа токенов (1 токен ≈ 4 символа).
        """
        return int(len(text) / 4)

    def set_generation_parameters(self, temperature: float, top_p: float = None, top_k: int = None):
        """
        Устанавливает параметры генерации.
        """
        self.temperature = temperature
        self.logger.info(f"Set generation temperature to {temperature}")
        if top_p:
            self.logger.info(f"Set top_p to {top_p}")
        if top_k:
            self.logger.info(f"Set top_k to {top_k}")

    def get_generation_stats(self) -> dict:
        """
        Возвращает параметры генерации (для мониторинга/отладки).
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_chars": self.max_chars,
            "max_chars_with_media": self.max_chars_with_media,
            "history_limit": self.history_limit,
            "system_message": self.system_message
        }

    def clear_conversation_history(self):
        """
        Очищает историю диалога.
        """
        self.history = []
        self.logger.debug("Conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        """
        Добавляет сообщения в историю диалога.
        """
        self.history.append({"user": user_message, "bot": bot_message})
        if self.history_limit and len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

    def health_check(self) -> dict:
        """
        Проверяет работоспособность сервера LM Studio.
        """
        try:
            resp = requests.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.warning("Health check failed", exc_info=True)
            return {"status": "unreachable"}

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class PromptBuilder:
    # Обязательные плейсхолдеры, которые должны встречаться в итоговом склеенном шаблоне до подстановки
    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]

    def __init__(self, prompt_folders: List[str]):
        """
        Инициализация PromptBuilder.
        :param prompt_folders: список путей к папкам с шаблонами промптов.
        """
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        """
        Загружает все шаблоны из указанных папок в self.templates.
        Если папка не существует — пишет warning.
        """
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                continue
            self.templates[str(folder)] = self.scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def scan_prompt_folder(self, folder_path: Path) -> List[str]:
        """
        Возвращает список путей к .txt-файлам в папке.
        """
        return [str(p) for p in folder_path.glob("*.txt")]

    def select_random_templates(self) -> List[str]:
        """
        Случайно выбирает по одному шаблону из каждой папки.
        Если в папке нет шаблонов — вместо файла будет пустая строка.
        """
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            if templates:
                selected.append(random.choice(templates))
            else:
                selected.append("")
        return selected

    def read_template_file(self, file_path: str) -> str:
        """
        Читает содержимое файла шаблона.
        Если файл не читается — пишет ошибку в лог, возвращает пустую строку.
        """
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read prompt file: {file_path}", exc_info=True)
            return ""

    def build_prompt(self, topic: str, context: str, media_file: Optional[str] = None) -> str:
        """
        Собирает итоговый промпт:
        - Случайно выбирает по одному шаблону из каждой папки
        - Склеивает их содержимое
        - Проверяет, что до подстановки есть все обязательные плейсхолдеры
        - Подставляет значения плейсхолдеров
        :param topic: Тема для подстановки
        :param context: Контекст для подстановки
        :param media_file: Путь к медиафайлу (необязательно)
        :return: Сгенерированный промпт
        :raises ValueError: если не хватает плейсхолдеров
        """
        template_paths = self.select_random_templates()
        template_texts = [self.read_template_file(path) for path in template_paths]
        content = "\n\n".join(template_texts)

        # Критически важно: Проверяем до замены — все ли плейсхолдеры есть в склеенном шаблоне
        if not self.validate_prompt_structure(content):
            self.logger.error("Prompt structure validation failed. Missing required placeholders in templates.")
            raise ValueError("Prompt structure validation failed. Missing required placeholders in templates.")

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
            "{UPLOADFILE}": media_file.strip() if media_file else ""
        }

        prompt = self.replace_placeholders(content, replacements)

        # Необязательная проверка: не остались ли неиспользованные плейсхолдеры (help debug)
        unused = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph in prompt]
        if unused:
            self.logger.warning(f"Prompt still contains unused placeholders: {unused}")

        return prompt

    def replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        """
        Заменяет все плейсхолдеры на их значения.
        """
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def validate_prompt_structure(self, template: str) -> bool:
        """
        Проверяет, что в шаблоне (до замены) есть все обязательные плейсхолдеры.
        """
        for required in self.REQUIRED_PLACEHOLDERS:
            if required not in template:
                self.logger.error(f"Prompt missing required placeholder: {required}")
                return False
        return True

    def check_placeholder_presence(self, template: str) -> Dict[str, bool]:
        """
        Возвращает словарь вида {плейсхолдер: True/False} — есть ли плейсхолдер в шаблоне.
        """
        all_ph = self.REQUIRED_PLACEHOLDERS + ["{UPLOADFILE}"]
        return {ph: (ph in template) for ph in all_ph}

    def get_template_stats(self) -> Dict[str, int]:
        """
        Возвращает статистику (количество шаблонов) по каждой папке.
        """
        return {folder: len(templates) for folder, templates in self.templates.items()}

    def reload_templates(self) -> None:
        """
        Перезагружает шаблоны из файловой системы.
        """
        self.logger.info("Reloading prompt templates...")
        self.load_prompt_templates()

    def test_template_combination(self, topic: str, context: str) -> str:
        """
        Генерирует промпт с тестовым медиафайлом (для отладки).
        """
        return self.build_prompt(topic, context, media_file="media/sample.jpg")


# modules/utils/config_manager.py

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

class ConfigManager:
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = {}
        self.logger = logging.getLogger("ConfigManager")
        self.load_config()

    def load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                self.config = json.load(file)
                self.logger.info(f"Configuration loaded from {self.config_path}")
        except json.JSONDecodeError as e:
            self.logger.exception("Failed to decode JSON config")
            raise ValueError("Invalid JSON format in configuration") from e

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"Missing config key: {key_path}. Using default: {default}")
            return default

    def get_telegram_token(self) -> str:
        token_path = Path(self.get_config_value("telegram.bot_token_file", "config/telegram_token.txt"))
        return self._read_text_file(token_path, "Telegram Token")

    def get_telegram_channel_id(self) -> str:
        channel_path = Path(self.get_config_value("telegram.channel_id_file", "config/telegram_channel.txt"))
        return self._read_text_file(channel_path, "Telegram Channel ID")

    def get_lm_studio_config(self) -> dict:
        return self.get_config_value("lm_studio", {})

    def get_rag_config(self) -> dict:
        return self.get_config_value("rag", {})

    def get_serper_api_key(self) -> str:
        """Получает ключ Serper API из config.json или файла, согласно структуре."""
        # 1. Новый стиль: external_apis.serper.api_key
        key = self.get_config_value("external_apis.serper.api_key", None)
        if key:
            return key
        # 2. Старый стиль: external_apis["serper.api_key"]
        key = self.get_config_value("external_apis.serper.api_key", None)
        if key:
            return key
        # 3. Через файл: serper.api_key_file
        api_key_file = self.get_config_value("serper.api_key_file", None)
        if api_key_file:
            try:
                return self._read_text_file(Path(api_key_file), "Serper API Key")
            except FileNotFoundError:
                self.logger.error(f"Serper API key file not found: {api_key_file}")
        # 4. Через файл: external_apis.serper.api_key_file
        api_key_file = self.get_config_value("external_apis.serper.api_key_file", None)
        if api_key_file:
            try:
                return self._read_text_file(Path(api_key_file), "Serper API Key (external_apis)")
            except FileNotFoundError:
                self.logger.error(f"Serper API key file not found: {api_key_file}")
        self.logger.critical("Serper API key not found in config or file!")
        return ""

    def update_config_value(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        ref = self.config
        for key in keys[:-1]:
            ref = ref.setdefault(key, {})
        ref[keys[-1]] = value
        self.logger.info(f"Updated config key: {key_path} = {value}")

    def save_config(self) -> None:
        try:
            with self.config_path.open("w", encoding="utf-8") as file:
                json.dump(self.config, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.exception("Failed to save configuration")
            raise IOError("Unable to save configuration") from e

    def validate_config(self) -> bool:
        required_keys = ["lm_studio", "rag", "telegram", "serper", "processing"]
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required section in config: '{key}'")
                return False
        self.logger.info("Configuration validated successfully")
        return True

    def _read_text_file(self, file_path: Path, label: str) -> str:
        if not file_path.exists():
            self.logger.error(f"{label} file not found: {file_path}")
            raise FileNotFoundError(f"{label} file missing: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            self.logger.debug(f"{label} loaded: {content[:10]}...")
            return content
        except Exception as e:
            self.logger.exception(f"Failed to read {label} from file")
            raise IOError(f"Error reading {label}") from e


# log.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import platform
import psutil
import time
import json

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Создать и вернуть логгер с файловым и консольным обработчиком.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = setup_file_handler(log_file, level)
        console_handler = setup_console_handler(level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False  # Не передаём в root

    return logger

def setup_file_handler(log_file: str, level=logging.INFO) -> RotatingFileHandler:
    """
    Создать файловый обработчик с ротацией.
    """
    file_path = LOG_DIR / log_file
    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler

def setup_console_handler(level=logging.INFO) -> logging.StreamHandler:
    """
    Создать консольный обработчик логов.
    """
    console = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    console.setLevel(level)
    return console

def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер с файловым и консольным выводом.
    """
    return setup_logger(name, f"{name.lower()}.log")

# ----------------------------- SYSTEM METRICS -----------------------------

def log_system_info(logger: logging.Logger):
    """
    Логировать информацию о системе (платформа, память, CPU, аптайм).
    """
    info = {
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
        "boot_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(psutil.boot_time())),
    }
    logger.info(f"System Info: {json.dumps(info)}")

def log_processing_stats(topics_processed: int, errors: int, success_rate: float, logger: logging.Logger):
    """
    Логировать статистику обработки тем.
    """
    logger.info(
        f"Processing Stats | Topics: {topics_processed}, Errors: {errors}, Success Rate: {success_rate:.2f}"
    )

def log_rag_performance(retrieval_time: float, context_length: int, logger: logging.Logger):
    """
    Логировать производительность RAG поиска.
    """
    logger.debug(
        f"RAG | Time: {retrieval_time:.2f}s, Context length: {context_length} chars"
    )

def log_telegram_status(message_sent: bool, logger: logging.Logger, error_details: str = None):
    """
    Логировать статус отправки сообщения в Telegram.
    """
    if message_sent:
        logger.info("Telegram message sent successfully.")
    else:
        logger.error(f"Telegram message failed. Reason: {error_details}")

# state_manager.py

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union

class StateManager:
    """
    Класс для управления состоянием системы: трекинг тем, статистики,
    резервное копирование, восстановление, обработка ошибок чтения/записи состояния.
    """

    def __init__(self, state_file: Union[str, Path] = "data/state.json"):
        self.state_file = Path(state_file)
        self.logger = logging.getLogger("StateManager")

        self.default_state = {
            "processed": {},
            "unprocessed": [],
            "failed": {},
            "statistics": {
                "topics_processed": 0,
                "success_count": 0,
                "error_count": 0,
                "start_time": datetime.utcnow().isoformat()
            },
            "system_status": "INIT",
            "last_activity": datetime.utcnow().isoformat()
        }
        self.state: Dict[str, Any] = {}

        self.load_state()

    # ----------- Группа: Загрузка/сохранение состояния -----------
    def load_state(self) -> None:
        """
        Загружает состояние из файла (если он есть и валиден).
        В случае ошибки или пустого файла - инициализирует дефолтное состояние.
        """
        if not self.state_file.exists() or self.state_file.stat().st_size == 0:
            self.logger.warning(f"State file {self.state_file} missing or empty. Initializing default state.")
            self.state = self.default_state.copy()
            self.save_state()
            return
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            # Проверка обязательных полей и "дополнение" до актуальной структуры
            self._ensure_state_integrity()
            self.logger.info(f"State loaded from {self.state_file}")
        except Exception as e:
            self.logger.error("Failed to load state. Initializing default state.", exc_info=True)
            self.state = self.default_state.copy()
            self.save_state()

    def save_state(self) -> None:
        """
        Сохраняет текущее состояние в файл. Атомарно (сначала tmp, затем заменяет основной).
        """
        tmp_file = self.state_file.with_suffix(".tmp")
        try:
            with tmp_file.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4, ensure_ascii=False)
            tmp_file.replace(self.state_file)
            self.logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save state to {self.state_file}", exc_info=True)

    def _ensure_state_integrity(self) -> None:
        """
        Проверяет, что все обязательные поля есть в self.state (для обратной совместимости).
        Если чего-то не хватает - добавляет из default_state.
        """
        for key, value in self.default_state.items():
            if key not in self.state:
                self.state[key] = value if not isinstance(value, dict) else value.copy()
        # Для вложенных статистик
        for k, v in self.default_state["statistics"].items():
            if "statistics" not in self.state:
                self.state["statistics"] = {}
            if k not in self.state["statistics"]:
                self.state["statistics"][k] = v

    # ----------- Группа: Методы для работы с темами -----------
    def mark_topic_processed(self, topic: str, success: bool, details: Optional[dict] = None) -> None:
        """
        Отмечает тему как успешно обработанную или failed.
        Перемещает из unprocessed в processed/failed, обновляет статистику.
        """
        now = datetime.utcnow().isoformat()
        details = details or {}

        if success:
            self.state["processed"][topic] = {
                "status": "success",
                "timestamp": now,
                "details": details
            }
            self.state["statistics"]["success_count"] += 1
        else:
            self.state["failed"][topic] = {
                "status": "failed",
                "timestamp": now,
                "error": details
            }
            self.state["statistics"]["error_count"] += 1

        if topic in self.state["unprocessed"]:
            self.state["unprocessed"].remove(topic)

        self.state["statistics"]["topics_processed"] += 1
        self.state["last_activity"] = now
        self.save_state()

    def add_topic(self, topic: str) -> None:
        """
        Добавляет новую тему в список необработанных, если она ещё не встречалась.
        """
        if topic not in self.state["unprocessed"] \
           and topic not in self.state["processed"] \
           and topic not in self.state["failed"]:
            self.state["unprocessed"].append(topic)
            self.save_state()

    def add_topics(self, topics: List[str]) -> None:
        """
        Добавляет список новых тем.
        """
        for topic in topics:
            self.add_topic(topic)

    def get_next_unprocessed_topic(self) -> Optional[str]:
        """
        Возвращает следующую необработанную тему, либо None.
        """
        return self.state["unprocessed"][0] if self.state["unprocessed"] else None

    def get_unprocessed_topics(self) -> List[str]:
        """
        Возвращает список необработанных тем.
        """
        return list(self.state["unprocessed"])

    def get_processed_topics(self) -> List[str]:
        """
        Возвращает список успешно обработанных тем.
        """
        return list(self.state["processed"].keys())

    def get_failed_topics(self) -> List[str]:
        """
        Возвращает список тем с ошибками.
        """
        return list(self.state["failed"].keys())

    def reset_failed_topics(self) -> None:
        """
        Переносит все failed темы обратно в unprocessed, очищает failed.
        """
        failed = self.get_failed_topics()
        # Не добавлять дубликаты
        for topic in failed:
            if topic not in self.state["unprocessed"]:
                self.state["unprocessed"].append(topic)
        self.state["failed"] = {}
        self.save_state()
        self.logger.info("Failed topics reset to unprocessed.")

    # ----------- Группа: Методы для статистики и статуса -----------
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику обработки тем.
        """
        return dict(self.state["statistics"])

    def add_processing_stats(self, stats: dict) -> None:
        """
        Обновляет (дополняет) статистику.
        """
        self.state["statistics"].update(stats)
        self.save_state()

    def get_system_uptime(self) -> float:
        """
        Возвращает аптайм системы (в секундах) с момента старта.
        """
        try:
            start = datetime.fromisoformat(self.state["statistics"]["start_time"])
        except Exception:
            start = datetime.utcnow()
        return (datetime.utcnow() - start).total_seconds()

    def set_system_status(self, status: str) -> None:
        """
        Устанавливает строковый статус системы ("RUNNING", "SHUTDOWN", "ERROR" и т.д.).
        """
        self.state["system_status"] = status
        self.state["last_activity"] = datetime.utcnow().isoformat()
        self.save_state()

    def get_system_status(self) -> str:
        """
        Возвращает текущий статус системы.
        """
        return self.state.get("system_status", "UNKNOWN")

    def get_last_activity(self) -> datetime:
        """
        Возвращает timestamp последней активности.
        """
        try:
            return datetime.fromisoformat(self.state["last_activity"])
        except Exception:
            return datetime.utcnow()

    # ----------- Группа: Бэкапы, восстановление и диагностика -----------
    def backup_state(self) -> str:
        """
        Делаем резервную копию состояния (сохраняет как <state_file>.backup.json)
        """
        backup_path = self.state_file.with_suffix(".backup.json")
        try:
            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4, ensure_ascii=False)
            self.logger.info(f"State backup saved to {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error("Failed to backup state.", exc_info=True)
            return ""

    def restore_state(self, backup_path: Union[str, Path]) -> bool:
        """
        Восстанавливает состояние из резервной копии.
        """
        path = Path(backup_path)
        if not path.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            with path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            self._ensure_state_integrity()
            self.save_state()
            self.logger.info("State restored from backup.")
            return True
        except Exception as e:
            self.logger.error("Failed to restore state.", exc_info=True)
            return False

    # ----------- Группа: Вспомогательные методы доступа (get/set/delete) -----------
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение по ключу из состояния (верхний уровень).
        """
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Установить значение по ключу в состоянии (верхний уровень).
        """
        self.state[key] = value
        self.save_state()

    def delete(self, key: str) -> None:
        """
        Удалить ключ из состояния (верхний уровень).
        """
        if key in self.state:
            del self.state[key]
            self.save_state()

    # ----------- Группа: Диагностика и вывод состояния -----------
    def dump_state(self) -> str:
        """
        Возвращает строковое представление состояния (удобно для отладки).
        """
        return json.dumps(self.state, indent=2, ensure_ascii=False)

    def print_state(self) -> None:
        """
        Печатает состояние в лог (информационный уровень).
        """
        self.logger.info(f"STATE DUMP:\n{self.dump_state()}")
