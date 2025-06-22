# main.py

import sys
import signal
import time
import logging

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

        self.initialize_components()

    def setup_logging(self):
        log_system_info(self.logger)

    def validate_configuration(self):
        if not self.config_manager.validate_config():
            self.logger.critical("Configuration validation failed.")
            sys.exit(1)
        self.logger.info("Configuration validated successfully.")

    def initialize_components(self):
        self.setup_logging()
        self.validate_configuration()

        try:
            # Инициализация RAG и состояния
            self.rag_retriever = RAGRetriever(config=self.config["rag"])
            self.state_manager = StateManager(state_file="data/state.json")

            # Генерация контента
            self.lm_client = LMStudioClient(
                base_url=self.config["lm_studio"]["base_url"],
                model=self.config["lm_studio"]["model"],
                config=self.config["lm_studio"]
            )
            self.prompt_builder = PromptBuilder(prompt_folders=["data/prompt_1", "data/prompt_2", "data/prompt_3"])
            self.content_validator = ContentValidator(config=self.config)

            # Веб-поиск и медиа
            self.web_search = WebSearchClient(api_key=self.config["serper"].get("api_key", ""), config=self.config["serper"])
            self.media_handler = MediaHandler(media_folder="media", config=self.config)

            # Telegram клиент
            token = self.config_manager.get_telegram_token()
            channel_id = self.config_manager.get_telegram_channel_id()
            self.telegram_client = TelegramClient(token=token, channel_id=channel_id, config=self.config["telegram"])

        except Exception as e:
            self.logger.critical("Component initialization failed", exc_info=True)
            sys.exit(1)

    def build_initial_knowledge_base(self):
        try:
            folder = self.config["rag"].get("inform_folder", "inform/")
            self.rag_retriever.process_inform_folder(folder)
            self.rag_retriever.build_knowledge_base()
            self.logger.info("Initial RAG knowledge base built.")
        except Exception as e:
            self.logger.error("Knowledge base build failed", exc_info=True)

    def graceful_shutdown(self):
        self.shutdown_requested = True
        self.logger.warning("Shutdown signal received. Exiting loop...")

    def get_next_topic(self) -> str:
        topic = self.state_manager.get_next_unprocessed_topic()
        if topic:
            self.logger.info(f"Next topic selected: {topic}")
        return topic

    def combine_contexts(self, rag_context: str, web_context: str) -> str:
        return f"{rag_context}\n\n[Доп. контекст из поиска]\n\n{web_context}"

    def update_processing_state(self, topic: str, success: bool):
        self.state_manager.mark_topic_processed(topic, success)
        self.logger.info(f"Topic '{topic}' marked as {'processed' if success else 'failed'}.")

    def handle_error(self, topic: str, error: Exception):
        self.logger.error(f"Error processing topic '{topic}': {str(error)}", exc_info=True)
        self.update_processing_state(topic, success=False)

    def main_processing_loop(self):
        while not self.shutdown_requested:
            topic = self.get_next_topic()
            if not topic:
                self.logger.info("No more topics to process.")
                break

            try:
                # 1. Поиск контекста в RAG
                rag_context = self.rag_retriever.retrieve_context(topic)

                # 2. Web-поиск дополнительной информации
                web_context = self.web_search.extract_content(
                    self.web_search.search(topic)
                )

                # 3. Объединение контекста
                full_context = self.combine_contexts(rag_context, web_context)

                # 4. Выбор медиафайла (опционально)
                media_file = self.media_handler.get_random_media_file()

                # 5. Сборка промпта
                prompt = self.prompt_builder.build_prompt(
                    topic=topic,
                    context=full_context,
                    media_file=media_file
                )

                # 6. Генерация контента
                content = self.lm_client.generate_content(prompt)

                # 7. Валидация
                validated_content = self.content_validator.validate_content(content, has_media=bool(media_file))

                # 8. Публикация в Telegram
                success = self.telegram_client.send_media_message(validated_content, media_file) \
                          if media_file else self.telegram_client.send_text_message(validated_content)

                # 9. Состояние
                self.update_processing_state(topic, success)

                # 10. Пауза
                time.sleep(self.config["telegram"]["post_interval"])

            except Exception as e:
                self.handle_error(topic, e)
                continue

    def run(self):
        self.logger.info("System starting up...")
        signal.signal(signal.SIGINT, lambda s, f: self.graceful_shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self.graceful_shutdown())

        self.build_initial_knowledge_base()
        self.main_processing_loop()
        self.logger.info("System shut down gracefully.")
