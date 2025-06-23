import os
import sys
import time
import signal
import logging

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


class TelegramRAGSystem:
    def __init__(self, config_path: str = "config/config.json"):
        self.logger = get_logger("Main")
        self.shutdown_requested = False
        self.heartbeat_counter = 0

        try:
            log_system_info(self.logger)
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
            self._validate_config()
            self._init_services()
            self._load_topics()
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

    def _combine_context(self, topic: str) -> str:
        rag = self.rag_retriever.retrieve_context(topic)
        web_results = self.web_search.search(topic)
        web = self.web_search.extract_content(web_results)
        return f"{rag}\n\n[WEB]\n{web}"

    def _shorten_if_needed(self, text: str, prompt: str, has_media: bool) -> str:
        try:
            self.content_validator.validate_content(text, has_media=has_media)
            return text
        except ValueError:
            self.logger.warning("Content too long, requesting shortened version...")
            return self.lm_client.request_shorter_version(prompt, current_length=len(text), target_length=1024 if has_media else 4096)

    def _process_topic(self, topic: str) -> bool:
        self.logger.info(f"Processing topic: {topic}")
        context = self._combine_context(topic)
        media_file = self.media_handler.get_random_media_file()
        prompt = self.prompt_builder.build_prompt(topic, context, media_file)

        try:
            response = self.lm_client.generate_with_retry(prompt)
            response = self._shorten_if_needed(response, prompt, has_media=bool(media_file))
            validated = self.content_validator.validate_content(response, has_media=bool(media_file))

            if media_file:
                result = self.telegram_client.send_media_message(validated, media_file)
            else:
                result = self.telegram_client.send_text_message(validated)

            if result:
                self.logger.info(f"Topic '{topic}' successfully posted.")
            else:
                self.logger.error(f"Failed to publish topic '{topic}'.")

            return result

        except Exception as e:
            self.logger.error(f"Unhandled error on topic '{topic}'", exc_info=True)
            return False

    def run(self):
        self.logger.info("Bot started.")
        while not self.shutdown_requested:
            topic = self.state_manager.get_next_unprocessed_topic()
            if not topic:
                self.logger.info("No more topics in queue.")
                break

            success = self._process_topic(topic)
            self.state_manager.mark_topic_processed(topic, success=success)

            self.heartbeat_counter += 1
            if self.heartbeat_counter % 5 == 0:
                self.logger.info(f"Heartbeat: {self.heartbeat_counter} topics processed.")
                self.state_manager.save_state()

            interval = self.config["telegram"].get("post_interval", 900)
            self.logger.debug(f"Sleeping for {interval}s before next topic...")
            time.sleep(interval)

        self.logger.info("Bot shutdown complete.")
