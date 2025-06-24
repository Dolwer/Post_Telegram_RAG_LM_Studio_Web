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

def clean_context(text: str) -> str:
    """Очищает контекст от мусора, nan, пустых строк."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if line.strip() and
           "nan" not in line.lower() and
           "data too large for file format" not in line.lower()
    ]
    return "\n".join(cleaned)

def slugify(text: str, maxlen=64) -> str:
    import re
    slug = re.sub(r'[^\w\s-]', '', text).strip().lower()
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug[:maxlen]

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
        # Загружаем новые топики только если очередь пуста
        if not self.state_manager.get_unprocessed_topics():
            topics_file = os.path.join(self.config["paths"]["data_dir"], "topics.txt")
            if os.path.isfile(topics_file):
                with open(topics_file, "r", encoding="utf-8") as f:
                    topics = [line.strip() for line in f if line.strip()]
                self.state_manager.add_topics(topics)
                self.logger.info(f"Loaded {len(topics)} topics from {topics_file} into state.json")
            else:
                self.logger.warning("No topics file found.")
        else:
            self.logger.info("Unprocessed topics already present in state.json")

    def _graceful_shutdown(self, *_):
        self.logger.warning("Shutdown signal received. Exiting loop...")
        self.shutdown_requested = True

    def _combine_context(self, topic: str, topic_logger) -> str:
        """Собирает и очищает объединённый контекст для промпта."""
        rag, web = "", ""
        try:
            rag = self.rag_retriever.retrieve_context(topic)
            topic_logger.info(f"RAG context:\n{rag[:2000]}")
        except Exception as e:
            topic_logger.error(f"Failed to retrieve RAG context: {e}\n{traceback.format_exc()}")
            rag = ""
        try:
            web_results = self.web_search.search(topic)
            web = self.web_search.extract_content(web_results)
            # Сохраняем web-материал для аудита
            if web.strip():
                self.web_search.save_to_inform(web, topic, source="web")
            topic_logger.info(f"Web search context:\n{web[:2000]}")
        except Exception as e:
            topic_logger.error(f"Failed to retrieve web context: {e}\n{traceback.format_exc()}")
            web = ""
        full_context = f"{rag}\n\n[WEB]\n{web}"
        full_context = clean_context(full_context)
        topic_logger.debug(f"Combined context (truncated): {full_context[:1000]}")
        return full_context

    def _shorten_if_needed(self, text: str, prompt: str, has_media: bool, topic_logger) -> str:
        """Если контент слишком длинный — просит нейросеть сократить его."""
        try:
            self.content_validator.validate_content(text, has_media=has_media)
            return text
        except ValueError as e:
            topic_logger.warning(f"Content too long, requesting shortened version... {e}")
            return self.lm_client.request_shorter_version(prompt, current_length=len(text), target_length=1024 if has_media else 4096)

    def _log_topic_step(self, topic_slug, step, msg, level="info"):
        os.makedirs("logs/topics", exist_ok=True)
        logfile = os.path.join("logs/topics", f"{topic_slug}.log")
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{step.upper()}] {msg}\n")
        getattr(self.logger, level)(f"[{topic_slug}] [{step.upper()}] {msg}")

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
            max_len = 1024 if has_media else 4096

            min_context_length = 512
            for attempt in range(3):
                try:
                    topic_logger.info(f"Generation attempt {attempt+1}: prompt len={len(prompt)}, context len={len(context)}")
                    response = self.lm_client.generate_with_retry(prompt)
                    if not response or not response.strip():
                        raise ValueError("Empty response from LM Studio")
                    if "prediction-error" in response.lower() or "error:" in response.lower():
                        raise ValueError(f"LM Studio error: {response.strip()[:100]}")
                    topic_logger.info(f"LM Studio response received, len={len(response)}")
                    break  # успех
                except Exception as e:
                    msg = str(e)
                    topic_logger.error(f"LM Studio generation error on attempt {attempt+1}: {msg}")
                    # Если ошибка из-за длины prompt, сокращаем context и пытаемся снова
                    if ("Prompt too long" in msg or "prompt/messages too long" in msg.lower()) and len(context) > min_context_length:
                        new_len = max(min_context_length, int(len(context) * 0.7))
                        topic_logger.warning(f"Prompt too long. Shortening context from {len(context)} to {new_len} chars and retrying...")
                        context = context[:new_len]
                        try:
                            prompt = self.prompt_builder.build_prompt(topic, context, media_file)
                            topic_logger.debug(f"Prompt rebuilt for retry (truncated): {prompt[:1500]}")
                        except Exception as e2:
                            err_msg = f"Prompt rebuild failed after context cut: {e2}"
                            topic_logger.error(err_msg)
                            topic_result["status"] = "fail"
                            topic_result["error"] = err_msg
                            self.stats["fail"] += 1
                            self.stats["topics"].append(topic_result)
                            return False
                        continue
                    else:
                        # Любая другая ошибка или уже минимальный контекст
                        topic_logger.error(f"LM Studio unrecoverable error: {msg}")
                        topic_result["status"] = "fail"
                        topic_result["error"] = msg
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

            formatted = self.content_validator.format_for_telegram(validated, max_len=max_len)
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
            try:
                topic = self.state_manager.get_next_unprocessed_topic()
                if not topic:
                    self.logger.info("No more topics in queue.")
                    break

                self.logger.info(f"Next topic selected: {topic}")
                success = self._process_topic(topic)
                self.state_manager.mark_topic_processed(topic, success=success)

                heartbeat_counter += 1
                if heartbeat_counter % 5 == 0:
                    self._heartbeat(heartbeat_counter)
                    self.state_manager.save_state()

                interval = self.config["telegram"].get("post_interval", 900)
                self.logger.debug(f"Sleeping for {interval}s before next topic...")
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Exception in processing loop: {e}\n{traceback.format_exc()}")
                continue

        self._log_summary()
        self.logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    system = TelegramRAGSystem()
    system.run()
