# modules/external_apis/telegram_client.py

import logging
import time
import requests
from typing import Optional, Dict

class TelegramClient:
    TELEGRAM_API_URL = "https://api.telegram.org"

    def __init__(self, token: str, channel_id: str, config: dict):
        self.token = token
        self.channel_id = channel_id
        self.config = config
        self.logger = logging.getLogger("TelegramClient")

        self.api_base = f"{self.TELEGRAM_API_URL}/bot{self.token}"

    def send_text_message(self, text: str) -> bool:
        if not self.validate_message_length(text, has_media=False):
            self.logger.warning("Text message too long. Skipping.")
            return False
        payload = {
            "chat_id": self.channel_id,
            "text": self.format_message(text),
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        return self._post("sendMessage", payload)

    def send_media_message(self, text: str, media_path: str) -> bool:
        if not self.validate_message_length(text, has_media=True):
            self.logger.warning("Caption too long. Skipping media post.")
            return False

        media_type = self.get_media_type(media_path)
        method = {
            "photo": "sendPhoto",
            "video": "sendVideo",
            "document": "sendDocument"
        }.get(media_type)

        if not method:
            self.logger.error(f"Unsupported media type for {media_path}")
            return False

        files = {
            media_type: open(media_path, "rb")
        }
        data = {
            "chat_id": self.channel_id,
            "caption": self.format_message(text),
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(f"{self.api_base}/{method}", data=data, files=files)
            response.raise_for_status()
            self.logger.info(f"Media message sent successfully: {media_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send media: {media_path}", exc_info=True)
            return False
        finally:
            files[media_type].close()

    def _post(self, method: str, payload: dict) -> bool:
        try:
            response = requests.post(f"{self.api_base}/{method}", json=payload)
            response.raise_for_status()
            self.logger.info(f"Telegram message sent: {method}")
            return True
        except Exception as e:
            self.logger.error(f"Telegram API error: {method}", exc_info=True)
            return False

    def retry_send_message(self, message_data: dict, max_retries: int = 3) -> bool:
        for attempt in range(max_retries):
            success = self._post("sendMessage", message_data)
            if success:
                return True
            time.sleep(2 ** attempt)
        return False

    def format_message(self, text: str) -> str:
        return self.escape_markdown(text)

    def escape_markdown(self, text: str) -> str:
        # Telegram HTML-compatible safe text
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def validate_message_length(self, text: str, has_media: bool) -> bool:
        limit = 1024 if has_media else 4096
        return len(text) <= limit

    def get_media_type(self, file_path: str) -> Optional[str]:
        ext = file_path.lower().split('.')[-1]
        if ext in ["jpg", "jpeg", "png"]:
            return "photo"
        elif ext in ["mp4", "mov"]:
            return "video"
        elif ext in ["pdf", "docx", "txt"]:
            return "document"
        return None

    def handle_telegram_errors(self, error: Exception) -> bool:
        self.logger.error(f"Telegram error handled: {str(error)}")
        return False

    def check_bot_permissions(self) -> Dict[str, any]:
        try:
            resp = requests.get(f"{self.api_base}/getMe")
            info = resp.json()
            self.logger.info(f"Bot Info: {info}")
            return info
        except Exception as e:
            self.logger.error("Permission check failed", exc_info=True)
            return {}

    def get_channel_info(self) -> Dict[str, any]:
        try:
            resp = requests.get(f"{self.api_base}/getChat", params={"chat_id": self.channel_id})
            return resp.json()
        except Exception as e:
            self.logger.warning("Failed to get channel info", exc_info=True)
            return {}

    def get_send_stats(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "token_hash": hash(self.token),
        }
