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
        token_path = Path("config/telegram_token.txt")
        return self._read_text_file(token_path, "Telegram Token")

    def get_telegram_channel_id(self) -> str:
        channel_path = Path("config/telegram_channel.txt")
        return self._read_text_file(channel_path, "Telegram Channel ID")

    def get_lm_studio_config(self) -> dict:
        return self.get_config_value("lm_studio", {})

    def get_rag_config(self) -> dict:
        return self.get_config_value("rag", {})

    def get_serper_api_key(self) -> str:
        return self.get_config_value("serper.api_key", "")

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
                json.dump(self.config, file, indent=4)
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
