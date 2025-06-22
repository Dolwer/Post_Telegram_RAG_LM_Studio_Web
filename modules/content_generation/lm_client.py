import logging
import requests
from typing import Dict, Any, List

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
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            models = response.json().get("data", [])
            is_ok = any(m["id"] == self.model for m in models)
            self.logger.info("LM Studio connection OK" if is_ok else "Model not found in LM Studio")
            return is_ok
        except Exception as e:
            self.logger.error("Failed to connect to LM Studio", exc_info=True)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/models")
            return resp.json()
        except Exception as e:
            self.logger.error("Model info retrieval failed", exc_info=True)
            return {}

    def generate_content(self, prompt: str, max_tokens: int = None, with_media=False) -> str:
        max_tokens = max_tokens or self.max_tokens
        # Compose prompt with system message
        full_prompt = ""
        if self.system_message:
            full_prompt += f"{self.system_message}\n"
        full_prompt += prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                json=payload,
                timeout=self.timeout
            )
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

    def generate_with_retry(self, prompt: str, max_retries: int = 3, with_media=False) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                return self.generate_content(prompt, with_media=with_media)
            except Exception as e:
                self.logger.warning(f"Retry {attempt}/{max_retries} for prompt generation")
                if attempt == max_retries:
                    raise

    def validate_response_length(self, text: str, max_length: int = None) -> bool:
        if max_length is None:
            max_length = self.max_chars
        if len(text) <= max_length:
            return True
        self.logger.warning(f"Response exceeds max length ({len(text)} > {max_length})")
        return False

    def request_shorter_version(self, original_prompt: str, current_length: int, target_length: int) -> str:
        instruction = (
            f"\n\n[ИНСТРУКЦИЯ]: Сократи предыдущий текст до {target_length} символов. "
            "Сохрани ключевые идеи и структуру, но сделай более компактным."
        )
        new_prompt = original_prompt + instruction
        return self.generate_with_retry(new_prompt)

    def estimate_tokens(self, text: str) -> int:
        # Примерная оценка: 1 токен ≈ 4 символа
        return int(len(text) / 4)

    def set_generation_parameters(self, temperature: float, top_p: float = None, top_k: int = None):
        self.temperature = temperature
        self.logger.info(f"Set generation temperature to {temperature}")
        if top_p:
            self.logger.info(f"Set top_p to {top_p}")
        if top_k:
            self.logger.info(f"Set top_k to {top_k}")

    def get_generation_stats(self) -> dict:
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
        self.history = []
        self.logger.debug("Conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        self.history.append({"user": user_message, "bot": bot_message})
        if self.history_limit and len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/health")
            return resp.json()
        except Exception as e:
            self.logger.warning("Health check failed", exc_info=True)
            return {"status": "unreachable"}
