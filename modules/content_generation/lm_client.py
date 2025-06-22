# modules/content_generation/lm_client.py

import logging
import requests
from typing import Dict, Any

class LMStudioClient:
    def __init__(self, base_url: str, model: str, config: Dict[str, Any]):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        self.logger = logging.getLogger("LMStudioClient")

        self.logger.info(f"LMStudioClient initialized with model '{model}'")

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

    def generate_content(self, prompt: str, max_tokens: int = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            response = requests.post(f"{self.base_url}/completions", json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            return text.strip()
        except Exception as e:
            self.logger.error("Content generation failed", exc_info=True)
            raise

    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                return self.generate_content(prompt)
            except Exception as e:
                self.logger.warning(f"Retry {attempt}/{max_retries} for prompt generation")
                if attempt == max_retries:
                    raise

    def validate_response_length(self, text: str, max_length: int) -> bool:
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
            "timeout": self.timeout
        }

    def clear_conversation_history(self):
        # LM Studio не поддерживает историю — заглушка
        self.logger.debug("clear_conversation_history(): no-op")

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/health")
            return resp.json()
        except Exception as e:
            self.logger.warning("Health check failed", exc_info=True)
            return {"status": "unreachable"}
