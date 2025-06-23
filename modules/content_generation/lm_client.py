import logging
import requests
import time
from typing import List, Optional


class LMStudioClient:
    def __init__(self, base_url: str, model: str, config: dict):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.config = config
        self.logger = logging.getLogger("LMStudioClient")

        # Параметры генерации
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self.top_k = config.get("top_k", 40)
        self.top_p = config.get("top_p", 0.95)
        self.system_message = config.get("system_message", "You are a helpful assistant.")

        # История диалога
        self.history: List[dict] = []
        self.history_limit = config.get("history_limit", 5)

    def check_connection(self) -> bool:
        try:
            response = requests.get(self.base_url + "/v1/models", timeout=10)
            if response.status_code == 200:
                self.logger.info("Connected to LM Studio.")
                return True
            self.logger.warning(f"LM Studio connection failed: {response.status_code}")
            return False
        except requests.RequestException as e:
            self.logger.error("LM Studio not reachable.", exc_info=True)
            return False

    def clear_conversation_history(self):
        self.history.clear()
        self.logger.info("LM history cleared.")

    def estimate_tokens(self, text: str) -> int:
        return len(text.split()) * 1.3  # очень грубая оценка

    def _build_payload_chat(self, prompt: str) -> dict:
        messages = [{"role": "system", "content": self.system_message}]
        messages += self.history[-self.history_limit:]
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def _build_payload_completion(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": None,
        }

    def _parse_response(self, response: requests.Response, mode: str) -> Optional[str]:
        try:
            result = response.json()
        except ValueError:
            self.logger.error("Failed to decode JSON from LM response.", exc_info=True)
            return None

        if "error" in result:
            self.logger.error(f"LMStudio error: {result['error']}")
            return None

        try:
            if mode == "chat":
                return result["choices"][0]["message"]["content"]
            else:
                return result["choices"][0]["text"]
        except (KeyError, IndexError):
            self.logger.error("LMStudio response missing required fields.", exc_info=True)
            return None

    def generate_content(self, prompt: str) -> Optional[str]:
        mode = "chat"
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload_chat(prompt)

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            self.logger.info(f"[LM] POST {url} [{response.status_code}] in {response.elapsed.total_seconds():.2f}s")
            content = self._parse_response(response, mode)
            if content:
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": content})
                return content
        except requests.RequestException as e:
            self.logger.error(f"Request failed to LMStudio at {url}", exc_info=True)

        # fallback to completions
        self.logger.warning("Falling back to /v1/completions")
        fallback_url = f"{self.base_url}/v1/completions"
        fallback_payload = self._build_payload_completion(prompt)

        try:
            response = requests.post(fallback_url, json=fallback_payload, timeout=self.timeout)
            self.logger.info(f"[LM] Fallback POST {fallback_url} [{response.status_code}] in {response.elapsed.total_seconds():.2f}s")
            return self._parse_response(response, "completion")
        except requests.RequestException as e:
            self.logger.error(f"Fallback request also failed.", exc_info=True)
            return None

    def generate_with_retry(self, prompt: str, max_retries: int = 3, delay: float = 3.0) -> str:
        for attempt in range(max_retries):
            result = self.generate_content(prompt)
            if result:
                return result
            self.logger.warning(f"LM generation failed (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
            time.sleep(delay)

        self.logger.critical("All LMStudio attempts failed. Raising exception.")
        raise RuntimeError("LMStudio generation failed after retries.")

    def request_shorter_version(self, original_prompt: str, current_length: int, target_length: int) -> str:
        shorten_instruction = (
            f"{original_prompt}\n\nPlease shorten the response to less than {target_length} characters."
        )
        try:
            result = self.generate_with_retry(shorten_instruction)
            return result if result else ""
        except Exception as e:
            self.logger.error("Failed to generate shorter version.", exc_info=True)
            return ""

    def get_generation_stats(self) -> dict:
        return {
            "history_length": len(self.history),
            "history_tokens": sum(self.estimate_tokens(m['content']) for m in self.history)
        }

    def health_check(self) -> dict:
        return {
            "connected": self.check_connection(),
            "model": self.model,
            "stats": self.get_generation_stats()
        }
