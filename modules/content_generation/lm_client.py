import logging
import requests
import time
from typing import List, Optional, Dict, Any

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

        # История диалога (как список пар user/assistant)
        self.history: List[Dict[str, str]] = []
        self.history_limit = config.get("history_limit", 5)

    def check_connection(self) -> bool:
        try:
            url = f"{self.base_url}/v1/models"
            self.logger.debug(f"Checking LM Studio at: {url}")
            response = requests.get(url, timeout=10)
            self.logger.debug(f"Model info raw response: {response.text[:1000]}")
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [m.get("id") for m in models]
                if self.model in model_ids:
                    self.logger.info("Connected to LM Studio. Model OK.")
                    return True
                else:
                    self.logger.error(f"Model '{self.model}' not found in LM Studio. Available: {model_ids}")
                    return False
            self.logger.warning(f"LM Studio connection failed: {response.status_code}")
            return False
        except requests.RequestException as e:
            self.logger.error("LM Studio not reachable.", exc_info=True)
            return False

    def clear_conversation_history(self):
        self.history.clear()
        self.logger.info("LM history cleared.")

    def estimate_tokens(self, text: str) -> int:
        # Оценка: 1 токен ≈ 4 символа
        return max(1, int(len(text) / 4))

    def _build_payload_chat(self, prompt: str) -> dict:
        # История — последние self.history_limit*2 сообщений (user/assistant)
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        # Добавляем пары user/assistant из истории (если есть)
        if self.history_limit > 0:
            history_pairs = self.history[-self.history_limit*2:]
            for m in history_pairs:
                # m всегда dict с role и content
                messages.append(m)
        # Текущий user prompt
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
            self.logger.debug(f"Raw LMStudio response: {response.text[:1000]}")
            result = response.json()
        except ValueError:
            self.logger.error("Failed to decode JSON from LM response.", exc_info=True)
            return None

        if "error" in result:
            self.logger.error(f"LMStudio error: {result['error']}")
            # prediction-error: main.py сможет обработать как "надо сокращать"
            return None

        try:
            if mode == "chat":
                return result["choices"][0]["message"]["content"]
            else:
                return result["choices"][0]["text"]
        except (KeyError, IndexError):
            self.logger.error("LMStudio response missing required fields.", exc_info=True)
            return None

    def _validate_prompt_length(self, messages: List[Dict[str, str]]) -> bool:
        # Лимит токенов для всей истории и запроса
        total_tokens = sum(self.estimate_tokens(m["content"]) for m in messages)
        if total_tokens > self.max_tokens:
            self.logger.warning(f"Prompt too long for LM Studio ({total_tokens} > {self.max_tokens} tokens)")
            return False
        return True

    def generate_content(self, prompt: str) -> Optional[str]:
        mode = "chat"
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload_chat(prompt)
        self.logger.debug(f"LMStudio chat payload (truncated): {str(payload)[:1200]}")
        # Валидация длины prompt/messages
        if not self._validate_prompt_length(payload["messages"]):
            self.logger.warning("Prompt/messages too long, try to shorten before sending.")
            return None

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            self.logger.info(f"[LM] POST {url} [{response.status_code}] in {response.elapsed.total_seconds():.2f}s")
            content = self._parse_response(response, mode)
            if content:
                # Добавляем только последнего user+assistant в историю
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": content})
                # Ограничиваем размер истории
                if len(self.history) > self.history_limit*2:
                    self.history = self.history[-self.history_limit*2:]
                return content
        except requests.RequestException as e:
            self.logger.error(f"Request failed to LMStudio at {url}", exc_info=True)

        # fallback to completions
        self.logger.warning("Falling back to /v1/completions")
        fallback_url = f"{self.base_url}/v1/completions"
        fallback_payload = self._build_payload_completion(prompt)
        self.logger.debug(f"LMStudio completion payload (truncated): {str(fallback_payload)[:1200]}")

        try:
            response = requests.post(fallback_url, json=fallback_payload, timeout=self.timeout)
            self.logger.info(f"[LM] Fallback POST {fallback_url} [{response.status_code}] in {response.elapsed.total_seconds():.2f}s")
            return self._parse_response(response, "completion")
        except requests.RequestException as e:
            self.logger.error(f"Fallback request also failed.", exc_info=True)
            return None

    def generate_with_retry(self, prompt: str, max_retries: int = 3, delay: float = 3.0) -> str:
        last_exception = None
        for attempt in range(max_retries):
            try:
                result = self.generate_content(prompt)
                if result:
                    return result
                else:
                    self.logger.warning(f"LM generation failed (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
            except Exception as e:
                last_exception = e
                self.logger.warning(f"LMStudio internal exception: {e}. Retrying in {delay}s...")
            time.sleep(delay)

        self.logger.critical("All LMStudio attempts failed. Raising exception.")
        raise RuntimeError(f"LMStudio generation failed after {max_retries} retries. Last error: {last_exception}")

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
