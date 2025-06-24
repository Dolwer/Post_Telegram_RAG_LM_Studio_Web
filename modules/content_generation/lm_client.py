import logging
import requests
from typing import Dict, Any, List, Optional

class LMStudioClient:
    """
    Клиент для взаимодействия с LM Studio (локальная LLM).
    Контролирует лимит payload (prompt + history + system_message) для LM Studio.
    Не занимается Telegram-валидированием и не режет по лимитам Telegram — только лимит самой LLM!
    """

    LM_MAX_TOTAL_CHARS = 20000  # Лимит для LM Studio (под свою модель!)

    def __init__(self, base_url: str, model: str, config: Dict[str, Any]):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        self.history_limit = config.get("history_limit", 3)
        self.system_message = config.get("system_message", None)
        self.top_p = config.get("top_p", None)
        self.top_k = config.get("top_k", None)
        self.logger = logging.getLogger("LMStudioClient")
        self.history: List[Dict[str, str]] = []
        self._validate_config()
        self._check_health_on_init()

    def _validate_config(self):
        assert isinstance(self.max_tokens, int) and self.max_tokens > 0, "max_tokens must be positive integer"
        assert isinstance(self.LM_MAX_TOTAL_CHARS, int) and self.LM_MAX_TOTAL_CHARS > 1000, "LM_MAX_TOTAL_CHARS must be > 1000"
        assert isinstance(self.temperature, (float, int)), "temperature must be float"
        if self.top_p is not None:
            assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0,1]"
        if self.top_k is not None:
            assert isinstance(self.top_k, int) and self.top_k >= 0, "top_k must be non-negative int"

    def _check_health_on_init(self):
        try:
            status = self.health_check()
            if status.get("status") != "ok":
                raise RuntimeError(f"LM Studio health check failed: {status}")
            self.logger.info(f"LMStudioClient connected to model '{self.model}'. Health OK.")
        except Exception as e:
            self.logger.error(f"LM Studio health check failed: {e}")
            raise

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unreachable"}

    def clear_conversation_history(self):
        self.history = []
        self.logger.debug("LMStudioClient: conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        if user_message and isinstance(user_message, str) and user_message.strip():
            self.history.append({"role": "user", "content": user_message})
        if bot_message and isinstance(bot_message, str) and bot_message.strip():
            self.history.append({"role": "assistant", "content": bot_message})
        # Trim to the last N exchanges
        if self.history_limit > 0 and len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def _clean_history(self) -> List[Dict[str, str]]:
        """
        Возвращает только валидные последние сообщения, проверяет структуру.
        """
        clean = []
        for m in self.history[-self.history_limit * 2:]:
            if (
                isinstance(m, dict)
                and m.get("role") in {"user", "assistant", "system"}
                and isinstance(m.get("content"), str)
                and m["content"].strip()
                and "nan" not in m["content"]
            ):
                clean.append(m)
            else:
                self.logger.warning(f"Skipping invalid message in LLM history: {m}")
        return clean

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str, media_file: Optional[str]) -> str:
        """
        Подбирает длину context так, чтобы prompt + history + system_message влезли в LM Studio лимит.
        Обрезает context, если надо.
        """
        # Подставляем все плейсхолдеры кроме {CONTEXT}
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": "CONTEXT_PLACEHOLDER",
            "{UPLOADFILE}": media_file.strip() if media_file else "",
        }
        prompt_wo_context = prompt_template
        for key, value in replacements.items():
            prompt_wo_context = prompt_wo_context.replace(key, value)
        prompt_wo_context_len = len(prompt_wo_context.replace("CONTEXT_PLACEHOLDER", ""))

        # Считаем размер истории и system_message
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        static_chars = sum(len(m["content"]) for m in messages) + prompt_wo_context_len
        available = self.LM_MAX_TOTAL_CHARS - static_chars
        if available <= 0:
            self.logger.warning(f"No room for context: static_chars={static_chars} > limit={self.LM_MAX_TOTAL_CHARS}")
            return ""
        context = context.strip()
        if len(context) > available:
            self.logger.warning(f"Context too long for LLM ({len(context)} > {available}), truncating context")
            context = context[:available]
        return context

    def _build_messages(self, prompt_template: str, topic: str, context: str, media_file: Optional[str]) -> List[Dict[str, str]]:
        """
        Собирает сообщения для LLM API, применяет лимиты.
        """
        context = self._truncate_context_for_llm(prompt_template, topic, context, media_file)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context,
            "{UPLOADFILE}": media_file.strip() if media_file else "",
        }
        prompt = prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        prompt = prompt.replace("nan", "").strip()
        # Формируем историю
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        messages.append({"role": "user", "content": prompt})
        # Ещё раз проверяем лимит
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > self.LM_MAX_TOTAL_CHARS:
            self.logger.warning(f"Total LLM payload too long ({total_chars} > {self.LM_MAX_TOTAL_CHARS}), trimming prompt/history")
            # Обрезаем prompt
            excess = total_chars - self.LM_MAX_TOTAL_CHARS
            if len(messages[-1]["content"]) > excess:
                messages[-1]["content"] = messages[-1]["content"][:len(messages[-1]["content"]) - excess]
            else:
                # Убираем старые истории
                while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                    removed = messages.pop(1)  # после system_message
                    self.logger.warning(f"Removed old history message to fit LM payload: {removed}")
                    total_chars = sum(len(m["content"]) for m in messages)
                if total_chars > self.LM_MAX_TOTAL_CHARS:
                    last = messages[-1]
                    last["content"] = last["content"][:max(0, len(last["content"]) - (total_chars - self.LM_MAX_TOTAL_CHARS))]
        return messages

    def generate_content(
        self,
        prompt_template: str,
        topic: str,
        context: str,
        media_file: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Генерирует текст с помощью LM Studio, ограничивает весь payload до LM_MAX_TOTAL_CHARS.
        Возвращает строку или бросает ValueError.
        """
        max_tokens = max_tokens or self.max_tokens
        messages = self._build_messages(prompt_template, topic, context, media_file)
        chat_url = f"{self.base_url}/v1/chat/completions"
        payload_chat = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        if self.top_p is not None:
            payload_chat["top_p"] = self.top_p
        if self.top_k is not None:
            payload_chat["top_k"] = self.top_k

        self.logger.debug(f"LMStudioClient: Sending chat payload to {chat_url}: {str(payload_chat)[:800]}")

        try:
            response = requests.post(chat_url, json=payload_chat, timeout=self.timeout)
            self.logger.debug(f"LMStudioClient: raw response: {response.text[:1000]}")
            response.raise_for_status()
            try:
                result = response.json()
            except Exception as e:
                self.logger.error("Failed to decode LM Studio response as JSON", exc_info=True)
                result = {}
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Fallback если пусто или endpoint не поддержан
            if not text:
                self.logger.warning("Empty chat response, fallback to completions endpoint.")
                comp_url = f"{self.base_url}/v1/completions"
                payload = {
                    "model": self.model,
                    "prompt": messages[-1]['content'],
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "stream": False
                }
                if self.top_p is not None:
                    payload["top_p"] = self.top_p
                if self.top_k is not None:
                    payload["top_k"] = self.top_k
                comp_resp = requests.post(comp_url, json=payload, timeout=self.timeout)
                comp_resp.raise_for_status()
                try:
                    comp_result = comp_resp.json()
                except Exception as e:
                    self.logger.error("Failed to decode completions response as JSON", exc_info=True)
                    comp_result = {}
                text = comp_result.get("choices", [{}])[0].get("text", "")

            if text and text.strip():
                self.add_to_history(messages[-1]['content'], text)
            else:
                self.logger.warning("LM Studio returned empty text from both endpoints.")
            if not isinstance(text, str):
                raise ValueError("LM Studio returned non-string result.")
            return text.strip()
        except Exception as e:
            self.logger.error("Content generation failed on both endpoints", exc_info=True)
            self.clear_conversation_history()
            raise ValueError(f"LM Studio content generation failed: {e}")

    def generate_with_retry(
        self,
        prompt_template: str,
        topic: str,
        context: str,
        media_file: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Автоматический повтор генерации при ошибках, с учётом лимита на payload.
        При ошибке 400 — уменьшает context и очищает историю.
        """
        last_err = None
        original_context = context
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"LMStudioClient: generation attempt {attempt}/{max_retries}")
                text = self.generate_content(prompt_template, topic, context, media_file)
                if text and text.strip():
                    return text
                self.logger.warning(f"LM Studio generation returned empty text (attempt {attempt})")
            except Exception as e:
                last_err = e
                msg = str(e)
                self.logger.warning(f"LMStudioClient: error on attempt {attempt}: {msg}")
                # Если payload слишком большой — сокращаем context
                if "413" in msg or "400" in msg or "payload" in msg:
                    context = context[:max(100, len(context) // 2)]
                    self.logger.warning("Reducing context and retrying...")
                # При повторяющейся ошибке — сбрасываем историю
                if attempt == max_retries or (self.history_limit and len(self.history) > self.history_limit * 4):
                    self.logger.warning("Clearing conversation history due to repeated failures.")
                    self.clear_conversation_history()
        # Последняя попытка — с минимальным context и чистой историей
        try:
            return self.generate_content(prompt_template, topic, original_context[:256], media_file)
        except Exception as e:
            self.logger.error("Final fallback attempt failed", exc_info=True)
            raise ValueError(f"LM Studio did not generate content after {max_retries} attempts: {last_err}")

    def set_generation_parameters(self, temperature: float, top_p: Optional[float] = None, top_k: Optional[int] = None):
        assert 0.0 <= temperature <= 2.0, "temperature must be in [0,2]"
        self.temperature = temperature
        if top_p is not None:
            assert 0.0 <= top_p <= 1.0, "top_p must be in [0,1]"
            self.top_p = top_p
        if top_k is not None:
            assert isinstance(top_k, int) and top_k >= 0, "top_k must be non-negative int"
            self.top_k = top_k
        self.logger.info(f"Set generation parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}")

    def get_generation_stats(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "history_limit": self.history_limit,
            "system_message": self.system_message,
            "top_p": self.top_p,
            "top_k": self.top_k
        }
