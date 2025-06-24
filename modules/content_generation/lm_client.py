import logging
import requests
from typing import Dict, Any, List, Optional

class LMStudioClient:
    LM_MAX_TOTAL_CHARS = 20000
    TELEGRAM_LIMIT = 4096

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
        status = self.health_check()
        if status.get("status") != "ok":
            self.logger.warning(f"LM Studio health check: {status}")
        else:
            self.logger.info(f"LMStudioClient connected to model '{self.model}'. Health OK.")

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and any(self.model in m.get("id", "") for m in data["data"]):
                return {"status": "ok"}
            return {"status": "model_not_found"}
        except Exception as e:
            self.logger.warning(f"Health check fallback: /v1/models endpoint not found, trying /v1/chat/completions dummy call.")
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=10)
                if resp.status_code in (200, 400):
                    return {"status": "ok"}
            except Exception as e2:
                self.logger.error(f"Health check failed: {e2}")
            return {"status": "unreachable"}

    def clear_conversation_history(self):
        self.history = []
        self.logger.debug("LMStudioClient: conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        if user_message and isinstance(user_message, str) and user_message.strip():
            self.history.append({"role": "user", "content": user_message})
        if bot_message and isinstance(bot_message, str) and bot_message.strip():
            self.history.append({"role": "assistant", "content": bot_message})
        if self.history_limit > 0 and len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def _clean_history(self) -> List[Dict[str, str]]:
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

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str) -> str:
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": "CONTEXT_PLACEHOLDER",
        }
        prompt_wo_context = prompt_template
        for key, value in replacements.items():
            prompt_wo_context = prompt_wo_context.replace(key, value)
        prompt_wo_context_len = len(prompt_wo_context.replace("CONTEXT_PLACEHOLDER", ""))

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

    def _build_messages(self, prompt_template: str, topic: str, context: str) -> List[Dict[str, str]]:
        context = self._truncate_context_for_llm(prompt_template, topic, context)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context,
        }
        prompt = prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        prompt = prompt.replace("nan", "").strip()
        
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        messages.append({"role": "user", "content": prompt})
        
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > self.LM_MAX_TOTAL_CHARS:
            self.logger.warning(f"Total LLM payload too long ({total_chars} > {self.LM_MAX_TOTAL_CHARS}), trimming prompt/history")
            excess = total_chars - self.LM_MAX_TOTAL_CHARS
            if len(messages[-1]["content"]) > excess:
                messages[-1]["content"] = messages[-1]["content"][:len(messages[-1]["content"]) - excess]
            else:
                while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                    removed = messages.pop(1)
                    self.logger.warning(f"Removed old history message to fit LM payload: {removed}")
                    total_chars = sum(len(m["content"]) for m in messages)
                if total_chars > self.LM_MAX_TOTAL_CHARS:
                    last = messages[-1]
                    last["content"] = last["content"][:max(0, len(last["content"]) - (total_chars - self.LM_MAX_TOTAL_CHARS))]
        return messages

    def _request_shorter_content(self, prompt_template: str, topic: str, context: str, current_length: int) -> str:
        shorter_prompt = f"{prompt_template}\n\nВАЖНО: Твой предыдущий ответ был слишком длинный ({current_length} символов). Напиши более короткий ответ, который поместится в {self.TELEGRAM_LIMIT} символов. Сохрани все важные детали, но сделай текст более сжатым."
        messages = self._build_messages(shorter_prompt, topic, context)
        return self._make_request(messages)

    def _make_request(self, messages: List[Dict[str, str]]) -> str:
        chat_url = f"{self.base_url}/v1/chat/completions"
        payload_chat = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if self.top_p is not None:
            payload_chat["top_p"] = self.top_p
        if self.top_k is not None:
            payload_chat["top_k"] = self.top_k

        self.logger.debug(f"LMStudioClient: Sending chat payload to {chat_url}: {str(payload_chat)[:800]}")

        response = requests.post(chat_url, json=payload_chat, timeout=self.timeout)
        self.logger.debug(f"LMStudioClient: raw response: {response.text[:1000]}")
        response.raise_for_status()
        
        try:
            result = response.json()
        except Exception as e:
            self.logger.error("Failed to decode LM Studio response as JSON", exc_info=True)
            result = {}
        
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not text:
            self.logger.warning("Empty chat response, fallback to completions endpoint.")
            comp_url = f"{self.base_url}/v1/completions"
            payload = {
                "model": self.model,
                "prompt": messages[-1]['content'],
                "max_tokens": self.max_tokens,
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

        if not isinstance(text, str):
            raise ValueError("LM Studio returned non-string result.")
        return text.strip()

    def generate_content(self, prompt_template: str, topic: str, context: str, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        
        try:
            messages = self._build_messages(prompt_template, topic, context)
            text = self._make_request(messages)
            
            if len(text) > self.TELEGRAM_LIMIT:
                self.logger.warning(f"Generated content too long ({len(text)} chars), requesting shorter version")
                attempts = 0
                max_attempts = 3
                
                while len(text) > self.TELEGRAM_LIMIT and attempts < max_attempts:
                    attempts += 1
                    self.logger.info(f"Attempt {attempts} to get shorter content")
                    try:
                        text = self._request_shorter_content(prompt_template, topic, context, len(text))
                    except Exception as e:
                        self.logger.warning(f"Failed to get shorter content on attempt {attempts}: {e}")
                        if attempts == max_attempts:
                            self.logger.warning("Max attempts reached, truncating content")
                            text = text[:self.TELEGRAM_LIMIT-10] + "..."
                            break
                
                if len(text) <= self.TELEGRAM_LIMIT:
                    self.logger.info(f"Successfully got shorter content ({len(text)} chars)")

            if text and text.strip():
                self.add_to_history(messages[-1]['content'], text)
            else:
                self.logger.warning("LM Studio returned empty text from both endpoints.")
            
            return text.strip()
        finally:
            self.max_tokens = original_max_tokens

    def generate_with_retry(self, prompt_template: str, topic: str, context: str, max_retries: int = 3) -> str:
        last_err = None
        original_context = context
        
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"LMStudioClient: generation attempt {attempt}/{max_retries}")
                text = self.generate_content(prompt_template, topic, context)
                if text and text.strip():
                    return text
                self.logger.warning(f"LM Studio generation returned empty text (attempt {attempt})")
            except Exception as e:
                last_err = e
                msg = str(e)
                self.logger.warning(f"LMStudioClient: error on attempt {attempt}: {msg}")
                if "413" in msg or "400" in msg or "payload" in msg:
                    context = context[:max(100, len(context) // 2)]
                    self.logger.warning("Reducing context and retrying...")
                if attempt == max_retries or (self.history_limit and len(self.history) > self.history_limit * 4):
                    self.logger.warning("Clearing conversation history due to repeated failures.")
                    self.clear_conversation_history()
        
        try:
            return self.generate_content(prompt_template, topic, original_context[:256])
        except Exception as e:
            self.logger.error("Final fallback attempt failed", exc_info=True)
            raise ValueError(f"LM Studio did not generate content after {max_retries} attempts: {last_err}")
