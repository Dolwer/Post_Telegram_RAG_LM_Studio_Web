import os
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class PromptBuilder:
    """
    Сборка промпта из шаблонов с жестким контролем структуры, подстановки плейсхолдеров,
    и надежным возвратом данных для всей цепочки генерации.
    """

    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]
    OPTIONAL_PLACEHOLDERS = ["{UPLOADFILE}"]
    PLACEHOLDER_PATTERN = re.compile(r"\{[A-Z_]+\}")

    def __init__(self, prompt_folders: List[str]):
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self.error_history: List[Dict] = []
        self._last_prompt_template: Optional[str] = None
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        """Загружает шаблоны из всех указанных папок, сохраняет пути."""
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                self.templates[str(folder)] = []
                continue
            self.templates[str(folder)] = self._scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def _scan_prompt_folder(self, folder_path: Path) -> List[str]:
        return [str(p) for p in folder_path.glob("*.txt")]

    def _select_random_templates(self) -> List[str]:
        """Из каждой папки выбирает случайный шаблон. Если нет файлов — None."""
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            selected.append(random.choice(templates) if templates else None)
        return selected

    def _read_template_file(self, file_path: Optional[str]) -> str:
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            msg = f"Failed to read prompt file: {file_path}"
            self._log_error(msg, exc=e)
            return ""

    def _validate_prompt_structure(self, template: str) -> None:
        """Проверяет наличие всех обязательных плейсхолдеров. Кидает ошибку если невалидно."""
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            msg = f"Prompt missing required placeholders: {missing}"
            self._log_error(msg)
            self.logger.error(msg)
            raise ValueError(msg)
        # Логируем дубли, не валим работу
        for ph in self.REQUIRED_PLACEHOLDERS + self.OPTIONAL_PLACEHOLDERS:
            if ph * 2 in template:
                self.logger.warning(f"Prompt contains duplicated placeholder: {ph}{ph}")
        # Проверка на неизвестные плейсхолдеры
        all_placeholders = set(self.PLACEHOLDER_PATTERN.findall(template))
        supported = set(self.REQUIRED_PLACEHOLDERS + self.OPTIONAL_PLACEHOLDERS)
        unsupported = [ph for ph in all_placeholders if ph not in supported]
        if unsupported:
            self.logger.warning(f"Prompt contains unsupported placeholders: {unsupported}")

    def _find_unresolved_placeholders(self, text: str) -> List[str]:
        return list(set(self.PLACEHOLDER_PATTERN.findall(text)))

    def _compact_whitespace(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)

    def _replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def _has_uploadfile_placeholder(self, template: str) -> bool:
        return "{UPLOADFILE}" in template

    def build_prompt(
        self,
        topic: str,
        context: str,
        media_file: Optional[str] = None
    ) -> Tuple[str, bool, str]:
        """
        Возвращает:
            - prompt (str): итоговый текст промпта без неразрешённых плейсхолдеров
            - has_uploadfile (bool): True, если в шаблоне был {UPLOADFILE} и он был заменён
            - prompt_template (str): оригинальный текст шаблона (до подстановки)
        """
        # 1. Проверка входных данных
        if not topic or not isinstance(topic, str):
            msg = "Topic for prompt_builder is empty or not a string."
            self._log_error(msg, context={"topic": topic})
            self.logger.error(msg)
            raise ValueError(msg)
        if not context or not isinstance(context, str):
            msg = "Context for prompt_builder is empty or not a string."
            self._log_error(msg, context={"context": context})
            self.logger.error(msg)
            raise ValueError(msg)

        # 2. Выбор шаблонов
        template_paths = self._select_random_templates()
        template_texts = [self._read_template_file(path) for path in template_paths if path]
        # Если все шаблоны пусты — дефолт
        if not any(template_texts):
            prompt_template = self._default_template()
            self.logger.warning("No prompt templates found, using default.")
        else:
            prompt_template = "\n\n".join(filter(None, template_texts)).strip()

        self._last_prompt_template = prompt_template

        # 3. Проверка структуры ДО подстановки
        self._validate_prompt_structure(prompt_template)

        # 4. Определяем, нужен ли uploadfile
        has_uploadfile = self._has_uploadfile_placeholder(prompt_template)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
            "{UPLOADFILE}": media_file.strip() if (has_uploadfile and media_file) else "",
        }
        prompt = self._replace_placeholders(prompt_template, replacements)

        # 5. Проверка после подстановки: не осталось ли плейсхолдеров
        unresolved = self._find_unresolved_placeholders(prompt)
        critical_unresolved = [
            ph for ph in unresolved if
            ph in self.REQUIRED_PLACEHOLDERS or
            (ph == "{UPLOADFILE}" and has_uploadfile)
        ]
        if critical_unresolved:
            msg = f"Prompt contains unresolved placeholders after replacement: {critical_unresolved}"
            self._log_error(msg, context={"prompt": prompt, "replacements": replacements})
            self.logger.error(msg)
            raise ValueError(msg)

        # 6. Очистка whitespace
        prompt = self._compact_whitespace(prompt)

        self.logger.debug(
            f"PromptBuilder: topic='{topic[:100]}', context_len={len(context)}, media_file='{media_file}', has_uploadfile={has_uploadfile}")
        self.logger.debug(f"PromptBuilder: template_paths={template_paths}")
        self.logger.debug(f"PromptBuilder: prompt after replacement (truncated): {prompt[:1000]}")

        return prompt, has_uploadfile, prompt_template

    def _log_error(self, msg: str, context: Optional[dict] = None, exc: Exception = None):
        entry = {
            "message": msg,
            "context": context,
            "exception": repr(exc) if exc else None
        }
        self.error_history.append(entry)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

    def get_error_history(self) -> List[Dict]:
        return self.error_history.copy()

    def check_placeholder_presence(self, template: str) -> Dict[str, bool]:
        all_ph = self.REQUIRED_PLACEHOLDERS + self.OPTIONAL_PLACEHOLDERS
        return {ph: (ph in template) for ph in all_ph}

    def get_template_stats(self) -> Dict[str, int]:
        return {folder: len(templates) for folder, templates in self.templates.items()}

    def reload_templates(self) -> None:
        self.logger.info("Reloading prompt templates...")
        self.load_prompt_templates()

    def test_template_combination(self, topic: str, context: str) -> Tuple[str, bool, str]:
        return self.build_prompt(topic, context, media_file="media/sample.jpg")

    def _default_template(self) -> str:
        return (
            "Ты опытный механик грузовой техники с 15-летним стажем работы.\n"
            "Работал в крупных автопарках, ремонтировал краны, фургоны, бортовые машины. "
            "Знаешь все подводные камни эксплуатации. Говоришь простым языком, приводишь примеры из практики.\n\n"
            "Тема: {TOPIC}\n\n"
            "Контекст для анализа: {CONTEXT}\n"
            "{UPLOADFILE}"
        )

    @property
    def last_prompt_template(self) -> Optional[str]:
        return self._last_prompt_template
