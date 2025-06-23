import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class PromptBuilder:
    """
    Отвечает за сборку промпта из шаблонов, подстановку переменных и валидацию.
    """
    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]

    def __init__(self, prompt_folders: List[str]):
        """
        :param prompt_folders: список путей к папкам с шаблонами промптов.
        """
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        """
        Загружает все шаблоны из указанных папок в self.templates.
        """
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                self.templates[str(folder)] = []
                continue
            self.templates[str(folder)] = self.scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def scan_prompt_folder(self, folder_path: Path) -> List[str]:
        """
        Возвращает список путей к .txt-файлам в папке.
        """
        return [str(p) for p in folder_path.glob("*.txt")]

    def select_random_templates(self) -> List[str]:
        """
        Случайно выбирает по одному шаблону из каждой папки. 
        Если в папке нет шаблонов — вместо файла будет пустая строка.
        """
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            if templates:
                selected.append(random.choice(templates))
            else:
                selected.append("")
        return selected

    def read_template_file(self, file_path: str) -> str:
        """
        Читает содержимое файла шаблона.
        Если файл не читается — пишет ошибку в лог, возвращает пустую строку.
        """
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content
        except Exception as e:
            self.logger.error(f"Failed to read prompt file: {file_path}", exc_info=True)
            return ""

    def validate_prompt_structure(self, template: str) -> bool:
        """
        Проверяет, что в шаблоне (до замены) есть все обязательные плейсхолдеры.
        """
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            self.logger.error(f"Prompt missing required placeholders: {missing}")
            return False
        return True

    def replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        """
        Заменяет все плейсхолдеры на их значения.
        """
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def build_prompt(self, topic: str, context: str, media_file: Optional[str] = None) -> str:
        """
        Собирает итоговый промпт:
        - Случайно выбирает по одному шаблону из каждой папки
        - Склеивает их содержимое
        - Проверяет, что до подстановки есть все обязательные плейсхолдеры
        - Подставляет значения плейсхолдеров
        :param topic: Тема для подстановки
        :param context: Контекст для подстановки
        :param media_file: Путь к медиафайлу (необязательно)
        :return: Сгенерированный промпт
        :raises ValueError: если не хватает плейсхолдеров
        """
        template_paths = self.select_random_templates()
        template_texts = [self.read_template_file(path) for path in template_paths]
        content = "\n\n".join(template_texts).strip()

        # Проверяем до замены — все ли плейсхолдеры есть в склеенном шаблоне
        if not self.validate_prompt_structure(content):
            raise ValueError("Prompt structure validation failed. Missing required placeholders in templates.")

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
            "{UPLOADFILE}": media_file.strip() if media_file else ""
        }

        prompt = self.replace_placeholders(content, replacements)

        # Логируем неиспользованные плейсхолдеры для отладки
        unused = [ph for ph in self.REQUIRED_PLACEHOLDERS + ["{UPLOADFILE}"] if ph in prompt]
        if unused:
            self.logger.warning(f"Prompt still contains unused placeholders: {unused}")

        # Дополнительно: очистка двойных/тройных пустых строк для компактности
        prompt = self._compact_whitespace(prompt)
        self.logger.debug(f"Final prompt (truncated): {prompt[:1000]}")
        return prompt

    def _compact_whitespace(self, text: str) -> str:
        """Удаляет избыточные пустые строки для компактности и чистоты промпта."""
        return os.linesep.join([line for line in text.splitlines() if line.strip()])

    def check_placeholder_presence(self, template: str) -> Dict[str, bool]:
        """
        Возвращает словарь вида {плейсхолдер: True/False} — есть ли плейсхолдер в шаблоне.
        """
        all_ph = self.REQUIRED_PLACEHOLDERS + ["{UPLOADFILE}"]
        return {ph: (ph in template) for ph in all_ph}

    def get_template_stats(self) -> Dict[str, int]:
        """
        Возвращает статистику (количество шаблонов) по каждой папке.
        """
        return {folder: len(templates) for folder, templates in self.templates.items()}

    def reload_templates(self) -> None:
        """
        Перезагружает шаблоны из файловой системы.
        """
        self.logger.info("Reloading prompt templates...")
        self.load_prompt_templates()

    def test_template_combination(self, topic: str, context: str) -> str:
        """
        Генерирует промпт с тестовым медиафайлом (для отладки).
        """
        return self.build_prompt(topic, context, media_file="media/sample.jpg")
