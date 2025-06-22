# modules/content_generation/prompt_builder.py

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict

class PromptBuilder:
    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]

    def __init__(self, prompt_folders: List[str]):
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates = {}
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                continue

            self.templates[str(folder)] = self.scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def scan_prompt_folder(self, folder_path: Path) -> List[str]:
        return [str(p) for p in folder_path.glob("*.txt")]

    def select_random_templates(self) -> Tuple[str, str, str]:
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            if templates:
                selected.append(random.choice(templates))
            else:
                selected.append("")
        return tuple(selected)

    def read_template_file(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read prompt file: {file_path}", exc_info=True)
            return ""

    def build_prompt(self, topic: str, context: str, media_file: str = None) -> str:
        t1, t2, t3 = self.select_random_templates()
        content = "\n\n".join([
            self.read_template_file(t1),
            self.read_template_file(t2),
            self.read_template_file(t3)
        ])

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
            "{UPLOADFILE}": media_file.strip() if media_file else ""
        }

        prompt = self.replace_placeholders(content, replacements)

        if not self.validate_prompt_structure(prompt):
            raise ValueError("Prompt structure validation failed. Missing required placeholders.")

        return prompt

    def replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def validate_prompt_structure(self, prompt: str) -> bool:
        for required in self.REQUIRED_PLACEHOLDERS:
            if required not in prompt:
                self.logger.error(f"Prompt missing required placeholder: {required}")
                return False
        return True

    def check_placeholder_presence(self, template: str) -> Dict[str, bool]:
        return {ph: (ph in template) for ph in self.REQUIRED_PLACEHOLDERS + ["{UPLOADFILE}"]}

    def get_template_stats(self) -> Dict[str, int]:
        return {folder: len(templates) for folder, templates in self.templates.items()}

    def reload_templates(self) -> None:
        self.logger.info("Reloading prompt templates...")
        self.load_prompt_templates()

    def test_template_combination(self, topic: str, context: str) -> str:
        return self.build_prompt(topic, context, media_file="media/sample.jpg")
