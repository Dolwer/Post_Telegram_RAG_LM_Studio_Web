# modules/utils/media_handler.py

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
from contextlib import suppress
import mimetypes

class MediaHandler:
    def __init__(self, media_folder: str, config: dict):
        self.media_folder = Path(media_folder)
        self.config = config
        self.logger = logging.getLogger("MediaHandler")

        self.max_file_size_mb = config["processing"].get("max_file_size_mb", 50)
        self.max_image_size = (1280, 1280)

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {
            "image": [".jpg", ".jpeg", ".png"],
            "video": [".mp4", ".mov"],
            "document": [".pdf", ".docx", ".txt"]
        }

    def get_random_media_file(self) -> Optional[str]:
        files = self.get_media_files_list()
        if not files:
            self.logger.warning("No media files available.")
            return None
        selected = random.choice(files)
        self.logger.info(f"Selected media file: {selected}")
        return selected

    def get_media_files_list(self) -> List[str]:
        all_files = []
        for ext_list in self.get_supported_formats().values():
            for ext in ext_list:
                all_files.extend(self.media_folder.glob(f"*{ext}"))
        return [str(f) for f in all_files if self.validate_media_file(f)]

    def validate_media_file(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        if file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
            self.logger.warning(f"File too large: {file_path}")
            return False
        return True

     def send_media_message(self, text: str, media_path: str) -> bool:
        if not self.validate_message_length(text, has_media=True):
            self.logger.warning("Caption too long. Skipping media post.")
            return False

        media_type = self.get_media_type(media_path)
        method = {
            "photo": "sendPhoto",
            "video": "sendVideo",
            "document": "sendDocument"
        }.get(media_type)

        if not method:
            self.logger.error(f"Unsupported media type for {media_path}")
            return False

        try:
            with open(media_path, "rb") as media_file:
                files = {media_type: media_file}
                data = {
                    "chat_id": self.channel_id,
                    "caption": self.format_message(text),
                    "parse_mode": "HTML"
                }
                response = requests.post(f"{self.api_base}/{method}", data=data, files=files)
                response.raise_for_status()
                self.logger.info(f"Media message sent: {media_path}")
                return True
        except Exception as e:
            self.logger.error(f"Telegram API failed to send media", exc_info=True)
            return False

    def get_media_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type.startswith("image/"):
                return "photo"
            elif mime_type.startswith("video/"):
                return "video"
            else:
                return "document"
        return "document"

    def get_file_size(self, file_path: str) -> int:
        return Path(file_path).stat().st_size

    def resize_image_if_needed(self, file_path: str, max_size: Tuple[int, int] = None) -> str:
        max_size = max_size or self.max_image_size
        try:
            img = Image.open(file_path)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size)
                img.save(file_path)
                self.logger.info(f"Image resized: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to resize image: {file_path}", exc_info=True)
        return file_path

    def compress_video_if_needed(self, file_path: str) -> str:
        # Заглушка: Для продакшена требуется ffmpeg
        self.logger.debug("compress_video_if_needed(): Not implemented")
        return file_path

    def get_image_dimensions(self, file_path: str) -> Tuple[int, int]:
        try:
            img = Image.open(file_path)
            return img.size
        except Exception as e:
            self.logger.warning(f"Can't read image: {file_path}", exc_info=True)
            return (0, 0)

    def get_video_duration(self, file_path: str) -> float:
        # Заглушка: Для продакшена требуется ffprobe
        return 0.0

    def create_thumbnail(self, file_path: str) -> Optional[str]:
        try:
            img = Image.open(file_path)
            thumb_path = str(Path(file_path).with_name(f"thumb_{Path(file_path).name}"))
            img.thumbnail((320, 320))
            img.save(thumb_path)
            return thumb_path
        except Exception as e:
            self.logger.warning(f"Failed to create thumbnail: {file_path}", exc_info=True)
            return None

     def get_media_stats(self) -> dict:
        stats = {
            "total": 0,
            "photos": 0,
            "videos": 0,
            "documents": 0
        }
        for file in self.get_media_files_list():
            mtype = self.get_media_type(file)
            stats["total"] += 1
            if mtype == "photo":
                stats["photos"] += 1
            elif mtype == "video":
                stats["videos"] += 1
            else:
                stats["documents"] += 1
        return stats

    def compress_video_if_needed(self, file_path: str) -> str:
        self.logger.warning("Video compression not implemented. File returned as-is.")
        return file_path

    def get_media_type(self, file_path: str) -> str:
        ext = file_path.lower().split('.')[-1]
        if ext in ["jpg", "jpeg", "png"]:
            return "photo"
        elif ext in ["mp4", "mov"]:
            return "video"
        elif ext in ["pdf", "docx", "txt"]:
            return "document"
        else:
            self.logger.warning(f"Unknown media extension: {file_path}")
            return "document"

    def cleanup_processed_media(self):
        for f in self.media_folder.glob("thumb_*"):
            try:
                f.unlink()
                self.logger.debug(f"Deleted: {f}")
            except Exception as e:
                self.logger.warning(f"Can't delete thumbnail: {f}", exc_info=True)
