# modules/utils/state_manager.py

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

class StateManager:
    def __init__(self, state_file: str = "data/state.json"):
        self.state_file = Path(state_file)
        self.logger = logging.getLogger("StateManager")

        self.state = {
            "processed": {},
            "unprocessed": [],
            "failed": {},
            "statistics": {
                "topics_processed": 0,
                "success_count": 0,
                "error_count": 0,
                "start_time": datetime.utcnow().isoformat()
            },
            "system_status": "INIT",
            "last_activity": datetime.utcnow().isoformat()
        }

        self.load_state()

    def load_state(self) -> None:
        if self.state_file.exists():
            try:
                with self.state_file.open("r", encoding="utf-8") as f:
                    self.state = json.load(f)
                    self.logger.info("State loaded.")
            except Exception as e:
                self.logger.error("Failed to load state.", exc_info=True)

    def save_state(self) -> None:
        try:
            with self.state_file.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
            self.logger.debug("State saved.")
        except Exception as e:
            self.logger.error("Failed to save state.", exc_info=True)

    def mark_topic_processed(self, topic: str, success: bool, details: dict = None) -> None:
        if success:
            self.state["processed"][topic] = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {}
            }
            self.state["statistics"]["success_count"] += 1
        else:
            self.state["failed"][topic] = {
                "status": "failed",
                "timestamp": datetime.utcnow().isoformat(),
                "error": details or {}
            }
            self.state["statistics"]["error_count"] += 1

        if topic in self.state["unprocessed"]:
            self.state["unprocessed"].remove(topic)

        self.state["statistics"]["topics_processed"] += 1
        self.state["last_activity"] = datetime.utcnow().isoformat()
        self.save_state()

    def get_next_unprocessed_topic(self) -> Optional[str]:
        return self.state["unprocessed"][0] if self.state["unprocessed"] else None

    def get_unprocessed_topics(self) -> List[str]:
        return self.state["unprocessed"]

    def get_processed_topics(self) -> List[str]:
        return list(self.state["processed"].keys())

    def get_failed_topics(self) -> List[str]:
        return list(self.state["failed"].keys())

    def reset_failed_topics(self) -> None:
        self.state["unprocessed"].extend(self.get_failed_topics())
        self.state["failed"] = {}
        self.save_state()
        self.logger.info("Failed topics reset to unprocessed.")

    def get_processing_statistics(self) -> Dict[str, Any]:
        return self.state["statistics"]

    def add_processing_stats(self, stats: dict) -> None:
        self.state["statistics"].update(stats)
        self.save_state()

    def get_system_uptime(self) -> float:
        start = datetime.fromisoformat(self.state["statistics"]["start_time"])
        return (datetime.utcnow() - start).total_seconds()

    def set_system_status(self, status: str) -> None:
        self.state["system_status"] = status
        self.save_state()

    def get_last_activity(self) -> datetime:
        return datetime.fromisoformat(self.state["last_activity"])

    def backup_state(self) -> str:
        backup_path = self.state_file.with_suffix(".backup.json")
        try:
            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
            self.logger.info(f"State backup saved to {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error("Failed to backup state.", exc_info=True)
            return ""

    def restore_state(self, backup_path: str) -> bool:
        path = Path(backup_path)
        if not path.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            with path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            self.logger.info("State restored from backup.")
            self.save_state()
            return True
        except Exception as e:
            self.logger.error("Failed to restore state.", exc_info=True)
            return False
