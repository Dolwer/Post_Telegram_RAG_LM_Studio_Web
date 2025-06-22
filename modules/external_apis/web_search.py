# modules/external_apis/web_search.py

import logging
import requests
from typing import List, Dict
from pathlib import Path

class WebSearchClient:
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
        self.logger = logging.getLogger("WebSearchClient")
        self.endpoint = "https://google.serper.dev/search"

    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json().get("organic", [])
            self.logger.info(f"Found {len(results)} search results for: {query}")
            return results
        except Exception as e:
            self.logger.error(f"Search failed for: {query}", exc_info=True)
            return []

    def extract_content(self, search_results: List[Dict]) -> str:
        contents = [res.get("snippet", "") for res in search_results if "snippet" in res]
        return "\n\n".join(contents)

    def filter_relevant_results(self, results: List[Dict], topic: str) -> List[Dict]:
        filtered = [r for r in results if topic.lower() in r.get("title", "").lower()]
        return filtered or results

    def build_search_query(self, topic: str) -> str:
        return f"{topic} site:medium.com OR site:forbes.com OR site:wikipedia.org"

    def format_search_context(self, results: List[Dict]) -> str:
        formatted = []
        for r in results:
            url = r.get("link", "#")
            snippet = r.get("snippet", "")
            formatted.append(f"{snippet}\nИсточник: {url}")
        return "\n\n".join(formatted)

    def save_to_inform(self, content: str, topic: str, source: str = "web") -> None:
        folder = Path("inform/web")
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{topic.replace(' ', '_')}_{source}.txt"
        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Web content saved: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save web content for {topic}", exc_info=True)

    def validate_search_results(self, results: List[Dict]) -> bool:
        return len(results) > 0

    def handle_rate_limits(self, response: Dict) -> bool:
        if response.get("error", "").lower().find("rate") != -1:
            self.logger.warning("API rate limit reached.")
            return True
        return False

    def clean_search_content(self, content: str) -> str:
        return content.replace("\u200b", "").strip()

    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for r in results:
            url = r.get("link")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)
        return unique

    def get_search_stats(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "limit": self.config.get("results_limit", 10)
        }
