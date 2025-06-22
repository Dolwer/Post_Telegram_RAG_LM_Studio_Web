# modules/external_apis/web_search.py

import logging
import requests
import time
from typing import List, Dict
from pathlib import Path

class WebSearchClient:
    def __init__(self, api_key: str, endpoint: str = "https://google.serper.dev/search", results_limit: int = 10):
        self.api_key = api_key
        self.endpoint = endpoint
        self.results_limit = results_limit
        self.logger = logging.getLogger("WebSearchClient")

    def search(self, query: str, num_results: int = None) -> List[Dict]:
        num = num_results if num_results is not None else self.results_limit
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num}
        try:
            for attempt in range(2):
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=15)
                if response.status_code == 403:
                    self.logger.error("403 Forbidden: Invalid or expired Serper API key")
                    self.logger.error(f"Response: {response.text}")
                    return []
                if response.status_code == 429:
                    self.logger.warning("429 Too Many Requests: Rate limit exceeded, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    self.logger.error(f"API error: {data.get('error')}")
                    return []
                results = data.get("organic", [])
                self.logger.info(f"Found {len(results)} search results for: {query}")
                return results
        except Exception as e:
            self.logger.error(f"Search failed for: {query}", exc_info=True)
        return []

    def build_search_query(self, topic: str) -> str:
        return topic

    def extract_content(self, search_results: List[Dict]) -> str:
        contents = [res.get("snippet", "") for res in search_results if "snippet" in res]
        return "\n\n".join(contents)

    def filter_relevant_results(self, results: List[Dict], topic: str) -> List[Dict]:
        filtered = [r for r in results if topic.lower() in r.get("title", "").lower()]
        return filtered or results

    def save_to_inform(self, content: str, topic: str, source: str = "web") -> None:
        folder = Path("inform/web")
        folder.mkdir(parents=True, exist_ok=True)
        safe_topic = "".join([c if c.isalnum() or c in " _-" else "_" for c in topic])
        file_path = folder / f"{safe_topic}_{source}.txt"
        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Web content saved: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save web content for {topic}", exc_info=True)

    def format_search_context(self, results: List[Dict]) -> str:
        formatted = []
        for r in results:
            url = r.get("link", "#")
            snippet = r.get("snippet", "")
            formatted.append(f"{snippet}\nИсточник: {url}")
        return "\n\n".join(formatted)

    def get_search_stats(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "results_limit": self.results_limit
        }

    def validate_search_results(self, results: List[Dict]) -> bool:
        return len(results) > 0

    def handle_rate_limits(self, response: Dict) -> bool:
        if "error" in response and "rate" in response["error"].lower():
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
