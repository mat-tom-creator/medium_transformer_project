import datasets
from typing import List, Dict
import logging

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def collect_wikitext(self, max_samples: int = 100000) -> List[str]:
        """Collect data from WikiText dataset."""
        try:
            dataset = datasets.load_dataset('wikitext', 
                                         'wikitext-103-raw-v1', 
                                         split=f'train[:{max_samples}]')
            return [item['text'] for item in dataset if item['text'].strip()]
        except Exception as e:
            self.logger.error(f"Error loading WikiText: {e}")
            return []

    def collect_books(self, max_samples: int = 10000) -> List[str]:
        """Collect data from BookCorpus dataset."""
        try:
            dataset = datasets.load_dataset('bookcorpus', 
                                         split=f'train[:{max_samples}]')
            return [item['text'] for item in dataset if item['text'].strip()]
        except Exception as e:
            self.logger.error(f"Error loading BookCorpus: {e}")
            return []

    def save_dataset(self, texts: List[str], output_path: str):
        """Save collected texts to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n\n')