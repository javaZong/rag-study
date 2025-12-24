import openai
from typing import List
import numpy as np
from config.settings import Settings

class EmbeddingClient:
    def __init__(self):
        self.api_key = Settings.EMBEDDING_API_KEY
        self.model = Settings.EMBEDDING_MODEL
        openai.api_key = self.api_key

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using text-embedding-v4
        """
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts
            )
            
            embeddings = []
            for item in response['data']:
                embeddings.append(item['embedding'])
            
            return embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []

    def create_single_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        """
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=[text]
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error creating single embedding: {e}")
            return []