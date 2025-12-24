import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")
    EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")  # Using OpenAI embedding API
    
    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
    
    # Model Configuration
    LLM_MODEL = "qwen3-max"  # Using qwen3-max as specified
    EMBEDDING_MODEL = "text-embedding-v4"  # Using text-embedding-v4 as specified
    EMBEDDING_DIM = 1536  # Dimension for text-embedding-v4
    
    # Processing Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 5  # Number of similar documents to retrieve