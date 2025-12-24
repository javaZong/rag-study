from typing import List, Dict, Any
from src.llm_client import QwenLLMClient
from src.vector_store import MilvusVectorStore
from config.settings import Settings
import re

class RAGEngine:
    def __init__(self):
        self.settings = Settings()
        self.llm_client = QwenLLMClient()
        self.vector_store = MilvusVectorStore()
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document to the knowledge base
        """
        try:
            # Split content into chunks
            chunks = self._split_text(content)
            
            # Prepare documents for insertion
            documents = []
            for chunk in chunks:
                doc = {
                    "content": chunk,
                    "metadata": metadata or {}
                }
                documents.append(doc)
            
            # Insert into vector store
            ids = self.vector_store.insert_documents(documents)
            
            return len(ids) > 0
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def _split_text(self, text: str) -> List[str]:
        """
        Simple text splitting function
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.settings.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too long, split it
                if len(para) > self.settings.CHUNK_SIZE:
                    sub_chunks = self._split_long_paragraph(para)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk) > 20]  # Filter out very short chunks
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """
        Split a long paragraph into smaller chunks
        """
        sentences = re.split(r'[.!?。！？]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > self.settings.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk) > 20]
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the knowledge base and generate answer
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.search_similar(
                question, 
                top_k=self.settings.TOP_K
            )
            
            # Extract content from retrieved documents
            context = [doc["content"] for doc in retrieved_docs]
            
            # Generate response using LLM
            answer = self.llm_client.generate_response(question, context)
            
            return {
                "answer": answer,
                "context": context,
                "retrieved_docs": retrieved_docs
            }
        except Exception as e:
            print(f"Error querying: {e}")
            return {
                "answer": "抱歉，我在处理您的查询时遇到了问题。",
                "context": [],
                "retrieved_docs": []
            }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat interface for conversation-style interaction
        """
        return self.llm_client.chat_completion(messages)