from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict, Any
import numpy as np
from config.settings import Settings
from src.embedding_client import EmbeddingClient

class MilvusVectorStore:
    def __init__(self):
        self.settings = Settings()
        self.embedding_client = EmbeddingClient()
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.settings.MILVUS_HOST,
            port=self.settings.MILVUS_PORT,
            user=self.settings.MILVUS_USER,
            password=self.settings.MILVUS_PASSWORD
        )
        
        self.collection_name = self.settings.COLLECTION_NAME
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """
        Create collection if it doesn't exist
        """
        try:
            # Check if collection exists
            if not self.collection_exists():
                # Define schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.settings.EMBEDDING_DIM),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                
                schema = CollectionSchema(
                    fields=fields,
                    description="Knowledge base collection"
                )
                
                # Create collection
                collection = Collection(name=self.collection_name, schema=schema)
                
                # Create index
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                }
                
                collection.create_index(field_name="embedding", index_params=index_params)
                print(f"Collection '{self.collection_name}' created successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def collection_exists(self) -> bool:
        """
        Check if collection exists
        """
        try:
            from pymilvus import utility
            return utility.has_collection(self.collection_name)
        except:
            return False
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """
        Insert documents into the collection
        """
        try:
            collection = Collection(self.collection_name)
            
            # Prepare data
            contents = []
            embeddings = []
            metadata_list = []
            
            for doc in documents:
                contents.append(doc['content'])
                metadata_list.append(doc.get('metadata', {}))
            
            # Create embeddings
            embeddings = self.embedding_client.create_embeddings(contents)
            
            # Insert data
            data = [contents, embeddings, metadata_list]
            insert_result = collection.insert(data)
            
            # Load collection for search
            collection.load()
            
            return insert_result.primary_keys
        except Exception as e:
            print(f"Error inserting documents: {e}")
            return []
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query
        """
        try:
            collection = Collection(self.collection_name)
            
            # Create embedding for query
            query_embedding = self.embedding_client.create_single_embedding(query)
            
            # Load collection if not loaded
            if not collection.is_empty:
                collection.load()
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "metadata"]
            )
            
            # Format results
            retrieved_docs = []
            for hits in results:
                for hit in hits:
                    retrieved_docs.append({
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "distance": hit.distance
                    })
            
            return retrieved_docs
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def delete_collection(self):
        """
        Delete the collection (useful for testing)
        """
        try:
            from pymilvus import utility
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' deleted.")
        except Exception as e:
            print(f"Error deleting collection: {e}")