import chromadb
import uuid
import json
from typing import List, Dict, Any
from litellm import completion, embedding
from .config import Config

class MemoryLayer:
    def __init__(self):
        # Initialize ChromaDB
        # PersistentClient saves data to disk at the path specified in Config
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        
        # Get or create a collection for memories
        self.collection = self.client.get_or_create_collection(name="project_memories")
        
        print(f"[Memory] Initialized ChromaDB at {Config.VECTOR_DB_PATH}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding using the configured provider via LiteLLM."""
        
        # Prepare kwargs
        kwargs = {
            "model": Config.EMBEDDING_MODEL_STRING,
            "input": [text]
        }
        
        # If using Ollama, explicit api_base is often helpful
        if Config.EMBEDDING_PROVIDER == "ollama":
            kwargs["api_base"] = Config.OLLAMA_API_BASE

        response = embedding(**kwargs)
        # generic response structure for litellm
        return response.data[0]["embedding"]

    def _extract_facts(self, user_input: str) -> str:
        """Uses LLM to extract granular facts/rules from user input."""
        prompt = f"""
        Extract the key constraints, rules, or facts from the following text.
        Focus on numerical limits, banned items, preferred vendors, or site-specific instructions.
        Return ONLY the extracted facts as a single concise sentence.

        Input: "{user_input}"
        Facts:
        """
        response = completion(
            model=Config.LLM_MODEL_STRING,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def add(self, user_input: str, metadata: Dict[str, Any] = None):
        """Extracts facts from input, embeds them, and stores them in ChromaDB."""
        fact = self._extract_facts(user_input)
        vector = self._get_embedding(fact)
        
        # ChromaDB requires unique IDs
        mem_id = str(uuid.uuid4())
        
        # Prepare metadata (ensure flat dict for Chroma)
        meta = metadata or {}
        meta["original_input"] = user_input
        meta["timestamp"] = "2024-01-01T12:00:00" # Placeholder
        
        self.collection.add(
            documents=[fact],
            embeddings=[vector],
            metadatas=[meta],
            ids=[mem_id]
        )
        
        print(f"Stored memory: {fact}")
        return fact

    def get_all(self) -> List[Dict]:
        """Retrieves all stored memories."""
        count = self.collection.count()
        if count == 0:
            return []
            
        # Retrieve all (up to a reasonable limit for UI)
        results = self.collection.get(include=["documents", "metadatas"])
        
        memories = []
        if results["documents"]:
             for i in range(len(results["documents"])):
                 memories.append({
                     "fact": results["documents"][i],
                     "metadata": results["metadatas"][i]
                 })
        return memories

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Semantic search for relevant memories using ChromaDB."""
        count = self.collection.count()
        if count == 0:
            return []

        # Generate embedding for the query
        query_vector = self._get_embedding(query)
        
        # Perform query
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(limit, count) # Prevent asking for more than exist
        )
        
        # Reformat results to match previous interface (List of Dicts with 'fact')
        # Chroma returns lists of lists (one list per query)
        retrieved_memories = []
        
        if results["documents"]:
            # Iterate through the first query's results
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i] if results["distances"] else 0
                
                # Reconstruct memory object
                # Note: Chroma distance is usually L2 or cosine distance. 
                # Smaller is better for distance. Similarity = 1 - distance (approx).
                # We can filter if needed, but for now return top k.
                mem_item = {
                    "fact": doc,
                    "metadata": meta,
                    "score": dist
                }
                retrieved_memories.append(mem_item)
                
        return retrieved_memories

if __name__ == "__main__":
    try:
        mem = MemoryLayer()
        print("Memory Layer Initialized Successfully.")
    except Exception as e:
        print(f"Failed to init memory layer: {e}")
