
import logging
import os
from typing import List, Dict, Optional, Any
from config import settings

logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
    from langchain.docstore.document import Document
except ImportError:
    logger.error("LangChain or ChromaDB not installed. Vector Store will not function.")
    Chroma = None
    OpenAIEmbeddings = None
    OllamaEmbeddings = None
    Document = Any

class VectorMemory:
    """
    Vector Store memory using LangChain and ChromaDB.
    Stores and retrieves interview context, previous answers, and behavioral patterns.
    """
    
    def __init__(self, collection_name: str = "interview_memory"):
        self.collection_name = collection_name
        self.persist_directory = str(settings.data_dir / "chroma_db")
        self.embeddings = self._get_embeddings()
        
        if Chroma and self.embeddings:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Vector Store initialized at {self.persist_directory}")
        else:
            self.vectorstore = None
            logger.warning("Vector Store failed to initialize.")

    def _get_embeddings(self):
        """
        Determine which embedding model to use based on settings.
        """
        if not OpenAIEmbeddings:
            return None
            
        try:
            # 1. Try OpenAI if key is present
            if settings.openai_api_key:
                logger.info("Using OpenAI Embeddings")
                return OpenAIEmbeddings(
                    openai_api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url
                )
            
            # 2. Try Ollama if explicitly set or as fallback (using nomic-embed-text or similar)
            if settings.llm_provider == "ollama":
                logger.info("Using Ollama Embeddings (nomic-embed-text)")
                # Assumption: user has nomic-embed-text or similar pulled
                return OllamaEmbeddings(
                    base_url=settings.ollama_base_url,
                    model="nomic-embed-text" 
                )
            
            # 3. Fallback to minimal local HuggingFace if available
            logger.info("Using HuggingFace Embeddings (all-MiniLM-L6-v2)")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return None

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add texts to the vector store.
        """
        if not self.vectorstore:
            return
            
        try:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"Added {len(texts)} documents to Vector Store")
        except Exception as e:
            logger.error(f"Failed to add texts to Vector Store: {e}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for relevant documents.
        """
        if not self.vectorstore:
            return []
            
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_context_string(self, query: str, k: int = 4) -> str:
        """
        Retrieve context as a single string for LLM injection.
        """
        docs = self.similarity_search(query, k=k)
        return "\n\n".join([d.page_content for d in docs])

# Global instance
vector_memory = VectorMemory()
