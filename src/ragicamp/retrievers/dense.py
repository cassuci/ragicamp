"""Dense retriever using vector similarity."""

from typing import Any, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragicamp.retrievers.base import Document, Retriever


class DenseRetriever(Retriever):
    """Dense retriever using neural embeddings and vector similarity.
    
    Uses sentence transformers to encode queries and documents,
    then performs similarity search using FAISS.
    """
    
    def __init__(
        self,
        name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        **kwargs: Any
    ):
        """Initialize dense retriever.
        
        Args:
            name: Retriever identifier
            embedding_model: Sentence transformer model name
            index_type: FAISS index type (flat, ivf, hnsw)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.documents: List[Document] = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents by computing and storing their embeddings."""
        self.documents = documents
        
        # Compute embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        if self.index_type == "ivf":
            self.index.train(embeddings.astype('float32'))
        
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any
    ) -> List[dict]:
        """Retrieve documents using dense similarity search."""
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.score = float(score)
                results.append(doc.to_dict())
        
        return results

