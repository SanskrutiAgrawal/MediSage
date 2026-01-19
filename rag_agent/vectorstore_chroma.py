import os
import re
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Simple in-memory document store to replace LocalFileStore
class SimpleDocStore:
    def __init__(self, path):
        self.path = path
        self.store = {}
        
    def mset(self, items):
        for key, value in items:
            self.store[key] = value
    
    def mget(self, keys):
        return [self.store.get(k) for k in keys]

class VectorStore:
    """
    Manages ChromaDB vector store: ingest documents, retrieve relevant documents.
    """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config.rag
        self.collection_name = self.config.collection_name
        self.embedding_model = self.config.embedding_model
        self.retrieval_top_k = self.config.top_k
        self.vectorstore_local_path = self.config.vector_local_path
        self.docstore_local_path = self.config.doc_local_path

        # Initialize the ChromaDB client
        self.client = chromadb.PersistentClient(path=self.vectorstore_local_path)

    def load_vectorstore(self) -> Tuple[Chroma, SimpleDocStore]:
        """
        Load existing vectorstore and docstore for retrieval.
        """
        self.logger.info(f"Loading existing vectorstore from path: {self.vectorstore_local_path}")
        
        vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )
        
        docstore = SimpleDocStore(self.docstore_local_path)
        
        self.logger.info("Successfully loaded vectorstore and docstore.")
        return vectorstore, docstore

    def create_vectorstore(self, document_chunks: List[str], document_path: str):
        """
        Create or upsert documents into the ChromaDB vector store.
        """
        self.logger.info(f"Ingesting {len(document_chunks)} chunks into ChromaDB collection '{self.collection_name}'...")

        doc_ids = [str(uuid4()) for _ in document_chunks]
        
        langchain_documents = []
        for i, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[i],
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )

        # Initialize LangChain's Chroma wrapper
        vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )

        # Add documents to ChromaDB
        vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)

        # Store the raw text chunks in a file-based store
        docstore = SimpleDocStore(self.docstore_local_path)
        encoded_chunks = [chunk.encode('utf-8') for chunk in document_chunks]
        docstore.mset(list(zip(doc_ids, encoded_chunks)))
        
        self.logger.info("Ingestion complete.")

    def retrieve_relevant_chunks(self, query: str, vectorstore: Chroma, docstore: SimpleDocStore) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from ChromaDB based on a query.
        """
        self.logger.info(f"Retrieving top {self.retrieval_top_k} chunks for query: '{query}'")

        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )
        
        retrieved_docs = []
        for doc, score in results:
            doc_dict = {
                "id": doc.metadata.get('doc_id', str(uuid4())),
                "content": doc.page_content,
                "score": score,
                "source": doc.metadata.get('source', 'Unknown'),
                "source_path": doc.metadata.get('source_path', 'Unknown'),
            }
            retrieved_docs.append(doc_dict)
        
        return retrieved_docs