import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator, Field


class BaseDatabase(BaseModel):
    @abstractmethod
    def add_documents(self, documents, embeddings) -> None:
        """Store documents and corresponding embeddings."""
        ...

    @abstractmethod
    def retrieve_documents(self, query_embedding, n=3) -> dict[str, float]:
        """
        return n documents, sorted by a score.
        {doc: score}
        """
        ...

class SimpleDatabase(BaseDatabase):
    """This database stores documents and embeddings in memory"""
    documents: list[str] = Field(default = [])
    embeddings: list[list[float]] = Field(default = []) # vectors

    @model_validator(mode='after')
    def validate_data(self):
        assert len(self.documents) == len(self.embeddings), "Embeddings are not the same size as documents"

    def add_documents(self, documents, embeddings) -> None:
        for d, e in zip(documents, embeddings):
            if d not in self.documents:
                self.documents.append(d)
                self.embeddings.append(e)
        self.validate_data()

    def retrieve_documents(self, query_embedding: list[float], n=3)-> dict[str, float]:
        if self.documents:
            if isinstance(query_embedding[0], list):
                assert len(query_embedding) == 1, "query contains multiple vectors"
                # assert len(query_embedding[0]) == len(self.embeddings[0]), "query embedding is the wrong dimension"
                query_embedding = query_embedding[0]
            assert len(query_embedding) == len(self.embeddings[0]), "query embedding is the wrong dimension"
            # TODO
            # dot product using numpy
            scores = np.dot(self.embeddings, query_embedding, )
            # sort by scores using sorted and slice up to n documents
            retrieved = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)[0:n]
            # dict([(a, b), (c,d)]) = {a: b, c: d}
            return dict(retrieved)
        else:
            # no docs
            return {}
        










