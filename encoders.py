from abc import ABC, abstractmethod
from collections import Counter
from pydantic import BaseModel, Field, field_validator, model_validator
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class BaseEncoder(BaseModel, ABC):
    # x: np.ndarray

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class SillyEncoder(BaseEncoder):
    """
    The silly encoder is for debugging. 
    A text is encoded by counting each letter and dividing by the length.
    """
    alphabet: str = 'abcdefghijklmnopqrstuvwxyz'
    def encode(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        counters_and_lengths = [(Counter(t.lower()), len(t))for t in texts] # count characters in string
        vectors = [
            [
                counter[letter]/L
                for letter in self.alphabet
            ] 
            for counter, L in counters_and_lengths
        ]
        return vectors


class HuggingFaceEncoder(BaseEncoder):
    """
    A local encoder alternative. 
    CPU is PLENTY, but GPU will speed perfomance if you need it - see handout.

    Model usage: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    Sentence-transformers Library: https://www.sbert.net/
    """
    model_name: str
    model: PreTrainedModel  = Field(default=None)
    tokenizer: PreTrainedTokenizer = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model.cuda()
        return self

    def encode(self, texts: list[str], pooling='mean') -> list[list[float]]:
        model_inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        model_inputs.to(self.model.device)
        model_outputs = self.model(**model_inputs)
        if pooling == 'mean':
            embeddings = self._mean_pooling(model_inputs, model_outputs)
        elif pooling == 'cls':
            embeddings = self._cls_pooling(model_outputs)
        else:
            raise ValueError('pooling must be "mean" or "cls"')
        return embeddings.tolist()
    
    def _mean_pooling(self, model_inputs, model_outputs) -> torch.Tensor:
        """Represent the document with the average of its tokens"""
        attention_mask: torch.Tensor = model_inputs['attention_mask'] # [samples, tokens]
        token_embeddings = model_outputs['last_hidden_state'] # [samples, tokens, dim]

        # Increase the dimension of mask so we can multiply by embeddings
        # Then, find how many tokens per sample
        attention_mask = attention_mask.unsqueeze(-1) # [samples, tokens, 1]
        weight = attention_mask.sum(1) # [samples, 1]

        masked_embeddings = token_embeddings * attention_mask # [samples, tokens, dim]
        mean_pooled = masked_embeddings.sum(1) # [samples, dim]
        mean_pooled /= torch.clamp(weight, min=1e-9) # [samples, dim], prevent divide by zero
        return mean_pooled

    
    def _cls_pooling(self, model_outputs) -> torch.Tensor:
        """Represent the document with the first token, [CLS]"""
        return model_outputs['last_hidden_state'][:, 0] # [samples, dim]
