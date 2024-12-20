from abc import ABC, abstractmethod
from pydantic import Field, BaseModel
import openai
import ollama

class BaseLLM(BaseModel):
    """
    A generative model that takes in chat messages.
    """

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a text response from past messages"""


class SillyLLM(BaseLLM):
    
    def generate(self, messages: list[dict[str, str]]) -> str:
        list_of_past = [f'''{m['role']}: "{m['content'][0:40]}..."''' for m in messages]
        yield from "DEBUG LLM RESPONSE \n\n I see {} past messages starting with: \n\n  {}".format(
            len(messages), ',\n\n'.join(list_of_past)
        )   
    

class LocalLLM(BaseLLM):
    """
    Requires Ollama installed on device.

    CPU should be fine but runtime might be slow.
    """
    model_name: str = 'mistral'

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self.generate_ollama_directly(messages)

    def generate_ollama_directly(self, messages):

        input_data = {
            "messages": messages,
            "model": self.model_name
        }

        # Call Ollama's streaming chat API
    def generate_ollama_directly(self, messages):
        for m in ollama.chat( 
            model = 'mistral',
            messages = messages,
            stream = True
        ):
            yield m['message']['content']
       