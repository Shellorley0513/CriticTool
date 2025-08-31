from openai import OpenAI
from typing import Union, List
from critictool.utils.chat_template import Meta_Template
from concurrent.futures import ThreadPoolExecutor

class GPT:
    """Model wrapper around OpenAI's models.

    Args:
        model (str): The name of OpenAI's model.
        key (str): OpenAI key(s).
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo-16k',
        key: Union[str, List[str]] = '',
        openai_api_base: str = 'https://api.openai.com/v1/chat/completions',
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=key,
            base_url=openai_api_base,
        )
        self.template = Meta_Template()
    
    def _chat(self, prompt:List[dict]):
        """Generate responses for chat inputs."""
        response = self.client.chat.completions.create(
            model=self.model,  
            messages=self.template.apply_api_role(prompt),
            max_tokens=512,
            temperature=0 
        )
        
        output = response.choices[0].message.content
        if not isinstance(output, str):
            output = str(output)

        return output

    def chat(self, inputs:Union[List[dict], List[List[dict]]]):
        assert isinstance(inputs, list)
        if not isinstance(inputs[0], list):
            inputs = [inputs]
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat,
                                messages,)
                for messages in inputs
            ]
        ret = [task.result() for task in tasks]
        return ret
