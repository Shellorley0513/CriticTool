import torch
from typing import Union, List
from vllm import LLM, SamplingParams
from critictool.utils.chat_template import Meta_Template

class HFTransformer:
    """Wrapper around HuggingFace and vLLM inference.
    Args:
        path (str): The name or path to HuggingFace's model.
    """
    
    def __init__(self,
                 template: str,
                 model_path: str,
                ) -> None:
        self.template = Meta_Template(template_type=template)
        self.model = LLM(model=model_path, tensor_parallel_size=2, max_model_len=32768, trust_remote_code=True, gpu_memory_utilization=0.7)

    def _chat(self, inputs):
        """Generate responses for chat inputs."""
        with torch.no_grad():
            _inputs = list()
            for prompt in inputs:
                _inputs.append(self.template.apply_hf_template(prompt))
            if isinstance(_inputs, str):
                _inputs = [_inputs]

            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0,
            )
            outputs = self.model.generate(_inputs, sampling_params, use_tqdm=False)
            responses = [out.outputs[0].text.strip() for out in outputs]

        return responses
    
    def chat(self, inputs:Union[List[dict], List[List[dict]]]):
        assert isinstance(inputs, list), "Inputs must be a list of dict or list of dicts."
        return self._chat(inputs)



