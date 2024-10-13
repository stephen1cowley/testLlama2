import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import Literal
from transformers.generation.utils import GenerateOutput

class HybridMethod:
    def __init__(self, model_name: str, device: Literal['cpu', 'cuda']):
        self.model_name: str = model_name
        self.device: Literal['cpu', 'cuda'] = device
        self.tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )

    def generate(self, input_text: str, dola_layers: Literal['high', 'low', None] = None) -> str:
        """
        Generate either with DoLa, depending on whether `dola_layers` is set to the higher or lower
        layer setting. DoLa is turned off if `dola_layers=None`.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs: GenerateOutput = self.model.generate(
            **inputs, 
            output_scores=False,
            return_dict_in_generate=True,
            output_hidden_states=False,
            dola_layers=dola_layers,
        )
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
