from typing import Literal, Any, List
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.generation.utils import GenerateOutput

class HybridMethod:
    """
    Sets up an LLM that we can do inference on.
    """
    def __init__(self, model_name: str, device: Literal['cpu', 'cuda']):
        self.model_name: str = model_name
        self.device: Literal['cpu', 'cuda'] = device
        self.tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )

    def generate(
            self,
            input_text: str,
            dola_layers: Literal['high', 'low'] | None = None,
        ) -> str | None:
        """
        Generate either with DoLa, depending on whether `dola_layers` is set to the higher or lower
        layer setting. DoLa is turned off if `dola_layers=None`.
        """
        inputs: Any = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            output_scores=False,
            return_dict_in_generate=True,
            output_hidden_states=False,
            dola_layers=dola_layers,
        )

        if isinstance(outputs, GenerateOutput):
            return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return None

    def log_probs(
            self,
            prompt: str,
            answer: str
        ) -> float:
        """
        Returns log probs of the answer bit
        """
        input_text = prompt + answer
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prefix_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        continue_ids = input_ids[0, prefix_ids.shape[-1]:]

        outputs = self.model(input_ids)[0].squeeze(0)
        outputs = outputs.log_softmax(-1)  # logits to log probs

        # skip tokens in the prompt -- we only care about the answer
        outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

        # get logprobs for each token in the answer
        log_probs = outputs[range(outputs.shape[0]), continue_ids].mean().item()
        return log_probs
