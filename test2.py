from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer
import torch

model_id = '4bit/Llama-2-7b-chat-hf'

# Check if CUDA is available
if not torch.cuda.is_available():
    raise ValueError("GPU not available. Please check your CUDA installation.")

# Load the model with low memory usage
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained(model_id)

# Testing the model
input_text = "Hello, Llama!"
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
outputs = model.generate(
    **inputs, 
    output_scores=True,
    return_dict_in_generate=True,
    output_hidden_states=True,
    dola_layers="high",
)

# Decode and print the generated tokens
generated_tokens = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print(generated_tokens)

# Print the probabilities of each token
for i, token_id in enumerate(outputs.sequences[0, inputs['input_ids'].shape[1]:]):
    token_probabilities = torch.softmax(outputs.scores[i], dim=-1)
    token_probability = token_probabilities[0, token_id]
    topk = torch.topk(token_probabilities, 5)

    print("Distribution")
    for j, idtok in enumerate(topk.indices[0]):
        print(repr(tokenizer.decode([idtok])))
        print(token_probabilities[0, idtok].item())


    token = tokenizer.decode([token_id])
    print(f"Token: {repr(token)}, Probability: {token_probability.item()}")
