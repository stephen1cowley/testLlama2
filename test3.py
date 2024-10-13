from hybrid_method import HybridMethod

llm = HybridMethod(
    model_name='4bit/Llama-2-7b-chat-hf',
    device='cuda'
)

print(llm.generate("Q: What is 2+2? A: "))
