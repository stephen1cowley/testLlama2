from typing import List, Literal
import ast
import pandas as pd
import torch
from hybrid_method import HybridMethod

MEMOTRAP_DATAPATH = 'memotrap/1-proverb-ending.csv'
MODEL = '4bit/Llama-2-7b-chat-hf'

df = pd.read_csv(MEMOTRAP_DATAPATH)

def unmarshall_list(data: str) -> List[str]:
    # Use ast.literal_eval to convert the string into a list
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        # Handle cases where the data is not properly formatted
        return []

llm = HybridMethod(
    model_name=MODEL,
    device='cuda'
)

with torch.no_grad():
    for idx, row in df.iterrows():
        context: str = row['prompt'].split(":")[0]
        prompt: str = row['prompt'].split(":")[1][1:]
        classes: List[str] = unmarshall_list(row['classes'])
        answer_index: Literal[0, 1] = row['answer_index']

        answer = llm.cad_generate(
            context=context,
            prompt=prompt,
            dola_layers=None
        )

        if isinstance(answer, str):
            answer = answer[len(context + ": " + prompt):]
            print("Question:", context + ": " + prompt)
            print("Correct answer:", repr(classes[answer_index]))
            print("CAD Given answer:", repr(answer))
            print("CAD Success\n" if answer == classes[answer_index] else "CAD Failure\n")

        answer = llm.generate(
            input_text=context + ": " + prompt,
            dola_layers=None
        )

        if isinstance(answer, str):
            answer = answer[len(context + ": " + prompt):]
            print("Question:", context + ": " + prompt)
            print("Correct answer:", repr(classes[answer_index]))
            print("Reg Given answer:", repr(answer))
            print("Reg Success\n" if answer == classes[answer_index] else "Reg Failure\n")

