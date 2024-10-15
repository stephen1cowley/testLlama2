from typing import List, Literal
import pandas as pd
import ast

MEMOTRAP_DATAPATH = 'memotrap/1-proverb-ending.csv'

df = pd.read_csv(MEMOTRAP_DATAPATH)

def unmarshall_list(data: str) -> List[str]:
    # Use ast.literal_eval to convert the string into a list
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        # Handle cases where the data is not properly formatted
        return []

for idx, row in df.iterrows():
    prompt: str = row['prompt']
    classes: List[str] = unmarshall_list(row['classes'])
    answer_index: Literal[0, 1] = row['answer_index']
    print(classes[0], classes[1])
    print(answer_index, prompt)
