# ref: https://github.com/voidism/DoLa/blob/main/tfqa_mc_eval.py

import os, gzip
import pandas as pd
from typing import List, Dict

def load_csv(file_path, is_gzip=False) -> List[Dict[str, str]]:
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only

    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {'question': df['Question'][idx], 
                    'answer_best': df['Best Answer'][idx],
                    'answer_true': df['Correct Answers'][idx],
                    'answer_false': df['Incorrect Answers'][idx]}
            list_data.append(data)

    return list_data

if __name__ == "__main__":


    fp = os.path.join('TruthfulQA.csv')

    list_data_dict = load_csv(fp)

