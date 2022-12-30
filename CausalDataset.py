from DataPreppers import *
from datasets import DatasetDict, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

class CausalDataset():
    def __init__(self, datasets, tokenizer, max_length):
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.workers = 4
        self.attention_mask = np.ones((max_length,))
        
    def __len__(self):
        return len(self.data)

    def _tokenizeAndExtend(self, dp):
        _out = self.tokenizer(dp["causal"]["context"])
        return {"input_ids":_out["input_ids"]}

    def doSplit(self, dataset, split):
        current_ = []
        nd =  dataset[split].map(lambda dp:self._tokenizeAndExtend(dp), num_proc=self.workers)
        print("Serializing for the causal LM")
        _c = []
        for i in tqdm(nd):            
            _c.extend(i["input_ids"])
        current_ = [_c[i:i+self.max_length] for i in range(0, len(_c), self.max_length)]
        return current_

    def genTokenizedDataset(self):
        splits = ["train", "test", "validation"]
        f_data = DatasetDict()
        for split in splits:
            _x = []

            for dataset in self.datasets:
                print("Tokenizing dataset: ", dataset, " split: ", split)
                ds = self.datasets[dataset]
                final = self.doSplit(ds, split)
                _x.extend(final)

            final = []
            for item in _x:
                final.append({"input_ids":item, "attention_mask":self.attention_mask, "labels":item}) #Note that the HF trainer expects labels to be the same as input_ids. It will automatically left shift. 
            f_data[split] = Dataset.from_pandas(pd.DataFrame(final))
            
        return f_data



        
            

