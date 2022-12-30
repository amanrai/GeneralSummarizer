try:
    from .DataPrep import *
except:
    from DataPrep import *

from datasets import load_dataset
import pandas as pd

class DataPrepCNN(DataPrep):

    def __init__(self, hf_datasetName = None, workers=1):
        #CNN is acquired from huggingface datasets
        #You must pass a dataset name
        assert hf_datasetName is not None, "You must pass a dataset name"
        assert len(hf_datasetName) > 1, "You must pass the name of the dataset and the version as a tuple"
        self.hf_datasetName = hf_datasetName
        self.workers = workers
    
    def acquire(self, from_hf=True, from_file=False, from_url=False):
        self.dataset = load_dataset(self.hf_datasetName[0], self.hf_datasetName[1])

    def _genCausal(self, dp, context_prompt, generator_prompt):
        _f =  context_prompt + ": " + dp["article"] + " " + generator_prompt + ": " + dp["highlights"]
        return {"causal": {
            "context": _f,
        }}

    def _genSeq2Seq(self, dp, context_prompt, generator_prompt):
        return {
            "seq2seq": {
                "context": context_prompt + ": " + dp["article"],
                "output": generator_prompt + ": " + dp["highlights"]
            }
        }

    def convertToTask(self, context_prompt, generator_prompt, causal = False):
        """
        The causal parameter defines if the output should be a single string per datapoint. 
        If True, the output will be:
            context_prompt + ": " + article + " " + generator_prompt + ": " + highlights
            highlights will be split by the \n Token. 
        If False, the output will be:
            {
                "context": "context_prompt" + ": " article + " " + generator_prompt + ": ",
                "output": highlights
            }
        """
        generator = self._genCausal if causal else self._genSeq2Seq
        _desc = "Causal" if causal else "Seq2Seq"
        print("Generating the Train Split for a {} network".format(_desc))
        train = self.dataset["train"].map(lambda dp: generator(dp, context_prompt, generator_prompt), num_proc=self.workers)
        print("Generating the Test Split for a {} network".format(_desc))
        test  =self.dataset["test"].map(lambda dp: generator(dp, context_prompt, generator_prompt), num_proc=self.workers)
        print("Generating the Validation Split for a {} network".format(_desc))
        validation = self.dataset["validation"].map(lambda dp: generator(dp, context_prompt, generator_prompt), num_proc=self.workers)
        self.dataset = {"train": train, "test": test, "validation": validation}
        return self.dataset
        

if __name__ == "__main__":
    dp = DataPrepCNN(hf_datasetName=("cnn_dailymail", "3.0.0"), workers=4)
    dp.acquire()
    dataset = dp.convertToTask("context", "summary", causal=True)
    print(dataset)
    