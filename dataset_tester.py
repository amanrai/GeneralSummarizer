from CausalDataset import CausalDataset
from DataPreppers.DataPrepCNN import DataPrepCNN
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import DatasetDict, load_from_disk, Dataset
import json
import os

max_seq_length = 512
eos_token = "<eoseq>"
dataset_dict_file_name = "./final_dataset.pkl"
force_regenerate = False
tokenizer_local_save_path = "./tokenizer/"

model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.add_tokens([eos_token])

if (not os.path.exists(dataset_dict_file_name)) or force_regenerate:
    cnn = DataPrepCNN(hf_datasetName=("cnn_dailymail", "3.0.0"), workers=4, eos_token = "<eoseq>")
    cnn.acquire()
    cnn_dataset = cnn.convertToTask("context", "summary", causal=True)
    f_dataset = CausalDataset(datasets={"cnn":cnn_dataset}, tokenizer=tokenizer, max_length=512)
    f_dataset = f_dataset.genTokenizedDataset()
    f_dataset.save_to_disk("./final_dataset.pkl")

dataset = load_from_disk("./final_dataset.pkl")
tokenizer.save_pretrained(tokenizer_local_save_path)
config.save_pretrained(tokenizer_local_save_path)

print(dataset)

# for dp in dataset["train"]:
#     print(dp)
