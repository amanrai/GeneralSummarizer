from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
import wandb


model_name = "facebook/opt-125m"

wandb.init(project="{model_name}-finetuning-summarization")

print(model_name)
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer_local_save_path = "./tokenizer/"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_save_path, config=AutoConfig.from_pretrained(model_name))

c_dataset = load_from_disk("./final_dataset.pkl")

training_args = TrainingArguments(
    "./models/" + f"{model_name}-finetuned-c4",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    report_to="wandb",
    run_name=f"{model_name}-finetuning-summarization",
    logging_steps = 10,
    weight_decay=0.01,
    per_device_train_batch_size=12,
    save_steps=10000,
    eval_steps=10000,
    save_total_limit=5,
    push_to_hub=False,
)

trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=c_dataset["train"],
    eval_dataset=c_dataset["test"]
)

print("Training...")
trainer.train()