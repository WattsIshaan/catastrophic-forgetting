from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import argparse

# ==== Configuration ====

# Create the parser
parser = argparse.ArgumentParser(description="Finetuning parameters")
parser.add_argument("--model_path", help="The path of the model")
parser.add_argument("--output_dir", help="Path to output model")
parser.add_argument("--learning_rate",type = float, help="learning rate")
args = parser.parse_args()

# Set variable names
MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir
LEARNING_RATE = args.learning_rate

# ==== Load Model and Tokenizer ====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token  
dataset = load_dataset("OLMo-Coding/starcoder-python-instruct")


#Formatting function for prompt completion data
def formatting_func(example):
    prompt = f"Instruction: {example['instruction']}\nResponse: "
    return {
        "prompt": prompt,
        "completion": example["text"]
    }

dataset = dataset.map(formatting_func, remove_columns=["instruction", "text", "id", "metadata", "added", "created", "source"])


#global batch size 64
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,  

    learning_rate=LEARNING_RATE,  
    
    logging_steps=10,
    save_strategy="epoch",
    
    max_seq_length=1024,
    packing=True,  # Enable packing to combine short sequences
    
    # KEY FIX 4: Gradient clipping to prevent explosion
    max_grad_norm=1.0, 
    
    # # KEY FIX 5: More conservative training settings
    warmup_steps=100,  # More warmup steps for stability
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    
    # KEY FIX 6: Disable mixed precision to avoid NaN
    fp16=False,
    bf16=False,
    
    # KEY FIX 7: Training stability options
    dataloader_drop_last=True,
    gradient_checkpointing=False,  # Disable to avoid potential issues
    
    # Better logging to monitor training
    logging_first_step=True,
    load_best_model_at_end=False,

    completion_only_loss=True,
)

# ==== Train ====

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    args = training_args,

)

trainer.train()

# ==== Save Final Model ====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)