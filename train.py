import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# For CPU training, use a smaller model
MODEL_NAME = "microsoft/DialoGPT-medium"  # Smaller model for CPU training
DATA_PATH = "data/prepared/c1_html_instructions.jsonl"
OUTPUT_DIR = "adapters/c1-html-dialogpt-qlora"
MAX_LEN = 1024  # Smaller context for CPU

# Check if CUDA is available, but don't fail if it's not
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    print("⚠️  CUDA GPU not available - training will be slower on CPU")
    print("💡 Consider using Google Colab or a cloud GPU for faster training")
else:
    print("✅ CUDA GPU detected - using GPU acceleration")

# Configure quantization based on available hardware
if CUDA_AVAILABLE:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    # CPU-only configuration - no quantization
    bnb_config = None
    print("📝 Using CPU-only training without quantization")

if CUDA_AVAILABLE:
    print("Loading base model in 4-bit…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    print("Loading base model for CPU training…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Preparing PEFT/LoRA…")
if CUDA_AVAILABLE:
    model = prepare_model_for_kbit_training(model)
else:
    print("📝 Skipping k-bit preparation for CPU training")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

print("Loading dataset…")
# JSONL with fields: instruction, input, output
raw = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(ex):
    # Simple SFT: instruction → output (without special tokens for DialoGPT)
    prompt = ex['instruction'].strip()
    text = prompt + "\n" + ex["output"].strip()
    tokens = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = raw.map(format_example, remove_columns=raw.column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,                # More epochs for better learning
    per_device_train_batch_size=1,      # Small batch size
    gradient_accumulation_steps=4,      # Effective batch size 4
    learning_rate=5e-5,                 # Lower learning rate for CPU
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,                    # More frequent logging
    save_steps=100,                     # Save more frequently
    save_total_limit=3,
    optim="adamw_torch",                # Standard optimizer for CPU
    gradient_checkpointing=False,       # Disable for CPU
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to=["none"],
    dataloader_pin_memory=False,        # Disable for CPU
)

print("Starting training…")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
)
trainer.train()

print("Saving adapter…")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")