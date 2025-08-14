from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

BASE = "codellama/CodeLlama-7b-Instruct-hf"
ADAPTER = "adapters/c1-html-7b-qlora"
OUT = "outputs/c1-html-7b-merged"

os.makedirs(OUT, exist_ok=True)

print("Loading base…")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
print("Merging…")
model = PeftModel.from_pretrained(base, ADAPTER)
merged = model.merge_and_unload()
print("Saving…")
merged.save_pretrained(OUT)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print(f"Merged model saved to {OUT}")