import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Use the same model and adapter from training
BASE = "microsoft/DialoGPT-medium"
ADAPTER = "adapters/c1-html-dialogpt-qlora"

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    print("⚠️  CUDA GPU not available - using CPU for inference")
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
    bnb_config = None

print("Loading base model...")
if CUDA_AVAILABLE:
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb_config, device_map="auto")
else:
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu")

print("Loading adapter...")
model = PeftModel.from_pretrained(base, ADAPTER)

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test prompt - use the same format as training
prompt = "[INST] Create a responsive cafe landing page with hero, 3x2 menu grid, and contact form. Use minimal CSS and accessible HTML. [/INST]\n"
inputs = tokenizer(prompt, return_tensors="pt")

# Move to appropriate device
if CUDA_AVAILABLE:
    inputs = inputs.to("cuda")
    model = model.to("cuda")
else:
    inputs = inputs.to("cpu")
    model = model.to("cpu")

print(f"\n🎯 Generating HTML for: {prompt}")
print("=" * 60)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
print(generated_text)