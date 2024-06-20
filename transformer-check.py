import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = os.path.abspath('./models/vicuna-7b-v1.5/')
print("Model directory:", model_path)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("!!!!!!!!! Tokenizer loaded successfully !!!!!!!!!")
except Exception as e:
    print(f"We Failed to load tokenizer: {e}")

try:
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    print("!!!!!!!!! Model loaded successfully !!!!!!!!!")
except Exception as e:
    print(f"We Failed to load model: {e}")
