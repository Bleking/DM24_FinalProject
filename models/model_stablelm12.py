from typing import List, Union
import random
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import ShopBenchBaseModel

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class StableLM(ShopBenchBaseModel):
    def __init__(self):
        random.seed(AICROWD_RUN_SEED)
        
        model_path = 'stabilityai/stablelm-2-12b-chat'

        self.tokenizer = AutoTokenizer.from_pretrained('./models/stablelm-2-12b-chat/', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('./models/stablelm-2-12b-chat/', device_map='auto', torch_dtype='auto', trust_remote_code=True, do_sample=True)
        self.system_prompt = "Please listen to the given task carefully and pay special attention on how the answers should be give.\n\n"
        
    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        prompt = self.system_prompt + prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs.input_ids = inputs.input_ids.cuda()
        
        if is_multiple_choice:
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=1, temperature=1)
        else:
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=100, temperature=1)
        
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation = result[len(prompt):]
        
        return generation