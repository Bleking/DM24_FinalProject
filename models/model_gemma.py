from typing import List, Union
import random
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from auto_gptq import AutoGPTQForCausalLM

from .base_model import ShopBenchBaseModel

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class GoogleGemma(ShopBenchBaseModel):
    def __init__(self):
        random.seed(AICROWD_RUN_SEED)

        model_path = 'google/gemma-7b'

        self.tokenizer = AutoTokenizer.from_pretrained('./models/gemma-7b/', trust_remote_code=True)
        self.model = AutoGPTQForCausalLM.from_pretrained('./models/gemma-7b/', quantize_config=None, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, do_sample=True).to('cuda')
        self.system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"


    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        prompt = self.system_prompt + prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        inputs.input_ids = inputs.input_ids.cuda()
        
        if is_multiple_choice:
            if "Which of the following product" or "Select the category" in prompt:  #task 2
                prompt = ("You should choose the option which fits best for the given attribute or product "
                          "type. \n\n") + prompt
            elif "e-commerce website" and ("total" or "How many") in prompt:  #task 5
                prompt = (("You will be given a product. Answer the given question based on the information "
                           "you have. You have to calculate the answer based on the product name. \n\n") +
                          prompt)
            elif "e-commerce website" in prompt:  #task 8
                prompt = ("You will be given a product. Answer the given question based on the information "
                          "you have. \n\n") + prompt
            elif "Which of the following statements" in prompt:  #task 11
                prompt = ("You will be given two queries. You have to decide, in which way they are "
                          "related. Take special attention what the general categories of the given queries "
                          "are and how they relate towards each other. \n\n") + prompt
            else:
                prompt = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n\n" + prompt

            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=1, temperature=1)
        
        else:
            if " rank " in prompt:  # task 12
                prompt = ("You have to rank the given products based on the query that is about to be "
                          "mentioned. The way you should answer is described at the end of the "
                          "task.\n\n") + prompt
            elif "xplain " in prompt:  # task 1
                prompt = ("You have to explain the given product category or type. Your answer should be "
                          "in one sentence of a length of about 20 words.\n\n") + prompt
            elif "potential review" in prompt:  # task 3
                prompt = ("Listen to the given task carefully and pay attention how the formatting will "
                          "be. Take special attention on which kind of product it is and which of the "
                          "reviews fits the aspect that should be mentioned. \n\n") + prompt
            elif "extract phrases" in prompt:  # task 4
                prompt = ("Listen to the given task in the next paragraph carefully. The output of that task should "
                          "just be a single word from the given query."
                          "This word should be the best fitting  given entity type in inverted commas.\n\n") + prompt
            elif "title" in prompt:  # task 17
                prompt = ("You will be given a product title in inverted commas. You have to translate it based on the "
                          "language given in the description. You should not explain the answer. \n\n") + prompt
            else:
                prompt = self.system_prompt + prompt
            
            inputs = self.tokenizer(prompt, return_tensors='pt')
            generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=100, temperature=1)
        
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation = result[len(prompt):]
        
        return generation
