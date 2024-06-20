from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset("NingLab/ECInstruct")

# Load pre-trained model and tokenizer
model_name = "BarraHome/Mistroll-7B-v2.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Preprocess data
def preprocess_function(examples):
    # combined_input = [
    #    f"Instruction: {instr}\nInput: {inp}\nOptions: {opts}"
    #    for instr, inp, opts in zip(examples["Instruction"], examples["Input"], examples["Options"])
    # ]
    combined_input = [
        f"Instruction: {instr}\nInput: {inp}\nOptions: {opts}"
        # f"Input:{instr} + {inp} + {opts}"
        for instr, inp, opts in zip(examples["instruction"], examples["input"], examples["options"])
    ]
    model_inputs = tokenizer(combined_input, padding="max_length", truncation=True, max_length=256)

    outputs = [ex for ex in examples["output"]]
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=256)["input_ids"]
    model_inputs["labels"] = labels
    
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision if supported by your GPU
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'],
)

# Train the model
trainer.train()

# Evaluate and save the model
results = trainer.evaluate()
print(results)

model.save_pretrained("./fine-tuned-mistroll-7B-v2.2")
tokenizer.save_pretrained("./fine-tuned-mistroll-7B-v2.2")
