import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
from peft import PolyConfig, get_peft_model,LoraConfig, TaskType, PeftModel, PeftConfig

model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"

r = 8  # rank of lora in poly
n_tasks = 4  # number of tasks
n_skills = 2  # number of skills (loras)
n_splits = 4  # number of heads

batch_size = 1
lr = 5e-5
num_epochs = 8

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    device_map="auto"
)

peft_config = PolyConfig(
    task_type=TaskType.CAUSAL_LM,  # Changed from SEQ_2_SEQ_LM
    poly_type="poly",
    r=r,
    n_tasks=n_tasks,
    n_skills=n_skills,
    n_splits=n_splits,
)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# Format prompts for chat-based instruction following
def format_prompt(instruction, choices=None):
    if choices:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# boolq
boolq_dataset = (
    load_dataset("super_glue", "boolq")
    .map(
        lambda x: {
            "input": format_prompt(f"{x['passage']}\nQuestion: {x['question']}\nChoose A or B:\nA. Yes\nB. No\nAnswer:"),
            # 0 - False, 1 - True
            "output": ["B", "A"][int(x["label"])],
            "task_name": "boolq",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("boolq example: ")
print(boolq_dataset["train"][0])

# multirc
multirc_dataset = (
    load_dataset("super_glue", "multirc")
    .map(
        lambda x: {
            "input": format_prompt(
                f"{x['paragraph']}\nQuestion: {x['question']}\nAnswer: {x['answer']}\nIs it true?\nChoose A or B:\nA. Yes\nB. No\nAnswer:"
            ),
            # 0 - False, 1 - True
            "output": ["B", "A"][int(x["label"])],
            "task_name": "multirc",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("multirc example: ")
print(multirc_dataset["train"][0])

# rte
rte_dataset = (
    load_dataset("super_glue", "rte")
    .map(
        lambda x: {
            "input": format_prompt(
                f"Premise: {x['premise']}\nHypothesis: {x['hypothesis']}\nIs the hypothesis entailed by the premise?\nChoose A or B:\nA. Yes\nB. No\nAnswer:"
            ),
            # 0 - entailment, 1 - not_entailment
            "output": ["A", "B"][int(x["label"])],
            "task_name": "rte",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("rte example: ")
print(rte_dataset["train"][0])

# wic
wic_dataset = (
    load_dataset("super_glue", "wic")
    .map(
        lambda x: {
            "input": format_prompt(
                f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}\nAre '{x['word']}' in the above two sentences used in the same way?\nChoose A or B:\nA. Yes\nB. No\nAnswer:"
            ),
            # 0 - False, 1 - True
            "output": ["B", "A"][int(x["label"])],
            "task_name": "wic",
        }
    )
    .select_columns(["input", "output", "task_name"])
)
print("wic example: ")
print(wic_dataset["train"][0])

# define a task2id map
TASK2ID = {
    "boolq": 0,
    "multirc": 1,
    "rte": 2,
    "wic": 3,
}

def tokenize(examples):
    inputs, targets = examples["input"], examples["output"]
    
    # Tokenize the full prompt
    full_texts = [inp + target for inp, target in zip(inputs, targets)]
    
    # Tokenize input and full text - don't return tensors here
    input_encodings = tokenizer(inputs, max_length=510, padding=False, truncation=True)
    full_encodings = tokenizer(full_texts, max_length=512, padding="max_length", truncation=True)
    
    # Convert to lists for processing
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    task_ids_list = []
    
    for i, (full_input_ids, full_attention_mask, input_length, task_name) in enumerate(
        zip(full_encodings["input_ids"], full_encodings["attention_mask"], 
            [len(enc) for enc in input_encodings["input_ids"]], examples["task_name"])
    ):
        # Create labels - mask the input part, only train on the output
        labels = full_input_ids.copy()
        # Mask input tokens with -100 so they're not included in loss
        for j in range(min(input_length, len(labels))):
            labels[j] = -100
        # Also mask padding tokens
        for j in range(len(labels)):
            if full_input_ids[j] == tokenizer.pad_token_id:
                labels[j] = -100
        
        input_ids_list.append(full_input_ids)
        attention_mask_list.append(full_attention_mask)
        labels_list.append(labels)
        task_ids_list.append([TASK2ID[task_name]])
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "task_ids": task_ids_list
    }

def get_superglue_dataset(split="train", n_samples=500):
    ds = concatenate_datasets(
        [
            boolq_dataset[split].shuffle().select(range(min(n_samples, len(boolq_dataset[split])))),
            multirc_dataset[split].shuffle().select(range(min(n_samples, len(multirc_dataset[split])))),
            rte_dataset[split].shuffle().select(range(min(n_samples, len(rte_dataset[split])))),
            wic_dataset[split].shuffle().select(range(min(n_samples, len(wic_dataset[split])))),
        ]
    )
    ds = ds.map(
        tokenize,
        batched=True,
        remove_columns=["input", "output", "task_name"],
        load_from_cache_file=False,
    )
    return ds

superglue_train_dataset = get_superglue_dataset(split="train", n_samples=1000)
superglue_eval_dataset = get_superglue_dataset(split="validation", n_samples=100)

# training and evaluation
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = []
    decoded_labels = []
    
    for pred_seq, label_seq in zip(preds, labels):
        # Find non-masked labels
        label_mask = label_seq != -100
        if label_mask.any():
            pred_tokens = pred_seq[label_mask]
            label_tokens = label_seq[label_mask]
            
            decoded_pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            decoded_label = tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
            
            decoded_preds.append(decoded_pred)
            decoded_labels.append(decoded_label)

    correct = 0
    total = len(decoded_preds)
    for pred, true in zip(decoded_preds, decoded_labels):
        if pred.strip().upper() == true.strip().upper():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}

training_args = TrainingArguments(  # Changed from Seq2SeqTrainingArguments
    "output",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    num_train_epochs=num_epochs,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    report_to="tensorboard",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    # gradient_checkpointing=True,  # Save memory
    fp16=True,  # Use mixed precision
)

trainer = Trainer(  # Changed from Seq2SeqTrainer
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=superglue_train_dataset,
    eval_dataset=superglue_eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# saving model
peft_model_id = f"{model_name_or_path.replace('/', '_')}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

# Testing the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.eval()

# Test with RTE example
i = 5
test_input = rte_dataset["validation"]["input"][i]
test_output = rte_dataset["validation"]["output"][i]

inputs = tokenizer(test_input, return_tensors="pt")
inputs["task_ids"] = torch.LongTensor([TASK2ID["rte"]])
inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask", "task_ids"]}

print("Input prompt:")
print(test_input)
print("Expected output:")
print(test_output)

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=5,
        do_sample=False,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("Generated output:")
    print(generated_text)