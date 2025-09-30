# Stage 1: Doc-Param Pairs Collection
import os
import gc
import time
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel, PolyConfig
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, load_data, predict_poly

import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class TrainingDataWithPoly(Dataset):
    ignored_id = -100
    def __init__(self, prompt_ids, tokenizer, task_ids, max_length=3000):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        for input_ids, task_id in zip(prompt_ids, task_ids):
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                    "task_ids": task_id,
                }
            )
        self.total_len = len(self.dataset)
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]




class TrainingDataCollatorWithPoly(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, task_ids = tuple(
            map(
                lambda x: [example[x] for example in examples],
                ["input_ids", "labels", "attention_mask", "task_ids"],
            )
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
            "task_ids": torch.tensor(task_ids).to(self.device),
        }
def get_train_data(aug_model, augments, tokenizer, args):
    from prompt_template import get_prompt

    prompt_ids = []
    for aug in augments:
        psg = aug["passage"]
        rew = aug[f"{aug_model}_rewrite"]
        qas = aug[f"{aug_model}_qa"]

        qpa_cnt = (len(qas) + 1) // 2
        for qid, qa in enumerate(qas):
            # if args.dataset == "ragtruth":
            # qa['answer'] = qa['full_answer']
            if qid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(
                        get_prompt(
                            tokenizer,
                            qa["question"],
                            [ppp],
                            qa["answer"] if not args.with_cot else qa["full_answer"],
                            with_cot=args.with_cot,
                        )
                    )
            else:
                prompt_ids.append(
                    get_prompt(
                        tokenizer,
                        qa["question"],
                        None,
                        qa["answer"] if not args.with_cot else qa["full_answer"],
                        with_cot=args.with_cot,
                    )
                )
    return prompt_ids

# get the input ids and the task ids, each test sample is corresponding to a task ids
def get_prompt_ids_task_ids(augments, args, tokenizer, sample_idx):
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    task_ids = [sample_idx] * len(prompt_ids)
    return prompt_ids, task_ids

def train_with_poly(train_dataloader, args, model, tokenizer, init_adapter_path, save_path):

    
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    torch.cuda.empty_cache()
    gc.collect()
    return model



def main(args):
    if args.projector:
        data_list = load_data(
            args.dataset, args.data_type, args.augment_model, "./data_aug_projector"
        )
    else:
        data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, _generation_config = get_model(args.model_name)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    init_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name+"_poly",
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        "base_weight",
    )
    
    print("No LoRA base weight, creating...")
    peft_config = PolyConfig(
        task_type=TaskType.CAUSAL_LM,
        poly_type="poly",
        target_modules=["down_proj", "gate_proj", "up_proj"],
        r=args.lora_rank,
        n_tasks=args.n_tasks,
        n_skills=args.n_skills,
        n_splits=args.n_splits
    )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    print(f"Save LoRA base weight to {init_adapter_path}")
    os.makedirs(init_adapter_path, exist_ok=True)
    model.save_pretrained(init_adapter_path)
    time.sleep(2)
    assert os.path.exists(
        os.path.join(init_adapter_path, "adapter_model.safetensors")
    )

    cot_name = "cot" if args.with_cot else "direct"

    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        if args.projector:
            output_dir = os.path.join(
                ROOT_DIR,
                "offline",
                args.model_name+"_poly",
                f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                args.dataset,
                f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
                f"aug_model={args.augment_model}_projector",
                filename,
            )
        else:
            output_dir = os.path.join(
                ROOT_DIR,
                "offline",
                args.model_name+"_poly",
                f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                args.dataset,
                f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
                f"aug_model={args.augment_model}",
                filename,
            )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[: args.sample]

        # create the dataset for a poly model, each pid is a task_ids
        poly_dataset = {"prompt_ids": [], "task_ids": []}
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            for pid in range(len(augment)):
                # get the prompt_ids and task_ids
                prompt_ids, task_ids = get_prompt_ids_task_ids([augment[pid]], args, tokenizer, did)
                poly_dataset["prompt_ids"].extend(prompt_ids)
                poly_dataset["task_ids"].extend(task_ids)
        train_data = TrainingDataWithPoly(poly_dataset["prompt_ids"], tokenizer, poly_dataset["task_ids"])
        train_data = train_data[:10]
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.per_device_train_batch_size,
            collate_fn=TrainingDataCollatorWithPoly(tokenizer, model.device),
            shuffle=False,
        )
        # train the poly model
        model = train_with_poly(
            train_dataloader,
            args,
            model,
            tokenizer,
            init_adapter_path,
            output_dir,
        )
        # inference the poly model
        model = PeftModel.from_pretrained(model, output_dir, adapter_name="poly", is_trainable=False)
        model.eval()
        output = model.generate(torch.tensor([train_data[0]["input_ids"]]).to(model.device), task_ids=torch.tensor([train_data[0]["task_ids"]]).to(model.device))
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        print("inference sample: ", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=300)  # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--projector", action="store_true")
    parser.add_argument("--n_tasks", type=int, default=300, help="number of tasks")
    parser.add_argument("--n_skills", type=int, default=8, help="number of latent skills for each task")
    parser.add_argument("--n_splits", type=int, default=1)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
