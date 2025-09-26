# Adopted from PRAG: https://github.com/oneal2000/PRAG
# Add our implementaion of DyPRAG
import os
import re
import json
import torch
import string
import numpy as np
import torch.nn as nn
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.modeling_qwen2 import Qwen2ForCausalLM
from models.modeling_llama import LlamaForCausalLM
from root_dir_path import ROOT_DIR
from prompt_template import get_prompt

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")


class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None,
    ):
        ground_truths = (
            {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        )
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max(
            [
                int(cls.normalize_answer(prediction) == cls.normalize_answer(gt))
                for gt in ground_truths
            ]
        )
        return {"correct": correct, "incorrect": 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None,
    ):
        ground_truths = (
            {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        )
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if (
                normalized_prediction in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def load_ragtruth(data_name, data_type):
    data_dir = os.path.join(ROOT_DIR, "data", data_name, "source_info.jsonl")
    data_list = []
    with open(data_dir, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["task_type"] == data_type:
                data_list.append(data)
    return data_list


def load_data(data_name, data_type, model_name, projector=False, data_dir=None):
    solve_dataset = []

    if data_dir is None:
        data_dir = DATA_ROOT_DIR
    print(f"Load data_dir: {data_dir}")
    input_dir = os.path.join(data_dir, data_name, model_name)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    print(f"Loading data from {input_dir}")
    files = [f for f in os.listdir(input_dir)]
    if len(files) > 1:  # more types in dataset
        if data_type == "total":  # merge all types to total
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
            total_data = []
            idx = {filename: 0 for filename in files}
            for data in all_data["total.json"]:
                typ = data["type"] + ".json"
                if idx[typ] == len(all_data[typ]):
                    break
                aim_data = all_data[typ][idx[typ]]
                assert aim_data["question"] == data["question"]
                idx[typ] += 1
                total_data.append(aim_data)
            return [["total.json", total_data]]
        for filename in files:
            if filename != "total.json":
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else:
            data_type = data_type + ".json"
            if data_type not in [v[0] for v in solve_dataset]:
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin:
            solve_dataset.append(("total.json", json.load(fin)))
        return solve_dataset


def get_model_path(model_name):
    path = ""  # Your local path to the model
    if model_name == "llama3-8b-instruct":
        return path + "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return path + "Qwen/Qwen2.5-1.5B-Instruct"
    elif model_name == "llama3.2-1b-instruct":
        return path + "meta-llama/Llama-3.2-1B-Instruct"
    else:
        return model_name


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def get_model_class(model_name):
    """
    Get the modified model class based on model name
    """
    model_classes = {
        "llama": LlamaForCausalLM,
        "qwen": Qwen2ForCausalLM,
    }
    for prefix, model_class in model_classes.items():
        if prefix in model_name.lower():
            return model_class
    raise ValueError(f"Invalid model name: {model_name}")


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model_class = get_model_class(model_name)
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ),
    )
    return model, tokenizer, generation_config


# -------------------------------- for augmentation ----------------------------------------


def model_generate(prompt, model, tokenizer, generation_config):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones(input_ids.shape).to(model.device),
        **generation_config,
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text


# ------------------------------------------------------------------------------------


def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0


def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is") :]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def predict(model, tokenizer, generation_config, question, with_cot, passages=None):
    model.eval()
    input_ids = get_prompt(tokenizer, question, passages=passages, with_cot=with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones(input_ids.shape).to(model.device),
            **generation_config,
        )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text


# -------------------------------- for DyPRAG Injection----------------------------------------
def delta_inject(model, adapter_weights):
    """
    Injects delta weights into the model's layers.

    Args:
        model: The model to inject deltas into.
        adapter_weights: A dictionary containing the delta weights.
    """
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        lora_A = adapter_weights[module + ".lora_A.weight"]
        lora_B = adapter_weights[module + ".lora_B.weight"]
        # Calculate delta
        delta = lora_B @ lora_A
        # Set the delta in the module
        setattr(m, "delta", delta.to(torch.float32))


def delta_remove(model, adapter_weights):
    """
    Removes delta weights from the model's layers.

    Args:
        model: The model to remove deltas from.
        adapter_weights: A dictionary containing the delta weights.
    """
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        delattr(m, "delta")


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x
