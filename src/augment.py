# Adopted from PRAG: https://github.com/oneal2000/PRAG
import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

from retrieve.retriever import bm25_retrieve
from utils import get_model, model_generate
from root_dir_path import ROOT_DIR

random.seed(42)


def load_popqa(data_path):
    data_path = os.path.join(data_path, "popQA.tsv")
    dataset = pd.read_csv(data_path, sep="\t")
    new_dataset = []
    for did in range(len(dataset)):
        data = dataset.iloc[did]
        question = data["question"]
        answer = [data["obj"]] + eval(data["o_aliases"])
        val = {
            "test_id": did,
            "question": question,
            "answer": answer,
        }
        new_dataset.append(val)
    return {"total": new_dataset}


def load_complexwebquestions(data_path):
    data_path = os.path.join(data_path, "ComplexWebQuestions_dev.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        question = data["question"]
        answer = []
        for ans in data["answers"]:
            answer.append(ans["answer"])
            answer.extend(ans["aliases"])
        answer = list(set(answer))
        val = {
            "test_id": did,
            "question": question,
            "answer": answer,
        }
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def load_ragtruth(data_path):
    source_info_path = os.path.join(data_path, "source_info.jsonl")
    data_list = []
    with open(source_info_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["task_type"] == "QA":
                data_list.append(data)
    new_dataset = []
    for data in data_list:
        question = data["source_info"]["question"]
        passages = data["source_info"]["passages"]
        passages_list = [passages.split("passage")[i + 1][3:] for i in range(3)]
        val = {
            "test_id": data["source_id"],
            "question": question,
            "passages": passages_list,
        }
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def load_2wikimultihopqa(data_path):
    with open(os.path.join(data_path, "dev.json"), "r") as fin:
        dataset = json.load(fin)
    with open(os.path.join(data_path, "id_aliases.json"), "r") as fin:
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        ans_id = data["answer_id"]
        val = {
            "qid": data["_id"],
            "test_id": did,
            "question": data["question"],
            "answer": aliases[ans_id] if ans_id else data["answer"],
        }
        golden_passages = []
        contexts = {name: " ".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            golden_passages.append(psg)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_hotpotqa(data_path):
    data_path = os.path.join(data_path, "hotpot_dev_distractor_v1.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        val = {
            "qid": data["_id"],
            "test_id": did,
            "question": data["question"],
            "answer": data["answer"],
        }
        tmp = []
        contexts = {name: "".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            tmp.append(psg)
        golden_passages = []
        for p in tmp:
            if p not in golden_passages:
                golden_passages.append(p)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_iirc(data_path):
    dataset = []
    with open(os.path.join(data_path, "dev.json"), "r") as fin:
        js = json.load(fin)
        for tmp in tqdm(js):
            for example in tmp["questions"]:
                qid = example["qid"]
                question = example["question"]

                ans = example["answer"]

                if ans["type"] == "none":
                    continue
                elif ans["type"] == "value" or ans["type"] == "binary":
                    answer = [ans["answer_value"]]
                elif ans["type"] == "span":
                    answer = [v["text"].strip() for v in ans["answer_spans"]]

                context = [item["text"] for item in example["context"]]
                dataset.append(
                    {
                        "qid": qid,
                        "question": question,
                        "answer": answer,
                        "passages": context,
                    }
                )
    ret = {"total": dataset}
    return ret


def load_strategyqa(data_path):
    dataset = []
    with open(os.path.join(data_path, "strategyqa_train.json"), "r") as fin:
        dataset_1 = json.load(fin)
    with open(os.path.join(data_path, "strategyqa_train_paragraphs.json"), "r") as fin:
        dataset_2 = json.load(fin)
    for data in tqdm(dataset_1):
        example = {
            "qid": data["qid"],
            "question": data["question"],
            "cot": " ".join(data["facts"]),
            "answer": "yes" if data["answer"] == True else "no",
        }
        title = []
        ctxs = []
        for evi in data["evidence"][0]:
            if type(evi) == list:
                for t in evi:
                    if type(t) == list:
                        title.extend(t)
                    else:
                        title.append(t)
            else:
                title.append(evi)
        for tl in title:
            if tl == "operation" or tl == "no_evidence":
                continue
            if tl in dataset_2:
                ctxs.append(dataset_2[tl]["content"])
        example["passages"] = ctxs
        dataset.append(example)
    ret = {"total": dataset}
    return ret


def load_default_format_data(data_path):
    filename = data_path.split("/")[-1]
    assert filename.endswith(".json"), f"Need json data: {data_path}"
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    for did, data in enumerate(dataset):
        assert "question" in data, f'"question" not in data, {data_path}'
        question = data["question"]
        assert type(question) == str, f'"question": {question} should be a string'
        assert "answer" in data, f'"answer" not in data, {data_path}'
        answer = data["answer"]
        assert type(answer) == str or (
            type(answer) == list and (not any(type(a) != str for a in answer))
        ), f'"answer": {answer} should be a string or a list[str]'
        data["test_id"] = did
    return {filename: dataset}


def get_rewrite(
    passage, model_name, model=None, tokenizer=None, generation_config=None
):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(
        rewrite_prompt.format(passage=passage), model, tokenizer, generation_config
    )


qa_prompt_template = 'I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        "question": "What is the capital of France?",\n\
        "answer": "Paris"\n\
        "full_answer": "The capital of France is Paris."\n\
    }}, \n\
]\n\n\
This list should have at least three elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}'


def fix_qa(qa):
    if isinstance(qa, list):
        if len(qa) >= 3:
            qa = qa[:3]
            for data in qa:
                if (
                    "question" not in data
                    or "answer" not in data
                    or "full_answer" not in data
                ):
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa


def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None):

    def fix_json(output):
        if "3.2-1b-instruct" in model_name.lower():
            output = output[output.find("[") :]
            if output.endswith(","):
                output = output[:-1]
            if not output.endswith("]"):
                output += "]"
        elif "3-8b-instruct" in model_name.lower():
            if "[" in output:
                output = output[output.find("[") :]
            if "]" in output:
                output = output[: output.find("]") + 1]
        return output

    try_times = 100
    prompt = qa_prompt_template.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa)
            if ret:
                return qa
        except:
            try_times -= 1
    return output


def main(args):
    output_dir = os.path.join(
        ROOT_DIR, args.output_dir, args.dataset, args.model_name.split("/")[-1]
    )
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        load_func = globals()["load_default_format_data"]
    load_dataset = load_func(args.data_path)
    if args.projector:
        start_idx = 300
    else:
        start_idx = 0
    if len(load_dataset) == 1:
        solve_dataset = load_dataset
    else:
        solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"}
        with open(os.path.join(output_dir, "total.json"), "w") as fout:
            json.dump(
                load_dataset["total"][start_idx : start_idx + args.sample],
                fout,
                indent=4,
            )

    model, tokenizer, _ = get_model(args.model_name)
    args.model_name = args.model_name.split("/")[-1]
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ),
        temperature=0.7,
        top_k=50,
    )
    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")
        output_file = os.path.join(
            output_dir, filename if filename.endswith(".json") else filename + ".json"
        )
        print(output_file)
        ret = []
        dataset = dataset[start_idx : start_idx + args.sample]
        pbar = tqdm(total=args.sample * args.topk)
        for data in dataset:
            print(f"question: {data['question']}")
            if "passages" not in data or len(data["passages"]) == 0:
                passages = bm25_retrieve(data["question"], topk=args.topk + 10)
            else:
                passages = data["passages"]
            print(f"passages: {passages[0]}")
            final_passages = []
            data["augment"] = []
            for psg in passages:
                val = {
                    "pid": len(final_passages),
                    "passage": psg,
                    f"{args.model_name}_rewrite": get_rewrite(
                        psg, args.model_name, model, tokenizer, generation_config
                    ),
                }
                print(f"{args.model_name}_rewrite: {val[f'{args.model_name}_rewrite']}")
                qa = get_qa(psg, args.model_name, model, tokenizer, generation_config)
                if fix_qa(qa)[0] == False:  # skip error passage
                    continue
                print(f"{args.model_name}_qa: {qa}")
                val[f"{args.model_name}_qa"] = qa
                data["augment"].append(val)
                print(f"{args.model_name}_qa: {val[f'{args.model_name}_qa']}")
                final_passages.append(psg)
                pbar.update(1)
                if len(data["augment"]) == args.topk:
                    break
            data["passages"] = final_passages
            ret.append(data)
            import pdb

            pdb.set_trace()
            with open(output_file, "w") as fout:
                json.dump(ret, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="./llama-3.2-1b-instruct",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="2wikimultihopqa",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/2wikimultihopqa/",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=300,
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="data_aug")
    parser.add_argument("--projector", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)
