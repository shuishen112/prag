# Evaluate the knowledge internalization of the DyPRAG-Combine answer and the RAG answer
import json
import openai
import argparse
import random
from typing import Dict, Optional
import os
from root_dir_path import ROOT_DIR

def evaluate_truthfulness(dyprag_file: str, rag_file: str, output_file: str) -> Dict[str, str]:
    # Load results from JSON files
    with open(dyprag_file, 'r') as f:
        dyprag_results = json.load(f)
    with open(rag_file, 'r') as f:
        rag_results = json.load(f)
  
    RAGTRUTH_PROMPT_TEMPLATE = """Compare DyPRAG and RAG answers to assess which better internalizes knowledge—integrating its own knowledge with the given context for a natural, informed response.
Evaluation Criteria:
1. Internalization: Does the answer go beyond repetition to integrate knowledge seamlessly?
2. Fluency: Is the response well-structured and readable?
3. Relevance: Does it stay on topic while demonstrating depth?

Mark the Winner: Identify the superior response. If both are equally strong, mark it as a tie.

Question: {question}
Context: {context}
DyPRAG Answer: {dyprag_answer}
RAG Answer: {rag_answer}

Respond in the following format:
{{
  "win model": "DyPRAG or RAG or Tie",
  "reason": "Provide a concise explanation of why the selected answer demonstrates better knowledge integration, referencing the question, context, and specific details from both answers. If one answer has clear advantages in integration, explain them; if there are errors or weaknesses, specify them."
}}"""
    # Extract query, context and answers
    ret = []
    for dyprag_result, rag_result,  in zip(dyprag_results[94:], rag_results[94:], ):
        question = dyprag_result['question']
        context = rag_result['passages']
        dyprag_answer = " ".join(dyprag_result['text'].split("\n")).strip()
        dyprag_answer = dyprag_answer.split('assistant')[0]
        rag_answer = " ".join(rag_result['text'].split("\n")).strip()
        rag_answer = rag_answer.split('assistant')[0]
        prompt = RAGTRUTH_PROMPT_TEMPLATE.format(question=question, context=context, dyprag_answer=dyprag_answer, rag_answer=rag_answer)
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                result = json.loads(response.choices[0].message.content.replace("\n",""))
                break
            except Exception as e:
                print(response.choices[0].message.content)
                print(f"Error during evaluation: {str(e)}")
                continue
        
        
        # Save complete evaluation results
        output = {
            "test_id": dyprag_result['test_id'],
            "question": question,
            "context": context,
            "dyprag_answer": dyprag_answer,
            "rag_answer": rag_answer,
            "evaluation": result
        }
        ret.append(output)
        print(f"Winner: {result['win model']}")
        print(f"Reason: {result['reason']}")
        with open(output_file, "w") as f:
            json.dump(ret, f, indent=2)
        
def main(args):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "evaluation_results.json")
    result = evaluate_truthfulness(args.dyprag_path, args.rag_path, output_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dyprag_path", type=str, required=True)
    parser.add_argument("--rag_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()
        
    print(args)
    main(args)
