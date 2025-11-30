import os
import argparse
import json
from typing import List
import torch
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI
# from transformers import AutoTokenizer
# from PathFinderPRM import PathFinderPRM
# from QwenPRM import QwenPRM
# from prompts.policy_prompt import Qwen_Policy_Prompt

import regex

def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      data = [json.loads(line) for line in f]
    return data
# Define the allowed character set:
# - \p{Latin} matches all Latin letters
# - \p{Greek} matches all Greek letters
# - \d matches digits
# - \s matches whitespace characters
# - The following are common math symbols, punctuation, and symbols frequently used in LaTeX
allowed_pattern = regex.compile(
    r"[^\p{Latin}\p{Greek}\d\s"
    r"\+\-\*/=\^\_\'\".,:;?!\(\)\{\}\[\]\\\$%<>|&@#"
    r"√∞±×÷°]"
)

# SYSTEM_PROMPT="You are a Math Teacher. Given a question and a student's solution, evaluate the mathemetical correctness, logic consistency of the current step and whether it will lead to the correct final solution"
# EOS_TOKEN="<|im_end|>"

def extract_boxed(s):
    results = []
    i = 0
    while True:
        start = s.find(r'\boxed{', i)
        if start == -1:
            break
        # advance past the “\boxed{”
        j = start + len(r'\boxed{')
        depth = 1
        while j < len(s) and depth > 0:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            # everything from just after the first '{' up to j-1
            content = s[start + len(r'\boxed{') : j - 1]
            results.append(content)
            i = j
        else:
            # unbalanced braces: bail out
            break
    
    if results:
        return results[-1]
    else:
        return None


def find_illegal_chars(text: str) -> list:
    """
     Find characters in the text that are outside the allowed set, returning a list.
    """
    return allowed_pattern.findall(text)

def is_math_answer_valid(answer: str) -> bool:
    """
    Check whether the math answer contains illegal characters:
      - If returns True, it means the text has no disallowed characters
      - If returns False, the text contains illegal characters
    """
    illegal = find_illegal_chars(answer)
    if illegal:
        return False
    return True


def get_solutions_batch(policy_model, questions, temperature):

    messages = [[{"role":"user", "content":question}] for question in questions]

    if temperature == 0:
        sampling_params = SamplingParams(
            n=1,
            top_k=1,
            temperature=0,
            max_tokens=16384, #512*32
            seed=42
        )
    else:
        sampling_params = SamplingParams(
            n=1,
            top_p=0.95,
            temperature=temperature,
            max_tokens=16384, #512*32
            seed=42
        )

    llm_outputs = policy_model.chat(
                    messages, 
                    sampling_params,
                    chat_template_kwargs={"enable_thinking": False}
                )
    # breakpoint()
    outputs = []
    for output_obj in llm_outputs:
        # breakpoint()
        gen_text = output_obj.outputs[0].text
        outputs.append(gen_text)
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Greedy Search reasoning pipeline with reward model")

    parser.add_argument("--policy_model_path", type=str, required=True, help="Path to the policy model.")
    
    parser.add_argument("--data", type=str, required=True, help="Dataset to Evaluate on", 
                        choices = ["math", "amc23", "aime25", "aime24", "college_math", "minerva_math", "olympiadbench"])

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=0, help="the temperature of the policy model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size for inference")
    parser.add_argument("--data_begin", type=int, default=0, help="Starting index of the dataset to process.")
    parser.add_argument("--data_end", type=int, default=None, help="Ending index of the dataset to process.")

    parser.add_argument("--prompt", type=str, default="base", help="Dataset to Evaluate on", 
                        choices = ["base", "boxed"])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    DATA_PATHS = {
        "math": "./eval_data/MATH/Math-OAI.jsonl", 
        "amc23": "./eval_data/AMC23/test.jsonl", 
        "aime25": "./eval_data/AIME25/AIME25_train.jsonl", 
        "aime24": "./eval_data/AIME24/test.jsonl",
        "college_math": "./eval_data/College_Math/college_math_200.jsonl", 
        "minerva_math": "./eval_data/Minerva-MATH/minerva-math.jsonl",
        "olympiadbench" : "./eval_data/OlympiadBench/olympiadbench_200.jsonl"
    }

    if args.data == "minerva_math":
        ans_key = "solution"
    elif args.data == "olympiadbench":
        ans_key = "final_answer"
    else:
        ans_key = "answer"

    dataset = read_jsonl(DATA_PATHS[args.data])

    print("Number of test set samples:", len(dataset))
    
    if args.data_begin!=0 or args.data_end != None:
        dataset = dataset[args.data_begin:args.data_end] #[dataset[i] for i in range(args.data_begin, args.data_end)]

    policy_model = LLM(
        model=args.policy_model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )
    
    new_dataset = []
    # for i, data in tqdm(enumerate(dataset), desc="Processing dataset", total=len(dataset)):
    num_samples = len(dataset)
    batch_size = args.batch_size
    
    for i in tqdm(range(0,num_samples, batch_size)):
        
        if i+batch_size < num_samples:
            batch_data = dataset[i:i+batch_size]
        else:
            batch_data = dataset[i:]
        # breakpoint()
        if args.prompt == "base":
            questions = [sample["problem"] for sample in batch_data]
        elif args.prompt == "boxed":
            questions = [sample["problem"] + "Present the final answer enclosed in \\boxed{}." for sample in batch_data]
        else:
            raise ValueError(f"Unknown prompt type {args.prompt}")

        solutions = get_solutions_batch(policy_model, questions, args.temperature)

        for j in range(len(batch_data)):        
            new_dataset.append({
                "question": batch_data[j]["problem"],
                "generated_solution": {"solution": solutions[j], "pred_answer": extract_boxed(solutions[j])}, #solutions[j],
                "gt_answer": batch_data[j][ans_key],
            })

        output_file = os.path.join(args.output_dir, f"result-{args.data_begin}-{args.data_end}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"Done! Results are saved to {output_file}.")


if __name__ == "__main__":
    main()