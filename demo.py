import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
    )
    return parser.parse_args()

def main(args):    
    max_seq_length = 1024  # Choose any sequence length
    dtype = None  # This will auto-detect the best data type for your GPU
    load_in_4bit = True  # Use 4-bit quantization to save memory

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B", # Competition-approved model
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    save_path = args.checkpoint_path
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = save_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    print(f"Model and tokenizer loaded from: {save_path}")


    def clean_text(x):
        def _clean_one(text):
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)

            text = str(text)
            text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
            text = re.sub(r"`([^`]*)`", r"\1", text)
            text = re.sub(r"(?<=\d),(?=\d)", "", text)
            text = re.sub(r"[０-９]", lambda m: chr(ord(m.group(0)) - 65248), text)
            text = re.sub(r"\s+", " ", text).strip()

            return text

        if isinstance(x, list):
            return [_clean_one(item) for item in x]
        else:
            return _clean_one(x)
    

    # Load the full training dataset
    full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

    # Shuffle the dataset for randomness and create our smaller splits
    shuffled_dataset = full_dataset.shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(50000))      # Use the first 5,000 for training
    validation_dataset = shuffled_dataset.select(range(50000, 50500)) # Use the next 500 for validation


    # # The instructional prompt template for training
    training_prompt = best_prompt_candidate = """You are a great mathematician. You must decide if the Solution to the Question is correct.
    Your response should be exactly 'True' if the Solution's final answer is correct.
    Your response should be exactly 'False' if the Solution's final answer is not correct.
    Question:
    {}
    Solution:
    {}
    Output:
    {}"""



    # We must add an End Of Sequence (EOS) token to tell the model when a completion is finished.
    EOS_TOKEN = tokenizer.eos_token

    # This function formats our data samples into the prompt template.
    def formatting_prompts_func(examples):
        questions = clean_text(examples["question"])
        solutions = clean_text(examples["solution"])
        outputs = examples["is_correct"]
        texts = []
        for question, solution, output in zip(questions, solutions, outputs):
            # Format the prompt and add the EOS token
            text = training_prompt.format(question, str(solution), str(output)) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts }

    # Apply the formatting function to our training dataset
    formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=True)


    from trl import SFTTrainer
    from transformers import TrainingArguments


    # Create the prompt template for inference (no answer included)
    inference_prompt = best_prompt_candidate = """You are a great mathematician. You must decide if the Solution to the Question is correct.
    Your response should be exactly 'True' if the Solution's final answer is correct.
    Your response should be exactly 'False' if the Solution's final answer is not correct.
    Question:
    {}
    Solution:
    {}
    Output:
    """

    print("\n===== Calculating Validation Accuracy =====")

    correct = 0
    total = 0

    for i, example in enumerate(validation_dataset):
        question = clean_text(example["question"])
        solution = clean_text(example["solution"])
        true_label = str(example["is_correct"]).strip().lower()

        inputs = tokenizer(
            [inference_prompt.format(question, str(solution))],
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True, do_sample=False, temperature=0.0)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if "Output:" in response:
            pred_text = response.split("Output:")[-1].strip()
        else:
            pred_text = response.strip()

        pred_label = "true" if "true" in pred_text.lower() else "false"

        if pred_label == true_label:
            correct += 1
        else:
            with open("/scratch/wz3008/dl/wrong_case.txt", "a", encoding="utf-8") as f:
                f.write(f"Question: {question}\n"
                        f"Solution: {solution}\n"
                        f"Pred label: {pred_label}\n"
                        f"True label: {true_label}\n"
                        f"{'-'*40}\n")
            # print(f"Question: {question}\nSolution: {solution}\nPred label: {pred_label}\nTrue label: {true_label}")
            # import pdb; pdb.set_trace()
        total += 1

        if i % 25 == 0 and i > 0:
            print(f"Processed {i}/{len(validation_dataset)} examples...")

    val_accuracy = 100 * correct / total
    print(f"\nValidation Accuracy: {val_accuracy:.2f}%  ({correct}/{total})")



if __name__=="__main__":
    args = parse_args()
    main(args)