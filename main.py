import pandas as pd
from tqdm import tqdm

def main(max_step):    
    # !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    # !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 1024  # Choose any sequence length
    dtype = None  # This will auto-detect the best data type for your GPU
    load_in_4bit = True  # Use 4-bit quantization to save memory

    # Load the model and tokenizer from Hugging Face
    # Note: We use the base model, not a 4-bit pre-quantized one,
    # to ensure we start from the official weights.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B", # Competition-approved model
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )


    import re

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


    from datasets import load_dataset

    # Load the full training dataset
    full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

    # Shuffle the dataset for randomness and create our smaller splits
    shuffled_dataset = full_dataset.shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(50000))      # Use the first 5,000 for training
    validation_dataset = shuffled_dataset.select(range(50000, 50500)) # Use the next 500 for validation


    # The instructional prompt template for training
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



    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # A small rank for lighter training
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_alpha = 64, # A common practice is to set alpha = 2 * r
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )


    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            # num_train_epochs = 3,
            per_device_train_batch_size = 32,
            gradient_accumulation_steps = 4,
            warmup_steps = 15,
            max_steps = max_step,
            learning_rate = 5e-5,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    trainer.train()


    # Prepare the model for faster inference
    FastLanguageModel.for_inference(model)

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

    # Select a sample from the validation set
    example = validation_dataset[10] # You can change the index (e.g., to 1, 2, 50)
    question = clean_text(example["question"])
    solution = clean_text(example["solution"])

    # Format the prompt with the validation data
    inputs = tokenizer(
    [
        inference_prompt.format(question, str(solution))
    ], return_tensors = "pt").to("cuda")

    # Generate the model's response
    outputs = model.generate(**inputs, max_new_tokens = 8, use_cache = True, do_sample=False, temperature=0.0)
    response = tokenizer.batch_decode(outputs)

    # Print the results
    print("#### QUESTION ####")
    print(question)
    print("\n#### SOLUTION ####")
    print(solution)
    print("\n#### MODEL'S PREDICTION ####")
    # We process the output to show only the generated text
    print(response[0].split("Output:\n")[1])
    print("\n#### CORRECT ANSWER ####")
    print(example["is_correct"])

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
        total += 1

        if i % 25 == 0 and i > 0:
            print(f"Processed {i}/{len(validation_dataset)} examples...")

    val_accuracy = 100 * correct / total
    print(f"\nValidation Accuracy: {val_accuracy:.2f}%  ({correct}/{total})")




    # Load the official test set
    test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
    predictions = []

    # A simple function to parse 'True' or 'False' from the model's raw output
    def parse_output(response_text):
        # Find the text after "Output:"
        output_part = response_text.split("Output:\n")[-1]
        # Check if "True" is in that part, case-insensitively
        if 'true' in output_part.lower():
            return True
        return False

    # Loop through the test dataset and generate a prediction for each example
    for example in tqdm(test_dataset):
        question = clean_text(example["question"])
        solution = clean_text(example["solution"])

        # Format the prompt
        prompt = inference_prompt.format(question, str(solution))
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate the prediction
        outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True, do_sample=False, temperature=0.0)
        response_text = tokenizer.batch_decode(outputs)[0]

        # Parse the prediction and add it to our list
        prediction = parse_output(response_text)
        predictions.append(prediction)
        # break

    # Create the submission DataFrame
    submission = pd.DataFrame({
        'ID': range(len(predictions)),
        'is_correct': predictions
    })

    # Save the DataFrame to a CSV file
    submission.to_csv(f'{max_step}_submission.csv', index=False)

    print("\nSubmission file 'submission.csv' created successfully!")
    print("You can now download this file and submit it to the Kaggle competition.")




if __name__=="__main__":
    # main(250)
    # main(350)
    # main(450)
    main(1200)