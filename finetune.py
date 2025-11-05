import pandas as pd
from tqdm import tqdm


def main(max_step, finetune_step):    
    # # !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    # # !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 1024  # Choose any sequence length
    dtype = None  # This will auto-detect the best data type for your GPU
    load_in_4bit = True  # Use 4-bit quantization to save memory

    # # Load the model and tokenizer from Hugging Face
    # # Note: We use the base model, not a 4-bit pre-quantized one,
    # # to ensure we start from the official weights.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B", # Competition-approved model
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    save_path = "/scratch/wz3008/dl/outputs/checkpoint-350"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = save_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    print(f"Model and tokenizer loaded from: {save_path}")

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



    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 32, # A small rank for lighter training
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj", "lm_head"],
    #     lora_alpha = 64, # A common practice is to set alpha = 2 * r
    #     lora_dropout = 0.05,
    #     bias = "none",
    #     use_gradient_checkpointing = "unsloth",
    #     random_state = 42,
    # )


    from trl import SFTTrainer
    from transformers import TrainingArguments

    # trainer = SFTTrainer(
    #     model = model,
    #     tokenizer = tokenizer,
    #     train_dataset = formatted_train_dataset,
    #     dataset_text_field = "text",
    #     max_seq_length = max_seq_length,
    #     args = TrainingArguments(
    #         # num_train_epochs = 3,
    #         per_device_train_batch_size = 32,
    #         gradient_accumulation_steps = 4,
    #         warmup_steps = 15,
    #         max_steps = max_step,
    #         learning_rate = 5e-5,
    #         fp16 = not torch.cuda.is_bf16_supported(),
    #         bf16 = torch.cuda.is_bf16_supported(),
    #         logging_steps = 10,
    #         optim = "adamw_8bit",
    #         weight_decay = 0.01,
    #         lr_scheduler_type = "linear",
    #         seed = 42,
    #         output_dir = "outputs",
    #         report_to = "none",
    #     ),
    # )

    # trainer.train()


    # # Prepare the model for faster inference
    # FastLanguageModel.for_inference(model)

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

    # # Select a sample from the validation set
    # example = validation_dataset[10] # You can change the index (e.g., to 1, 2, 50)
    # question = clean_text(example["question"])
    # solution = clean_text(example["solution"])

    # # Format the prompt with the validation data
    # inputs = tokenizer(
    # [
    #     inference_prompt.format(question, str(solution))
    # ], return_tensors = "pt").to("cuda")

    # # Generate the model's response
    # outputs = model.generate(**inputs, max_new_tokens = 8, use_cache = True, do_sample=False, temperature=0.0)
    # response = tokenizer.batch_decode(outputs)

    # # Print the results
    # print("#### QUESTION ####")
    # print(question)
    # print("\n#### SOLUTION ####")
    # print(solution)
    # print("\n#### MODEL'S PREDICTION ####")
    # # We process the output to show only the generated text
    # print(response[0].split("Output:\n")[1])
    # print("\n#### CORRECT ANSWER ####")
    # print(example["is_correct"])

    # print("\n===== Calculating Validation Accuracy =====")

    # correct = 0
    # total = 0

    # for i, example in enumerate(validation_dataset):
    #     question = clean_text(example["question"])
    #     solution = clean_text(example["solution"])
    #     true_label = str(example["is_correct"]).strip().lower()

    #     inputs = tokenizer(
    #         [inference_prompt.format(question, str(solution))],
    #         return_tensors="pt"
    #     ).to("cuda")

    #     outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True, do_sample=False, temperature=0.0)
    #     response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    #     if "Output:" in response:
    #         pred_text = response.split("Output:")[-1].strip()
    #     else:
    #         pred_text = response.strip()

    #     pred_label = "true" if "true" in pred_text.lower() else "false"

    #     if pred_label == true_label:
    #         correct += 1
    #     total += 1

    #     if i % 25 == 0 and i > 0:
    #         print(f"Processed {i}/{len(validation_dataset)} examples...")

    # val_accuracy = 100 * correct / total
    # print(f"\nValidation Accuracy: {val_accuracy:.2f}%  ({correct}/{total})")

    ########################################################
    #####                Start Finetune                #####
    ########################################################
    print("Start Finetuning")
    from datasets import load_dataset
    import torch
    test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
    # # import pdb; pdb.set_trace()
    # def _pick_token_id(tokenizer, text_options):
    #     """
    #     兼容句首/前置空格等 BPE 差异：在若干候选字符串中择其一的“首 token id”。
    #     text_options: 如 ["True", " True"] 或 ["False", " False"]
    #     返回：首 token id
    #     """
    #     cands = []
    #     for t in text_options:
    #         ids = tokenizer.encode(t, add_special_tokens=False)
    #         if len(ids) > 0:
    #             cands.append(ids[0])
    #     # 兜底：如果都没切出 token，就直接编码不加 special，取第一个
    #     if not cands:
    #         ids = tokenizer.encode(text_options[0], add_special_tokens=False)
    #         cands = [ids[0]] if ids else []
    #     return cands[0]
    # def _score_true_false_logits(model, tokenizer, prompt, device="cuda"):
    #     """
    #     给定已包含 'Output:\\n' 的 prompt，仅比较下一步 True/False 的 logit。
    #     返回：pred(bool), prob_of_pred(float), margin(float)
    #     """
    #     inputs = tokenizer([prompt], return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         out = model(**inputs)
    #         logits = out.logits[:, -1, :]  # 最后一位置的下一 token 分布

    #     # 兼容 token 化差异：同时考虑带空格或不带空格
    #     tid = _pick_token_id(tokenizer, ["True", " True"])
    #     fid = _pick_token_id(tokenizer, ["False", " False"])

    #     lt = logits[0, tid].item()
    #     lf = logits[0, fid].item()
    #     # 二分类 softmax 概率
    #     pt = torch.softmax(torch.tensor([lt, lf]), dim=0)[0].item()  # P(True)
    #     pred_is_true = (lt > lf)
    #     prob_pred = pt if pred_is_true else (1.0 - pt)
    #     margin = abs(lt - lf)
    #     return (True if pred_is_true else False), float(prob_pred), float(margin)
    
    # def run_pseudo_label_routine(
    #     finetune_step,
    #     conf_threshold=0.90,
    #     margin_threshold=1.5,
    #     short_ft_lr=2e-5,
    #     short_ft_weight_decay=0.02,
    #     lora_dropout_new=None  # 可选：需要时可在再次微调前调大 dropout
    # ):

    #     # 如果想在再训练前加大 dropout（可选）
    #     if lora_dropout_new is not None and hasattr(model, "peft_config"):
    #         try:
    #             # unsloth 的 LoRA 通常通过 peft_config 保存超参；不同版本字段名可能不同，谨慎处理
    #             for _, cfg in model.peft_config.items():
    #                 if hasattr(cfg, "lora_dropout"):
    #                     cfg.lora_dropout = lora_dropout_new
    #             print(f"[Info] Updated LoRA dropout to {lora_dropout_new}.")
    #         except Exception as e:
    #             print(f"[Warn] Failed to update LoRA dropout: {e}")

    #     # 1) 在官方 test（无标签）上打分并筛高置信伪标签
    #     test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
    #     pseudo_texts = []
    #     kept = 0

    #     for ex in test_dataset:
    #         q = clean_text(ex["question"])
    #         s = clean_text(ex["solution"])
    #         prompt = inference_prompt.format(q, str(s))  # 以你的推理模板构造，末尾含 "Output:\n"
    #         pred, conf, margin = _score_true_false_logits(model, tokenizer, prompt)
    #         if conf >= conf_threshold and margin >= margin_threshold:
    #             txt = training_prompt.format(q, str(s), "True" if pred else "False") + tokenizer.eos_token
    #             pseudo_texts.append(txt)
    #             kept += 1

    #     print(f"[Pseudo-Label] Collected {kept} high-confidence test samples "
    #         f"(conf >= {conf_threshold}, margin >= {margin_threshold}), from {len(test_dataset)} samples in test dataset.")

    #     if kept == 0:
    #         print("[Pseudo-Label] No pseudo labels passed thresholds. Skipping short fine-tuning.")
    #     else:
    #         # 2) 与原训练文本合并（你原先的 formatted_train_dataset['text']）
    #         mixed_train = Dataset.from_dict({
    #             "text": list(formatted_train_dataset["text"]) + pseudo_texts
    #         })

    #         # 3) 短程再微调（低学习率+较强正则）
    #         from trl import SFTTrainer
    #         from transformers import TrainingArguments

    #         short_trainer = SFTTrainer(
    #             model = model,
    #             tokenizer = tokenizer,
    #             train_dataset = mixed_train,
    #             dataset_text_field = "text",
    #             max_seq_length = max_seq_length,
    #             args = TrainingArguments(
    #                 per_device_train_batch_size = 32,
    #                 gradient_accumulation_steps = 4,
    #                 warmup_steps = 10,
    #                 max_steps = finetune_step,
    #                 learning_rate = short_ft_lr,
    #                 fp16 = not torch.cuda.is_bf16_supported(),
    #                 bf16 = torch.cuda.is_bf16_supported(),
    #                 logging_steps = 10,
    #                 optim = "adamw_8bit",
    #                 weight_decay = short_ft_weight_decay,
    #                 lr_scheduler_type = "linear",
    #                 seed = 42,
    #                 output_dir = "outputs_pseudo_shortft",
    #                 report_to = "none",
    #             ),
    #         )
    #         print("[Pseudo-Label] Start short fine-tuning...")
    #         short_trainer.train()
    #         print("[Pseudo-Label] Short fine-tuning done.")
    # run_pseudo_label_routine(finetune_step)
        

    def score_one(q, s):
        prompt = inference_prompt.format(clean_text(q), str(clean_text(s)))
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
            # 只看 "Output:\n" 后的下一个位置
            logits = out.logits[:, -1, :]  # (1, vocab)
        tid = tokenizer.convert_tokens_to_ids("True")
        fid = tokenizer.convert_tokens_to_ids("False")
        lt, lf = logits[0, tid].item(), logits[0, fid].item()
        # softmax 概率 & 置信度（margin）
        mx = max(lt, lf)
        pt = torch.softmax(torch.tensor([lt, lf]), dim=0)[0].item()
        pred = True if lt > lf else False
        margin = abs(lt - lf)
        return pred, pt if pred else (1-pt), margin

    pseudo_texts = []
    for ex in test_dataset:
        pred, conf, margin = score_one(ex["question"], ex["solution"])
        if conf >= 0.90 and margin >= 1.5:  # 可按需调整阈值
            # 伪标签样本与训练模板一致，**必须保留 "Output:" 锚点**
            txt = training_prompt.format(
                clean_text(ex["question"]),
                str(clean_text(ex["solution"])),
                "True" if pred else "False"
            ) + tokenizer.eos_token
            pseudo_texts.append(txt)

    print("Pseudo-labeled test samples:", len(pseudo_texts))

    from datasets import Dataset
    pseudo_ds = Dataset.from_dict({"text": pseudo_texts})
    # 2) 与原 formatted_train_dataset 混合（也可给伪标签样本 downweight）
    mixed_train = Dataset.from_dict({
        "text": list(formatted_train_dataset["text"]) + pseudo_texts
    })

    # 3) 用较小 lr + 较少步数再训一小会儿
    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = mixed_train,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 32,
            warmup_steps = 5,
            max_steps = finetune_step,                # 短程
            learning_rate = 2e-5,           # 降 lr
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.02,            # 稍强正则
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs_pseudo",
            report_to = "none",
        ),
    )
    trainer.train()

    
    ###################################################################
    #####                Finetune Finish, Evaluate                #####
    ###################################################################
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
    # import pdb; pdb.set_trace()


    #####################################################################
    #####                Finally, Generate CSV, File                #####
    #####################################################################

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
    submission.to_csv(f'{max_step}_finetune_{finetune_step}_submission.csv', index=False)

    print("\nSubmission file 'submission.csv' created successfully!")
    print("You can now download this file and submit it to the Kaggle competition.")




if __name__=="__main__":
    # main(250)
    # main(350)
    # main(450)
    main(350,100)