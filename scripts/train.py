import os
import sys

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, LORA_DATASET_DIR, OUTPUT_DIR, MASTER_MODEL_DIR, MAX_STEPS

# 1. Configuration
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters (Optimized for Qwen)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # High rank for precision on 1.7B model
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
)

# 3. Format Dataset for Qwen
def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"]):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

dataset = load_dataset("json", data_files={
    "train": os.path.join(LORA_DATASET_DIR, "train.jsonl"),
    "test": os.path.join(LORA_DATASET_DIR, "val.jsonl")
})
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. Training Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # 1. Warmup 변경
        warmup_ratio=0.05,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        # 2. Scheduler 변경
        lr_scheduler_type="cosine",
        seed=1557,
        output_dir=OUTPUT_DIR,

        # --- 3 & 4. 검증 및 저장(Checkpoint) 설정 추가 ---
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
    ),
)

# 5. Execute Training
trainer.train()

# 6. Export to GGUF
print("in Windows, You need to convert manually")

model.save_pretrained_merged(MASTER_MODEL_DIR, tokenizer, save_method = "merged_16bit")
# model.save_pretrained_gguf(output_model_name, tokenizer, quantization_method = "q6_k")

print(f"Training and conversion finished! Your Q4_K_M GGUF model is saved in: {MASTER_MODEL_DIR}")