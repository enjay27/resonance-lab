import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, LORA_DATASET_DIR, OUTPUT_DIR, MASTER_MODEL_DIR, MAX_STEPS, MAX_SEQ_LENGTH

# 1. Configuration

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters (Optimized for Qwen)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # High rank for precision on 1.7B model
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth"
)

# 3. Format Dataset for Qwen
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Qwen3 템플릿 적용
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "assistant", "content": output_text}
        ], tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # 텍스트를 반환하면 SFTTrainer가 내부적으로 토큰화할 때 max_seq_length를 참조합니다.
    return {"text": texts, }

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
    max_seq_length = MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        # 1. Warmup 변경
        warmup_ratio=0.1,
        max_steps=MAX_STEPS,
        learning_rate=5e-5,
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        # 2. Scheduler 변경
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,

        # --- 3 & 4. 검증 및 저장(Checkpoint) 설정 추가 ---
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
    ),
)

# 5. Execute Training
trainer.train()

# 6. Export to GGUF
print("in Windows, You need to convert manually")

model.save_pretrained_merged(MASTER_MODEL_DIR, tokenizer, save_method = "merged_16bit")
# model.save_pretrained_gguf(output_model_name, tokenizer, quantization_method = "q6_k")

print(f"Training and conversion finished! Your Q4_K_M GGUF model is saved in: {MASTER_MODEL_DIR}")