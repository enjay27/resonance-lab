from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
max_seq_length = 512
model_name = "rd211/Qwen3-1.7B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
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
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files={"train": "lora_dataset/train.jsonl", "test": "lora_dataset/val.jsonl"})
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. Training Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 1000,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir ="../outputs",
    ),
)

# 5. Execute Training
trainer.train()

# 6. Export to GGUF (Change your output name here)
# This will save a file like: BlueProtocol_JP_KO_Translator.Q8_0.gguf
output_model_name = "BlueProtocol_JP_KO_Translator-merged-16bit"

model.save_pretrained_merged("model_f16", tokenizer, save_method = "merged_16bit")
# model.save_pretrained_gguf(output_model_name, tokenizer, quantization_method = "q6_k")

print(f"Training finished! Your 1.7B GGUF model is saved as: {output_model_name}")