import os
# 抑制 bitsandbytes 库的欢迎信息和CUDA版本警告
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BNB_CUDA_VERSION"] = ""

# ==========================================================
# 注意: 依赖安装命令已移除，请在运行脚本前在终端手动执行：
# pip install --upgrade accelerate peft transformers
# 以确保所有库版本兼容。
# ==========================================================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch

# -------------------------------
# 1. Model & Tokenizer
# -------------------------------
# ❌ 当前错误提示网络连接不可达。
# 解决方案：请将模型 Qwen/Qwen2.5-1.5B-Instruct 完整下载到本地，
# 并将 MODEL_NAME 修改为该本地目录的绝对路径。
# 
# 示例：假设您已下载模型到 /root/models/Qwen2.5-1.5B-Instruct
# MODEL_NAME = "/root/models/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    # 对于 Causal LM 训练，right padding 是常用且兼容 DataCollator 的设置
    padding_side="right" 
)

# pad_token 必须设置，否则 DataCollatorForLanguageModeling 会报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型，使用 float16 减少显存占用
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# -------------------------------
# 2. Load Dataset
# -------------------------------
dataset = load_dataset(
    "json",
    # *** 修复点: 确保使用正确的 .jsonl 文件扩展名 ***
    data_files="dataset/train1.jsonl", 
    split="train"
)

# -------------------------------
# 3. Build Prompt (Qwen2.5 messages 格式)
# -------------------------------
def build_prompt(example):
    """
    将 messages 列表转成 Qwen2.5 指令格式：
    <|im_start|>role\ncontent<|im_end|>\n
    """
    messages = example["messages"]
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # 注意：这里需要确保末尾有一个额外的换行符（\n）用于分隔不同的对话回合
        prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n" 
    
    # 确保最终的 prompt 字符串以 EOS 标记结束 (虽然tokenizer会自动添加，但显式包含更安全)
    # 不过在训练时，DataCollatorForLanguageModeling 会自动处理，保持现有逻辑即可
    return prompt

# -------------------------------
# 4. Tokenize & labels
# -------------------------------
def preprocess(example):
    """Tokenize function, used for dataset mapping."""
    prompt = build_prompt(example)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=1024, # 最大序列长度
        padding="max_length"
    )
    # 对于 Causal LM，输入 ID 即为标签 ID
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 对数据集进行预处理和 tokenization
train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# -------------------------------
# 5. LoRA configuration
# -------------------------------
lora_config = LoraConfig(
    r=8, # LoRA 秩
    lora_alpha=32,
    lora_dropout=0.05,
    # 针对 Qwen 模型，通常 target q_proj 和 v_proj
    target_modules=["q_proj", "v_proj"], 
    bias="none",
    task_type="CAUSAL_LM",
)

# 将 LoRA 适配器添加到模型
model = get_peft_model(model, lora_config)

# 打印模型可训练参数信息
model.print_trainable_parameters()

# -------------------------------
# 6. Trainer arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="model/lora_adapter", # LoRA 适配器保存路径
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # 梯度累积步数，有效批次大小为 2 * 4 = 8
    logging_steps=10,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True, # 启用 16 位浮点训练，加速并节省显存
    bf16=False,
    save_strategy="epoch", # 每 epoch 结束保存一次
    save_total_limit=2, # 只保留最新的 2 个检查点
    report_to="none",
)

# -------------------------------
# 7. DataCollator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # 非 Masked Language Modeling，用于 Causal LM
)

# -------------------------------
# 8. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# -------------------------------
# 9. Train
# -------------------------------
if __name__ == "__main__":
    print("Starting LoRA training...")
    # 打印 LoRA 模型可训练参数信息
    model.print_trainable_parameters()
    trainer.train()
    print("Training finished!")

    # -------------------------------
    # 10. Save LoRA Adapter
    # -------------------------------
    # 确保在训练完成后保存最终的适配器
    final_output_dir = "model/final_lora_adapter"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"LoRA adapter and tokenizer saved to {final_output_dir}")