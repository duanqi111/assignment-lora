import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------
# 1. 模型路径
# -------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# !!! 修正路径: 加载训练脚本最后保存的最终适配器 !!!
LORA_ADAPTER = "model/final_lora_adapter" 

# -------------------------------
# 2. 加载 tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right"
)

if tokenizer.pad_token is None:
    # 确保 pad_token 存在，避免生成时的警告
    tokenizer.pad_token = tokenizer.eos_token 

# -------------------------------
# 3. 加载基础模型 + LoRA adapter
# -------------------------------
print(f"Loading base model: {BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print(f"Loading LoRA adapter from: {LORA_ADAPTER}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()
print("Model and adapter loaded successfully.")

# -------------------------------
# 4. 构建 prompt
# -------------------------------
def build_prompt(messages):
    """
    messages: [{"role": "system|user|assistant", "content": "..."}]
    Qwen2.5 格式:
    <|im_start|>role\ncontent<|im_end|>\n
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # 确保遵循 Qwen2.5 的多轮对话格式
        prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n" 
    
    # 显式添加 assistant 的起始标记，以便模型知道轮到它回答
    prompt += "<|im_start|>assistant\n"
    return prompt

# -------------------------------
# 5. 推理函数
# -------------------------------
def ask(question):
    # 保持与训练数据中的角色一致
    messages = [
        {"role": "system", "content": "你是一名网络安全专家，回答要专业简洁。"},
        {"role": "user", "content": question}
    ]

    prompt = build_prompt(messages)
    # 移到 GPU 上
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False, # 使用贪婪搜索（确定性输出）
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id # 确保 pad token 在生成中正确使用
        )

    # 模型输出通常包含整个 prompt，需要去除
    # output_ids[0] 是包含 [prompt_tokens + generated_tokens] 的张量
    full_output_ids = output_ids[0]
    
    # 完整文本
    text = tokenizer.decode(full_output_ids, skip_special_tokens=False)

    # 提取 assistant 回答内容
    # Qwen2.5 assistant 起始标记 <|im_start|>assistant\n
    # 由于我们在 build_prompt 中添加了它，我们需要从它后面开始提取
    split_token = "<|im_start|>assistant\n"
    if split_token in text:
        assistant_text = text.split(split_token)[-1]
        # 去掉末尾的 <|im_end|> 以及可能的其他内容
        assistant_text = assistant_text.split("<|im_end|>")[0].strip()
    else:
        # 如果模型输出格式混乱，则返回整个解码结果
        assistant_text = text.strip()

    return assistant_text

# -------------------------------
# 6. 测试
# -------------------------------
if __name__ == "__main__":
    question = "什么是 SQL 注入？"
    print("====================")
    print(f"问题: {question}")
    print("--------------------")
    answer = ask(question)
    print(f"回答: {answer}")
    print("====================")

    # 第二个问题测试
    question_2 = "如何配置WAF来抵御XSS攻击？"
    print(f"问题: {question_2}")
    print("--------------------")
    answer_2 = ask(question_2)
    print(f"回答: {answer_2}")
    print("====================")
