import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# -------------------------------
# 0. 屏蔽警告
# -------------------------------
# 屏蔽关于 temperature/top_p/top_k 在 do_sample=False 时无效的警告
warnings.filterwarnings("ignore", ".*is set to `False`. However, `temperature` is set to.*", category=UserWarning)
warnings.filterwarnings("ignore", ".*is set to `False`. However, `top_p` is set to.*", category=UserWarning)
warnings.filterwarnings("ignore", ".*is set to `False`. However, `top_k` is set to.*", category=UserWarning)
warnings.filterwarnings("ignore", ".*`resume_download` is deprecated.*", category=FutureWarning)


# -------------------------------
# 1. 模型路径配置
# -------------------------------
BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# -------------------------------
# 2. 加载 tokenizer
# -------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# 3. 加载 Base 模型 (不加载 LoRA 适配器)
# -------------------------------
print("Loading BASE model (No LoRA adapter)...")
# 使用 device_map="auto" 自动将模型加载到 CUDA 或 CPU
# 使用 float32 确保稳定性，如果显存不足请尝试 torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()
print("Base Model loaded successfully.")

# -------------------------------
# 4. 问答函数 - 保持和微调模型相同的推理逻辑
# -------------------------------
def ask_base_model(question, max_new_tokens=128):
    """
    question: str, 用户问题
    max_new_tokens: int, 最大生成长度
    """
    # 保持与微调模型相同的系统 Prompt
    messages = [
        {"role": "system", "content": "你是一名网络安全专家，回答要专业简洁。"},
        {"role": "user", "content": question}
    ]

    # 使用官方 chat template 构建 prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 转为 tensor
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    im_end_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    pad_token_id = tokenizer.pad_token_id
    
    stop_tokens = [tokenizer.eos_token_id]
    if im_end_token_id is not None and im_end_token_id not in stop_tokens:
        stop_tokens.append(im_end_token_id)

    # 贪心生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_tokens,
            pad_token_id=pad_token_id, 
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # 清理逻辑：找到 <|im_start|>assistant\n 标记后的内容
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in full_text:
        answer_part = full_text.split(assistant_marker)[-1]
        
        if "<|im_end|>" in answer_part:
            answer = answer_part.split("<|im_end|>")[0]
        else:
             answer = answer_part

        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        final_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return final_answer.strip()
    
    return full_text.strip()


# -------------------------------
# 5. 主函数测试
# -------------------------------
if __name__ == "__main__":
    

    # 示例问题
    question_1 = "什么是 SQL 注入？"
    
    print(f"用户提问 1：{question_1}")
    try:
        answer_1 = ask_base_model(question_1)
        print(f"基础模型回答：\n{answer_1}\n")
    except Exception as e:
        print(f"发生错误：{e}")
        print("请检查模型路径和环境依赖是否正确安装。")
    question_2 = "如何配置WAF来抵御XSS攻击？"
    
    print(f"用户提问 2：{question_2}")
    try:
        answer_2 = ask_base_model(question_2)
        print(f"基础模型回答：\n{answer_2}\n")
    except Exception as e:
        print(f"发生错误：{e}")
        print("请检查模型路径和环境依赖是否正确安装。")

    # TODO: 您可以在此处添加更多测试问题，以构建您的定量数据集
    # question_2 = "请简述 OAuth 2.0 中的授权码流程。"
    # print(f"用户提问 2：{question_2}")
    # answer_2 = ask_base_model(question_2)
    # print(f"基础模型回答：\n{answer_2}\n")