import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import os
from fastapi.middleware.cors import CORSMiddleware 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# 1. 模型路径和初始化
# -------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER = "model/final_lora_adapter" 

tokenizer = None
model = None

# 定义请求体的数据结构
class QuestionRequest(BaseModel):
    question: str

# -------------------------------
# 2. 模型加载函数 (同步执行)
# -------------------------------
def load_model():
    """加载基础模型和 LoRA 适配器，确保只加载一次。"""
    global tokenizer, model

    if model is not None and tokenizer is not None:
        logging.info("Model already loaded.")
        return

    try:
        logging.info(f"Loading tokenizer from: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 

        logging.info(f"Loading base model: {BASE_MODEL}")
        # 使用 device_map="auto" 自动将模型加载到 CUDA 或 CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        logging.info(f"Loading LoRA adapter from: {LORA_ADAPTER}")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
        model.eval()
        logging.info("Model and adapter loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        model = None
        tokenizer = None
        # 如果模型加载失败，打印错误但不退出，留给 uvicorn 处理
        print("MODEL LOADING FAILED. API will return 500 errors.")

# -------------------------------
# 3. 构建 prompt
# -------------------------------
def build_prompt(messages):
    """
    Qwen2.5 格式: <|im_start|>role\ncontent<|im_end|>\n
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n" 
    
    prompt += "<|im_start|>assistant\n"
    return prompt

# -------------------------------
# 4. FastAPI 应用实例与 CORS
# -------------------------------


# 模型加载将在 if __name__ == "__main__" 中强制同步执行，避免 Uvicorn 挂起。
app = FastAPI(
    title="LoRA Qwen Security Expert API",
    description="Backend API for Qwen2.5-1.5B with LoRA adapter for security Q&A."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
# -------------------------------
# 5. API 路由
# -------------------------------
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    接收用户问题并返回模型回答
    """
    # 在推理时检查模型是否成功加载
    if model is None or tokenizer is None:
        return {"error": "Model not ready. Server failed to load the model on startup."}, 500

    try:
        question = request.question
        messages = [
            {"role": "system", "content": "你是一名网络安全专家，回答要专业简洁。"},
            {"role": "user", "content": question}
        ]

        prompt = build_prompt(messages)
        # 将输入移动到模型所在设备
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        full_output_ids = output_ids[0]
        text = tokenizer.decode(full_output_ids, skip_special_tokens=False)

        # 提取 assistant 回答内容
        split_token = "<|im_start|>assistant\n"
        if split_token in text:
            assistant_text = text.split(split_token)[-1]
            assistant_text = assistant_text.split("<|im_end|>")[0].strip()
        else:
            assistant_text = text.strip()

        return {"question": question, "answer": assistant_text}

    except Exception as e:
        logging.error(f"Inference error: {e}")
        return {"error": f"An error occurred during inference: {e}"}, 500


# -------------------------------
# 6. 运行服务器 
# -------------------------------
if __name__ == "__main__":
    
  
    load_model() 

    try:
        uvicorn.run(
            app, # 直接传递 app 对象
            host="127.0.0.1", 
            port=6007, 
            workers=1,
            log_level="info"
        )
    except RuntimeError:
        logging.error("Server startup aborted due to model loading failure.")