import json
import os

# --- 文件路径配置 (请确保这些路径在您的环境中是正确的) ---
INPUT_FILE_PATH = "data/csbersec-qa.json"
OUTPUT_FILE_PATH = "dataset/train1.jsonl"
# ----------------------------------------------------

# ----------------------------------------------------
# 1. 读取原始数据
# ----------------------------------------------------
# 关键修复: 在 try/except 外部初始化 raw_data，确保变量在任何情况下都能被定义。
raw_data = []
try:
    print(f"正在读取原始数据文件: {INPUT_FILE_PATH}")
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"✅ 成功读取 {len(raw_data)} 条原始数据。")

except FileNotFoundError:
    print(f"❌ 错误: 未找到输入文件 {INPUT_FILE_PATH}。请检查路径是否正确。")
    raw_data = [] # 确保即使出错，raw_data 也是一个列表
except json.JSONDecodeError:
    print("❌ 错误: 输入文件格式不是有效的 JSON。")
    raw_data = []
except Exception as e:
    print(f"❌ 读取文件时发生未知错误: {e}")
    raw_data = []

# ----------------------------------------------------
# 2. 转换逻辑
# ----------------------------------------------------

formatted = []
for i, item in enumerate(raw_data):
    # 构建包含系统角色的对话结构
    messages = [
        {"role": "system", "content": "你是一名网络安全专家。"},
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["output"]}
    ]
    # 添加一个 id 字段，方便在训练中追踪数据
    formatted.append({"id": f"security_qa_{i+1}", "messages": messages})

# ----------------------------------------------------
# 3. 写入 JSONL 文件
# ----------------------------------------------------
if formatted:
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        
        # 写入 jsonl，每行一条
        with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            for line in formatted:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"✅ 转换完成！共 {len(formatted)} 条数据，已保存到 {os.path.abspath(OUTPUT_FILE_PATH)}")

    except Exception as e:
        print(f"❌ 写入文件时发生错误: {e}")
else:
    print("跳过写入操作，因为没有有效数据被转换。")

# ----------------------------------------------------
# 打印转换后的数据结构示例（前两条）
# ----------------------------------------------------
if formatted:
    print("\n--- 转换后的数据结构示例 (前 2 条) ---")
    print(json.dumps(formatted[:2], ensure_ascii=False, indent=4))