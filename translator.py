import os
import json
import re
from openai import OpenAI

# =========================
# 配置区
# =========================
DATA_DIR = "/path/to/data"
TRANSLATION_OUTPUT_FILE = "jp_translation_map.json"

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"

# =========================
# 初始化 vLLM Client
# =========================
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# =========================
# 日语检测
# =========================
JP_PATTERN = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]"
)

def is_japanese(text):
    return isinstance(text, str) and bool(JP_PATTERN.search(text))

# =========================
# 调用 vLLM 翻译
# =========================
def translate_with_vllm(text: str) -> str:
    """
    单条翻译（日 → 中，可自行调整 prompt）
    """
    prompt = f"""请将下面的日语翻译成简体中文，只输出翻译结果，不要添加任何解释：

    {text}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        extra_body={
            "top_k": 20
        }
    )

    return response.choices[0].message.content.strip()

# =========================
# 递归收集 value（不碰 key）
# =========================
def collect_value_strings(obj, result: set):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_value_strings(v, result)
    elif isinstance(obj, list):
        for item in obj:
            collect_value_strings(item, result)
    elif isinstance(obj, str):
        result.add(obj)

# =========================
# 只替换 value（格式完全不变）
# =========================
def replace_value_strings(raw_text, translation_map):
    def replacer(match):
        value_text = match.group(1)
        if value_text in translation_map:
            translated = translation_map[value_text].replace('"', '\\"')
            return f': "{translated}"'
        return match.group(0)

    # 只匹配 value，不匹配 key
    pattern = re.compile(
        r':\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
    )

    return pattern.sub(replacer, raw_text)

# =========================
# 主流程
# =========================
def main():
    value_strings = set()
    translation_map = {}

    # ---------- 第一阶段：结构识别 ----------
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        data = json.loads(raw_text)
        collect_value_strings(data, value_strings)

    # ---------- 第二阶段：翻译（日语去重） ----------
    jp_texts = [t for t in value_strings if is_japanese(t)]

    print(f"✅ 发现日语 value 数量: {len(jp_texts)}")

    for idx, text in enumerate(jp_texts, 1):
        print(f"[{idx}/{len(jp_texts)}] 翻译中...")
        translation_map[text] = translate_with_vllm(text)

    # ---------- 第三阶段：原文级替换 ----------
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        new_text = replace_value_strings(raw_text, translation_map)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)

    # ---------- 保存翻译映射 ----------
    with open(TRANSLATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(translation_map, f, ensure_ascii=False, indent=2)

    print("✅ 翻译完成")
    print("✅ key 完全未修改")
    print("✅ JSON 原始格式完全保留")

if __name__ == "__main__":
    main()
