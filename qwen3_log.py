from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

# 使用 Qwen3-4B-Thinking-2507
model_name = "Qwen/Qwen3-4B-Thinking-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 訊息
messages = [
    {"role": "user", "content": 
     """
You are an art style analysis assistant. 
Your task is to take a simple user prompt describing an art style (e.g., "van Gogh style") and expand it into a precise and detailed description of the style, 
if there are multiple style, you need to propose a method that can trying to merge two styles, and conclude with a final prompt. 
Always structure your analysis according to the following criteria: 

1. **Form & Composition** – Describe the typical structure, shapes, spatial arrangement, and overall organization of artworks in this style. 
2. **Color & Tonality** – Explain the common color palette, tonal range, and use of contrast or harmony. 
3. **Brushwork & Technique** – Identify characteristic brushstrokes, textures, and technical methods used in this style. 
4. **Expression & Theme** – Describe the mood, atmosphere, or emotional quality conveyed through the artworks. 
5. **Historical & Cultural Context** – Provide the cultural background, historical period, and artistic influences that define this style. 

Your output should always be clear, structured, and specific, even if the user prompt is vague. If the style has multiple phases (e.g., early vs. late period), highlight key distinctions. Do not invent unrelated attributes—base your elaboration on widely recognized features of the style. now here's the user prompt "Fauvism, Miyazaki Hayao"
"""},
]

# 準備輸入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 嘗試切開思考內容
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# Latex 符號轉換
latex_to_unicode = {
    r"\\Delta": "Δ",
    r"\\hbar": "ℏ",
    r"\\frac{\\hbar}{2}": "ℏ/2",
    r"\\psi": "ψ",
    r"\\nabla": "∇",
    r"\\ge": "≥"
}

for latex, uni in latex_to_unicode.items():
    content = re.sub(latex, uni, content)

content = re.sub(r"analysis.*?assistantfinal", "", content, flags=re.S|re.I).strip()

# print("thinking content:", thinking_content)
print("content:", content)

with open("ref_log.json", "w", encoding="utf-8") as f:
    json.dump({"role": "assistant", "content": content}, f, ensure_ascii=False, indent=2)
