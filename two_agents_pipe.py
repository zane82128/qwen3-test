# merge_two_agents_pipeline.py
# 功能：
# 1) 讀入 user prompt K 與 L
# 2) agent A 回覆 X，存成 JSON
# 3) 將 X 與 L 一起給 agent B，得到 Y，存成 JSON
# 4) 最後把 K、L、X、Y 全部存成一個彙整 JSON
#
# 依賴：pip install transformers accelerate torch
# 模型：Qwen/Qwen3-4B-Thinking-2507（可依需求替換）

"""
python two_agents_pipe.py \
  --p "Fauvism, Miyazaki Hayao" \
  --outdir runs/0905
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"

# 訊息
style_messages ="""
You are an art style analysis assistant. 
Your task is to take a simple user prompt describing an art style (e.g., "van Gogh style") and expand it into a precise and detailed description of the style, 
if there are multiple style, you need to propose a method that can trying to merge two styles, and conclude with a final prompt. 
Always structure your analysis according to the following criteria: 

1. **Form & Composition** – Describe the typical structure, shapes, spatial arrangement, and overall organization of artworks in this style. 
2. **Color & Tonality** – Explain the common color palette, tonal range, and use of contrast or harmony. 
3. **Brushwork & Technique** – Identify characteristic brushstrokes, textures, and technical methods used in this style. 
4. **Expression & Theme** – Describe the mood, atmosphere, or emotional quality conveyed through the artworks. 
5. **Historical & Cultural Context** – Provide the cultural background, historical period, and artistic influences that define this style. 

Your output should always be clear, structured, and specific, even if the user prompt is vague. 
If the style has multiple phases (e.g., early vs. late period), highlight key distinctions. 
Do not invent unrelated attributes—base your elaboration on widely recognized features of the style. 
"""

summarize_messages = """
請用繁體中文回答。請總結上述對話重點。
"""

# generator
def build_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = "auto",
        device_map = "auto"
    )
    return tok, model

def apply_chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# make the respone content clean, delete redundant text (Ex. |assistant|, |thinking content|)
def decode_without_think(tok, output_ids):
    text = tok.decode(output_ids, skip_special_tokens=False)
    text = re.sub(r"analysis.*?assistantfinal", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    
    text = tok.decode(tok.encode(text), skip_special_tokens=True).strip()
    return text

def run_agent(tok, model, system_prompt: str, user_prompt: str,
              max_new_tokens=4096, temperature=0.7, top_p=0.9):
    messages = []
    if isinstance(system_prompt,str):
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    text = apply_chat(tok, messages)
    inputs = tok([text], return_tensors="pt").to(model.device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_p = top_p,
    )[0]

    # get new tokens
    new_ids = gen_ids[len(inputs.input_ids[0]):]
    content = decode_without_think(tok, new_ids)
    return content

def dump_json(obj, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, required=True, help="user prompt K 給 agent A")
    # parser.add_argument("--L", type=str, required=True, help="user prompt L 給 agent B（會與 X 一起提供）")
    parser.add_argument("--outdir", type=str, default="logs_run", help="輸出資料夾")
    # parser.add_argument("--spa", type=str, default="請用精簡、具體的中文回答。", help="agent A 的 system prompt")
    # parser.add_argument("--spb", type=str, default="請用條列與結論給出可執行建議。", help="agent B 的 system prompt")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    date_str = time.strftime("%Y%m%d")
    time_str = time.strftime("%H%M%S")
    outdir = Path(args.outdir) / date_str / time_str

    tok, model = build_model_and_tokenizer(MODEL_NAME)

    # 1. user prompt -> style agent
    resp_sty = run_agent(tok, model, style_messages, args.p)

    # 2. save response
    dump_json(
        {"TIME": ts, "AGENT": "style", "INPUT": args.p, "RESPONSE": resp_sty},
        outdir / "style_agent_response.json"
    )
    
    # 3. summarize agent
    prompt_sum = "請總結上述對話重點。"
    combined_prompt_sum = f"【RESPONSE FROM STYLE AGENT】\n{resp_sty}\n\n【USER PROMPT】\n{prompt_sum}"
    resp_sum = run_agent(tok, model, summarize_messages, combined_prompt_sum)

    # 4. save response
    dump_json(
        {"TIME": ts, "AGENT": "summarize", "INPUT": combined_prompt_sum, "RESPONSE": resp_sum},
        outdir / "summarize_agent_response.json"
    )

    # 5. summary
    summary = {
        "TIME": ts,
        "MODEL": MODEL_NAME,
        "STYLE AGENT": {
            "SYSTEM PROMPT": style_messages,
            "INPUT": args.p,
            "RESPONSE": resp_sty
        },
        "SUMMARIZE AGENT": {
            "SYSTEM PROMPT": summarize_messages,
            "INPUT": prompt_sum,
            "RESPONSE": resp_sum
        }
    }
    dump_json(summary, outdir / "combined_log.json")

    print("Done.")

if __name__ == "__main__":
    main()
