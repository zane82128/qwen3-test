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
  --prompt "Fauvism, Miyazaki Hayao" \
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

# system message that is used in the first round
style_sys_msg ="""
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

ask_sys_msg = """
請用繁體中文回答。請根據history中，style agent的回答提出對應的問題。
"""
# message that is used from second rounds
style_rounds_msg="""
You need to study the message history log, which includes the few rounds of your answer and the question from ask agent.
You need to take the initial prompt which is from the user into consideratrion, 
and find out is there anything about style is that you missed before, 
then answer the question from the asking agent last time.
"""
ask_rounds_msg="""
your job is to ask questions about style,
based on the message history log, you need to find out what details is missed and ask about style agent.
the details should be as many as possible.
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

def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    s = s.replace("\\n", "\n")  # 把字串中的 \n 展開為真正換行
    path.write_text(s, encoding="utf-8")


def fmt_hist(history: list) -> str:
    # 依序展開：每輪 STYLE / SUM
    lines = []
    for h in history:
        lines.append(f"[Round {h['ROUND']}][STYLE AGENT]\n{h['STYLE_RESPONSE']}")
        lines.append(f"[Round {h['ROUND']}][ASK AGENT]\n{h['ASK_RESPONSE']}")
    return "\n\n".join(lines)

def run_rounds(tok, model,
               sys_sty: str, sys_sum: str,
               response_sty: str, response_ask: str,
               init_prompt:str, 
               rounds: int, outdir: Path):
    history = []
    hist_path = outdir / "history.json"

    # first round
    prompt_sty1 = f"【USER PROMPT (STYLE)】\n{init_prompt}"
    response_sty_this_round = run_agent(tok, model, style_sys_msg, prompt_sty1)
    hist_txt1 = fmt_hist([{"ROUND": 1, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": ""}])
    
    prompt_ask1 = f"【HISTORY】\n{hist_txt1}\n\n【USER PROMPT (ASK)】\n{response_ask}"
    response_ask_this_round = run_agent(tok, model, ask_sys_msg, prompt_ask1)  

    history.append({"ROUND": 1, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": response_ask_this_round})
    dump_json(history, hist_path)
    print("-----Round 1 finished.-----")

    for r in range(2, rounds + 1):
        # --- Style：看前面所有輪（若無則只看使用者 style prompt）---
        hist_txt = fmt_hist(history)
        prompt_sty = (
            (f"【HISTORY】\n{hist_txt}\n\n" if hist_txt else "") +
            f"【USER PROMPT (style)】\n{response_sty}"
        )
        response_sty_this_round = run_agent(tok, model, sys_sty, prompt_sty)

        # --- Summarize：看「前面所有輪 + 本輪新 X」---
        tmp_hist = history + [{"ROUND": r, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": ""}]
        hist_txt2 = fmt_hist(tmp_hist)
        prompt_sum = (
            f"【HISTORY】\n{hist_txt2}\n\n"
            f"【USER PROMPT (summary)】\n{response_ask}"
        )
        response_ask_this_round = run_agent(tok, model, sys_sum, prompt_sum)

        # 累積並即時寫檔
        history.append({"ROUND": r, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": response_ask_this_round})
        dump_json(history, hist_path)

        print(f"-----Round {r} finished.-----")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="user prompt K 給 agent A")
    # parser.add_argument("--L", type=str, required=True, help="user prompt L 給 agent B（會與 X 一起提供）")
    parser.add_argument("--outdir", type=str, default="logs_run", help="輸出資料夾")
    # parser.add_argument("--spa", type=str, default="請用精簡、具體的中文回答。", help="agent A 的 system prompt")
    # parser.add_argument("--spb", type=str, default="請用條列與結論給出可執行建議。", help="agent B 的 system prompt")
    parser.add_argument("--rounds", type=str, default="3", help="debating rounds")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    date_str = time.strftime("%Y%m%d")
    time_str = time.strftime("%H%M%S")
    outdir = Path(args.outdir) / date_str / time_str

    tok, model = build_model_and_tokenizer(MODEL_NAME)

    history = run_rounds(
        tok, model,
        sys_sty=style_sys_msg,
        sys_sum=ask_sys_msg,
        init_prompt=args.prompt,
        response_sty=style_rounds_msg,
        response_ask=ask_rounds_msg,
        rounds=args.rounds,
        outdir=outdir
    )

    # dump_json({
    #     "t": f"{date_str}-{time_str}",
    #     "model": MODEL_NAME,
    #     "sys_sty": style_sys_msg,
    #     "sys_sum": ask_sys_msg,
    #     "response_sty": args.p,
    #     "response_ask": style_sys_msg,
    #     "history": history
    # }, outdir / "combined_log.json")

    # 1. user prompt -> style agent
    # resp_sty = run_agent(tok, model, style_sys_msg, args.p)

    # # 2. save response
    # dump_json(
    #     {"TIME": ts, "AGENT": "style", "INPUT": args.p, "RESPONSE": resp_sty},
    #     outdir / "style_agent_response.json"
    # )
    
    # # 3. summarize agent
    # prompt_sum = "請總結上述對話重點。"
    # combined_prompt_sum = f"【RESPONSE FROM STYLE AGENT】\n{resp_sty}\n\n【USER PROMPT】\n{prompt_sum}"
    # resp_sum = run_agent(tok, model, ask_sys_msg, combined_prompt_sum)

    # # 4. save response
    # dump_json(
    #     {"TIME": ts, "AGENT": "summarize", "INPUT": combined_prompt_sum, "RESPONSE": resp_sum},
    #     outdir / "summarize_agent_response.json"
    # )

    # 5. summary
    # summary = {
    #     "TIME": ts,
    #     "MODEL": MODEL_NAME,
    #     "STYLE AGENT": {
    #         "SYSTEM PROMPT": style_sys_msg,
    #         "INPUT": args.p,
    #         "RESPONSE": resp_sty
    #     },
    #     "SUMMARIZE AGENT": {
    #         "SYSTEM PROMPT": ask_sys_msg,
    #         "INPUT": prompt_sum,
    #         "RESPONSE": resp_sum
    #     }
    # }
    # dump_json(summary, outdir / "combined_log.json")


    print("Done.")

if __name__ == "__main__":
    main()
