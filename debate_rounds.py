"""
python debate_rounds.py \
  --prompt "Fauvism, Miyazaki Hayao" \
  --rounds 3 \
  --outdir runs/0907
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from setting import (
    MODEL_NAME,
    SYS_MSG_STYLE, SYS_MSG_OBJECT, SYS_MSG_STY_ASK, SYS_MSG_OBJ_ASK,
    USER_MSG_STY_ROUND, USER_MSG_OBJ_ROUND, USER_MSG_STY_ASK_ROUND, USER_MSG_OBJ_ASK_ROUND,
    SYS_MSG_FINAL_STYLE, SYS_MSG_FINAL_OBJECT
)

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
    # 依序展開：每輪 STYLE / OBJECT / RESPONSE
    lines = []
    for h in history:
        if(h.get('STYLE_RESPONSE')):
            lines.append(f"[Round {h['ROUND']}][STYLE AGENT]\n{h['STYLE_RESPONSE']}")
        if(h.get('OBJECT_RESPONSE')):
            lines.append(f"[Round {h['ROUND']}][OBJECT AGENT]\n{h['OBJECT_RESPONSE']}")
        lines.append(f"[Round {h['ROUND']}][ASK AGENT]\n{h['ASK_RESPONSE']}")
    return "\n\n".join(lines)

# def run_pair(tok, model, *,
#              name: str,                  # "STYLE" 或 "OBJECT"
#              sys_msg_main: str,              # 主 agent 的 system
#              sys_msg_ask: str,               # 對應 ask agent 的 system
#              init_prompt: str,           # 第 1 輪專用使用者初始題
#              response_main: str,         # 第 2 輪起，每輪提供主 agent 的使用者指令
#              response_ask: str,          # 每輪提供 ask agent 的使用者指令
#              rounds: int,
#              history: list,              # 共享 history（就地累積）
#              outdir: Path):
#     """
#     會把每輪兩個輸出都 append 進 history：
#       {"ROUND": r, f"{name}_RESPONSE": 主回覆, "ASK_RESPONSE": 詢問回覆}
#     也會寫入 round_{r}/ 檔案。
#     """
#     name = name.isupper() # name should be upper type
#     hist_path_all = outdir / "history.json"

#     # first round
#     print("-----Round 1 started.-----")
#     print(f"-----{name} agent start.-----")
#     # user prompt into style agent
#     prompt_main = f"【USER PROMPT ({name})】\n{init_prompt}"
#     response = run_agent(tok, model, sys_msg_main, prompt_main)
#     hist_txt1 = fmt_hist([{"ROUND": 1, f"{name}_RESPONSE": response, "ASK_RESPONSE": ""}])

#     prompt_ask = f"【HISTORY】\n{hist_txt1}\n\n【USER PROMPT (ASK)】\n{response_ask}"
#     response_ask_this_round = run_agent(tok, model, sys_msg_ask, prompt_ask_sty1) 

def run_rounds(tok, model,
               sys_sty: str, sys_ask_sty: str, sys_ask_obj: str,
               response_sty: str, response_ask_sty: str,
               response_obj: str, response_ask_obj: str,
               init_prompt:str, 
               rounds: int, outdir: Path):
    history = []
    history_style = []
    history_object = []
    hist_path_all = outdir / "history.json"
    hist_path_style = outdir / "history_style.json"
    hist_path_object = outdir / "history_object.json"

    # first round
    print("-----Round 1 started.-----")

    # style agent and asking agent
    print("-----Style agent start.-----")
    
    prompt_sty1 = f"【USER PROMPT (STYLE)】\n{init_prompt}"
    response_sty_this_round = run_agent(tok, model, SYS_MSG_STYLE, prompt_sty1)
    hist_txt1 = fmt_hist([{"ROUND": 1, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": ""}])
    
    prompt_ask_sty1 = f"【HISTORY】\n{hist_txt1}\n\n【USER PROMPT (ASK)】\n{response_ask_sty}"
    response_ask_sty_this_round = run_agent(tok, model, SYS_MSG_STY_ASK, prompt_ask_sty1)  

    history.append({"ROUND": 1, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": response_ask_sty_this_round})
    history_style.append({"ROUND": 1, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": response_ask_sty_this_round})
    dump_json(history, hist_path_all)
    dump_json(history_style, hist_path_style)
    
    print("-----Style agent finished.-----")

    # object agent and asking agent
    print("-----Object agent start.-----")
    
    prompt_obj1 = f"【USER PROMPT (OBJECT)】\n{init_prompt}"
    response_obj_this_round = run_agent(tok, model, SYS_MSG_OBJECT, prompt_obj1)
    hist_txt1 = fmt_hist([{"ROUND": 1, "OBJECT_RESPONSE": response_obj_this_round, "ASK_RESPONSE": ""}])
    
    prompt_ask_obj1 = f"【HISTORY】\n{hist_txt1}\n\n【USER PROMPT (ASK)】\n{response_ask_obj}"
    response_ask_obj_this_round = run_agent(tok, model, SYS_MSG_OBJ_ASK, prompt_ask_obj1)  

    history.append({"ROUND": 1, "OBJECT_RESPONSE": response_obj_this_round, "ASK_RESPONSE": response_ask_obj_this_round})
    history_object.append({"ROUND": 1, "OBJECT_RESPONSE": response_obj_this_round, "ASK_RESPONSE": response_ask_obj_this_round})
    dump_json(history, hist_path_all)
    dump_json(history_object, hist_path_object)
    
    print("-----Object agent finished.-----")

    print("-----Round 1 finished.-----")

    for r in range(2, rounds + 1):
        print(f"-----Round {r} started.-----")

        # --- Style：看前面所有輪（若無則只看使用者 style prompt）---
        print("-----Style agent start.-----")

        hist_txt = fmt_hist(history)
        prompt_sty = (
            (f"【HISTORY】\n{hist_txt}\n\n" if hist_txt else "") +
            f"【USER PROMPT (style)】\n{response_sty}"
        )
        response_sty_this_round = run_agent(tok, model, sys_sty, prompt_sty)

        # --- Ask：看「前面所有輪 + 本輪新 X」---
        tmp_hist = history + [{"ROUND": r, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": ""}]
        hist_txt2 = fmt_hist(tmp_hist)
        prompt_sum = (
            f"【HISTORY】\n{hist_txt2}\n\n"
            f"【USER PROMPT】\n{response_ask_sty}"
        )
        response_ask_sty_this_round = run_agent(tok, model, sys_ask_sty, prompt_sum)

        # 累積並即時寫檔
        history.append({"ROUND": r, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": response_ask_sty_this_round})
        history_style.append({"ROUND": r, "STYLE_RESPONSE": response_sty_this_round, "ASK_RESPONSE": response_ask_sty_this_round})
        dump_json(history, hist_path_all)
        dump_json(history_style, hist_path_style)

        print("-----Style agent finished.-----")

        # --- Object：看前面所有輪（若無則只看使用者 style prompt）---
        print("-----Object agent start.-----")

        hist_txt = fmt_hist(history)
        prompt_obj = (
            (f"【HISTORY】\n{hist_txt}\n\n" if hist_txt else "") +
            f"【USER PROMPT (object)】\n{response_obj}"
        )
        response_obj_this_round = run_agent(tok, model, sys_ask_obj, prompt_obj)

        # --- Ask：看「前面所有輪 + 本輪新 X」---
        tmp_hist = history + [{"ROUND": r, "OBJECT_RESPONSE": response_obj_this_round, "ASK_RESPONSE": ""}]
        hist_txt2 = fmt_hist(tmp_hist)
        prompt_sum = (
            f"【HISTORY】\n{hist_txt2}\n\n"
            f"【USER PROMPT】\n{response_ask_obj}"
        )
        response_ask_obj_this_round = run_agent(tok, model, sys_ask_obj, prompt_sum)

        # 累積並即時寫檔
        history.append({"ROUND": r, "OBJECT_RESPONSE": response_obj_this_round, "ASK_RESPONSE": response_ask_obj_this_round})
        history_object.append({"ROUND": r, "OBJECT_RESPONSE": response_obj_this_round, "ASK_RESPONSE": response_ask_obj_this_round})
        dump_json(history, hist_path_all)
        dump_json(history_object, hist_path_object)

        print("-----Object agent finished.-----")

        print(f"-----Round {r} finished.-----")

    hist_txt_style = fmt_hist(history_style)  
    user_prompt_style = (
        f"USER INITIAL PROMPT: {init_prompt}\n"
        f"HISTORY:\n{hist_txt_style}\n\n"
        f"Study HISTORY and produce one precise English prompt that clearly and specifically describes the painting style implied by the USER PROMPT."
    )
    final_prompt_style = run_agent(tok, model, SYS_MSG_FINAL_STYLE, user_prompt_style)
    (outdir / f"final_style_prompt.json").write_text(final_prompt_style, encoding="utf-8")

    hist_txt_object = fmt_hist(history_object)  
    user_prompt_object = (
        f"USER INITIAL PROMPT: {init_prompt}\n"
        f"HISTORY:\n{hist_txt_object}\n\n"
        f"Study HISTORY and produce one precise English prompt line that clearly specifies the key objects/motifs characteristic of the style(s), adding only essential cues (form, color palette, lighting, composition role) when critical."
    )
    final_prompt_object = run_agent(tok, model, SYS_MSG_FINAL_OBJECT, user_prompt_object)
    (outdir / f"final_object_prompt.json").write_text(final_prompt_object, encoding="utf-8")


    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="user prompt K 給 agent A")
    parser.add_argument("--outdir", type=str, default="logs_run", help="輸出資料夾")
    parser.add_argument("--rounds", type=int, default="3", help="debating rounds")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    date_str = time.strftime("%Y%m%d")
    time_str = time.strftime("%H%M%S")
    outdir = Path(args.outdir) / date_str / time_str

    tok, model = build_model_and_tokenizer(MODEL_NAME)

    history = run_rounds(
        tok, model,
        sys_sty=SYS_MSG_STYLE,
        sys_ask_sty=SYS_MSG_STY_ASK,
        sys_ask_obj=SYS_MSG_OBJ_ASK,
        init_prompt=args.prompt,
        response_sty=USER_MSG_STY_ROUND,
        response_ask_sty=USER_MSG_STY_ASK_ROUND,
        response_obj=USER_MSG_OBJ_ROUND,
        response_ask_obj=USER_MSG_OBJ_ASK_ROUND,
        rounds=args.rounds,
        outdir=outdir
    )

    print("Done.")

if __name__ == "__main__":
    main()
