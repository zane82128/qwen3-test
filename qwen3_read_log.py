from transformers import pipeline, AutoTokenizer
import json, re

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
pipe = pipeline("text-generation", model=MODEL_ID, torch_dtype="auto", device_map="auto")

# save the conversation log to file
def save_history():
    with open("response_log.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_messages(path, keep_last=40):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        msgs = data
    elif isinstance(data, dict) and "messages" in data:
        msgs = data["messages"]
    else:
        msgs = [data]   # 單一物件時包成陣列

    assert isinstance(msgs, list)

    # 清掉 assistant 內嵌的分析雜訊（若有）
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m["content"] = re.sub(r"analysis.*?assistantfinal", "", m["content"], flags=re.S|re.I).strip()

    # 去除 system，僅取最後 keep_last 則
    msgs = [m for m in msgs if m.get("role") != "system"][-keep_last:]
    return msgs

# 讀入既有對話作為「前情」
history = load_messages("ref_log.json", keep_last=40)

# 建立最終 messages（可保留你需要的 system 指令）
messages = [{"role": "system", "content": "請用繁體中文回答。"}]
messages.extend(history)

# 新的使用者問題
messages.append({"role": "user", "content": "請總結上述對話重點？"})

# 串接到 chat template 並生成
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = pipe(prompt, max_new_tokens=2048, temperature=0.7, top_p=0.9, return_full_text=False)




reply = out[0]["generated_text"]
reply = re.sub(r"analysis.*?assistantfinal", "", reply, flags=re.S|re.I).strip()
print(reply)

messages.append({"role": "assistant", "content": reply})
save_history()
