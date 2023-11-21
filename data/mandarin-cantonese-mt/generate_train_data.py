import json


raw_data = []

with open("./cleaned_parallel_sentences.txt", "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        c, m = line.strip().split("|||")[1:]
        raw_data.append((m, c))

# prompts = ["请把下面的话翻译成粤语：", "请用粤语转述下面的话："]
prompts = [""]
chat_data = []
for d in raw_data:
    c, m = d
    for p in prompts:
        user = f"{p}{c}。"
        response = m
        chat_data.append({"prompt": user, "response": response, "history": []})

with open("chat_train_data.json", "w", encoding="utf-8") as fw:
    for data in chat_data:
        fw.write(json.dumps(data, ensure_ascii=False))
        fw.write("\n")
