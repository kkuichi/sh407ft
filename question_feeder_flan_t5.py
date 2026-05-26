from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# Cesta pre lokalne uložený model FLAN-T5
model_path = r"tu vlož cestu k modelu"


# Načítanie tokenizera a natrénovaného modelu
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# Nastavenie zariadenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_file = "questions_flan_t5.txt"

questions = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()

    if not line:
        continue

    if re.match(r"^\d+\.", line):
        continue

    line = line.strip('"')

    questions.append(line)


# vhodný kontext
context = "Newcastle United played against Crystal Palace. They scored 4 goals and conceded 2. Possession was 59.0% and they had 10.0 shots."


# Generovanie odpovedí pomocou modelu
output_file = "answers_r_flan_t5.txt"

with open(output_file, "w", encoding="utf-8") as f:

    for q in questions:

        q_clean = q.replace('"team name"', "Newcastle").replace("'team name'", "Newcastle")

        # Vytvorenie vstupu pre model
        input_text = f"question: {q_clean} context: {context}"

        # Tokenizácia vstupu
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        ).to(device)

        # Generovanie odpovede modelom
        outputs = model.generate(
            **inputs,
            max_length=32,
            num_beams=4,
            early_stopping=True
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        f.write(f"Question: {q_clean}\n")
        f.write(f"Answer: {answer}\n")
        f.write("-" * 50 + "\n")

print("Answers are in file: " + output_file)