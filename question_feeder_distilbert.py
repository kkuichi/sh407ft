from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import re


# Cesta pre lokalne uložený model FLAN-T5
model_path = r"tu vlož cestu k modelu"


# Načítanie tokenizera a natrénovaného modelu
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)


# Nastavenie zariadenia
device = 0 if torch.cuda.is_available() else -1


# Vytvorenie Question Answering pipeline
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=device
)

input_file = "questions_distilbert.txt"

questions = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()

    if not line:
        continue

    # Preskočenie očíslovaných riadkov
    if re.match(r"^\d+\.", line):
        continue

    # Odstránenie úvodzoviek
    line = line.strip('"')

    questions.append(line)


# Vhodný kontext
context = "Newcastle United played against Crystal Palace. They scored 4 goals and conceded 2. Possession was 59.0% and they had 10.0 shots."


# Generovanie odpovedí pomocou modelu
output_file = "answers_s_distilbert.txt"

with open(output_file, "w", encoding="utf-8") as f:

    for q in questions:

        q_clean = q.replace('"team name"', "Newcastle").replace("'team name'", "Newcastle")

        # Generovanie odpovede modelom
        result = qa_pipeline(
            question=q_clean,
            context=context
        )
        answer = result["answer"]
        score = result["score"]

        f.write(f"Question: {q_clean}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Score: {score:.4f}\n")
        f.write("-" * 50 + "\n")


print("Answers are in file: " + output_file)