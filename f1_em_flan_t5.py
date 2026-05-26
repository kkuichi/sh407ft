import re
from collections import Counter


# Kontextové dáta použité pri evaluácii modelu
contexts = [
    {
        "team": "Newcastle",
        "score": "4-2",
        "goals": "4",
        "conceded": "2",
        "possession": "59.0%",
        "shots": "10.0",
        "opponent": "Crystal Palace"
    },
    {
        "team": "Chelsea",
        "score": "3-2",
        "goals": "3",
        "conceded": "2",
        "possession": "67.0%",
        "shots": "12.0",
        "opponent": "Nott'ham Forest"
    },
    {
        "team": "Aston Villa",
        "score": "4-0",
        "goals": "4",
        "conceded": "0",
        "possession": "60.0%",
        "shots": "12.0",
        "opponent": "Everton"
    },
    {
        "team": "Arsenal",
        "score": "-2--2.5",
        "goals": "-2",
        "conceded": "-2.5",
        "possession": "120.0%",
        "shots": "13.4",
        "opponent": "Luton Town"
    }
]


# Počet otázok prislúchajúcich jednému kontextu
QUESTIONS_PER_CONTEXT = 31


# Normalizácia textu pre evaluáciu odpovedí
def normalize(text):

    text = text.lower()

    text = text.replace("–", "-")

    text = re.sub(r"[^a-z0-9.\-% ]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# Výpočet F1 Score medzi predikciou a správnou odpoveďou
def compute_f1(prediction, truth):

    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)

    return f1


# Výpočet Exact Match metriky
def compute_em(prediction, truth):

    return int(normalize(prediction) == normalize(truth))


# Získanie správnej odpovede na základe otázky a kontextu
def get_ground_truth(question, c):

    q = question.lower()

    if "final score and how many times" in q:
        return f"{c['score']} {c['goals']}"

    if "goal count" in q and "possession" in q:
        return f"{c['goals']} {c['possession']}"

    if "possession was recorded and who was the opponent" in q:
        return f"{c['possession']} {c['opponent']}"

    if "opponent of" in q and "shots" in q:
        return f"{c['opponent']} {c['shots']}"

    if "shots were taken" in q and "final score" in q:
        return f"{c['shots']} {c['score']}"

    if any(x in q for x in [
        "outcome of the match",
        "match end",
        "final result",
        "match finish",
        "match result",
        "final score"
    ]):
        return c["score"]

    if "conceded" in q:
        return c["conceded"]

    if any(x in q for x in [
        "how many times",
        "goal count",
        "number of goals",
        "manage to score",
        "goals were scored",
        "goals did"
    ]):
        return c["goals"]

    if any(x in q for x in [
        "possession",
        "ball possession",
        "share of possession",
        "controlled in terms of possession",
        "percentage of time was the ball in possession",
        "time was the ball in possession"
    ]):
        return c["possession"]

    if any(x in q for x in [
        "shots",
        "shot count",
        "shooting attempts",
        "number of shots",
        "total number of shots",
        "shots were taken",
        "shots did",
        "shots were made",
        "shot"
    ]):
        return c["shots"]

    if any(x in q for x in [
        "opponent",
        "face",
        "faced",
        "against which team",
        "played against",
        "what team played against"
    ]):
        return c["opponent"]

    return ""


# Cesta k lokálne uloženým odpovediam v .txt súbore
file_path = r"tu vlož cestu k odpovediam"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

blocks = content.split("--------------------------------------------------")


# Inicializácia premenných pre evaluáciu
context_idx = 0
question_idx = 0

f1_scores = []
em_scores = []


# Vyhodnotenie odpovedí modelu
for block in blocks:

    lines = block.strip().split("\n")

    if len(lines) < 2:
        continue

    if question_idx >= QUESTIONS_PER_CONTEXT:
        context_idx += 1
        question_idx = 0

    if context_idx >= len(contexts):
        break

    context = contexts[context_idx]

    question = lines[0].replace("Question:", "").strip()
    prediction = lines[1].replace("Answer:", "").strip()

    truth = get_ground_truth(question, context)

    # Výpočet metrík
    f1 = compute_f1(prediction, truth)
    em = compute_em(prediction, truth)

    f1_scores.append(f1)
    em_scores.append(em)

    # Výpis výsledkov evaluácie
    print("=" * 60)
    print(f"TEAM: {context['team']}")
    print(f"QUESTION: {question}")
    print(f"PREDICTION: {prediction}")
    print(f"GROUND TRUTH: {truth}")
    print(f"F1: {f1:.2f}")
    print(f"EM: {em}")

    question_idx += 1


# Výpočet priemerných metrík
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_em = sum(em_scores) / len(em_scores)


# Výpis priemerných výsledkov
print("\n" + "=" * 60)
print(f"AVERAGE F1: {avg_f1:.2f}")
print(f"AVERAGE EM: {avg_em:.2f}")
print("=" * 60)