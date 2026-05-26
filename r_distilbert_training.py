# Inštalácia potrebných knižníc v kompatibilných verziách
!pip install --force-reinstall transformers==4.41.2 datasets==2.19.1 accelerate==0.31.0 peft==0.11.1


from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Názov predtrénovaného DistilBERT modelu
model_name = "distilbert/distilbert-base-uncased"

# Načítanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Načítanie predtrénovaného modelu pre Question Answering
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


from datasets import load_dataset

# Načítanie datasetu vo formáte JSON
dataset = load_dataset("json", data_files="/content/r_distilbert_data_fixed.json")


# Funkcia na tokenizáciu a prípravu tréningových dát
def preprocess(example):

    # Tokenizácia otázky a kontextu
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    # Získanie mapovania tokenov na pôvodný text
    offsets = tokenized["offset_mapping"]

    # Začiatočná pozícia správnej odpovede v texte
    answer_start = example["answers"]["answer_start"][0]

    # Text správnej odpovede
    answer_text = example["answers"]["text"][0]

    # Koncová pozícia odpovede
    answer_end = answer_start + len(answer_text)

    # Predvolené hodnoty pozícií odpovede
    start_positions = 0
    end_positions = 0

    # Vyhľadanie tokenov zodpovedajúcich začiatku a koncu odpovede
    for i, (start, end) in enumerate(offsets):

        if start <= answer_start < end:
            start_positions = i

        if start < answer_end <= end:
            end_positions = i

    # Pridanie pozície začiatku odpovede
    tokenized["start_positions"] = start_positions

    # Pridanie pozície konca odpovede
    tokenized["end_positions"] = end_positions

    # Odstránenie offset mappingu, ktorý už nie je potrebný
    tokenized.pop("offset_mapping")

    return tokenized


# Aplikovanie preprocess funkcie na celý dataset
tokenized_dataset = dataset["train"].map(preprocess)


from transformers import TrainingArguments

# Nastavenie parametrov tréningu
training_args = TrainingArguments(
    output_dir="./results",               # Priečinok pre výsledky
    learning_rate=3e-5,                   # Learning rate
    per_device_train_batch_size=8,        # Veľkosť batchu
    num_train_epochs=1,                   # Počet epoch
    max_steps=70,                         # Maximálny počet krokov
    logging_steps=10,                     # Logovanie každých 10 krokov
    save_strategy="epoch",                # Uloženie modelu po epoche
    report_to="none"                      # Vypnutie reportovania do externých služieb
)


from transformers import Trainer

# Vytvorenie Trainer objektu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Spustenie tréningu modelu
trainer.train()


# Uloženie natrénovaného modelu
trainer.save_model("/content/distilbert_r_data")

# Uloženie tokenizera
tokenizer.save_pretrained("/content/distilbert_r_data")


# Zabalenie modelu do ZIP archívu
!zip -r distilbert_r_data.zip /content/distilbert_r_data/


from google.colab import files

# Stiahnutie ZIP archívu do lokálneho prostredia
files.download("distilbert_r_data.zip")