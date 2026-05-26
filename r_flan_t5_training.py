# Inštalácia potrebných knižníc v kompatibilných verziách
!pip install --force-reinstall transformers==4.41.2 datasets==2.19.1 accelerate==0.31.0 peft==0.11.1 


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Načítanie predtrénovaného tokenizera
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Načítanie predtrénovaného modelu
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


from datasets import load_dataset

# Načítanie reálneho datasetu vo formáte JSON
dataset = load_dataset("json", data_files="/content/r_flan_dataset_fixed.json")

# Funkcia na tokenizáciu vstupov a výstupov
def preprocess(example):

    # Tokenizácia vstupného textu
    inputs = tokenizer(
        example["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenizácia cieľového textu
    targets = tokenizer(
        example["output"],
        max_length=32,
        truncation=True,
        padding="max_length"
    )

    # Pridanie cieľových tokenov ako labels
    inputs["labels"] = targets["input_ids"]

    return inputs

# Aplikovanie preprocess funkcie na celý dataset
tokenized_dataset = dataset["train"].map(preprocess, batched=True)


from transformers import TrainingArguments

# Nastavenie parametrov tréningu
training_args = TrainingArguments(
    output_dir="./results",                # Priečinok pre výsledky
    learning_rate=5e-5,                    # Learning rate
    per_device_train_batch_size=8,         # Veľkosť batchu
    num_train_epochs=1,                    # Počet epoch
    max_steps=250,                         # Maximálny počet krokov
    logging_dir="./logs",                  # Priečinok pre logy
    logging_strategy="steps",              # Logovanie po krokoch
    logging_steps=10,                      # Logovanie každých 10 krokov
    save_strategy="epoch",                 # Uloženie modelu po epoche
    eval_strategy="no"                     # Bez evaluácie počas tréningu
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
trainer.save_model("/content/flan_t5_r_data")

# Uloženie tokenizera
tokenizer.save_pretrained("/content/flan_t5_r_data")

# Zabalenie modelu do ZIP archívu
!zip -r flan_t5_r_data.zip /content/flan_t5_r_data/


from google.colab import files

# Stiahnutie ZIP archívu do lokálneho prostredia
files.download("flan_t5_r_data.zip")