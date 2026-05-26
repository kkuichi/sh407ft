# Systémová príručka
## Použité súčasti

Zoznam použitých súčastí spolu s ich verziami:

- Python 3.12.13
- Google Colab
- Transformers verzia 4.41.2
- Datasets verzia 2.19.1
- Accelerate verzia 0.31.0
- PEFT verzia 0.11.1

Pre inštaláciu potrebných knižníc je možné použiť: `pip install --force-reinstall transformers==4.41.2 datasets==2.19.1 accelerate==0.31.0 peft==0.11.1`

## Popis skriptov
1. Skript `r_flan_t5_training.py` slúži na fine-tuning modelu FLAN-T5 Base pre úlohu Question Answering pomocou reálneho datasetu vo formáte JSON.
2. Skript `s_flan_t5_training.py` slúži na fine-tuning modelu FLAN-T5 Base pre úlohu Question Answering pomocou syntetického datasetu vo formáte JSON.
3. Skript `r_distilbert_training.py` slúži na fine-tuning modelu DistilBERT pre úlohu Question Answering pomocou reálneho datasetu vo formáte JSON.
4. Skript `s_distilbert_training.py` slúži na fine-tuning modelu DistilBERT pre úlohu Question Answering pomocou syntetického datasetu vo formáte JSON.
5. Skript `question_feeder_flan_t5.py` slúži na načítanie otázok zo súboru, generovanie odpovedí pomocou natrénovaného modelu FLAN-T5 Base a uloženie výsledkov do textového súboru.
6. Skript `question_feeder_distilbert.py` slúži na načítanie otázok zo súboru, generovanie odpovedí pomocou natrénovaného modelu DistilBERT a uloženie výsledkov do textového súboru.
7. Skript `f1_em_flan_t5.py` slúži na vyhodnotenie odpovedí generovaných modelom FLAN-T5 Base pomocou metrík F1 Score a Exact Match.
8. Skript `f1_em_distilbert.py` slúži na vyhodnotenie odpovedí generovaných modelom DistilBERT pomocou metrík F1 Score, Exact Match a Confidence Score.


## Zoznam použitých modelov

- FLAN-T5 Base — google/flan-t5-base
- DistilBERT Base Uncased — distilbert/distilbert-base-uncased

## Dôležité poznámky
- Dataset pre model `FLAN-T5 Base` musí byť nahratý do prostredia Google Colab vo formáte JSON a obsahovať položky `input` a `output`.
- Dataset pre model `DistilBERT Base Uncased` musí byť nahratý do prostredia Google Colab vo formáte JSON a obsahovať položky `question`, `context` a `answers`.
- Skripty `r_flan_t5_training.py`, `s_flan_t5_training.py`, `r_distilbert_training.py` a `s_distilbert_training.py` boli vytvorené pre prostredie Google Colab.
- Pri práci s modelmi `FLAN-T5 Base` a `DistilBERT Base Uncased` sa odporúča využívať GPU akceleráciu, pretože tréning na CPU môže byť výrazne pomalší.
- Veľkosť batchu, počet epoch a ďalšie tréningové parametre je možné meniť v objekte `TrainingArguments`.
- Otázky musia byť uložené v textovom súbore `.txt`, pričom každá otázka sa nachádza na samostatnom riadku.
- Výsledky generovania odpovedí sa ukladajú do výstupného textového súboru.
- Skripty `question_feeder_flan_t5.py` a `question_feeder_distilbert.py` boli vytvorené pre lokálne Python prostredie.