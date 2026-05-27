import pandas as pd
from src.ai_tech.methods import regex_gazetteer, llm_openai
from src.ai_tech.evaluate import evaluate, summary_row
import json

def print_summary(name, gold, preds):
    m = evaluate(gold, preds)
    s = summary_row(name, m)
    print(json.dumps(s, indent=2))

def test_ensemble():
    try:
        df = pd.read_csv("data/distilled/goldset_ai_tecnologie.csv")
    except FileNotFoundError:
        print("Goldset non trovato")
        return
        
    texts = df["DESCRIZIONE_PROGETTO"].tolist()
    
    # 1. Regex
    regex_preds = regex_gazetteer.extract_batch(texts)
    
    # 2. LLM
    print("Elaborazione LLM sul goldset...")
    try:
        llm_preds = llm_openai.extract_batch(texts, max_workers=2)
    except Exception as e:
        print(f"Errore: {e}")
        return
        
    # 3. Ensemble (Union)
    ensemble_preds = [list(set(r) | set(l)) for r, l in zip(regex_preds, llm_preds)]
    
    # Estraiamo gold
    gold = df["gold"].fillna("").apply(lambda x: x.split("|") if x else []).tolist()
    
    print_summary("Regex", gold, regex_preds)
    print_summary("LLM", gold, llm_preds)
    print_summary("Ensemble Regex U LLM", gold, ensemble_preds)

if __name__ == "__main__":
    test_ensemble()

def test_fallback():
    df = pd.read_csv("data/distilled/goldset_ai_tecnologie.csv")
    texts = df["DESCRIZIONE_PROGETTO"].tolist()
    regex_preds = regex_gazetteer.extract_batch(texts)
    llm_preds = llm_openai.extract_batch(texts, max_workers=2)
    
    # Fallback
    fallback_preds = [r if r else l for r, l in zip(regex_preds, llm_preds)]
    
    gold = df["gold"].fillna("").apply(lambda x: x.split("|") if x else []).tolist()
    print_summary("Ensemble Fallback (LLM se Regex vuota)", gold, fallback_preds)

test_fallback()
