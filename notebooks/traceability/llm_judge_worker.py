"""Giudice LLM zero-shot locale (LM Studio) per il concetto di "tracciabilita".

Riusa i pattern gia' presenti nel repo:
  - endpoint OpenAI-compatibile LM Studio (porta 1234), come
    `app/tui_regex_mismatch_classifier.py` e `src/ai_tech/methods/llm_openai.py`;
  - structured output via `response_format` json_schema con enum (forza la sola label);
  - strip dei blocchi di reasoning `<think>...</think>` dei modelli thinking (gpt-oss);
  - `extract_label()` robusto con fallback regex;
  - cache su disco per non rieseguire le stesse descrizioni.

Espone `judge_batch(...)` sincrono (usa AsyncOpenAI internamente) pensato per un
CAMPIONE di record discordanti, non per l'intera popolazione.
"""
import os
import re
import json
import hashlib
import asyncio
import threading

from openai import AsyncOpenAI


def _run_async(coro):
    """Esegue una coroutine sia da script sia da dentro Jupyter.

    In Jupyter il kernel ha gia' un event loop attivo, quindi `asyncio.run`
    solleverebbe RuntimeError. In quel caso eseguiamo la coroutine in un thread
    separato con un loop dedicato.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # nessun loop attivo: percorso standard

    result = {}
    def _worker():
        loop = asyncio.new_event_loop()
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as e:  # pragma: no cover - diagnostica runtime
            result["error"] = e
        finally:
            loop.close()
    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]

POS = "tracciabilita"
NEG = "altro"
_ALLOWED = {POS, NEG}

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"

_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "traceability",
    "validation", ".cache_llm_judge.json",
)

_SYSTEM = (
    "Sei un annotatore esperto di bandi e progetti finanziati in italiano. "
    "Devi decidere se un progetto riguarda la TRACCIABILITA', definita come la capacita' di "
    "tracciare, identificare o rintracciare PRODOTTI, MATERIE PRIME, LOTTI o MERCI lungo la "
    "filiera o il processo produttivo, dalla provenienza/origine fino al consumatore.\n\n"
    "Classifica 'tracciabilita' se il progetto riguarda almeno uno di questi:\n"
    "- tracciabilita'/rintracciabilita' di prodotti, lotti, materie o alimenti lungo la filiera "
    "o la produzione (inclusa la tracciabilita' della filiera agroalimentare e la sicurezza "
    "alimentare basata sul tracciamento);\n"
    "- provenienza o origine certificata del prodotto, passaporto digitale di prodotto, "
    "anticontraffazione lungo la supply chain;\n"
    "- blockchain, QR code, RFID, NFC o registri distribuiti USATI per tracciare prodotti/merci "
    "o la loro provenienza lungo la catena;\n"
    "- sistemi che registrano e seguono il percorso fisico di beni/merci nella logistica ai fini "
    "di rintracciabilita'.\n\n"
    "Classifica 'altro' se il progetto riguarda:\n"
    "- digitalizzazione, gestionali/ERP generici, automazione di processi aziendali, monitoraggio "
    "di dati/KPI SENZA tracciare il percorso di un prodotto;\n"
    "- e-commerce, booking, marketing, siti web, prenotazioni;\n"
    "- tracciamento di utenti, traffico web o presenze del personale;\n"
    "- filiera corta, valorizzazione o promozione territoriale senza un sistema di tracciamento;\n"
    "- adozione di tecnologie (IoT, AI, realta' aumentata, server) senza finalita' esplicita di "
    "tracciamento del prodotto lungo la filiera;\n"
    "- controllo qualita' o certificazioni che non implicano il tracciamento del percorso del prodotto.\n\n"
    "Criterio dirimente: conta COSA viene tracciato. Se si tracciano PRODOTTI/MATERIE/LOTTI lungo la "
    "filiera -> 'tracciabilita'. Se si tracciano solo PROCESSI o DATI aziendali -> 'altro'.\n"
    "Rispondi SOLO con JSON valido {\"label\": \"tracciabilita\"} oppure {\"label\": \"altro\"}."
)

# Few-shot ancorati a casi reali osservati (inclusi errori del prompt v1), per stabilizzare il confine.
_FEWSHOT = [
    ("VALORIZZAZIONE, TRACCIABILITA' E SICUREZZA ALIMENTARE DEI PRODOTTI DEL SUINO. "
     "Sostenere la nascita della prima filiera suinicola di origine Marche, tracciando le "
     "produzioni regionali lungo la filiera.", POS),
    ("Migliorare e garantire la salubrita' dei prodotti alimentari attraverso la tracciabilita' "
     "dei prodotti, che investe tutti i componenti della filiera alimentare dalla raccolta al "
     "consumatore.", POS),
    ("G.O. TRA.IN. filiera agroalimentare: tracciabilita' e blockchain per seguire i lotti dal "
     "campo alla tavola.", POS),
    ("ACQUISTO NUOVO GESTIONALE E SISTEMA DI ELABORAZIONE ORDINI interfacciato con una "
     "piattaforma di e-commerce per la gestione degli ordini aziendali.", NEG),
    ("STRATEGIE DI DIGITAL MARKETING per la commercializzazione dei prodotti in Italia ed Europa, "
     "con analisi del mercato e della concorrenza.", NEG),
    ("SMART GLASSES per applicazioni di realta' aumentata da sperimentare nel settore della "
     "logistica; tavolo multitouch e video wall.", NEG),
]

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "traceability_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": [POS, NEG]},
            },
            "required": ["label"],
            "additionalProperties": False,
        },
    },
}


# ----------------------------- cache --------------------------------------
def _load_cache():
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# Versione del prompt: entra nella chiave di cache cosi' un cambio di prompt invalida
# automaticamente le label vecchie (evita di restituire risposte del prompt precedente).
PROMPT_VERSION = "v2"


def _key(model, text):
    return hashlib.sha1(f"{PROMPT_VERSION}||{model}||{text}".encode("utf-8")).hexdigest()


# ----------------------------- parsing ------------------------------------
def _normalize(value):
    norm = re.sub(r"[^a-z]+", "", str(value).strip().lower())
    if norm in ("tracciabilita", "tracciabilit"):
        return POS
    if norm == "altro":
        return NEG
    return None


def extract_label(content):
    """Estrae la label da una risposta LLM, gestendo reasoning e markdown."""
    if not content:
        return "error: empty_response"
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 1) JSON strutturato
    if "{" in cleaned and "}" in cleaned:
        candidate = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "label" in parsed:
                norm = _normalize(parsed["label"])
                if norm:
                    return norm
        except json.JSONDecodeError:
            pass

    # 2) label: xxx
    m = re.search(r"(?is)\blabel\b\s*[:=]\s*[\"']?([a-zA-Z_\-\s]+)[\"']?", cleaned)
    if m:
        norm = _normalize(m.group(1))
        if norm:
            return norm

    # 3) match diretto della parola
    m = re.search(r"(?i)\b(tracciabilit[a\xe0]?|altro)\b", cleaned)
    if m:
        norm = _normalize(m.group(1))
        if norm:
            return norm

    return "error: unknown_label"


# ----------------------------- inference ----------------------------------
def _build_user_prompt(titolo, descrizione):
    titolo = (titolo or "").strip()
    descrizione = (descrizione or "").strip()[:2000]
    return f"TITOLO: {titolo}\n\nDESCRIZIONE: {descrizione}"


def _build_messages(titolo, descrizione):
    msgs = [{"role": "system", "content": _SYSTEM}]
    # Few-shot: coppie testo -> label per ancorare il confine decisionale.
    for text, label in _FEWSHOT:
        msgs.append({"role": "user", "content": _build_user_prompt("", text)})
        msgs.append({"role": "assistant", "content": json.dumps({"label": label})})
    msgs.append({"role": "user", "content": _build_user_prompt(titolo, descrizione)})
    return msgs


async def _classify_one(client, model, titolo, descrizione):
    kwargs = {
        "model": model,
        "messages": _build_messages(titolo, descrizione),
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    try:
        try:
            resp = await client.chat.completions.create(**kwargs, response_format=_SCHEMA)
        except Exception:
            try:
                resp = await client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            except Exception:
                resp = await client.chat.completions.create(**kwargs)
        return extract_label(resp.choices[0].message.content or "")
    except Exception as e:  # pragma: no cover - diagnostica runtime
        return f"error: {e}"


async def _judge_async(records, model, base_url, concurrency):
    cache = _load_cache()
    client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(records)

    async def worker(i, rec):
        text = rec.get("descrizione", "")
        k = _key(model, text)
        if k in cache:
            results[i] = cache[k]
            return
        async with sem:
            label = await _classify_one(
                client, model, rec.get("titolo", ""), text
            )
        results[i] = label
        if not str(label).startswith("error"):
            cache[k] = label

    await asyncio.gather(*(worker(i, r) for i, r in enumerate(records)))
    _save_cache(cache)
    return results


def ping(base_url=DEFAULT_BASE_URL, timeout=3.0):
    """Verifica sincrona che l'endpoint LM Studio risponda. Ritorna lista modelli o None."""
    async def _run():
        client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        resp = await asyncio.wait_for(client.models.list(), timeout=timeout)
        return [getattr(m, "id", str(m)) for m in getattr(resp, "data", [])]

    try:
        return _run_async(_run())
    except Exception:
        return None


def judge_batch(records, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, concurrency=4):
    """Classifica una lista di record via LLM locale.

    records: lista di dict con chiavi 'titolo' e 'descrizione'.
    Ritorna la lista di label ('tracciabilita' / 'altro' / 'error: ...').
    Usa cache su disco: rilanciare e' idempotente sui record gia' visti.
    """
    if not records:
        return []
    return _run_async(_judge_async(records, model, base_url, concurrency))
