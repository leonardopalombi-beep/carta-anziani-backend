"""Backend FastAPI per l'app Carta dei diritti degli anziani.

Endpoint:
- POST /api/chat  → { question: str, lang: 'it'|'en', history: [{role,content}] }
                  → { answer: str, citations: [{id, title, source}] }

Il retrieval usa BM25 su titolo+testo dei chunks. Passa top-K al LLM come contesto grounded.
Il modello risponde SOLO sulla base del contesto fornito.
"""
import json
import re
import pathlib
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import httpx
from anthropic import Anthropic

# --- Setup ---
HERE = pathlib.Path(__file__).parent
CORPUS = json.loads((HERE / 'corpus.json').read_text())

# Prepara BM25 tokenizzato per italiano (e inglese quando disponibile)
def tokenize(text: str) -> list:
    text = text.lower()
    # Rimuovi punteggiatura ma tieni numeri
    tokens = re.findall(r'[a-zàèéìòùüöä0-9]+', text)
    # Filtra stopwords comuni
    stopwords = {
        'il','la','le','lo','gli','i','e','o','ma','se','di','a','da','in','su','per','con',
        'del','della','delle','dei','degli','al','alla','alle','ai','agli','nel','nella','nelle','sui',
        'un','una','uno','che','chi','cui','non','anche','come','sono','sia','è','ed','the','of','and','a','an','to','in','for','on','with','as','is','are','be','it','this','that','which','from'
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]

# Documenti tokenizzati: uniamo title, title_en, text, text_en per matching multilingue
tokenized_corpus = []
for c in CORPUS:
    combined = ' '.join([c['title'], c.get('title_en',''), c['text'], c.get('text_en','')])
    tokenized_corpus.append(tokenize(combined))

BM25 = BM25Okapi(tokenized_corpus)
print(f"Corpus caricato: {len(CORPUS)} chunks")

# LLM client: due path supportati
# 1. DEPLOY AUTONOMO (Fly.io / Render / Railway): usa ANTHROPIC_API_KEY diretta ad api.anthropic.com
# 2. PREVIEW PERPLEXITY (sandbox web): usa SDK Anthropic con proxy Perplexity (llm-api:website)
import os

# Model id: Perplexity usa nomi diversi; Anthropic usa 'claude-sonnet-4-5-20250929'
DIRECT_ANTHROPIC_MODEL = 'claude-sonnet-4-5-20250929'
PPLX_PROXY_MODEL = 'claude_sonnet_4_5'
ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'

# Determina la modalità dall'ambiente
# - Sandbox Perplexity (preview /computer/a): ANTHROPIC_BASE_URL punta al proxy interno
# - Deploy autonomo (Fly.io/Render/Railway): solo ANTHROPIC_API_KEY reale sk-ant-...
# - Sandbox pubblicato pplx.app (senza deploy autonomo): nessuna delle due, disabilita chatbot
USE_PPLX_PROXY = bool(os.environ.get('ANTHROPIC_BASE_URL'))
_raw_key = os.environ.get('ANTHROPIC_API_KEY', '')
HAS_REAL_KEY = _raw_key.startswith('sk-ant-')
CHAT_ENABLED = USE_PPLX_PROXY or HAS_REAL_KEY
_pplx_client = Anthropic() if USE_PPLX_PROXY else None

def call_claude(system: str, messages: list, max_tokens: int = 2048) -> str:
    """Chiamata a Claude via il canale appropriato."""
    if not CHAT_ENABLED:
        raise RuntimeError("Chatbot disabilitato: manca ANTHROPIC_API_KEY reale (formato sk-ant-...).")
    if USE_PPLX_PROXY:
        # Preview Perplexity: usa SDK anthropic con proxy interno
        resp = _pplx_client.messages.create(
            model=PPLX_PROXY_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text
    # Deploy autonomo: HTTP diretto ad api.anthropic.com
    headers = {
        'content-type': 'application/json',
        'anthropic-version': '2023-06-01',
        'x-api-key': os.environ['ANTHROPIC_API_KEY'],
    }
    payload = {
        'model': DIRECT_ANTHROPIC_MODEL,
        'max_tokens': max_tokens,
        'system': system,
        'messages': messages,
    }
    with httpx.Client(timeout=90.0) as client:
        resp = client.post(ANTHROPIC_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data['content'][0]['text']


# --- API ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    lang: str = 'it'
    history: list = []


class Citation(BaseModel):
    id: str
    title: str
    source: str
    num: int


class ChatResponse(BaseModel):
    answer: str
    citations: list


GENERIC_KEYWORDS = {'sintesi', 'riassunto', 'pensiero', 'filosofia', 'visione', 'novita', 'novità', 'panoramica', 'introduzione', 'overview', 'summary', 'philosophy', 'principi', 'cultura', 'spiegami', 'raccontami', 'presenta', 'presentami', 'illustra'}

def retrieve(question: str, top_k: int = 6) -> list:
    """BM25 retrieval — restituisce i top-K chunks più rilevanti.

    Per domande generali/di sintesi, ricadiamo su TUTTI i 18 articoli della Carta.
    """
    tokens = tokenize(question)
    if not tokens:
        return []

    # Domanda generica? → usa tutta la Carta come contesto
    # Prima calcolo sempre BM25
    scores = BM25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_chunks = [CORPUS[i] for i in top_idx if scores[i] > 0]
    max_score = max(scores) if len(scores) > 0 else 0

    # Query generica: parole tipo 'sintesi', 'riassunto', 'panoramica', o query molto corta senza match forti
    has_generic_kw = any(t in GENERIC_KEYWORDS for t in tokens)
    is_short_weak = len(tokens) < 4 and max_score < 3.0
    if has_generic_kw or is_short_weak:
        # Panoramica generale: articoli Carta + testi introduttivi principali
        base = [c for c in CORPUS if c.get('source') == 'carta']
        front = [c for c in CORPUS if c.get('source') == 'front' and c.get('kind') in ('introduction', 'preface')]
        if base:
            return front + base

    return top_chunks


def build_prompt(question: str, chunks: list, lang: str, history: list) -> tuple:
    """Costruisce system prompt + messaggi per il LLM."""
    is_it = lang == 'it'

    if is_it:
        system = """Sei un assistente informativo della Fondazione Età Grande. Rispondi in italiano, in modo chiaro e conciso, basandoti ESCLUSIVAMENTE sui documenti forniti nel contesto:
1) la Prefazione di Mons. Vincenzo Paglia, la Premessa degli autori e l'Introduzione al sito
2) la Carta dei diritti degli anziani e dei doveri della società (di Mons. Vincenzo Paglia)
3) la Legge 23 marzo 2023, n. 33
4) il Decreto legislativo 15 marzo 2024, n. 29

REGOLE:
- Rispondi SOLO su temi legati alla Carta, ai suoi testi introduttivi o alla normativa italiana sull'assistenza agli anziani.
- Per la Carta e la normativa cita sempre articolo e comma (es. "Carta, art. 5, comma 2" o "DLgs 29/2024, art. 27").
- Per Prefazione, Premessa e Introduzione cita così: "Prefazione", "Premessa", "Introduzione al sito".
- Per domande su persone menzionate nella Prefazione o nella Premessa (es. autori, curatori, membri della Commissione), riporta fedelmente quanto scritto in quei testi.
- Se la risposta non è nei documenti forniti, dillo apertamente: "Su questo la Carta, i testi introduttivi e la normativa non offrono elementi diretti."
- Non aggiungere opinioni personali né interpretazioni giuridiche vincolanti.
- Usa un registro pacato, informativo, adatto a lettori non specialisti.
- Massimo 3-4 paragrafi brevi. Se serve una lista, tienila essenziale."""
    else:
        system = """You are an informational assistant of Fondazione Età Grande and AEGIS Foundation. Answer in English, clearly and concisely, based EXCLUSIVELY on the documents provided in context:
1) the Foreword by Bishop Vincenzo Paglia, the Introduction by the authors, and the About page of this website
2) the Charter of the Rights of Older Persons and Duties of Society (by Msgr. Vincenzo Paglia)
3) Italian Law 23 March 2023, no. 33
4) Italian Legislative Decree 15 March 2024, no. 29

RULES:
- Answer ONLY on topics related to the Charter, its introductory texts, or Italian legislation on care for older persons.
- For the Charter and legislation always cite article and paragraph (e.g. "Charter, art. 5, para. 2" or "Legislative Decree 29/2024, art. 27").
- For Foreword, Introduction and About page, cite as: "Foreword", "Introduction (Authors)", "About this website".
- For questions about people mentioned in the Foreword or Introduction (e.g. authors, curators, Commission members), report faithfully what those texts state.
- If the answer is not in the provided documents, say so openly: "The Charter, the introductory texts and the legislation do not directly address this."
- Do not add personal opinions or binding legal interpretations.
- Use a calm, informative tone suitable for non-specialist readers.
- Maximum 3-4 short paragraphs. If a list is needed, keep it essential."""

    # Contesto dai chunks
    context_parts = []
    for c in chunks:
        label = c['source_label_it'] if is_it else (c['source_label_en'] or c['source_label_it'])
        text = c['text'] if (is_it or not c.get('text_en')) else c['text_en']
        title = c['title'] if is_it else (c.get('title_en') or c['title'])
        # I chunk di 'front' (Prefazione/Premessa/Introduzione) non hanno numero articolo
        if c.get('source') == 'front':
            context_parts.append(f"--- {label}: {title} ---\n{text}")
        else:
            context_parts.append(f"--- {label}, Art. {c['num']} — {title} ---\n{text}")
    context = '\n\n'.join(context_parts)

    if is_it:
        user_msg = f"""CONTESTO (documenti pertinenti selezionati):

{context}

---

Domanda dell'utente: {question}"""
    else:
        user_msg = f"""CONTEXT (relevant documents selected):

{context}

---

User question: {question}"""

    messages = [{"role": "user", "content": user_msg}]
    return system, messages


@app.post('/api/chat', response_model=ChatResponse)
async def chat(req: ChatRequest):
    chunks = retrieve(req.question, top_k=6)
    if not chunks:
        # Nessun match: risposta polite
        msg = ("Sulla base della Carta e della normativa collegata (L. 33/2023, DLgs 29/2024) non trovo elementi diretti per rispondere a questa domanda. Prova a riformularla con termini pi\u00f9 specifici, ad esempio 'diritto alle cure palliative' o 'prestazione universale'."
               if req.lang == 'it' else
               "I did not find direct elements in the Charter or the related legislation (Law 33/2023, Legislative Decree 29/2024) to answer this question. Please try rephrasing it with more specific terms, e.g. 'right to palliative care' or 'universal benefit'.")
        return ChatResponse(answer=msg, citations=[])

    system, messages = build_prompt(req.question, chunks, req.lang, req.history)

    if not CHAT_ENABLED:
        msg = ("Il chatbot è configurato solo nell'anteprima interna. "
               "Per la pubblicazione al pubblico serve completare il deploy autonomo su Fly.io/Render. "
               "Vedi DEPLOY.md nel pacchetto scaricabile."
               if req.lang == 'it' else
               "Chatbot is available only in the internal preview. "
               "For public deployment, complete the autonomous deploy on Fly.io/Render — see DEPLOY.md.")
        return ChatResponse(answer=msg, citations=[])
    try:
        answer = call_claude(system, messages, max_tokens=2048)
    except httpx.HTTPStatusError as e:
        print(f"[chat] Anthropic HTTP error {e.response.status_code}: {e.response.text[:300]}")
        err_msg = ("Servizio temporaneamente non disponibile. Riprova tra qualche istante."
                   if req.lang == 'it' else
                   "Service temporarily unavailable. Please try again in a moment.")
        return ChatResponse(answer=err_msg, citations=[])
    except Exception as e:
        import traceback
        print(f"[chat] Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
        err_msg = ("Servizio temporaneamente non disponibile. Riprova tra qualche istante."
                   if req.lang == 'it' else
                   "Service temporarily unavailable. Please try again in a moment.")
        return ChatResponse(answer=err_msg, citations=[])

    citations = [
        Citation(
            id=c['id'],
            title=c['title'] if req.lang == 'it' else (c.get('title_en') or c['title']),
            source=c['source_label_it'] if req.lang == 'it' else (c['source_label_en'] or c['source_label_it']),
            num=c['num'],
        )
        for c in chunks[:4]  # Mostra max 4 fonti
    ]

    return ChatResponse(answer=answer, citations=citations)


@app.get('/api/health')
async def health():
    return {"status": "ok", "chunks": len(CORPUS)}
