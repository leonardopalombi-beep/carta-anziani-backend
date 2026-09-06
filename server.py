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
    articolo_id: Optional[str] = None
    area: Optional[str] = None
    ricerca_id: Optional[str] = None
    capitolo_id: Optional[str] = None
    excerpt: Optional[str] = None  # estratto ~300 char per export


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
        system = """Sei un assistente informativo della Fondazione Età Grande. Rispondi in italiano, in modo chiaro e conciso, basandoti ESCLUSIVAMENTE sui documenti forniti nel contesto. Il corpus include:
1) la Prefazione di Mons. Vincenzo Paglia, la Premessa degli autori e l'Introduzione al sito
2) la Carta dei diritti degli anziani e dei doveri della società (di Mons. Vincenzo Paglia)
3) la Legge 23 marzo 2023, n. 33 (delega al Governo in materia di politiche a favore delle persone anziane)
4) il Decreto legislativo 15 marzo 2024, n. 29 (riforma anziani non autosufficienti)
5) la Legge 8 novembre 2000, n. 328 (legge quadro per la realizzazione del sistema integrato di interventi e servizi sociali)
6) il DPCM 12 gennaio 2017 (Livelli essenziali di assistenza — LEA)
7) il DM 77/2022 (assistenza territoriale, Case della Comunità, ADI, Ospedali di Comunità)
8) il Piano Nazionale della Cronicità (2016)
9) il Piano Nazionale Demenze
10) la Legge 15 marzo 2010, n. 38 (cure palliative e terapia del dolore)
11) la ricognizione della normativa istitutiva e regolatoria delle Residenze Assistenziali (RA) e delle Residenze Sanitarie Assistenziali (RSA) — quadro nazionale e schede per ciascuna regione italiana e per le Province autonome di Trento e Bolzano
12) la ricognizione dell'Assistenza Domiciliare Integrata (ADI) e del Servizio di Assistenza Domiciliare (SAD) — quadro nazionale (DM 77/2022, DLgs 29/2024, LEPS, standard 10% over-65 in ADI) e schede per ciascuna regione italiana e per le Province autonome di Trento e Bolzano
13) la sezione «Il pensiero», organizzata in quattro sotto-aree:
   a) articoli, interviste ed editoriali di Mons. Vincenzo Paglia sugli anziani;
   b) ricerche scientifiche sul tema dell'invecchiamento (62 pubblicazioni verificate, 1990-2026) del gruppo di Leonardo Palombi, Giuseppe Liotta e Stefano Orlando (Dipartimento di Biomedicina e Prevenzione, Università di Roma Tor Vergata) e collaboratori, sui temi di fragilità, mortalità, valutazione geriatrica multidimensionale, RSA/ADI, screening territoriale;
   — inclusa nella sotto-area «articoli e interventi di Mons. Paglia» l'opera monografica «L'Età Grande: la nuova legge per gli anziani» (Edizioni LSWR 2024), volume in 7 capitoli che analizza la Legge 33/2023 e il decreto legislativo 29/2024 con la Carta dei diritti come orizzonte ideale, le aree di intervento, la visione, gli obiettivi politici e la sperimentazione del Progetto Anchise nella Regione Lazio;
   c) contributi della Comunità di Sant'Egidio;
   d) i documenti ufficiali della Commissione ministeriale per la riforma dell'assistenza agli anziani (presieduta da Mons. Paglia con il Prof. Palombi come Segretario), tra cui l'editoriale del 13 marzo 2021, la «Sintesi finale della proposta al Presidente Draghi» («L'abitazione come luogo di cura per gli anziani») e il DDL sulle deleghe in materia di politiche per gli anziani approvato dal Governo Draghi il 10 ottobre 2022, base della successiva Legge 33/2023.

REGOLE:
- Rispondi SOLO su temi legati alla Carta, ai suoi testi introduttivi, alla normativa italiana sull'assistenza agli anziani (incluse RA, RSA, ADI e SAD nazionali e regionali) o al contenuto degli articoli della sezione «Il pensiero».
- Per la Carta e la normativa nazionale cita sempre articolo e comma (es. "Carta, art. 5, comma 2", "L. 328/2000, art. 22", "DPCM 12/1/2017 (LEA), art. 30", "DM 77/2022").
- Per RA, RSA, ADI e SAD regionali cita così: "RA — Lombardia", "RSA — Emilia-Romagna", "ADI — Veneto", "SAD — Puglia", e — quando presenti nei documenti — le specifiche leggi regionali o DGR (es. "LR Piemonte 12/2009", "DGR Lazio 143/2019").
- Per i quadri nazionali cita "RSA — quadro nazionale", "ADI — quadro nazionale", "SAD — quadro nazionale" indicando la fonte primaria (per RSA art. 20 L. 67/1988, DPCM 22/12/1989, DPCM 14/2/2001, DPCM LEA 2017; per ADI DPCM LEA 2017 art. 22, DM 77/2022, DLgs 29/2024; per SAD L. 328/2000, L. 197/2022 sui LEPS, DLgs 29/2024).

- Per Prefazione, Premessa e Introduzione cita così: "Prefazione", "Premessa", "Introduzione al sito".
- Per i Piani nazionali cita: "Piano Nazionale della Cronicità", "Piano Nazionale Demenze".
- Per gli articoli della sezione «Il pensiero» cita così: «Il pensiero — [autore], “[titolo]”», indicando la testata quando disponibile (es. «Il pensiero — Vincenzo Paglia, “La cura che cambia: il Piemonte sceglie il territorio”»).
- Per il libro cita così: «V. Paglia, “L’Età Grande: la nuova legge per gli anziani”, LSWR 2024, cap. [titolo]».
- Per le pubblicazioni scientifiche cita autori, rivista e anno, es.: «Gilardi et al., European Journal of Public Health 2018», oppure «Liotta et al., PLoS ONE 2020».
- Riporta sempre i dati quantitativi (n=..., HR=..., IC95%, p-value) quando presenti negli abstract.
- Per domande su persone menzionate nella Prefazione o nella Premessa (es. autori, curatori, membri della Commissione), riporta fedelmente quanto scritto in quei testi.
- Se la risposta non è nei documenti forniti, dillo apertamente: "Su questo il corpus di riferimento non offre elementi diretti."
- Non aggiungere opinioni personali né interpretazioni giuridiche vincolanti.
- Usa un registro pacato, informativo, adatto a lettori non specialisti.
- Massimo 3-5 paragrafi brevi. Se serve una lista, tienila essenziale."""
    else:
        system = """You are an informational assistant of Fondazione Età Grande and AEGIS Foundation. Answer in English, clearly and concisely, based EXCLUSIVELY on the documents provided in context. The corpus includes:
1) the Foreword by Bishop Vincenzo Paglia, the Introduction by the authors, and the About page of this website
2) the Charter of the Rights of Older Persons and Duties of Society (by Msgr. Vincenzo Paglia)
3) Italian Law 23 March 2023, no. 33 (delegation on policies for older persons)
4) Italian Legislative Decree 15 March 2024, no. 29 (reform on non-self-sufficient older persons)
5) Law 8 November 2000, no. 328 (framework law on integrated social services)
6) PMCD 12 January 2017 (Essential Levels of Care — LEA)
7) Ministerial Decree 77/2022 (territorial healthcare standards)
8) National Chronicity Plan (2016)
9) National Dementia Plan
10) Law 15 March 2010, no. 38 (palliative care and pain therapy)
11) survey of the founding and regulatory legislation on Assisted-Living Residences (RA) and Nursing Homes (RSA) — national framework and profiles for each Italian region and for the Autonomous Provinces of Trento and Bolzano
12) survey of Integrated Home Care (ADI) and Municipal Home Care Service (SAD) — national framework (Ministerial Decree 77/2022, Legislative Decree 29/2024, LEPS, 10% over-65 ADI target) and profiles for each Italian region and for the Autonomous Provinces of Trento and Bolzano
13) the «Il pensiero» section, organised in four sub-areas:
   a) articles, interviews and editorials by Msgr. Vincenzo Paglia on older persons;
   b) scientific research on ageing (62 verified publications, 1990-2026) by the group of Leonardo Palombi, Giuseppe Liotta and Stefano Orlando (Department of Biomedicine and Prevention, University of Rome Tor Vergata) and collaborators, covering frailty, mortality, multidimensional geriatric assessment, RSA/ADI, community screening;
   c) contributions from the Community of Sant'Egidio;
   d) the official documents of the Ministerial Commission for the reform of elderly care (chaired by Msgr. Paglia, with Prof. Palombi as Secretary), including the editorial of 13 March 2021, the «Final Synthesis of the Proposal to Prime Minister Draghi» («The home as a place of care for older people») and the draft law on delegations for policies on older persons approved by the Draghi Government on 10 October 2022, which formed the basis for Law 33/2023.

RULES:
- Answer ONLY on topics related to the Charter, its introductory texts, or Italian legislation on care for older persons (including regional RA, RSA, ADI and SAD).
- For the Charter and national legislation always cite article and paragraph (e.g. "Charter, art. 5, para. 2", "Law 328/2000, art. 22", "LEA Decree 2017, art. 30", "MD 77/2022").
- For regional RA, RSA, ADI and SAD cite as: "RA — Lombardia", "RSA — Emilia-Romagna", "ADI — Veneto", "SAD — Puglia", and — when present in the documents — the specific regional laws or resolutions.
- For national frameworks cite "RSA — national framework", "ADI — national framework", "SAD — national framework" indicating the primary source (for RSA art. 20 Law 67/1988, PMCD 22/12/1989, PMCD 14/2/2001, LEA Decree 2017; for ADI LEA Decree 2017 art. 22, Ministerial Decree 77/2022, Legislative Decree 29/2024; for SAD Law 328/2000, Law 197/2022 on LEPS, Legislative Decree 29/2024).
- For Foreword, Introduction and About page, cite as: "Foreword", "Introduction (Authors)", "About this website".
- For National Plans cite as: "National Chronicity Plan", "National Dementia Plan".
- For questions about people mentioned in the Foreword or Introduction (e.g. authors, curators, Commission members), report faithfully what those texts state.
- If the answer is not in the provided documents, say so openly: "The reference corpus does not directly address this."
- Do not add personal opinions or binding legal interpretations.
- Use a calm, informative tone suitable for non-specialist readers.
- Maximum 3-5 short paragraphs. If a list is needed, keep it essential."""

    # Contesto dai chunks
    context_parts = []
    for c in chunks:
        label = c['source_label_it'] if is_it else (c['source_label_en'] or c['source_label_it'])
        text = c['text'] if (is_it or not c.get('text_en')) else c['text_en']
        title = c['title'] if is_it else (c.get('title_en') or c['title'])
        # I chunk di 'front' (Prefazione/Premessa/Introduzione) e i Piani non hanno numero articolo tradizionale
        if c.get('source') in ('front', 'pnc', 'pnd', 'dm77', 'ra_reg', 'rsa_reg', 'rsa_naz', 'pensiero', 'ricerca', 'libro'):
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
        msg = ("Sulla base della Carta e della normativa di riferimento (L. 328/2000, L. 33/2023, DLgs 29/2024, DPCM LEA 2017, DM 77/2022, L. 38/2010, Piani nazionali cronicit\u00e0 e demenze) non trovo elementi diretti per rispondere a questa domanda. Prova a riformularla con termini pi\u00f9 specifici, ad esempio 'assistenza domiciliare integrata', 'prestazione universale' o 'cure palliative'."
               if req.lang == 'it' else
               "I did not find direct elements in the reference corpus (Charter, Law 328/2000, Law 33/2023, Legislative Decree 29/2024, LEA Decree 2017, MD 77/2022, Law 38/2010, National Chronicity and Dementia Plans) to answer this question. Please try rephrasing it with more specific terms, e.g. 'integrated home care' or 'palliative care'.")
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

    # Deduplica citazioni per articolo_id o (source, num) per non mostrare
    # più chunk dello stesso articolo di pensiero
    seen = set()
    unique_chunks = []
    for c in chunks:
        key = c.get('articolo_id') or f"{c.get('source')}-{c.get('num')}"
        if key in seen: continue
        seen.add(key)
        unique_chunks.append(c)

    def _mk_excerpt(chunk: dict, max_chars: int = 320) -> str:
        """Ritorna i primi ~320 caratteri del testo del chunk, con ellissi se troncato."""
        txt = (chunk.get('text') or '').strip()
        # Normalizza whitespace multipli
        txt = ' '.join(txt.split())
        if len(txt) <= max_chars:
            return txt
        # Taglia all'ultimo confine di parola prima del limite
        cut = txt.rfind(' ', 0, max_chars)
        if cut < max_chars * 0.6:
            cut = max_chars
        return txt[:cut].rstrip() + '…'

    citations = [
        Citation(
            id=c['id'],
            title=c['title'] if req.lang == 'it' else (c.get('title_en') or c['title']),
            source=c['source_label_it'] if req.lang == 'it' else (c['source_label_en'] or c['source_label_it']),
            num=c['num'],
            articolo_id=c.get('articolo_id'),
            area=c.get('area'),
            ricerca_id=c.get('ricerca_id'),
            capitolo_id=c.get('capitolo_id'),
            excerpt=_mk_excerpt(c),
        )
        for c in unique_chunks[:4]  # Mostra max 4 fonti
    ]

    return ChatResponse(answer=answer, citations=citations)


@app.get('/api/health')
async def health():
    return {"status": "ok", "chunks": len(CORPUS)}
