"""Rigenera i chunk della Carta dal frontend data_it.json completo.

Il vecchio corpus 'carta' aveva solo 18 chunk (un piccolo estratto per articolo).
Il file frontend data_it.json contiene il TESTO COMPLETO della Carta: articoli,
commi, e commenti estensivi con dati OCSE, HelpAge, Oxford MPI, ecc.

Questo script produce chunk decenti (uno per articolo intero, con tutti i commi
e i commenti concatenati), da aggiungere al corpus RAG del backend.
"""
import json
import re
from pathlib import Path

DATA = Path('/home/user/workspace/carta-app/dist/public/data_it.json')
OUT = Path('/tmp/carta-backend/backend/carta_v2_chunks.json')

def _clean_text(t):
    """Normalizza spazi e rimuove residui HTML minimi."""
    t = re.sub(r'<[^>]+>', '', t or '')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_paragraphs(paragraphs):
    """Concatena i paragrafi di un comma, ignorando quelli vuoti."""
    out = []
    for p in paragraphs or []:
        if not isinstance(p, dict):
            continue
        t = _clean_text(p.get('text', ''))
        if t:
            out.append(t)
    return '\n\n'.join(out)

def chunkify_article(art):
    """Trasforma un articolo intero in uno o più chunk (uno per comma con commento)."""
    chunks = []
    num = art.get('num', '?')
    title = _clean_text(art.get('title', ''))
    sections = art.get('sections', [])
    
    # Per ciascun comma, creo un chunk che include: titolo articolo, numero comma, testo del comma, commento
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        n = sec.get('n', '')
        comma = _clean_text(sec.get('comma', ''))
        commento = extract_paragraphs(sec.get('paragraphs', []))
        
        if not comma and not commento:
            continue
        
        # Titolo del chunk: "Art. X.Y — Titolo articolo"
        chunk_title = f"Carta, Art. {num}.{n} — {title}" if n else f"Carta, Art. {num} — {title}"
        
        parts = [chunk_title]
        if comma:
            parts.append(f"Testo del comma: {comma}")
        if commento:
            parts.append(f"Commento: {commento}")
        
        text = '\n\n'.join(parts)
        
        def _make(part_suffix, body_text, part_num=None):
            """Costruisce un chunk con TUTTI i campi che il server si aspetta."""
            cid = f'carta-{num}-{n}' + (f'-p{part_num}' if part_num else '')
            ct = chunk_title + (f' (parte {part_num})' if part_num else '')
            return {
                'id': cid,
                'source': 'carta',
                'source_label_it': 'Carta dei diritti degli anziani',
                'source_label_en': 'Charter of the Rights of Older Persons',
                'article_id': f'{num}',
                'section_id': f'{num}.{n}',
                'articolo_id': f'{num}',
                'num': str(num),
                'title': ct,
                'title_en': ct,  # per ora IT usato in EN come fallback (traduzione EN può arrivare dopo)
                'text': body_text,
                'text_en': '',  # nessun testo EN separato: il server ricade su 'text'
            }

        # Split se troppo lungo (>3500 char): tengo articolo+comma intero, spezzo il commento
        if len(text) > 3500 and commento:
            paras = commento.split('\n\n')
            mid = len(paras) // 2
            first = '\n\n'.join(paras[:mid]) if mid > 0 else ''
            second = '\n\n'.join(paras[mid:])
            if first:
                text1 = f"{chunk_title}\n\nTesto del comma: {comma}\n\nCommento (parte 1): {first}"
                text2 = f"{chunk_title}\n\nTesto del comma: {comma}\n\nCommento (parte 2): {second}"
                chunks.append(_make(' (parte 1)', text1, part_num=1))
                chunks.append(_make(' (parte 2)', text2, part_num=2))
            else:
                chunks.append(_make('', text))
        else:
            chunks.append(_make('', text))
    
    return chunks

def main():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    all_chunks = []
    for art in data:
        all_chunks.extend(chunkify_article(art))
    
    print(f'Articoli: {len(data)}')
    print(f'Chunk generati: {len(all_chunks)}')
    print(f'Char totali: {sum(len(c["text"]) for c in all_chunks):,}')
    print(f'Char medi per chunk: {sum(len(c["text"]) for c in all_chunks) // len(all_chunks):,}')
    print()
    # Cerca menzioni povertà per verifica
    pov = [c for c in all_chunks if 'povert' in c['text'].lower() or 'indigen' in c['text'].lower() or 'ocse' in c['text'].lower() or 'helpage' in c['text'].lower()]
    print(f'Chunk con povertà/OCSE/HelpAge: {len(pov)}')
    for c in pov[:5]:
        print(f'  - {c["title"]}')
    
    OUT.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSalvato: {OUT}')

if __name__ == '__main__':
    main()
