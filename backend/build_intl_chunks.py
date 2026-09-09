"""Costruisce chunk RAG dai 7 report internazionali (WHO, UNDESA, Eurostat, HelpAge).

Chunking: uno per sezione riconoscibile o, dove non c'è struttura chiara, per
blocchi di ~2000-3000 char, preservando titoli/heading di pagina. Ogni chunk
riporta il PDF sorgente + range pagine per la citazione.

Tutti i report sono classificati come 'ricerca' (Ricerca scientifica) con
source_label che identifica l'agenzia e l'anno.
"""
import json
import re
from pathlib import Path
import pymupdf

SRC_DIR = Path('/tmp/intl_sources')
OUT = Path('/tmp/carta-backend/backend/intl_chunks.json')

# Metadati dei 7 report
REPORTS = [
    {
        'file': 'who_world_report_2015.pdf',
        'id_prefix': 'intl-who-wra2015',
        'source_label_it': 'WHO — World Report on Ageing and Health (2015)',
        'source_label_en': 'WHO — World Report on Ageing and Health (2015)',
        'title_short': 'WHO World Report on Ageing and Health (2015)',
        'url': 'https://iris.who.int/handle/10665/186463',
    },
    {
        'file': 'who_ageism_2021.pdf',
        'id_prefix': 'intl-who-ageism2021',
        'source_label_it': 'WHO — Global Report on Ageism (2021)',
        'source_label_en': 'WHO — Global Report on Ageism (2021)',
        'title_short': 'WHO Global Report on Ageism (2021)',
        'url': 'https://iris.who.int/handle/10665/340208',
    },
    {
        'file': 'undesa_wpa_2023.pdf',
        'id_prefix': 'intl-undesa-wpa2023',
        'source_label_it': 'UNDESA — World Population Ageing 2023',
        'source_label_en': 'UNDESA — World Population Ageing 2023',
        'title_short': 'UNDESA World Population Ageing 2023',
        'url': 'https://www.un.org/development/desa/pd/content/World-Population-Ageing-2023',
    },
    {
        'file': 'undesa_wsr_2023.pdf',
        'id_prefix': 'intl-undesa-wsr2023',
        'source_label_it': 'UNDESA — World Social Report 2023: Leaving No One Behind In An Ageing World',
        'source_label_en': 'UNDESA — World Social Report 2023: Leaving No One Behind In An Ageing World',
        'title_short': 'UNDESA World Social Report 2023',
        'url': 'https://desapublications.un.org/publications/world-social-report-2023-leaving-no-one-behind-ageing-world',
    },
    {
        'file': 'eurostat_ageing_2020.pdf',
        'id_prefix': 'intl-eurostat-ae2020',
        'source_label_it': 'Eurostat — Ageing Europe: looking at the lives of older people in the EU (2020 edition)',
        'source_label_en': 'Eurostat — Ageing Europe: looking at the lives of older people in the EU (2020 edition)',
        'title_short': 'Eurostat Ageing Europe 2020',
        'url': 'https://ec.europa.eu/eurostat/web/products-statistical-books/-/ks-02-20-655',
    },
    {
        'file': 'helpage_insights_2018.pdf',
        'id_prefix': 'intl-helpage-insights2018',
        'source_label_it': 'HelpAge International — Global AgeWatch Insights 2018: The right to health for older people, the right to be counted',
        'source_label_en': 'HelpAge International — Global AgeWatch Insights 2018: The right to health for older people, the right to be counted',
        'title_short': 'HelpAge Global AgeWatch Insights 2018',
        'url': 'https://globalagewatchindex.helpage.org/reports/global-agewatch-insights-2018-report-summary-and-country-profiles/',
    },
    {
        'file': 'helpage_out_of_sight.pdf',
        'id_prefix': 'intl-helpage-outofsight2022',
        'source_label_it': 'HelpAge International — Out of Sight, Out of Mind: The inclusion and use of older person data in humanitarian action (2022)',
        'source_label_en': 'HelpAge International — Out of Sight, Out of Mind: The inclusion and use of older person data in humanitarian action (2022)',
        'title_short': 'HelpAge Out of Sight, Out of Mind (2022)',
        'url': 'https://www.helpage.org/silo/files/out-of-sight-out-of-mindtechnical-report.pdf',
    },
]

TARGET_MIN = 1500  # char minimo per chunk (evita micro-chunk)
TARGET_MAX = 3200  # char massimo per chunk


def clean_text(text: str) -> str:
    """Normalizza spazi, rimuove hyphenation di riga."""
    # Congiungi parole spezzate a fine riga: "asses-\nment" -> "assessment"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Normalizza tab
    text = re.sub(r'\t+', ' ', text)
    # Collassa spazi multipli
    text = re.sub(r' +', ' ', text)
    # Preserva newline ma comprimi righe vuote multiple
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Estrae testo per pagina, saltando pagine praticamente vuote."""
    doc = pymupdf.open(pdf_path)
    out = []
    for i, page in enumerate(doc, start=1):
        txt = page.get_text()
        # Salta pagine molto brevi (bianche, solo header/footer)
        stripped = txt.strip()
        if len(stripped) < 100:
            continue
        out.append((i, clean_text(txt)))
    doc.close()
    return out


def chunk_pages_naive(pages: list[tuple[int, str]], report: dict) -> list[dict]:
    """Chunking: accumula pagine consecutive fino a raggiungere ~2500 char.

    Preserva il range pagine per la citazione. Se una singola pagina supera
    TARGET_MAX, la spezza in blocchi di paragrafi.
    """
    chunks = []
    buf = []
    buf_start_page = None
    buf_len = 0

    def flush():
        nonlocal buf, buf_start_page, buf_len
        if not buf:
            return
        end_page = buf[-1][0]
        page_range = f'p. {buf_start_page}' if buf_start_page == end_page else f'pp. {buf_start_page}-{end_page}'
        text_body = '\n\n'.join(b[1] for b in buf)
        # Header interno con contesto
        header = f'{report["title_short"]} — {page_range}'
        chunk_text = f'{header}\n\n{text_body}'
        chunks.append({
            'id': f'{report["id_prefix"]}-p{buf_start_page:04d}',
            'source': 'ricerca',
            'source_label_it': report['source_label_it'],
            'source_label_en': report['source_label_en'],
            'article_id': report['id_prefix'],
            'section_id': f'{report["id_prefix"]}-p{buf_start_page:04d}',
            'articolo_id': report['id_prefix'],
            'ricerca_id': report['id_prefix'],
            'num': str(buf_start_page),
            'title': header,
            'title_en': header,
            'text': chunk_text,
            'text_en': '',
            'meta': {'url': report['url'], 'page_start': buf_start_page, 'page_end': end_page},
        })
        buf = []
        buf_start_page = None
        buf_len = 0

    for page_num, page_text in pages:
        # Se la pagina da sola è oversize, la spezziamo in paragrafi
        if len(page_text) > TARGET_MAX:
            flush()  # svuota il buffer prima
            # Split by paragraphs
            paras = [p for p in page_text.split('\n\n') if p.strip()]
            acc = []
            acc_len = 0
            part_idx = 0
            for p in paras:
                if acc_len + len(p) > TARGET_MAX and acc:
                    part_idx += 1
                    header = f'{report["title_short"]} — p. {page_num} (parte {part_idx})'
                    chunks.append({
                        'id': f'{report["id_prefix"]}-p{page_num:04d}-{part_idx}',
                        'source': 'ricerca',
                        'source_label_it': report['source_label_it'],
                        'source_label_en': report['source_label_en'],
                        'article_id': report['id_prefix'],
                        'section_id': f'{report["id_prefix"]}-p{page_num:04d}-{part_idx}',
                        'articolo_id': report['id_prefix'],
                        'ricerca_id': report['id_prefix'],
                        'num': str(page_num),
                        'title': header,
                        'title_en': header,
                        'text': f'{header}\n\n' + '\n\n'.join(acc),
                        'text_en': '',
                        'meta': {'url': report['url'], 'page_start': page_num, 'page_end': page_num, 'part': part_idx},
                    })
                    acc = []
                    acc_len = 0
                acc.append(p)
                acc_len += len(p) + 2
            if acc:
                part_idx += 1
                header = f'{report["title_short"]} — p. {page_num} (parte {part_idx})' if part_idx > 1 else f'{report["title_short"]} — p. {page_num}'
                suffix = f'-{part_idx}' if part_idx > 1 else ''
                chunks.append({
                    'id': f'{report["id_prefix"]}-p{page_num:04d}{suffix}',
                    'source': 'ricerca',
                    'source_label_it': report['source_label_it'],
                    'source_label_en': report['source_label_en'],
                    'article_id': report['id_prefix'],
                    'section_id': f'{report["id_prefix"]}-p{page_num:04d}{suffix}',
                    'articolo_id': report['id_prefix'],
                    'ricerca_id': report['id_prefix'],
                    'num': str(page_num),
                    'title': header,
                    'title_en': header,
                    'text': f'{header}\n\n' + '\n\n'.join(acc),
                    'text_en': '',
                    'meta': {'url': report['url'], 'page_start': page_num, 'page_end': page_num, 'part': part_idx if part_idx > 1 else None},
                })
            continue

        # Pagina normale: prova ad accumulare
        if buf_len + len(page_text) > TARGET_MAX and buf:
            flush()
        if not buf:
            buf_start_page = page_num
        buf.append((page_num, page_text))
        buf_len += len(page_text) + 2

    flush()
    return chunks


def main():
    all_chunks = []
    for report in REPORTS:
        pdf_path = SRC_DIR / report['file']
        if not pdf_path.exists():
            print(f'MISSING: {pdf_path}')
            continue
        pages = extract_pages(pdf_path)
        chunks = chunk_pages_naive(pages, report)
        total_chars = sum(len(c['text']) for c in chunks)
        avg = total_chars // max(len(chunks), 1)
        print(f'{report["file"]:<38} → {len(chunks):>4} chunk ({total_chars:>7,} char, avg {avg:>4})')
        all_chunks.extend(chunks)

    print()
    print(f'TOTALE: {len(all_chunks)} chunk, {sum(len(c["text"]) for c in all_chunks):,} char')

    OUT.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Salvato: {OUT}')


if __name__ == '__main__':
    main()
