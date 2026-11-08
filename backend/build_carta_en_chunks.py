"""Genera i chunk inglesi della Carta (carta_en_chunks.json) da data_en.json del frontend.

Il corpus aveva i chunk della Carta solo in italiano (text_en vuoto): le domande in
inglese non li recuperavano. Un chunk per articolo, suddiviso in parti di ~3000 caratteri.
Uso: python build_carta_en_chunks.py path/to/data_en.json
"""
import json, sys, pathlib
src = pathlib.Path(sys.argv[1])
arts = json.loads(src.read_text())
out = []
for a in arts:
    n = str(a['num'])
    head = f"Charter, Art. {n} — {a['title']}"
    paras = [p['text'].strip() for p in a['paragraphs'] if p.get('text', '').strip()]
    parts, cur = [], ''
    for p in paras:
        if cur and len(cur) + len(p) > 3000:
            parts.append(cur); cur = ''
        cur += (('\n\n' if cur else '') + p)
    if cur: parts.append(cur)
    for k, body in enumerate(parts, 1):
        suffix = f" (part {k})" if len(parts) > 1 else ''
        out.append({
            'id': f'carta-en-{n}-p{k}', 'source': 'carta',
            'source_label_it': 'Carta dei diritti degli anziani',
            'source_label_en': 'Charter of the Rights of Older Persons',
            'article_id': n, 'articolo_id': n, 'num': n, 'lang': 'en',
            'title': head + suffix, 'title_en': head + suffix,
            'text': head + '\n\n' + body, 'text_en': '',
        })
pathlib.Path(__file__).with_name('carta_en_chunks.json').write_text(json.dumps(out, ensure_ascii=False))
print(len(out), 'chunk')
