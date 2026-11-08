"""Genera site_chunks.json: le sezioni del sito che mancavano nell'archivio della chat
(Costi, ADI/SAD nazionali e regionali, Sperimentazioni, L.33 applicazioni, testi
introduttivi, Conclusioni, Poesie), con testo italiano (text) e inglese (text_en),
più le traduzioni inglesi per i chunk RA/RSA già presenti (en_overlay).
Uso: python build_site_chunks.py <cartella frontend>
"""
import json, re, sys, html, pathlib
F = pathlib.Path(sys.argv[1])
def J(p):
    q = F / p
    return json.loads(q.read_text()) if q.exists() else None
def clean(h):
    h = re.sub(r'<(br|/p|/li|/h\d|/tr)[^>]*>', '\n', h or '')
    h = re.sub(r'<t[dh][^>]*>', ' | ', h)
    h = re.sub(r'<[^>]+>', '', h)
    h = html.unescape(h)
    return re.sub(r'\n\s*\n+', '\n\n', re.sub(r'[ \t]+', ' ', h)).strip()
def split(t, n=3000):
    paras, out, cur = t.split('\n\n'), [], ''
    for p in paras:
        if cur and len(cur) + len(p) > n:
            out.append(cur); cur = ''
        cur += ('\n\n' if cur else '') + p
    if cur: out.append(cur)
    return out or ['']
def split_like(t, k):
    if k <= 1: return [t]
    paras = t.split('\n\n'); size = max(1, -(-len(paras) // k))
    parts = ['\n\n'.join(paras[i*size:(i+1)*size]) for i in range(k)]
    return parts
OUT = []
def add(base_id, source, lab_it, lab_en, title, title_en, t_it, t_en, extra=None):
    parts = split(t_it); parts_en = split_like(t_en or '', len(parts))
    for k, (pi, pe) in enumerate(zip(parts, parts_en), 1):
        suf = f' (parte {k})' if len(parts) > 1 else ''
        suf_en = f' (part {k})' if len(parts) > 1 else ''
        c = {'id': f'{base_id}-p{k}', 'source': source, 'source_label_it': lab_it, 'source_label_en': lab_en,
             'title': title + suf, 'title_en': (title_en or title) + suf_en,
             'text': f'{title}\n\n{pi}', 'text_en': f'{title_en}\n\n{pe}' if pe else ''}
        if extra: c.update(extra)
        OUT.append(c)

def sections(fn, src, lab_it, lab_en, prefix):
    it, en = J(fn), J('en/' + fn) or []
    for i, s in enumerate(it or [], 1):
        e = en[i-1] if i-1 < len(en) else {}
        add(f'{prefix}-{i:02d}', src, lab_it, lab_en, f"{lab_it} — {s['title']}", f"{lab_en} — {e.get('title', s['title'])}",
            clean(s.get('html')), clean(e.get('html')), {'num': i, 'sezione': i})

sections('costi_nazionale.json', 'costi_nazionale', 'Costi del sistema di cura', 'Costs of the care system', 'costi')
sections('adi_nazionale.json', 'adi_naz', 'ADI — quadro nazionale', 'ADI (integrated home care) — national framework', 'adi-nazionale')
sections('sad_nazionale.json', 'sad_naz', 'SAD — quadro nazionale', 'SAD (social home care) — national framework', 'sad-nazionale')

def regional(fn, src, sig, sig_en, prefix):
    it, en = J(fn), J('en/' + fn) or []
    for i, r in enumerate(it or [], 1):
        e = en[i-1] if i-1 < len(en) else {}
        t_it = '\n\n'.join(f"{b['label']}: {clean(b['html'])}" for b in r['bullets'])
        t_en = '\n\n'.join(f"{b['label']}: {clean(b['html'])}" for b in e.get('bullets', []))
        add(f'{prefix}-{i:02d}', src, f'{sig} {r["regione"]}', f'{sig} {e.get("regione", r["regione"])}',
            f'{sig} — {r["regione"]}', f'{sig_en} — {e.get("regione", r["regione"])}', t_it, t_en, {'num': i})
regional('adi_per_regione.json', 'adi_reg', 'ADI', 'ADI (integrated home care)', 'adi')
regional('sad_per_regione.json', 'sad_reg', 'SAD', 'SAD (social home care)', 'sad')

sp, spe = J('sperimentazioni.json'), J('en/sperimentazioni.json') or {}
if sp:
    c = sp['cappello']
    add('sperim-intro', 'sperimentazioni', 'Sperimentazioni', 'Pilot projects', c['titolo'], 'Pilot projects',
        clean(c.get('descrizione', '')), clean((spe.get('cappello') or {}).get('descrizione', '')))
    for p in sp['progetti']:
        t_it = '\n\n'.join(clean(p.get(k) or '') for k in ('sottotitolo', 'quadro_normativo', 'descrizione', 'capofila', 'avvio') if p.get(k))
        t_en = '\n\n'.join(clean(p.get(k + '_en') or '') for k in ('sottotitolo', 'quadro_normativo', 'descrizione') if p.get(k + '_en'))
        add(f"sperim-{p['id']}", 'sperimentazioni', 'Sperimentazioni', 'Pilot projects',
            f"Sperimentazioni — {p['titolo']}", f"Pilot projects — {p.get('titolo_en') or p['titolo']}", t_it, t_en)

la, lae = J('l33_applicazioni.json'), J('en/l33_applicazioni.json') or {}
if la:
    se = lae.get('sezioni', [])
    for i, s in enumerate(la['sezioni'], 1):
        e = se[i-1] if i-1 < len(se) else {}
        def body(x):
            t = clean(x.get('corpo') if isinstance(x.get('corpo'), str) else '\n\n'.join(map(str, x.get('corpo') or [])))
            tab = x.get('tabella')
            if tab: t += '\n\n' + clean(json.dumps(tab, ensure_ascii=False))
            return t
        add(f'l33app-{i:02d}', 'l33_app', 'L. 33/2023 — Applicazioni e sperimentazioni', 'Law 33/2023 — Implementation and pilots',
            f"L. 33/2023, attuazione — {s['titolo']}", f"Law 33/2023, implementation — {e.get('titolo', s['titolo'])}", body(s), body(e) if e else '')

di = J('data_intro.json')
names = {'introduzione': ('Introduzione al sito', 'About this website', 'introduction'),
         'prefazione': ('Prefazione', 'Foreword', 'preface'), 'premessa': ('Premessa', 'Introduction', 'premise')}
if di:
    for k, (ti, te, kind) in names.items():
        add(f'front-{k}', 'front', ti, te, ti, te, '\n\n'.join(di['it'].get(k, [])), '\n\n'.join(di['en'].get(k, [])), {'kind': kind})

def embedded(id_):
    s = (F / 'app.html').read_text()
    m = re.search(r'<script id="%s" type="application/json">(.*?)</script>' % id_, s, re.S)
    return json.loads(m.group(1)) if m else None
ci, ce = embedded('data-conclusioni-it'), embedded('data-conclusioni-en')
def flat(x):
    if isinstance(x, str): return x
    if isinstance(x, list): return '\n\n'.join(flat(y) for y in x)
    if isinstance(x, dict): return '\n\n'.join(flat(v) for k, v in x.items() if k in ('title', 'text', 'paragraphs', 'sections', 'heading'))
    return ''
if ci:
    add('front-conclusioni', 'front', 'Conclusioni del libro', 'Conclusions of the book', 'Conclusioni', 'Conclusions',
        clean(flat(ci)), clean(flat(ce)) if ce else '', {'kind': 'conclusions'})

po, poe = J('poesie.json'), J('en/poesie.json') or {}
if po:
    pe = {p['id']: p for p in poe.get('poesie', [])}
    for p in po['poesie']:
        e = pe.get(p['id'], {})
        def ptxt(x):
            t = x.get('testo')
            return '\n'.join(t) if isinstance(t, list) else clean(str(t or ''))
        add(f"poesia-{p['id']}", 'poesie', 'Poesie sulla vecchiaia', 'Poems on old age',
            f"Poesie — {p['autore']}, «{p['titolo']}»", f"Poems — {p['autore']}, “{e.get('titolo_en') or e.get('titolo') or p['titolo']}”",
            (clean(p.get('nota') or '') + '\n\n' + ptxt(p)).strip(), (clean(e.get('nota') or '') + '\n\n' + ptxt(e)).strip() if e else '')

gl = J('glossario.json')
if gl:
    for v in gl['voci']:
        sig = (v['sigla'] + ' — ') if v.get('sigla') else ''
        OUT.append({'id': 'glossario-' + re.sub(r'[^a-z0-9]+', '-', (v.get('sigla') or v['it']).lower()).strip('-'),
                    'source': 'glossario', 'source_label_it': 'Glossario internazionale', 'source_label_en': 'International glossary',
                    'title': f"Glossario — {sig}{v['it']}", 'title_en': f"Glossary — {sig}{v['it']} ({v['en']})",
                    'text': f"{sig}{v['it']} (in inglese: {v['en']}). {v['def_it']}",
                    'text_en': f"{sig}{v['it']} — English: {v['en']}. {v['def_en']}"})

overlay = {}
for fn, prefix in (('rsa_per_regione.json', 'rsa'), ('ra_per_regione.json', 'ra')):
    for i, r in enumerate(J('en/' + fn) or [], 1):
        overlay[f'{prefix}-{i:02d}'] = '\n\n'.join(f"{b['label']}: {clean(b['html'])}" for b in r.get('bullets', []))
for i, s in enumerate(J('en/parte1_nazionale.json') or [], 1):
    overlay[f'rsa-nazionale-{i:02d}'] = s.get('title', '') + '\n\n' + clean(s.get('html'))

out = pathlib.Path(__file__).with_name('site_chunks.json')
out.write_text(json.dumps({'chunks': OUT, 'en_overlay': overlay}, ensure_ascii=False))
from collections import Counter
print(len(OUT), dict(Counter(c['source'] for c in OUT)), 'overlay', len(overlay), 'senza EN:', sum(1 for c in OUT if not c['text_en']))
