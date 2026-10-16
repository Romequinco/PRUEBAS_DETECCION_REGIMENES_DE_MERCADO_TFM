"""Extracción de texto y evidencia temporal sin usar fechas técnicas del archivo."""

import io
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .temporal import parse_date


def clean(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def soup(content):
    return BeautifulSoup(content, 'html.parser')


def publication_html(page, source):
    # Prioritize publication-specific fields; never use modified/created metadata.
    for name in ('citation_online_date', 'citation_publication_date', 'publication_date', 'article:published_time', 'DC.date.issued'):
        node = page.find('meta', attrs={'name': name}) or page.find('meta', attrs={'property': name})
        if node and node.get('content'):
            return node['content'], f'meta {name}'
    for script in page.select('script[type="application/ld+json"]'):
        try:
            stack = [json.loads(script.string or script.get_text())]
        except (ValueError, TypeError):
            continue
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if item.get('datePublished'):
                    return item['datePublished'], 'JSON-LD datePublished (contrastar versión)'
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    selectors = ['.pubdate', '.published-date', '.publication-date', '.ecb-publicationDate',
                 '.field--name-field-publication-date']
    if source == 'Fed':
        selectors.insert(0, '.article__time')
    if source in ('BCE', 'ESRB'):
        selectors.insert(0, '.date')
    if source == 'SNB (Suiza)':
        selectors.insert(0, '.cms-stage__date')
    for selector in selectors:
        node = page.select_one(selector)
        if node and clean(node.get_text()):
            return node.get('datetime') or clean(node.get_text()), f'fecha visible {selector}'
    text = clean((page.select_one('main, #content, article') or page).get_text(' ', strip=True))
    for pattern in (r'Published on\s+(\d{1,2} [A-Za-z]+ \d{4})',
                    r'Published\s+(\d{1,2} [A-Za-z]+ \d{4})',
                    r'Publication Date:\s*([A-Za-z]+ \d{1,2},? \d{4})',
                    r'Date\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                    r'Posted:\s*(\d{1,2} [A-Za-z]+ \d{4})',
                    r'Issue Date\s+([A-Za-z]+ \d{4})'):
        match = re.search(pattern, text, re.I)
        if match:
            return match[1], f'etiqueta {match[0]}'
    return None, 'Sin etiqueta explícita de publicación'


def extract(original, source, family, max_pages=40, qa=False):
    result = {'formato': 'HTML', 'paginas': None, 'texto_completo': True,
              'fecha_publicacion_raw': None, 'evidencia_publicacion': None}
    if original.content.startswith(b'%PDF'):
        reader = PdfReader(io.BytesIO(original.content))
        parts = [(p.extract_text() or '') for p in reader.pages[:max_pages]]
        text = '\n\n'.join(parts)
        result.update(formato='PDF', paginas=len(reader.pages),
                      texto_completo=len(reader.pages) <= max_pages,
                      titulo=(reader.metadata.title if reader.metadata else None) or family)
        embargo = re.search(r'Not to be released until\s+(.{0,180}?\d{4})', clean(text), re.I)
        if embargo:
            date_match = re.search(r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})', embargo[1])
            if date_match:
                result.update(fecha_publicacion_raw=date_match[1],
                              evidencia_publicacion=f'aviso PDF: {embargo[0]}')
    else:
        page = soup(original.content)
        title = page.find('meta', attrs={'name': 'citation_title'}) or page.find('meta', attrs={'property': 'og:title'})
        result['titulo'] = title.get('content') if title else clean((page.find('h1') or page.title or page).get_text())[:250]
        raw, evidence = publication_html(page, source)
        result.update(fecha_publicacion_raw=raw, evidencia_publicacion=evidence)
        node = None
        for selector in ('.ecb-pressContent', '#article', 'article', 'main', '#main-content', '#content', '.tablebg'):
            node = next((n for n in page.select(selector) if len(clean(n.get_text())) > 120), None)
            if node:
                break
        node = node or page.body or page
        copy = soup(str(node))
        for junk in copy.select('script, style, nav, header, footer, aside, form, .cookie-policy, .cookie-banner'):
            junk.decompose()
        if qa:
            anchor = copy.find(id='qa')
            if not anchor:
                raise ValueError('Comparecencia sin sección Q&A extraíble')
            text = '\n\n'.join(n.get_text(' ', strip=True) for n in anchor.parent.find_next_siblings())
            if '?' not in text:
                raise ValueError('Q&A sin preguntas reconocibles')
        else:
            blocks = copy.select('h1, h2, h3, h4, p, li, tr')
            text = '\n\n'.join(clean(n.get_text(' ', strip=True)) for n in blocks if not n.find_parent(['p', 'li', 'tr']))
            text = text or copy.get_text('\n', strip=True)
        result['texto_completo'] = not qa  # Extracted Q&A is a section, not the full parent.
    if len(clean(text)) < 80:
        raise ValueError('Sin texto suficiente: posible OCR, página vacía o protección de acceso')
    if re.search(r'just a moment|access denied|enable javascript and cookies|captcha', result['titulo'], re.I):
        raise ValueError('La respuesta contiene una página de protección, no el documento')
    result['texto'] = text
    return result


def event_date(url, text, source, family):
    """Cada regla documenta su significado; no interpreta cualquier fecha del cuerpo."""
    if source == 'Fed' and 'Actas FOMC' in family:
        match = re.search(r'fomcminutes(\d{8})', url)
        if match:
            return parse_date(f'{match[1][:4]}-{match[1][4:6]}-{match[1][6:]}')[0], 'fecha de reunión en identificador oficial FOMC'
    if source == 'BCE' and 'accounts' in family:
        match = re.search(r'(?:Meeting of|held on) (\d{1,2})(?:\s*(?:and|[-–])\s*(\d{1,2}))?\s+([A-Za-z]+)\s+(\d{4})', text[:6000], re.I)
        if match:
            return parse_date(f'{match[2] or match[1]} {match[3]} {match[4]}')[0], f'fin de reunión: {match[0]}'
    if source == 'Fed' and 'Comunicado y decisión FOMC' in family:
        match = re.search(r'monetary(\d{8})a\.htm', url)
        if match:
            return f'{match[1][:4]}-{match[1][4:6]}-{match[1][6:]}', 'fecha de decisión en identificador oficial FOMC'
    # Policy application is read from an explicit effective-date phrase, separately.
    for pattern in (r'(?:meeting held on|meeting on)\s+([A-Za-z]+)\s+(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?,?\s+(\d{4})',
                    r'(?:meeting held on|meeting on)\s+(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\s+([A-Za-z]+)\s+(\d{4})'):
        match = re.search(pattern, text[:6000], re.I)
        if match:
            if match[1].isalpha():
                value = f'{match[3] or match[2]} {match[1]} {match[4]}'
            else:
                value = f'{match[2] or match[1]} {match[3]} {match[4]}'
            return parse_date(value)[0], match[0]
    return None, 'Evento/periodo no identificado explícitamente'


def publication_clock(raw, text, source):
    """Hora solo cuando el aviso del documento identifica también su zona."""
    normalized, precision, _ = parse_date(raw)
    if source != 'Fed' or precision != 'día':
        return raw
    match = re.search(r'For release at\s+(\d{1,2}):(\d{2})\s*([ap])\.m\.\s*(EST|EDT)', text, re.I)
    if not match:
        return raw
    hour = int(match[1]) % 12 + (12 if match[3].lower() == 'p' else 0)
    day = datetime.fromisoformat(normalized).replace(hour=hour, minute=int(match[2]), tzinfo=ZoneInfo('America/New_York'))
    # Keep the issuer's explicit EST/EDT rather than assuming DST.
    offset = '-05:00' if match[4].upper() == 'EST' else '-04:00'
    return day.replace(tzinfo=None).isoformat() + offset


def application_date(text):
    match = re.search(r'(?:with effect from|effective (?:from|on)|shall apply from|applies from)\s+'
                      r'(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})', text, re.I)
    return (parse_date(match[1])[0], match[0]) if match else (None, 'Aplicación no indicada/no procede')


def chunks(text, version_id, max_words=300):
    """Párrafos y páginas/encabezados conservan orden y vínculo a su versión padre."""
    result = []
    for paragraph in text.split('\n\n'):
        words = paragraph.split()
        for start in range(0, len(words), max_words):
            result.append({'version_id': version_id, 'orden': len(result),
                           'texto': ' '.join(words[start:start + max_words])})
    return result
