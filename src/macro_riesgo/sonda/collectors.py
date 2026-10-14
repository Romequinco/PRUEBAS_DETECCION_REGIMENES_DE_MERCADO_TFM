"""Adaptadores de documentos, tablas y APIs para el inventario de la sonda."""

import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import quote, urljoin

import pandas as pd

from .parsing import clean, extract, soup
from .temporal import parse_date


class Collector:
    def __init__(self, http, config):
        self.http = http
        self.config = config

    def links(self, url, pattern):
        """Enlaces únicos en el orden del listado; no garantiza cobertura histórica."""
        original = self.http.get(url)
        page = soup(original.content)

        def find(tree, base):
            found = []
            node = tree.select_one('main, #main-content, #content') or tree
            for anchor in node.select('a[href]'):
                target = urljoin(base, anchor['href'])
                if anchor.find_parent(['nav', 'header', 'footer']) or not re.search(pattern, target, re.I):
                    continue
                context = anchor.find_parent('tr') or anchor.parent
                evidence = clean(context.get_text(' ', strip=True))
                if 'boj.or.jp' in base and anchor.find_parent('tr') and anchor.find_parent('td'):
                    row = anchor.find_parent('tr')
                    table = row.find_parent('table')
                    caption = table.find('caption') if table else None
                    year = re.search(r'\d{4}', caption.get_text()) if caption else None
                    cell = clean(anchor.find_parent('td').get_text())
                    released = re.match(r'([A-Za-z]+)\.?\s+(\d{1,2})', cell)
                    meeting_cell = row.find('td')
                    meeting = clean(meeting_cell.get_text()) if meeting_cell else ''
                    month = re.match(r'([A-Za-z]+)', meeting)
                    days = re.findall(r'\b(\d{1,2})\s*\(', meeting)
                    # Only release-schedule columns; the first cell is the meeting.
                    if year and released and anchor.find_parent('td') is not meeting_cell:
                        evidence += f' | PUBLICACION_BOJ={released[2]} {released[1]} {year[0]}'
                    if year and month and days:
                        evidence += f' | EVENTO_BOJ={days[-1]} {month[1]} {year[0]}'
                found.append((target, evidence))
            return list(dict(found).items())

        result = find(page, url)
        if not result:
            node = page.select_one('[data-snippets]')
            if node:
                fragment = urljoin(url, node['data-snippets'].split(',')[0])
                result = find(soup(self.http.get(fragment).content), fragment)
        if not result:
            raise ValueError(f'No se encontró un enlace con patrón {pattern!r}')
        return result

    def targets(self, spec):
        pattern = spec.get('patron')
        current = [(spec.get('url', ''), '')]
        for step in (pattern if isinstance(pattern, list) else [pattern]):
            if step:
                next_targets = []
                for url, context in current:
                    next_targets.extend(self.links(url, step)[:self.config.max_por_familia])
                current = next_targets[:self.config.max_por_familia]
        return current

    def document(self, spec, url, context):
        original = self.http.get(url)
        result = extract(original, spec['fuente'], spec['tipo'], self.config.max_paginas,
                         qa=spec['adaptador'] == 'qa')
        is_ficha = (spec['fuente'] in ('NBER', 'SSRN', 'FMI eLibrary')
                    or re.search(r'ficha|autores, serie y resumen', spec['tipo'], re.I)
                    or re.search(r'/publ/work\d+\.htm|/publications/(?:working-paper|qr-)', url))
        if original.content[:4] != b'%PDF' and is_ficha:
            result.update(texto_completo=False, naturaleza_contenido='ficha/resumen')
        # A WP's landing page dates its specific linked PDF. Never pick an arbitrary
        # chapter or translate the most recent listing date to a fixed older PDF.
        if original.content[:4] != b'%PDF' and 'working paper' in spec['tipo'].lower() and spec['fuente'] in ('BIS', 'BCE'):
            page = soup(original.content)
            pdf_meta = page.find('meta', attrs={'name': 'citation_pdf_url'})
            pdf_url = pdf_meta.get('content') if pdf_meta else None
            if not pdf_url:
                pattern = r'/(?:publ/work\d+|pub/pdf/scpwps/[^/?#]+)\.pdf(?:\?|$)'
                node = page.select_one('main, #main-content, #content') or page
                anchor = next((a for a in node.select('a[href]') if re.search(pattern, a['href'], re.I)), None)
                pdf_url = anchor['href'] if anchor else None
            if pdf_url:
                pdf = self.http.get(urljoin(url, pdf_url))
                full = extract(pdf, spec['fuente'], spec['tipo'], self.config.max_paginas)
                if result.get('fecha_publicacion_raw'):
                    full.update(fecha_publicacion_raw=result['fecha_publicacion_raw'],
                                evidencia_publicacion=f'ficha {url}: {result["evidencia_publicacion"]}')
                full['document_page_url'] = url
                original, result = pdf, full
        # Release label is adjacent to this document, not the page's last update.
        if not result['fecha_publicacion_raw']:
            released = re.search(r'Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})', context)
            if released:
                result.update(fecha_publicacion_raw=released[1],
                              evidencia_publicacion=f'listado {spec["url"]}: {released[0]}')
            boj = re.search(r'PUBLICACION_BOJ=([^|]+)', context)
            if boj:
                result.update(fecha_publicacion_raw=boj[1].strip(),
                              evidencia_publicacion=f'columna Release Schedule del calendario {spec["url"]}: {boj[1].strip()}')
        meeting = re.search(r'EVENTO_BOJ=([^|]+)', context)
        if meeting:
            result.update(fecha_evento=parse_date(meeting[1].strip())[0],
                          evidencia_evento=f'columna Date of MPM del calendario: {meeting[1].strip()}')
        result['contexto_listado'] = context[:600]
        return original, result

    def table(self, spec, url, context):
        original = self.http.get(url)
        fmt = spec.get('formato', 'xlsx')
        if fmt == 'zip_csv':
            with zipfile.ZipFile(io.BytesIO(original.content)) as archive:
                name = next(n for n in archive.namelist() if n.lower().endswith(('.csv', '.txt')))
                lines = archive.read(name).decode('utf-8-sig').splitlines()
            # Only monthly records, not the later annual section of French's file.
            rows = [line for line in lines if re.match(r'\s*\d{6}\s*,', line)]
            if not rows:
                raise ValueError('CSV sin filas mensuales reconocibles')
            table = pd.read_csv(io.StringIO('\n'.join(rows)), header=None)
        else:
            book = pd.ExcelFile(io.BytesIO(original.content), engine='openpyxl')
            sheet = spec.get('hoja') or next((s for s in book.sheet_names if re.search(r'data|cris', s, re.I)), book.sheet_names[0])
            table = pd.read_excel(book, sheet_name=sheet, header=spec.get('header', 0)).dropna(how='all')
        if spec['fuente'] == 'ESRB' and {'Country', 'Start date'}.issubset(table.columns):
            table = table.loc[table['Country'].notna() & table['Start date'].notna()]
        if spec['fuente'] == 'Banco Mundial' and len(table.columns) > 2:
            table = table.loc[table.iloc[:, 2].notna()]
        if table.empty:
            raise ValueError('Tabla vacía')
        rows = table.head(self.config.max_filas).to_dict('records')
        reference = None
        first = rows[0]
        if spec['fuente'] == 'ESRB' and first.get('Start date') is not None:
            reference = str(first['Start date'])[:10]
        return original, {'titulo': spec['tipo'], 'formato': fmt, 'filas_total': len(table),
                          'ejemplo_datos': rows, 'texto': '', 'texto_completo': False,
                          'fecha_evento': reference,
                          'evidencia_evento': 'Start date: etiqueta retrospectiva de crisis' if reference else None,
                          'evidencia_publicacion': 'Vintage/fecha de publicación de las cifras no acreditado'}

    def bis_csv(self, spec, url, context):
        original = self.http.get(url)
        with zipfile.ZipFile(io.BytesIO(original.content)) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith('.csv'))
            table = pd.read_csv(io.BytesIO(archive.read(name)), nrows=self.config.max_filas)
        return original, {'titulo': spec['tipo'], 'formato': 'ZIP/CSV', 'texto': '',
                          'texto_completo': False, 'ejemplo_datos': table.to_dict('records'),
                          'evidencia_publicacion': 'Fecha del discurso y publicación del paquete son distintas; pendiente de contraste'}

    def api(self, spec):
        adapter = spec['adaptador']
        identifier = spec.get('identificador', '')
        if adapter == 'crossref':
            original = self.http.get('https://api.crossref.org/works/' + quote(identifier, safe='/'))
            data = json.loads(original.content)['message']
            field = next((k for k in ('published-online', 'published-print', 'published', 'issued') if data.get(k)), None)
            parts = data.get(field, {}).get('date-parts', [[]])[0]
            date = '-'.join(str(v).zfill(4 if i == 0 else 2) for i, v in enumerate(parts))
            return original, {'titulo': (data.get('title') or [identifier])[0], 'formato': 'JSON/metadatos',
                              'texto': '', 'texto_completo': False, 'fecha_publicacion_raw': date,
                              'evidencia_publicacion': f'Crossref {field}: fecha editorial, no apertura del PDF',
                              'ejemplo_datos': {k: data.get(k) for k in ('DOI', 'author', 'published-online', 'published-print', 'published', 'issued')}}
        if adapter == 'arxiv':
            # Pin the text to v1: published must never date the latest revised PDF.
            identifier = re.sub(r'v\d+$', '', identifier) + 'v1'
            original = self.http.get('https://export.arxiv.org/api/query', params={'id_list': identifier, 'max_results': 1})
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            entry = ET.fromstring(original.content).find('a:entry', ns)
            if entry is None or '/errors' in (entry.findtext('a:id', '', ns) or ''):
                raise ValueError('arXiv no devuelve esa versión')
            if not entry.findtext('a:id', '', ns).endswith('/' + identifier):
                raise ValueError('arXiv no confirma la versión v1 solicitada; no se acredita el texto')
            pdf = self.http.get('https://arxiv.org/pdf/' + identifier)
            if not pdf.content.startswith(b'%PDF'):
                raise ValueError('arXiv devolvió contenido distinto del PDF solicitado')
            result = extract(pdf, 'arXiv', spec['tipo'], self.config.max_paginas)
            result.update(fecha_publicacion_raw=entry.findtext('a:published', namespaces=ns),
                          evidencia_publicacion=f'Atom published y PDF fijado a {identifier}',
                          version_acreditada=True, identificador=identifier,
                          fecha_actualizacion=entry.findtext('a:updated', namespaces=ns))
            return pdf, result
        if adapter == 'huggingface':
            headers = {'Authorization': 'Bearer ' + self.config.hf_token} if self.config.hf_token else None
            base = 'https://datasets-server.huggingface.co'
            original = self.http.get(base + '/splits', params={'dataset': identifier}, headers=headers)
            splits = json.loads(original.content)['splits']
            split = next((s for s in splits if s['config'] == spec.get('config')), splits[0])
            original = self.http.get(base + '/rows', params={'dataset': identifier, 'config': split['config'],
                                      'split': split['split'], 'offset': 0, 'length': self.config.max_filas}, headers=headers)
            rows = [r['row'] for r in json.loads(original.content)['rows']]
            if not rows:
                raise ValueError('Dataset sin filas')
            return original, {'titulo': identifier, 'formato': 'JSON/dataset', 'texto': '',
                              'texto_completo': False, 'ejemplo_datos': rows,
                              'dataset': identifier, 'config_dataset': split['config'], 'split': split['split'],
                              'fecha_publicacion_raw': rows[0].get('publication_date') or rows[0].get('published_at'),
                              'evidencia_publicacion': 'Campo del dataset, sin acreditar disponibilidad/versión original'}
        if adapter == 'unpaywall':
            if not self.config.unpaywall_email:
                raise ValueError('UNPAYWALL_EMAIL pendiente; completar variable de entorno con email real')
            original = self.http.get('https://api.unpaywall.org/v2/' + quote(identifier, safe='/'),
                                     params={'email': self.config.unpaywall_email})
            data = json.loads(original.content)
            return original, {'titulo': data.get('title'), 'formato': 'JSON/metadatos', 'texto': '',
                              'texto_completo': False, 'fecha_publicacion_raw': data.get('published_date'),
                              'evidencia_publicacion': 'Fecha editorial; no acredita fecha de apertura de la copia',
                              'ejemplo_datos': {k: data.get(k) for k in ('doi', 'is_oa', 'best_oa_location')}}
        if adapter == 'weo':
            original = self.http.get(spec['url'], params={'periods': '2025'})
            data = json.loads(original.content)
            return original, {'titulo': spec['tipo'], 'formato': 'JSON/serie', 'texto': '', 'texto_completo': False,
                              'ejemplo_datos': {'pais': 'ESP', 'periodo': '2025', 'unidad': '% anual',
                                               'valor': data['values']['NGDP_RPCH']['ESP']['2025']},
                              'evidencia_publicacion': 'DataMapper actual: vintage y publicación del punto no acreditados'}
        raise ValueError(f'Adaptador desconocido: {adapter}')
