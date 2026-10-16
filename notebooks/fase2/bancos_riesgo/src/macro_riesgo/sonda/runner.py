"""Orquesta muestras independientes y conserva texto, versiones y mediciones."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import re
import time

import pandas as pd
import requests

from .collectors import Collector
from .http import HttpClient
from .parsing import application_date, chunks, event_date, publication_clock
from .temporal import POLICY_VERSION, add_metrics, available_at, parse_date

ZONES = {'Fed': 'America/New_York', 'BCE': 'Europe/Berlin', 'BoE': 'Europe/London',
         'BoJ': 'Asia/Tokyo', 'BIS': 'Europe/Zurich', 'ESRB': 'Europe/Berlin',
         'Banco de España': 'Europe/Madrid', 'EUR-Lex': 'Europe/Brussels'}


@dataclass(frozen=True)
class ConfigSonda:
    raiz: Path
    max_por_familia: int = 3
    max_paginas: int = 40
    max_filas: int = 3
    usar_cache: bool = True
    corte_utc: str = '2026-10-14T00:00:00+00:00'
    hf_token: str = field(default_factory=lambda: os.getenv('HF_TOKEN', ''), repr=False)
    unpaywall_email: str = field(default_factory=lambda: os.getenv('UNPAYWALL_EMAIL', ''), repr=False)

    def __post_init__(self):
        if min(self.max_por_familia, self.max_paginas, self.max_filas) < 1:
            raise ValueError('Los límites de la sonda deben ser positivos')


class Sonda:
    def __init__(self, config):
        self.config = config
        self.inventory = json.loads((config.raiz / 'config/sources/sonda_fuentes.json').read_text(encoding='utf-8'))
        self.http = HttpClient(config.raiz / 'data/cache/sonda', config.usar_cache)
        self.collector = Collector(self.http, config)
        self.results = {}
        self.documents = {}

    def inventario(self, apartado):
        return pd.DataFrame(s for s in self.inventory if s['apartado'] == apartado).fillna('')

    @staticmethod
    def _base(spec, url):
        return {'apartado': spec['apartado'], 'fuente': spec['fuente'],
                'tipo': spec.get('identificador') if spec['adaptador'] == 'huggingface' else spec['tipo'],
                'url': url, 'estado': 'OK', 'error': '', 'fecha_publicacion': None,
                'precision_publicacion': 'desconocida', 'publicacion_utc': None,
                'fecha_publicacion_raw': None, 'evidencia_publicacion': '',
                'fecha_evento': None, 'evidencia_evento': '', 'fecha_aplicacion': None,
                'evidencia_aplicacion': '', 'texto_completo': False, 'version_acreditada': False,
                'disponible_desde_utc': None, 'palabras': 0, 'bytes': 0, 'bytes_gzip': 0,
                'fragmentos': 0, 'segundos': 0, 'consulta_utc': datetime.now(timezone.utc).isoformat(),
                'policy_version': POLICY_VERSION, 'zona_fuente': ZONES.get(spec['fuente']),
                'titulo': spec['tipo'], 'version_id': None, 'sha256': None, 'retrieved_at': None,
                'cache': False, 'formato': None}

    def _normalize(self, spec, original, extracted, started):
        record = self._base(spec, original.url)
        record.update({k: v for k, v in extracted.items() if k not in ('texto', 'ejemplo_datos')})
        text = extracted.get('texto', '')
        raw = publication_clock(record.get('fecha_publicacion_raw'), text, spec['fuente'])
        normalized, precision, utc = parse_date(raw)
        event, evidence = event_date(original.url, text, spec['fuente'], spec['tipo'])
        if spec.get('aplicacion_relevante'):
            application, application_evidence = application_date(text)
        else:
            application, application_evidence = None, 'No procede para esta familia; no se fecha la aplicación de políticas citadas dentro de un informe/paper'
        if event and not record.get('fecha_evento'):
            record.update(fecha_evento=event, evidencia_evento=evidence)
        record.update(fecha_publicacion=normalized, precision_publicacion=precision, publicacion_utc=utc,
                      fecha_aplicacion=application, evidencia_aplicacion=application_evidence,
                      retrieved_at=original.retrieved_at, sha256=original.sha256,
                      version_id=original.sha256, cache=original.cache,
                      bytes=len(original.content), bytes_gzip=len(gzip.compress(original.content)),
                      palabras=len(re.findall(r'\b\w+\b', text)), segundos=time.monotonic() - started)
        if raw != record.get('fecha_publicacion_raw'):
            record['evidencia_publicacion'] += ' + aviso For release at con EST/EDT'
        if spec['apartado'] == 14:
            record.update(fecha_evento='2008-09-15', evidencia_evento='Corte ilustrativo explícito GFC; no fecha de aplicación')
        record['disponible_desde_utc'] = available_at(record)
        record['estado_historico'] = ('acreditado' if record['disponible_desde_utc'] else
                                     'pendiente_versión' if normalized else 'sin_fecha')
        segments = chunks(text, original.sha256)
        record['fragmentos'] = len(segments)
        document = dict(record, texto=text, fragmentos_texto=segments,
                        ejemplo_datos=extracted.get('ejemplo_datos'))
        return record, document

    @staticmethod
    def _error(spec, url, error, started):
        record = Sonda._base(spec, url)
        # HTTP exceptions may contain personal query parameters; retain status only.
        if isinstance(error, requests.HTTPError):
            message = f'HTTP {error.response.status_code}; revisar acceso oficial'
        elif isinstance(error, requests.RequestException):
            message = f'Error de red: {type(error).__name__}'
        else:
            message = str(error)[:350]
        record.update(estado='ERROR_ACCESO' if isinstance(error, requests.RequestException) else 'PENDIENTE',
                      error=message, segundos=time.monotonic() - started, estado_historico='sin_muestra')
        return record

    def ejecutar(self, apartado):
        """Reejecutar un apartado sustituye sus resultados, sin duplicar registros."""
        specs = [s for s in self.inventory if s['apartado'] == apartado]
        if not specs:
            raise ValueError(f'Apartado desconocido: {apartado}')
        records, documents = [], []
        for spec in specs:
            started = time.monotonic()
            adapter = spec['adaptador']
            try:
                targets = (self.collector.targets(spec) if adapter in ('documento', 'qa', 'tabla', 'bis_csv')
                           else [(spec.get('url', spec.get('identificador')), '')])
            except Exception as error:
                records.append(self._error(spec, spec.get('url'), error, started))
                continue
            for url, context in targets:
                started = time.monotonic()
                try:
                    if adapter in ('documento', 'qa'):
                        original, extracted = self.collector.document(spec, url, context)
                    elif adapter == 'tabla':
                        original, extracted = self.collector.table(spec, url, context)
                    elif adapter == 'bis_csv':
                        original, extracted = self.collector.bis_csv(spec, url, context)
                    else:
                        original, extracted = self.collector.api(spec)
                    record, document = self._normalize(spec, original, extracted, started)
                    records.append(record)
                    documents.append(document)
                except Exception as error:
                    records.append(self._error(spec, url, error, started))
        frame = add_metrics(pd.DataFrame(records))
        self.results[apartado] = frame
        self.documents[apartado] = documents
        corpus = self.config.raiz / 'data/local/sonda'
        corpus.mkdir(parents=True, exist_ok=True)
        with (corpus / f'apartado_{apartado:02}.jsonl').open('w', encoding='utf-8') as stream:
            for document in documents:
                stream.write(json.dumps(document, ensure_ascii=False, default=str) + '\n')
                versions = corpus / 'versions'
                versions.mkdir(exist_ok=True)
                version_path = versions / (document['version_id'] + '.json')
                if not version_path.exists():
                    version_path.write_text(json.dumps(document, ensure_ascii=False, default=str), encoding='utf-8')
        return frame

    def guardar_tablas(self):
        directory = self.config.raiz / 'data/exports/sonda'
        directory.mkdir(parents=True, exist_ok=True)
        for section, frame in self.results.items():
            frame.to_csv(directory / f'apartado_{section:02}.csv', index=False, encoding='utf-8-sig')
        return directory

    def resumen(self):
        rows = []
        for section, frame in sorted(self.results.items()):
            ok = frame.loc[frame['estado'].eq('OK')]
            rows.append({'apartado': section, 'intentos': len(frame), 'muestras_OK': len(ok),
                         'fallos_pendientes': len(frame) - len(ok),
                         'con_fecha': int(ok['precision_publicacion'].isin(['día', 'hora', 'hora_sin_zona']).sum()),
                         'con_retraso_evento': int(ok['retraso_evento_dias'].notna().sum()),
                         'con_aplicacion': int(ok['retraso_aplicacion_dias'].notna().sum()),
                         'versiones_acreditadas': int(ok['disponible_desde_utc'].notna().sum()),
                         'MB_originales': ok.drop_duplicates('sha256')['bytes'].sum() / 1e6})
        return pd.DataFrame(rows)
