"""Fechas con precisión explícita y filtro histórico conservador por versión."""

from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo

import pandas as pd

POLICY_VERSION = 'sonda-timegate-v1'


def parse_date(raw):
    """Devuelve fecha ISO, precisión y timestamp UTC; nunca completa mes/año."""
    raw = str(raw or '').strip()
    if not raw:
        return None, 'desconocida', None
    raw = re.sub(r'^Published on\s+', '', raw, flags=re.I)
    if re.fullmatch(r'\d{4}[-/]\d{1,2}', raw) or re.fullmatch(r'[A-Za-z]+ \d{4}', raw):
        return raw, 'mes', None
    if re.fullmatch(r'\d{4}', raw):
        return raw, 'año', None
    try:
        parsed = pd.Timestamp(raw)
    except (ValueError, TypeError):
        return raw, 'sin_normalizar', None
    if parsed.tzinfo is not None:
        return parsed.isoformat(), 'hora', parsed.tz_convert('UTC').isoformat()
    # A clock without a timezone cannot support an intraday gate.
    if re.search(r'\d{1,2}:\d{2}', raw):
        return parsed.isoformat(), 'hora_sin_zona', None
    return parsed.date().isoformat(), 'día', None


def available_at(record):
    """No equivale a publicación: requiere acreditar la versión textual actual."""
    if not record.get('version_acreditada') or not record.get('texto_completo'):
        return None
    if record.get('precision_publicacion') == 'hora':
        return record.get('publicacion_utc')
    if record.get('precision_publicacion') == 'día' and record.get('zona_fuente'):
        # Explicit conservative policy: usable only from the following local day.
        day = datetime.fromisoformat(record['fecha_publicacion'][:10])
        bound = (day + timedelta(days=1)).replace(tzinfo=ZoneInfo(record['zona_fuente']))
        return pd.Timestamp(bound).tz_convert('UTC').isoformat()
    return None


def vista_a_fecha(frame, cutoff):
    """Texto completo acreditado de esa versión, disponible al corte UTC."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError('El corte debe incluir zona horaria (por ejemplo +00:00)')
    available = pd.to_datetime(frame['disponible_desde_utc'], utc=True, errors='coerce', format='mixed')
    eligible = (frame['estado'].eq('OK') & frame['version_acreditada'].fillna(False)
                & frame['texto_completo'].fillna(False) & available.notna() & available.le(cutoff))
    return frame.loc[eligible].copy()


def add_metrics(frame):
    """Lags de calendario firmados: una aplicación futura puede producir negativos."""
    frame = frame.copy()
    for target, reference in [('retraso_evento_dias', 'fecha_evento'),
                              ('retraso_aplicacion_dias', 'fecha_aplicacion')]:
        # Calendar lags use the source's local date, not the date after UTC conversion.
        pub_raw = frame['fecha_publicacion'].astype('string').str[:10]
        ref_raw = frame[reference].astype('string').str[:10]
        full_day = r'\d{4}-\d{2}-\d{2}'
        pub = pd.to_datetime(pub_raw.where(pub_raw.str.fullmatch(full_day)), errors='coerce', format='mixed')
        ref = pd.to_datetime(ref_raw.where(ref_raw.str.fullmatch(full_day)), errors='coerce', format='mixed')
        # Partial dates must not silently turn into the first of a month/year.
        valid = frame['precision_publicacion'].isin(['día', 'hora', 'hora_sin_zona']) & ref.notna()
        frame[target] = (pub.dt.normalize() - ref.dt.normalize()).dt.days.where(valid)
    return frame
