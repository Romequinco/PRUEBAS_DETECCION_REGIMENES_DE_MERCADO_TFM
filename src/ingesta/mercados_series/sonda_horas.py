"""Sondas de horas de publicación (bloque 2): cuándo aparece en FRED/ALFRED el dato del día D.

- ``primeras_publicaciones``: primera ``realtime_start`` de cada observación (ALFRED) → día real en
  que FRED publicó el dato de D (H.15, H.10, Moody's...).
- ``fechas_release``: calendario de un release de FRED (``fred/release/dates``).
- ``historial_en_vintage``: cobertura de una serie tal como estaba en un vintage (OAS de ICE BofA
  antes del capado de abril de 2026).
- ``hora_anunciada``: hora de publicación que declara la página oficial de un release de la Fed.

Solo se guardan FECHAS y recuentos, nunca valores (términos de FRED). La clave ``FRED_API_KEY``
sale de ``.env`` y nunca se imprime (``http.sin_clave`` en los mensajes).
CLI: ``python -m ingesta.mercados_series.sonda_horas`` (unas 15 peticiones, ~1 por segundo).
This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ingesta.comun import http
from ingesta.comun.rutas import env, resultados_dir
from ingesta.mercados_series import visibilidad

API_FRED = "https://api.stlouisfed.org/fred"
TEMA = "mercados_series"
COLUMNAS = ("sonda", "serie", "fecha_referencia", "fecha_publicacion_fred", "dias_naturales",
            "dias_habiles_us", "dia_semana_publicacion", "detalle")
_DIAS = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")


def ruta_csv() -> Path:
    """``results/ingesta/mercados_series/tablas/sondas_horas_bloque2.csv``."""
    carpeta = resultados_dir(TEMA) / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / "sondas_horas_bloque2.csv"


def _fred(endpoint: str, **params) -> dict:
    """GET a la API de FRED con la clave de ``.env`` (1 petición/s por host)."""
    clave = env("FRED_API_KEY")
    if not clave:
        raise RuntimeError("falta FRED_API_KEY en .env o en el entorno")
    params = {**params, "api_key": clave, "file_type": "json"}
    try:
        return http.get(f"{API_FRED}/{endpoint}", params=params, por_segundo=1.0).json()
    except Exception as e:  # el mensaje puede llevar la URL con la clave
        raise RuntimeError(http.sin_clave(f"FRED {endpoint}: {e}")) from None


def dias_habiles_entre(desde: pd.Timestamp, hasta: pd.Timestamp) -> int:
    """Días hábiles us_federal en ``(desde, hasta]`` (0 si ``hasta <= desde``)."""
    n, d = 0, desde
    while d < hasta:
        d += pd.Timedelta(days=1)
        n += visibilidad.es_habil(d, "us_federal")
    return n


def _fila(sonda: str, serie: str, ref: pd.Timestamp | None, pub: pd.Timestamp | None,
          detalle: str = "") -> dict:
    nat = hab = None
    if ref is not None and pub is not None:
        nat = int((pub - ref).days)
        hab = dias_habiles_entre(ref, pub)
    return {"sonda": sonda, "serie": serie,
            "fecha_referencia": None if ref is None else ref.date().isoformat(),
            "fecha_publicacion_fred": None if pub is None else pub.date().isoformat(),
            "dias_naturales": nat, "dias_habiles_us": hab,
            "dia_semana_publicacion": None if pub is None else _DIAS[pub.weekday()], "detalle": detalle}


def primeras_publicaciones(serie: str, obs_desde: str, obs_hasta: str, rt_desde: str,
                           sonda: str = "alfred_realtime_start") -> list[dict]:
    """Primera ``realtime_start`` (ALFRED) de cada observación de ``serie`` en ``[obs_desde, obs_hasta]``.

    Se piden todos los vintages desde ``rt_desde``; una observación cuya primera ``realtime_start``
    es justo ``rt_desde`` ya existía antes (truncada) y se descarta. Sin valores.
    """
    datos = _fred("series/observations", series_id=serie, observation_start=obs_desde,
                  observation_end=obs_hasta, realtime_start=rt_desde, realtime_end="9999-12-31")
    obs = pd.DataFrame(datos.get("observations", []))
    if obs.empty:
        return [_fila(sonda, serie, None, None, "sin observaciones")]
    obs = obs[obs["value"] != "."]
    primera = obs.groupby("date")["realtime_start"].min().sort_index()
    filas = []
    for ref, pub in primera.items():
        if pub == rt_desde:
            continue
        filas.append(_fila(sonda, serie, pd.Timestamp(ref), pd.Timestamp(pub)))
    return filas


def fechas_release(release_id: int, desde: str, hasta: str) -> list[dict]:
    """Fechas de publicación del release ``release_id`` entre ``desde`` y ``hasta``."""
    datos = _fred("release/dates", release_id=release_id, realtime_start=desde, realtime_end=hasta,
                  include_release_dates_with_no_data="true", sort_order="asc", limit=1000)
    return [_fila("release_dates", f"release:{release_id}", None, pd.Timestamp(d["date"]))
            for d in datos.get("release_dates", [])]


def historial_en_vintage(serie: str, vintage: str) -> list[dict]:
    """Cobertura de ``serie`` en el vintage ``vintage``, por ``vintage_dates`` y por realtime.

    Devuelve dos filas (una por método) con la primera observación como ``fecha_referencia``,
    el propio vintage como ``fecha_publicacion_fred`` y en ``detalle`` el número de observaciones
    y la última fecha. Sin valores.
    """
    filas = []
    for metodo, params in (("vintage_dates", {"vintage_dates": vintage}),
                           ("realtime", {"realtime_start": vintage, "realtime_end": vintage})):
        try:
            datos = _fred("series/observations", series_id=serie, **params)
        except RuntimeError as e:
            filas.append(_fila(f"alfred_{metodo}", serie, None, pd.Timestamp(vintage), f"error: {e}"))
            continue
        obs = [o for o in datos.get("observations", []) if o.get("value") != "."]
        if not obs:
            filas.append(_fila(f"alfred_{metodo}", serie, None, pd.Timestamp(vintage), "n_obs=0"))
            continue
        filas.append(_fila(f"alfred_{metodo}", serie, pd.Timestamp(obs[0]["date"]), pd.Timestamp(vintage),
                           f"n_obs={len(obs)}; ultima={obs[-1]['date']}"))
    return filas


def cobertura_actual(serie: str) -> dict:
    """Primera y última observación de ``serie`` hoy (sin valores)."""
    datos = _fred("series/observations", series_id=serie)
    obs = [o for o in datos.get("observations", []) if o.get("value") != "."]
    if not obs:
        return _fila("fred_actual", serie, None, None, "n_obs=0")
    return _fila("fred_actual", serie, pd.Timestamp(obs[0]["date"]), None,
                 f"n_obs={len(obs)}; ultima={obs[-1]['date']}")


_PATRON_HORA = re.compile(r"\b(\d{1,2}:\d{2})\s*(a\.?m\.?|p\.?m\.?)", re.IGNORECASE)


def hora_anunciada(url: str, nombre: str) -> dict:
    """Horas ``h:mm a.m./p.m.`` que cita la página oficial ``url`` (respeta robots.txt)."""
    if not http.robots_permite(url):
        return _fila("pagina_oficial", nombre, None, None, f"robots.txt no permite {url}")
    try:
        texto = http.get(url).text
    except Exception as e:
        return _fila("pagina_oficial", nombre, None, None, f"error: {http.sin_clave(str(e))}")
    horas = sorted({f"{h} {ap.lower().replace('.', '')}" for h, ap in _PATRON_HORA.findall(texto)})
    contexto = []
    for m in _PATRON_HORA.finditer(texto):
        trozo = re.sub(r"<[^>]+>", " ", texto[max(0, m.start() - 160):m.end() + 40])
        contexto.append(re.sub(r"\s+", " ", trozo).strip())
    return _fila("pagina_oficial", nombre, None, None,
                 f"horas={horas}; contexto={contexto[:2]}; url={url}")


def vintages_disponibles(serie: str) -> dict:
    """Primer y último vintage que ALFRED conserva de ``serie`` (``fred/series/vintagedates``)."""
    try:
        datos = _fred("series/vintagedates", series_id=serie, limit=10000)
    except RuntimeError as e:
        return _fila("alfred_vintagedates", serie, None, None, f"error: {e}")
    fechas = datos.get("vintage_dates", [])
    if not fechas:
        return _fila("alfred_vintagedates", serie, None, None, "n_vintages=0")
    return _fila("alfred_vintagedates", serie, pd.Timestamp(fechas[0]), pd.Timestamp(fechas[-1]),
                 f"n_vintages={len(fechas)} (fecha_referencia = primer vintage, publicacion = último)")


def ultima_actualizacion(serie: str) -> dict:
    """``last_updated`` de FRED (fecha y hora con huso): una muestra de la hora real de carga."""
    datos = _fred("series", series_id=serie)
    info = (datos.get("seriess") or [{}])[0]
    marca = info.get("last_updated")
    if not marca:
        return _fila("fred_last_updated", serie, None, None, "sin last_updated")
    t = pd.Timestamp(marca)
    ny = t.tz_convert("America/New_York") if t.tzinfo is not None else t
    return _fila("fred_last_updated", serie, None, ny.tz_localize(None).normalize(),
                 f"last_updated={marca}; hora_NY={ny:%H:%M}; ultima_obs={info.get('observation_end')}")


def metadatos_fred(ids: Iterable[str]) -> pd.DataFrame:
    """Metadatos de FRED (``fred/series``) de cada id: título, unidades, frecuencia, cobertura y
    última actualización. Un id que FRED ya no sirve queda con ``error`` (p. ej. HTTP 400). Sin valores.
    """
    filas = []
    for sid in ids:
        try:
            info = (_fred("series", series_id=sid).get("seriess") or [{}])[0]
            filas.append({"id_origen": sid, "titulo": info.get("title"), "unidades": info.get("units"),
                          "frecuencia": info.get("frequency"), "ajuste": info.get("seasonal_adjustment_short"),
                          "cobertura_inicio": info.get("observation_start"),
                          "cobertura_fin": info.get("observation_end"), "last_updated": info.get("last_updated"),
                          "error": None})
        except RuntimeError as e:
            filas.append({"id_origen": sid, "error": str(e)[:200]})
    return pd.DataFrame(filas)


def ejecutar(series_h15: Iterable[str] = ("DGS10", "DTB3", "DFF"),
             series_moodys: Iterable[str] = ("BAA10Y", "DBAA"),
             series_oas: Iterable[str] = ("BAMLH0A0HYM2", "BAMLC0A0CM"),
             vintages_oas: Iterable[str] = ("2026-03-31", "2023-01-03", "2019-01-02"),
             series_hora: Iterable[str] = ("DGS10", "DTB3", "DFF", "EFFR", "SOFR", "OBFR", "DBAA", "DAAA",
                                           "BAA10Y", "T10Y2Y", "T10Y3M", "T10YFF", "T5YFF", "T10YIE",
                                           "T5YIE", "T5YIFR", "DFII10", "IORB", "DPCREDIT", "DCPF3M",
                                           "CPFF", "DEXUSEU", "DTWEXBGS", "VIXCLS", "NASDAQ100",
                                           "BAMLH0A0HYM2", "NFCI", "STLFSI4", "ICSA", "WEI"),
             series_calculadas: Iterable[str] = ("T10Y2Y", "T10YIE", "CPFF"),
             hoy: pd.Timestamp | None = None) -> pd.DataFrame:
    """Todas las sondas del bloque 2; devuelve la tabla (solo fechas y recuentos)."""
    hoy = (hoy or pd.Timestamp.today()).normalize()
    obs_desde = (hoy - pd.Timedelta(days=24)).date().isoformat()
    obs_hasta = (hoy - pd.Timedelta(days=1)).date().isoformat()
    rt_desde = (hoy - pd.Timedelta(days=31)).date().isoformat()
    filas: list[dict] = []
    for s in series_h15:
        filas += primeras_publicaciones(s, obs_desde, obs_hasta, rt_desde, "h15_realtime_start")
    h10_desde = (hoy - pd.Timedelta(days=45)).date().isoformat()
    filas += primeras_publicaciones("DEXUSEU", h10_desde, obs_hasta,
                                    (hoy - pd.Timedelta(days=52)).date().isoformat(), "h10_realtime_start")
    for s in series_moodys:
        filas += primeras_publicaciones(s, obs_desde, obs_hasta, rt_desde, "moodys_realtime_start")
    for s in series_calculadas:
        filas += primeras_publicaciones(s, obs_desde, obs_hasta, rt_desde, "calculada_realtime_start")
    for rid in (17, 18):
        filas += fechas_release(rid, "2025-01-01", hoy.date().isoformat())
    for s in series_oas:
        filas.append(cobertura_actual(s))
        filas.append(vintages_disponibles(s))
        for vintage in vintages_oas:
            filas += historial_en_vintage(s, vintage)
    for s in series_hora:
        filas.append(ultima_actualizacion(s))
    filas.append(hora_anunciada("https://www.federalreserve.gov/releases/h15/", "fed_h15"))
    filas.append(hora_anunciada("https://www.federalreserve.gov/releases/h10/", "fed_h10"))
    filas.append(hora_anunciada("https://www.newyorkfed.org/markets/reference-rates/sofr", "nyfed_sofr"))
    filas.append(hora_anunciada("https://www.newyorkfed.org/markets/reference-rates/effr", "nyfed_effr"))
    return pd.DataFrame(filas, columns=list(COLUMNAS))


def main() -> None:
    tabla = ejecutar()
    ruta = ruta_csv()
    tabla.to_csv(ruta, index=False, encoding="utf-8")
    print(f"{len(tabla)} filas -> {ruta}")
    resumen = tabla.dropna(subset=["dias_habiles_us"]).groupby("sonda")["dias_habiles_us"]
    print(resumen.value_counts().to_string())


if __name__ == "__main__":
    main()
