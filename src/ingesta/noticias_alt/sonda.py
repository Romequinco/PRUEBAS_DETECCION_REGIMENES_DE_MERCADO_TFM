"""Sonda del Tema 4: orquesta los extractores y deja las tablas de medición en ``results``.

Cada módulo de ``MODULOS`` expone ``sondear(cfg) -> base.Resultado``; ``ejecutar`` los llama en orden
(un módulo que falla no para la sonda) y junta:

- ``registro_descargas``, ``cobertura`` y ``timegate`` (columnas de ``base``, más ``modulo``);
- ``tablas``: las de cada módulo con su nombre, más las de limpieza (``limpieza_embudo``,
  ``limpieza_idiomas``, ``limpieza_sindicacion``) calculadas sobre los titulares del corpus;
- ``fichas`` (una fila por ``FichaFuente``), ``series`` (resumen de cada parquet) y ``documentos``
  (nº de documentos por jsonl);
- ``volumen`` (muestra medida y extrapolación al corpus completo) y ``fuentes_resumen``.

``guardar`` escribe todo en ``results/ingesta/noticias_alt`` (en ``.../rapido`` con ``--rapido``, para no
pisar la pasada completa) y las fichas con ``ingesta.comun.fichas.guardar_ficha``.

Copyright: a ``results`` solo van metadatos, cifras y titulares cortos (``MAX_TITULAR`` caracteres).

Uso: ``python -m ingesta.noticias_alt.sonda [--rapido] [--sin-cache]``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import pandas as pd

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.comun.rutas import resultados_dir
from ingesta.noticias_alt import (base, ccnews, congreso, ews, gdelt, hf_datasets, indices, insiders, prediccion,
                                  prensa_api, rss, social, wikipedia)
from ingesta.noticias_alt.limpieza import (_fecha_doc, _medio_doc, deduplicar, embudo_duplicados)

# Orden: texto histórico, eventos y atención, vivo y redes, divulgaciones, predicción, índices, EWS.
MODULOS: list[ModuleType] = [ccnews, hf_datasets, prensa_api, gdelt, wikipedia, rss, social, congreso, insiders,
                             prediccion, indices, ews]

# Titulares que entran en el embudo de duplicados (jsonl del corpus, sin sufijo ``_rapido``).
DOCS_EMBUDO: tuple[str, ...] = ("ccnews_muestra", "gdelt_doc_titulares", "rss_titulares", "bing_news_titulares",
                                "gdelt_gkg_titulares")
# GKG trae ~10.000 titulares por día; el embudo es cuadrático: se toma una muestra fija por doc_id.
TOPE_DIA: dict[str, int] = {"gdelt_gkg_titulares": 500}
MAX_TITULAR = base.MAX_TITULAR
SINDICADOS_POR_DIA = 15
MIN_MB_VELOCIDAD = 0.1     # MB de red mínimos para medir la velocidad de descarga de una fuente

COLUMNAS_VOLUMEN = ["fuente", "n_items_muestra", "mb_muestra", "gb_corpus_estimado", "horas_descarga_estimadas",
                    "n_peticiones", "n_peticiones_red", "mb_s_medidos", "nota"]
COLUMNAS_RESUMEN = ["fuente", "nombre", "familia", "tipo", "estado_sonda", "cobertura_inicio", "cobertura_fin",
                    "retraso_publicacion", "lag_mediana_h", "destino", "backtest", "licencia", "agentes", "modulo"]


@dataclass
class ResultadoSonda:
    """Todo lo medido por la sonda (DataFrames listos para ``guardar``)."""

    registro_descargas: pd.DataFrame
    cobertura: pd.DataFrame
    timegate: pd.DataFrame
    tablas: dict[str, pd.DataFrame]
    fichas: pd.DataFrame
    series: pd.DataFrame
    documentos: pd.DataFrame
    modulos: pd.DataFrame
    volumen: pd.DataFrame = field(default_factory=pd.DataFrame)
    fuentes_resumen: pd.DataFrame = field(default_factory=pd.DataFrame)
    fichas_obj: list[FichaFuente] = field(default_factory=list)
    rapido: bool = False


def _nombre(m: Any) -> str:
    return str(getattr(m, "__name__", m)).rsplit(".", 1)[-1]


# --------------------------------------------------------------------------- ejecución

def ejecutar(cfg: base.ConfigSonda | None = None, modulos: Iterable[Any] | None = None) -> ResultadoSonda:
    """Llama a ``sondear`` de cada módulo y junta los resultados (sin escribir en ``results``)."""
    cfg = cfg or base.ConfigSonda()
    registro: list[dict] = []
    cobertura: list[dict] = []
    timegate: list[dict] = []
    tablas: dict[str, pd.DataFrame] = {}
    fichas: list[FichaFuente] = []
    fichas_mod: list[str] = []
    series: list[dict] = []
    documentos: list[dict] = []
    estado_mod: list[dict] = []
    for m in (MODULOS if modulos is None else list(modulos)):
        nombre = _nombre(m)
        t0 = base.cronometro()
        cfg.log(f"=== {nombre}")
        try:
            r: base.Resultado = m.sondear(cfg)
        except Exception as e:  # noqa: BLE001 — un módulo roto no para la sonda
            estado_mod.append(dict(modulo=nombre, ok=False, segundos=base.segundos(t0),
                                   error=f"{type(e).__name__}: {str(e)[:300]}"))
            cfg.log(f"  [{nombre}] FALLO {type(e).__name__}: {str(e)[:200]}")
            continue
        registro += [dict(r_, modulo=nombre) for r_ in r.registro]
        cobertura += [dict(c, modulo=nombre) for c in r.cobertura]
        timegate += [dict(t, modulo=nombre) for t in r.timegate]
        for k, t in r.tablas.items():
            tablas[k if k not in tablas else f"{nombre}_{k}"] = t
        fichas += r.fichas
        fichas_mod += [nombre] * len(r.fichas)
        series += [dict(modulo=nombre, serie=s) for s in r.series]
        documentos += [dict(modulo=nombre, jsonl=k, n_documentos=int(n)) for k, n in r.documentos.items()]
        estado_mod.append(dict(modulo=nombre, ok=True, segundos=base.segundos(t0), error="",
                               n_peticiones=len(r.registro), n_fichas=len(r.fichas), n_tablas=len(r.tablas),
                               n_series=len(r.series), n_jsonl=len(r.documentos)))
        cfg.log(f"  [{nombre}] {len(r.fichas)} fichas, {len(r.tablas)} tablas, {len(r.series)} series, "
                f"{sum(r.documentos.values())} documentos ({base.segundos(t0)} s)")

    res = ResultadoSonda(
        registro_descargas=_df(registro, base.COLUMNAS_REGISTRO + ["modulo"]),
        cobertura=_df(cobertura, base.COLUMNAS_COBERTURA + ["modulo"]),
        timegate=_df(timegate, base.COLUMNAS_TIMEGATE + ["modulo"]),
        tablas=tablas,
        fichas=_tabla_fichas(fichas, fichas_mod),
        series=resumen_series(series),
        documentos=_df(documentos, ["modulo", "jsonl", "n_documentos"]),
        modulos=pd.DataFrame(estado_mod),
        fichas_obj=fichas,
        rapido=cfg.modo_rapido,
    )
    try:
        tablas.update(limpieza_corpus(cfg))
    except Exception as e:  # noqa: BLE001
        cfg.log(f"  [limpieza] FALLO {type(e).__name__}: {str(e)[:200]}")
    res.volumen = volumen(res, conteo_items(res, cfg))
    res.fuentes_resumen = fuentes_resumen(res)
    return res


def _df(filas: list[dict], columnas: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(filas)
    extra = [c for c in df.columns if c not in columnas]
    return df.reindex(columns=columnas + extra)


def _tabla_fichas(fichas: list[FichaFuente], modulos: list[str]) -> pd.DataFrame:
    filas = []
    for f, m in zip(fichas, modulos):
        d = asdict(f)
        d["modulo"] = m
        d["agentes"] = ",".join(d["agentes"])
        d["notas"] = " | ".join(map(str, d["notas"]))
        d["mediciones"] = json.dumps(d["mediciones"], ensure_ascii=False, default=str)
        filas.append(d)
    cols = [x.name for x in FichaFuente.__dataclass_fields__.values()] + ["modulo"]
    return pd.DataFrame(filas).reindex(columns=cols)


def resumen_series(series: list[dict]) -> pd.DataFrame:
    """Una fila por parquet del corpus: filas, series internas, fechas y lag de publicación."""
    filas = []
    for s in series:
        df = base.leer_serie(s["serie"])
        fila = dict(s, n_filas=len(df), n_series=int(df["serie"].nunique()) if len(df) else 0)
        if len(df):
            ref, pub = pd.to_datetime(df["fecha_referencia"]), pd.to_datetime(df["fecha_publicacion"])
            lag = (pub - ref).dt.days
            fila.update(fuente=",".join(sorted(df["fuente"].dropna().astype(str).unique())),
                        referencia_inicio=ref.min().date().isoformat(), referencia_fin=ref.max().date().isoformat(),
                        publicacion_fin=pub.max().date().isoformat(), lag_mediana_d=float(lag.median()),
                        lag_max_d=int(lag.max()),
                        metodo_publicacion=",".join(sorted(df["metodo_publicacion"].dropna().astype(str).unique())))
        filas.append(fila)
    cols = ["modulo", "serie", "fuente", "n_filas", "n_series", "referencia_inicio", "referencia_fin",
            "publicacion_fin", "lag_mediana_d", "lag_max_d", "metodo_publicacion"]
    return pd.DataFrame(filas).reindex(columns=cols)


# --------------------------------------------------------------------------- limpieza (embudo)

def _dia_de(d: dict, claves: dict[str, str], fechas: dict[str, str]) -> str:
    """Día testigo de un documento: ``extra['dia_testigo']``; sin él (RSS, Bing), el de su fecha o «hoy»."""
    marca = (d.get("extra") or {}).get("dia_testigo")
    if marca in claves:
        return marca
    if marca and str(marca)[:10] in fechas:
        return fechas[str(marca)[:10]]
    return fechas.get(str(d.get("fecha_publicacion"))[:10], "hoy")


def leer_titulares(cfg: base.ConfigSonda, nombres: Iterable[str] = DOCS_EMBUDO) -> list[dict]:
    """Documentos de los jsonl de titulares/texto, con ``_jsonl`` y ``_dia`` (clave del día testigo)."""
    claves = {d.clave: d.fecha.isoformat() for d in cfg.dias}
    fechas = {v: k for k, v in claves.items()}
    docs = []
    for n in nombres:
        for d in base.leer_documentos(base.nombre_salida(n, cfg.modo_rapido)):
            d["_jsonl"] = n
            d["_dia"] = _dia_de(d, claves, fechas)
            docs.append(d)
    return docs


def limpieza_corpus(cfg: base.ConfigSonda, nombres: Iterable[str] = DOCS_EMBUDO) -> dict[str, pd.DataFrame]:
    """Embudo de duplicados por día testigo, idiomas por fuente y ejemplos de sindicación."""
    docs = leer_titulares(cfg, nombres)
    if not docs:
        return {}
    orden = {d.clave: i for i, d in enumerate(cfg.dias)}
    fecha = {d.clave: d.fecha.isoformat() for d in cfg.dias}
    embudos, sindicados = [], []
    for dia in sorted({d["_dia"] for d in docs}, key=lambda k: orden.get(k, 99)):
        grupo, notas = [], []
        for n in nombres:
            sub = sorted((d for d in docs if d["_dia"] == dia and d["_jsonl"] == n), key=lambda d: d["doc_id"])
            tope = TOPE_DIA.get(n)
            if tope and len(sub) > tope:
                notas.append(f"{n}: {tope} de {len(sub)} (muestra fija por doc_id)")
                sub = sub[:tope]
            grupo += sub
        if not grupo:
            continue
        cfg.log(f"  [limpieza] {dia}: {len(grupo)} documentos")
        comp = pd.Series([d["fuente"] for d in grupo]).value_counts()
        e = embudo_duplicados(grupo)
        e.insert(0, "dia", dia)
        e.insert(1, "fecha", fecha.get(dia))
        e["fuentes"] = "; ".join(f"{k}:{v}" for k, v in comp.items())
        e["nota"] = "; ".join(notas)
        embudos.append(e)
        for d in deduplicar(grupo):
            otros = (d.get("extra") or {}).get("sindicada_en") or []
            if otros:
                f = _fecha_doc(d)
                sindicados.append(dict(dia=dia, fuente=d["fuente"], titular=_corto(d.get("titulo")),
                                       n_medios=1 + len(otros), medio_original=_medio_doc(d),
                                       primera_fecha=f.isoformat() if f else d.get("fecha_publicacion"),
                                       otros_medios="; ".join(map(str, otros[:5]))))
    sind = pd.DataFrame(sindicados)
    if not sind.empty:
        sind = (sind.sort_values(["dia", "n_medios"], ascending=[True, False], key=lambda s: s.map(orden)
                                 if s.name == "dia" else s)
                .groupby("dia", sort=False).head(SINDICADOS_POR_DIA).reset_index(drop=True))
    idi = pd.DataFrame([dict(fuente=d["fuente"], idioma=d.get("idioma") or "und") for d in docs])
    idi = idi.value_counts().rename("n").reset_index()
    idi["pct_fuente"] = (100 * idi["n"] / idi.groupby("fuente")["n"].transform("sum")).round(1)
    idi = idi.sort_values(["fuente", "n"], ascending=[True, False]).reset_index(drop=True)
    return {"limpieza_embudo": pd.concat(embudos, ignore_index=True), "limpieza_idiomas": idi,
            "limpieza_sindicacion": sind.reindex(columns=["dia", "fuente", "titular", "n_medios", "medio_original",
                                                          "primera_fecha", "otros_medios"])}


def _corto(t: Any, n: int = MAX_TITULAR) -> str:
    return base.titular_corto(t, n)


# --------------------------------------------------------------------------- volumen

# Una cantidad seguida de «/…» es una tasa (por mes, por día, por 15 min...), no un total.
_CANTIDAD = re.compile(r"(\d+(?:[.,]\d+)?)\s*(TB|GB|MB|KB)\b(\s*/\s*(\w*))?", re.IGNORECASE)
_A_GB = {"TB": 1e3, "GB": 1.0, "MB": 1e-3, "KB": 1e-6}
_PERIODO_D = {"día": 1.0, "dia": 1.0, "semana": 7.0, "mes": 30.44, "trimestre": 91.31, "año": 365.25, "anio": 365.25}


def gb_de_texto(texto: str | None, inicio: str | None = None, fin: str | None = None) -> tuple[float | None, str]:
    """GB del corpus a partir del ``volumen_estimado`` de la ficha.

    Toma la ÚLTIMA cantidad absoluta del texto (en las fichas es el total tras las cuentas). Si solo hay
    tasas (``x GB/mes``, ``/día``, ``/año``) y la ficha tiene cobertura, la multiplica por el periodo.
    """
    absolutas, tasas = [], []
    for num, unidad, barra, per in _CANTIDAD.findall(texto or ""):
        gb = float(num.replace(",", ".")) * _A_GB[unidad.upper()]
        (tasas if barra else absolutas).append((gb, (per or "").lower()))
    if absolutas:
        return absolutas[-1][0], "volumen_estimado de la ficha"
    if tasas and inicio and fin:
        gb, per = tasas[0]
        if per in _PERIODO_D:
            dias = (pd.Timestamp(fin[:10]) - pd.Timestamp(inicio[:10])).days
            if dias > 0:
                return gb * dias / _PERIODO_D[per], f"tasa de la ficha ({gb:g} GB/{per}) × {dias} días de cobertura"
    return None, "sin cifra de volumen en la ficha"


def conteo_items(res: ResultadoSonda, cfg: base.ConfigSonda) -> dict[str, int]:
    """Nº de ítems de la muestra por fuente: documentos de los jsonl y filas de los parquet del corpus."""
    n: dict[str, int] = {}
    for jsonl in res.documentos.get("jsonl", pd.Series(dtype=str)):
        ruta = base.corpus("documentos") / f"{jsonl}.jsonl"
        if not ruta.exists():
            continue
        with ruta.open(encoding="utf-8") as fh:
            for linea in fh:
                m = re.search(r'"fuente":\s*"([^"]+)"', linea)
                if m:
                    n[m.group(1)] = n.get(m.group(1), 0) + 1
    for s in res.series.get("serie", pd.Series(dtype=str)):
        df = base.leer_serie(s)
        for f, k in df["fuente"].astype(str).value_counts().items():
            n[f] = n.get(f, 0) + int(k)
    return n


def volumen(res: ResultadoSonda, items: dict[str, int] | None = None) -> pd.DataFrame:
    """Por fuente: muestra medida (ítems, MB) y extrapolación al corpus (GB, horas de descarga).

    Las horas usan la velocidad medida de esa fuente en las peticiones de red (no de caché) si movieron al
    menos ``MIN_MB_VELOCIDAD``; si no, la velocidad conjunta (bytes / segundos) de las fuentes que sí (se indica
    en ``nota``). Incluye latencia y pausas de cortesía: con la caché caliente solo quedan peticiones pequeñas y
    la cifra es una cota pesimista (la descarga masiva de ficheros grandes va más rápido).
    """
    items = items or {}
    reg = res.registro_descargas.copy()
    reg["red"] = ~reg["item"].astype(str).str.contains("(caché)", regex=False) & (reg["segundos"].fillna(0) > 0)
    ok = reg.loc[reg["ok"].astype(bool)]
    vel, b_tot, s_tot = {}, 0.0, 0.0
    for f, g in ok.loc[ok["red"]].groupby("fuente"):
        if g["segundos"].sum() > 0 and g["bytes"].sum() >= MIN_MB_VELOCIDAD * 1e6:
            vel[f] = g["bytes"].sum() / g["segundos"].sum()
            b_tot, s_tot = b_tot + g["bytes"].sum(), s_tot + g["segundos"].sum()
    vel_global = b_tot / s_tot if s_tot else None
    filas = []
    for f in res.fichas.to_dict("records"):
        fu = f["fuente"]
        g, gr = ok.loc[ok["fuente"] == fu], reg.loc[reg["fuente"] == fu]
        gb, como = gb_de_texto(f["volumen_estimado"], f["cobertura_inicio"], f["cobertura_fin"])
        v, nota_v = vel.get(fu), "velocidad medida"
        if v is None and vel_global:
            v, nota_v = vel_global, f"velocidad conjunta de {len(vel)} fuentes (sin red propia suficiente)"
        horas = round(gb * 1e9 / v / 3600, 2) if gb is not None and v else None
        n_items = items.get(fu, int(len(g)))
        filas.append(dict(fuente=fu, n_items_muestra=n_items, mb_muestra=round(g["bytes"].sum() / 1e6, 3),
                          gb_corpus_estimado=round(gb, 4) if gb is not None else None,
                          horas_descarga_estimadas=horas, n_peticiones=int(len(gr)),
                          n_peticiones_red=int(gr["red"].sum()), mb_s_medidos=round(v / 1e6, 3) if v else None,
                          nota=f"{como}; {nota_v if horas is not None else 'horas sin estimar'} · "
                               f"{_corto(f['volumen_estimado'], 160)}"))
    return pd.DataFrame(filas).reindex(columns=COLUMNAS_VOLUMEN)


def _licencia_corta(t: Any, n: int = 80) -> str:
    t = " ".join(str(t or "").split())
    corte = re.split(r"[;(]", t, maxsplit=1)[0].strip()
    return _corto(corte or t, n)


def fuentes_resumen(res: ResultadoSonda) -> pd.DataFrame:
    """Una fila por ficha con lo esencial para la tabla de fuentes del plan de datos."""
    cob = res.cobertura.drop_duplicates("fuente").set_index("fuente") if not res.cobertura.empty else pd.DataFrame()
    filas = []
    for f in res.fichas.to_dict("records"):
        c = cob.loc[f["fuente"]] if f["fuente"] in cob.index else {}
        filas.append(dict(fuente=f["fuente"], nombre=f["nombre"], familia=c.get("familia"), tipo=c.get("tipo"),
                          estado_sonda=f["estado_sonda"], cobertura_inicio=f["cobertura_inicio"],
                          cobertura_fin=f["cobertura_fin"], retraso_publicacion=f["retraso_publicacion"],
                          lag_mediana_h=c.get("lag_mediana_h"), destino=c.get("destino") or f["rag"],
                          backtest=c.get("backtest"), licencia=_licencia_corta(f["licencia_y_terminos"]),
                          agentes=f["agentes"], modulo=f["modulo"]))
    return pd.DataFrame(filas).reindex(columns=COLUMNAS_RESUMEN)


# --------------------------------------------------------------------------- guardar

def carpeta_resultados(rapido: bool = False) -> Path:
    """``results/ingesta/noticias_alt`` (``.../rapido`` en el modo rápido)."""
    c = resultados_dir(base.TEMA)
    if rapido:
        c = c / "rapido"
        c.mkdir(parents=True, exist_ok=True)
    return c


def guardar(res: ResultadoSonda, carpeta: Path | None = None) -> list[Path]:
    """Escribe las tablas comunes, las de cada módulo y las fichas. Devuelve las rutas escritas."""
    carpeta = carpeta or carpeta_resultados(res.rapido)
    comunes = {"registro_descargas": res.registro_descargas, "cobertura": res.cobertura, "timegate": res.timegate,
               "volumen": res.volumen, "fuentes_resumen": res.fuentes_resumen, "sonda_modulos": res.modulos,
               "sonda_series": res.series, "sonda_documentos": res.documentos}
    rutas = [base.guardar_tabla(t, k, carpeta) for k, t in comunes.items()]
    for k, t in res.tablas.items():
        if isinstance(t, pd.DataFrame):
            rutas.append(base.guardar_tabla(t, k, carpeta))
    for f in res.fichas_obj:
        rutas.append(guardar_ficha(f, carpeta / "fichas"))
    return rutas


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ingesta.noticias_alt.sonda", description=__doc__.split("\n")[0])
    p.add_argument("--rapido", action="store_true", help="recorta cada fuente a lo mínimo (salidas en .../rapido)")
    p.add_argument("--sin-cache", action="store_true", help="vuelve a pedir todo a la red")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # consola cp1252: «≈», «→»... no rompen la sonda
    t0 = base.cronometro()
    cfg = base.ConfigSonda(modo_rapido=a.rapido, usar_cache=not a.sin_cache)
    res = ejecutar(cfg)
    rutas = guardar(res)
    reg = res.registro_descargas
    n_cache = int(reg["item"].astype(str).str.contains("(caché)", regex=False).sum()) if len(reg) else 0
    print(f"\nSonda del Tema 4 ({date.today().isoformat()}) en {base.segundos(t0)} s")
    print(f"  módulos: {int(res.modulos['ok'].sum())}/{len(res.modulos)} sin fallo")
    print(f"  peticiones: {len(reg)} ({n_cache} de caché, {int((~reg['ok'].astype(bool)).sum())} fallidas)")
    print(f"  fichas: {len(res.fichas)} · " + ", ".join(f"{k}={v}" for k, v in
                                                   res.fichas["estado_sonda"].value_counts().items()))
    print(f"  tablas: {len(res.tablas)} · series: {len(res.series)} · jsonl: {len(res.documentos)}")
    print(f"  {len(rutas)} ficheros en {carpeta_resultados(res.rapido)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
