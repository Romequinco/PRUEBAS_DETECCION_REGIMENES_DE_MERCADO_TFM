"""Base de series del Tema 1 (bloques 2 y 3): une los cargadores, resuelve conflictos y la escribe.

Cargadores (``CARGADORES``), cada uno ya validado contra el contrato y el catálogo:

- ``fase1``       ``data/raw`` de la Fase 1 (``fuentes.fase1.tabla_fase1``): lo que FRED no sirve
                  (yfinance, datahub) y respaldo de las series de FRED.
- ``fred``        FRED como fuente viva (``fuentes.fred.tabla_fred_vivo``): todas las series de FRED del
                  catálogo que no van por ALFRED, con su regla y las excepciones medidas.
- ``academicos``  Ken French, Shiller y Goyal-Welch con vintage por edición ``AAAA-MM`` y la OFR FSI
                  (``fuentes.academicos.tabla_academicos``).
- ``alfred``      vintages exactos de ALFRED de las series ``vintage_exacto`` (``fuentes.alfred.tabla_alfred``).

Una serie, una fuente: si dos cargadores traen el mismo id gana el de mayor ``PRIORIDAD``
(``alfred`` > ``academicos`` > ``fred`` > ``fase1``, el orden de ``catalogo.PRIORIDAD_ORIGENES``; las
derivadas propias se recalculan después). ``data/raw`` solo sirve una serie de FRED si FRED no trae
filas (``ResumenBase.respaldo`` cuenta las fechas que solo tenía la copia descartada). Las copias con
otro id de fuente (estado ``duplicada``) entran con su propio id. Derivadas propias de diferencia
(``deriv:A-B`` con ``id_origen`` ``'A - B'``): si sus dos componentes están en la base se recalculan
con ellos (valor ``A − B`` en las fechas con los dos valores; ``fecha_publicacion`` =
``contrato.fecha_publicacion_derivada``), y se compara con la copia de ``data/raw`` si la hay. Las
demás derivadas (volatilidad realizada) quedan como las fecha su cargador.

Empalme del PIB real de EE. UU. por vintage (``empalmar_pib_real_usa``, id ``EMPALME_PIB_USA``): los
vintages del RTDSM (``philly:ROUTPUT``, PNB real hasta el vintage de 1991-11) hasta el primer vintage de
ALFRED de ``fred:GDPC1`` (1991-12-04, primer vintage con PIB real) y los de ALFRED desde ahí. Cada fila
lleva ``NOTA_EMPALME_PIB_USA`` y el tramo (PNB o PIB): nunca se mezclan sin nota. Unidad: dólares
constantes de la base de CADA vintage (``unidad_por_vintage``): comparar tasas, no niveles.

Antes de derivar: ``catalogo.aplicar_excepciones`` (fecha = max(regla, primera vintage de ALFRED),
``config/excepciones_publicacion.csv``; solo filas ``unico``, idempotente), así las derivadas heredan
las fechas medidas. Después: ``catalogo.anotar_reconstruidas`` (índices publicados desde
``publicado_desde``) y ``anotar_anomalias`` (picos aislados que se revierten, cierres en fin de semana y
rachas planas: solo se anota la ``nota``, nunca se tocan valores).

``construir_base`` valida la unión (``contrato.validar_series`` + ``catalogo.validar_contra_catalogo``)
y la escribe con ``almacen.escribir_series``: base que solo crece, salvo el ``valor`` de las
revisables con vintage ``unico`` (``revisables``) y, con ``permitir_perdidas``, correcciones propias.
Las series guardadas que ya no se construyen (huérfanas) son un ``ErrorContrato`` salvo
``purgar=True``. ``desde_cero=True`` (cambio de reglas o fuentes, bloque 3) copia antes la base
guardada a ``RESPALDO`` y la sustituye entera. ``cobertura`` resume la base por serie (solo
metadatos). CLI: ``python -m ingesta.mercados_series construir-base [--sin-red] [--desde-cero]``.
This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""
from __future__ import annotations

import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun.rutas import resultados_dir
from ingesta.mercados_series import almacen
from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    ErrorContrato,
    fecha_publicacion_derivada,
    vacio_series,
    validar_series,
)

TEMA = "mercados_series"
CARGADORES = ("fase1", "philly_rtdsm", "curvas", "bancos_centrales", "oecd", "worldbank", "imf", "bis", "eurostat",
              "bce", "fred", "academicos", "alfred")
# Bloques 4 y 6: cargadores de fuentes oficiales (``catalogo.CARGADOR_POR_FUENTE``). Leen los crudos o las
# tablas ya descargadas en el corpus; las descargas van por la CLI (``descargar-<fuente>``).
CARGADORES_OFICIALES = ("bce", "eurostat", "bis", "imf", "worldbank", "oecd", "bancos_centrales", "curvas",
                        "philly_rtdsm")
_BLOQUE4 = {"bis": "bis", "imf": "imf_weo", "worldbank": "worldbank"}
# Orden de catalogo.PRIORIDAD_ORIGENES (sin 'derivadas', que no es un cargador: se recalculan tras unir).
PRIORIDAD = {c: i for i, c in enumerate(reversed([o for o in cat_mod.PRIORIDAD_ORIGENES if o in CARGADORES]))}
CARGADOR_DERIVADAS = "derivadas"
EMPALME_PIB_USA = "deriv:PIB_REAL_USA_VINTAGES"
NOTA_EMPALME_PIB_USA = ("empalme por vintage: RTDSM (philly:ROUTPUT) antes del primer vintage de ALFRED de fred:GDPC1 "
                        "y ALFRED desde él; nivel en dólares constantes de la base de cada vintage (comparar tasas)")
TRAMO_PNB = "tramo PNB real (GNP, vintages RTDSM anteriores a 1991-12)"
TRAMO_PIB = "tramo PIB real (GDP, vintages de ALFRED)"
RESPALDO = "respaldo_bloque2.parquet"   # copia de la base guardada antes de una reconstrucción desde cero
SUFIJO_PREVIA = "_previa.parquet"     # si RESPALDO ya existe: copia (rotatoria) de la base vigente
TOLERANCIA_DERIVADA = 1e-9
AVISO_FRED = ("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve "
              "Bank of St. Louis.")

__all__ = [
    "CARGADORES", "CARGADORES_OFICIALES", "EMPALME_PIB_USA", "empalmar_pib_real_usa", "tabla_oficial", "PRIORIDAD", "RESPALDO", "ResumenBase", "tablas_fuentes", "unir", "fechas_solo_respaldo",
    "derivar_diferencias",
    "catalogo_efectivo", "construir_base", "cobertura", "escribir_cobertura", "resumen_base",
    "anotar_anomalias", "NOTA_PICO", "NOTA_FIN_DE_SEMANA", "NOTA_RACHA",
]

# Anomalías que se anotan (sin tocar valores): umbrales medidos sobre la base del bloque 2.
NOTA_PICO = "posible dato erróneo de la fuente: pico aislado que se revierte al día siguiente"
NOTA_FIN_DE_SEMANA = "posible error de fecha de la fuente: cierre de mercado en fin de semana"
NOTA_RACHA = "posible dato congelado de la fuente: racha plana de"
UMBRAL_PICO = 0.4            # |Δlog| de ida y de vuelta
UMBRAL_REVERSION = 0.1       # |log(siguiente / anterior)|: el pico no cambia el nivel
MIN_RACHA = 120              # observaciones diarias seguidas con el mismo valor
# Tipos administrados (constantes entre reuniones del FOMC): una racha plana es normal.
SIN_RACHAS = ("fred:IORB", "fred:DPCREDIT", "fred:DFF", "fred:EFFR", "fred:OBFR", "fred:FEDFUNDS",
              # Bloque 4: tipos oficiales del BCE, del Banco de Canadá y del RBA (fecha de efecto).
              "ecb:DFR", "ecb:MRO", "boc:V39079", "rba:FIRMMCRTD")


@dataclass
class ResumenBase:
    """Lo que hizo ``construir_base`` (diccionario con ``como_dict``)."""

    ruta: str | None
    escrita: bool
    n_filas: int
    n_series: int
    n_valores: int
    n_huecos: int
    segundos: float
    por_cargador: dict[str, int] = field(default_factory=dict)
    filas_por_cargador: dict[str, int] = field(default_factory=dict)
    conflictos: dict[str, dict[str, Any]] = field(default_factory=dict)
    derivadas: dict[str, dict[str, Any]] = field(default_factory=dict)
    sin_datos: list[str] = field(default_factory=list)
    excepciones_aplicadas: int = 0
    huerfanas: list[str] = field(default_factory=list)
    purgadas: list[str] = field(default_factory=list)
    revisables: int = 0
    anomalias: dict[str, int] = field(default_factory=dict)
    detalle_anomalias: list[dict[str, str]] = field(default_factory=list, repr=False)
    por_estado: dict[str, int] = field(default_factory=dict)
    por_metodo: dict[str, int] = field(default_factory=dict)
    por_fuente: dict[str, int] = field(default_factory=dict)
    respaldo: dict[str, int] = field(default_factory=dict)
    alfred: dict[str, Any] = field(default_factory=dict)
    desde_cero: bool = False
    copia_previa: str | None = None
    bytes_disco: int | None = None

    def como_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Cargadores ---------------------------------------------------------------------------------
def tablas_fuentes(*, sin_red: bool = False, cargadores: Iterable[str] = CARGADORES,
                   catalogo: dict[str, dict] | None = None, raiz_fase1: Path | None = None,
                   raiz_crudo_academicos: Path | None = None, cliente_fred: Any = None,
                   cliente_alfred: Any = None, informe_alfred: dict[str, dict] | None = None,
                   registro: Callable[[str], None] | None = None) -> dict[str, pd.DataFrame]:
    """``{cargador: tabla validada}`` de los ``cargadores`` pedidos.

    ``sin_red=True``: FRED y ALFRED solo desde su archivo crudo y los académicos solo con los crudos
    ya descargados (y ``data/raw``). Sin ``sin_red``, FRED vivo baja lo que no tenga hoy en caché
    (incremental), ALFRED las series sin descarga completa archivada y los académicos la edición del
    día si no está archivada (un fallo de red en un académico no para la construcción: queda en su
    ``.meta.json`` y se usa el último crudo). Las descargas largas van antes y por tandas
    (``descargar-alfred``, ``descargar-fred-vivo``). ``informe_alfred``: se rellena con el informe
    por serie de ``fuentes.alfred.tabla_alfred``.
    """
    from ingesta.mercados_series.fuentes import academicos, alfred, fase1, fred

    log = registro or (lambda _m: None)
    pedidos = list(cargadores)
    desconocidos = sorted(set(pedidos) - set(CARGADORES))
    if desconocidos:
        raise ValueError(f"cargadores desconocidos {desconocidos}; válidos: {list(CARGADORES)}")
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    out: dict[str, pd.DataFrame] = {}
    for nombre in pedidos:
        t0 = time.monotonic()
        if nombre == "fase1":
            out[nombre] = fase1.tabla_fase1(raiz_fase1, cat)
        elif nombre == "fred":
            out[nombre] = fred.tabla_fred_vivo(sin_red=sin_red, cliente=cliente_fred, cat=cat, registro=log)
        elif nombre in CARGADORES_OFICIALES:
            out[nombre] = tabla_oficial(nombre, sin_red=sin_red, registro=log)
        elif nombre == "alfred":
            out[nombre] = alfred.tabla_alfred(cliente=cliente_alfred, descargar=not sin_red, cat=cat,
                                              informe=informe_alfred if informe_alfred is not None else {})
        else:
            if not sin_red:
                for f in academicos.FUENTES_CATALOGO:
                    for m in academicos.descargar(f, raiz=raiz_crudo_academicos):
                        if m.get("error"):
                            log(f"  aviso: {f} {m.get('clave')}: {m['error']}")
            out[nombre] = academicos.tabla_academicos(catalogo=cat, raiz_crudo=raiz_crudo_academicos,
                                                      raiz_fase1=raiz_fase1)
        log(f"{nombre}: {len(out[nombre])} filas, {out[nombre]['serie'].nunique()} series "
            f"({time.monotonic() - t0:.0f} s)")
    return out


def tabla_oficial(nombre: str, *, sin_red: bool = True, registro: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Tabla validada de un cargador de ``CARGADORES_OFICIALES`` (bloques 4 y 6) desde el corpus.

    ``bce``/``eurostat``: crudos archivados (``sin_red``) o descarga del día. ``bis``/``imf``/``worldbank``:
    ``<corpus>/mercados_series/bloque4/<fuente>.parquet``. ``oecd``: CSV de ediciones archivados.
    ``bancos_centrales``: crudo de cada serie. ``curvas``: ``series/curvas_b6.parquet`` o los crudos.
    ``philly_rtdsm``: matrices xlsx (solo vintages anteriores al primero de ALFRED). Lo que falta en
    el corpus se omite con aviso (una fuente sin descargar no para la construcción).
    """
    log = registro or (lambda _m: None)
    corpus = almacen.corpus_dir(TEMA)
    partes: list[pd.DataFrame] = []
    if nombre == "bce":
        from ingesta.mercados_series.fuentes import bce
        partes.append(bce.tabla_bce(sin_red=sin_red)[0])
    elif nombre == "eurostat":
        from ingesta.mercados_series.fuentes import eurostat
        partes.append(eurostat.tabla_eurostat(sin_red=sin_red)[0])
    elif nombre in _BLOQUE4:
        ruta = corpus / "bloque4" / f"{_BLOQUE4[nombre]}.parquet"
        if ruta.exists():
            partes.append(pd.read_parquet(ruta))
        else:
            log(f"  aviso: {nombre}: falta {ruta} (python -m ingesta.mercados_series descargar-{nombre})")
    elif nombre == "oecd":
        from ingesta.mercados_series.fuentes import oecd
        for area in oecd.AREAS:
            for medida in oecd.MEDIDAS:
                crudo = oecd.cargar_crudo(area, medida)
                if crudo is not None and len(crudo):
                    t = oecd.tabla_oecd(area, medida, crudo)
                    if len(t):
                        partes.append(t)
    elif nombre == "bancos_centrales":
        from ingesta.mercados_series.fuentes import b4_comun, bancos_centrales as bc
        for sid, info in bc.SERIES.items():
            ruta = b4_comun.ruta_crudo(info["fuente"]) / bc._nombre_crudo(sid)
            if ruta.exists():
                partes.append(bc.tabla_banco(sid, bc.leer(sid, ruta.read_bytes())))
            else:
                log(f"  aviso: bancos_centrales: {sid} sin crudo")
    elif nombre == "curvas":
        ruta = corpus / "series" / "curvas_b6.parquet"
        if ruta.exists():
            partes.append(pd.read_parquet(ruta))
        else:
            from ingesta.mercados_series.fuentes import curvas
            partes.append(curvas.tabla_curvas())
    elif nombre == "philly_rtdsm":
        from ingesta.mercados_series.fuentes import philly_rtdsm as ph
        for v in ph.MAPEO:
            ruta = ph._ruta_local(v)
            if ruta.exists():
                t = ph.tabla_philly(v, ruta)
                if len(t):
                    partes.append(t)
    else:
        raise ValueError(f"cargador oficial desconocido {nombre!r}; válidos: {list(CARGADORES_OFICIALES)}")
    partes = [p for p in partes if len(p)]
    if not partes:
        return vacio_series()
    return validar_series(pd.concat(partes, ignore_index=True)[list(COLUMNAS_SERIES)])


# --- Unión --------------------------------------------------------------------------------------
def unir(tablas: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, Any]]]:
    """Une las tablas de los cargadores con una sola fuente por serie.

    Devuelve ``(tabla, cargador_por_serie, conflictos)``. ``conflictos``: por id servido por más
    de un cargador, ``{'cargadores': [...], 'elegido': ...}`` (gana el de mayor ``PRIORIDAD``).
    """
    desconocidos = sorted(set(tablas) - set(PRIORIDAD))
    if desconocidos:
        raise ValueError(f"cargadores desconocidos {desconocidos}; válidos: {list(PRIORIDAD)}")
    servidores: dict[str, list[str]] = {}
    for nombre, t in tablas.items():
        for sid in pd.unique(t["serie"]):
            servidores.setdefault(str(sid), []).append(nombre)
    elegido = {sid: max(c, key=PRIORIDAD.__getitem__) for sid, c in servidores.items()}
    conflictos = {sid: {"cargadores": sorted(c, key=PRIORIDAD.__getitem__), "elegido": elegido[sid]}
                  for sid, c in sorted(servidores.items()) if len(c) > 1}
    partes = []
    for nombre, t in tablas.items():
        propias = [sid for sid, c in elegido.items() if c == nombre]
        if len(t) and propias:
            partes.append(t[t["serie"].isin(propias)])
    tabla = pd.concat(partes, ignore_index=True)[list(COLUMNAS_SERIES)] if partes else vacio_series()
    return tabla, dict(sorted(elegido.items())), conflictos


def fechas_solo_respaldo(tablas: Mapping[str, pd.DataFrame], elegido: Mapping[str, str]) -> dict[str, int]:
    """Por serie servida por varios cargadores: fechas con valor que solo traía un cargador descartado.

    Mide lo que se pierde al servir una serie desde una sola fuente (p. ej. una fecha de ``data/raw``
    que FRED ya no sirve). Solo las series con alguna; sin valores.
    """
    out: dict[str, int] = {}
    grupos = {n: t.loc[t["valor"].notna()].groupby("serie")["fecha_referencia"] for n, t in tablas.items() if len(t)}
    for sid, ganador in elegido.items():
        if ganador not in grupos or sid not in grupos[ganador].groups:
            continue
        propias = set(grupos[ganador].get_group(sid))
        perdidas: set = set()
        for n, g in grupos.items():
            if n != ganador and sid in g.groups:
                perdidas |= set(g.get_group(sid)) - propias
        if perdidas:
            out[sid] = len(perdidas)
    return dict(sorted(out.items()))


def _es_diferencia(sid: str, entrada: dict[str, Any]) -> bool:
    comp = entrada.get("componentes") or []
    return (entrada["metodo"] == "derivada" and len(comp) == 2
            and str(entrada.get("id_origen", "")).replace(" ", "") == f"{comp[0]}-{comp[1]}")


def derivar_diferencias(base: pd.DataFrame, catalogo: dict[str, dict]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Recalcula las derivadas ``A − B`` del catálogo cuyos componentes están en ``base``.

    Para cada una (estado cargable, ``_es_diferencia``): valor ``A − B`` en las fechas en que
    los dos componentes tienen valor (vintage ``unico``); ``fecha_publicacion`` y
    ``disponible_utc`` con ``contrato.fecha_publicacion_derivada``. Si la derivada ya estaba en
    ``base`` (copia de ``data/raw``), se sustituye y se mide: fechas comunes, valores distintos
    (más de ``TOLERANCIA_DERIVADA``), diferencia máxima, fechas solo en la copia o solo en el
    cálculo y fechas de publicación distintas. Una derivada con algún componente ausente o con
    vintages que no son ``unico`` se deja como estaba (``info['recalculada'] = False``).
    Devuelve ``(base, info)``.
    """
    info: dict[str, dict[str, Any]] = {}
    nuevas, quitar = [], []
    for sid, e in sorted(catalogo.items()):
        if not _es_diferencia(sid, e) or e["estado"] not in cat_mod.ESTADOS_CARGABLES:
            continue
        a, b = e["componentes"]
        comp = base[base["serie"].isin([a, b]) & base["valor"].notna()]
        presentes = set(comp["serie"])
        if presentes != {a, b} or (comp["vintage"] != "unico").any():
            motivo = ("componentes sin datos: " + ", ".join(sorted({a, b} - presentes)) if presentes != {a, b}
                      else "componentes con vintages")
            info[sid] = {"recalculada": False, "motivo": motivo}
            continue
        fechado = fecha_publicacion_derivada(comp, [a, b]).set_index("fecha_referencia")
        va = comp[comp["serie"] == a].set_index("fecha_referencia")["valor"]
        vb = comp[comp["serie"] == b].set_index("fecha_referencia")["valor"]
        valor = (va.reindex(fechado.index) - vb.reindex(fechado.index)).to_numpy()
        nota = cat_mod.NOTA_REVISABLE if e["revisable"] else None
        filas = pd.DataFrame({
            "serie": sid, "fecha_referencia": fechado.index, "fecha_publicacion": fechado["fecha_publicacion"].to_numpy(),
            "disponible_utc": fechado["disponible_utc"].to_numpy(), "vintage": "unico", "valor": valor,
            "unidad": e["unidad"], "divisa": e["divisa"], "fuente": e["fuente"], "metodo_publicacion": e["metodo"],
            "nota": nota,
        })[list(COLUMNAS_SERIES)]
        medida: dict[str, Any] = {"recalculada": True, "componentes": [a, b], "n_filas": int(len(filas)),
                                  "inicio": _iso(filas["fecha_referencia"].min()),
                                  "fin": _iso(filas["fecha_referencia"].max())}
        copia = base[base["serie"] == sid]
        if len(copia):
            c = copia[copia["valor"].notna()].set_index("fecha_referencia")
            f = filas.set_index("fecha_referencia")
            comunes = c.index.intersection(f.index)
            dif = (c.loc[comunes, "valor"] - f.loc[comunes, "valor"]).abs()
            medida["copia_fase1"] = {
                "fechas_comunes": int(len(comunes)),
                "valores_distintos": int((dif > TOLERANCIA_DERIVADA).sum()),
                "dif_max": float(dif.max()) if len(dif) else None,
                "solo_copia": int(len(c.index.difference(f.index))),
                "solo_calculo": int(len(f.index.difference(c.index))),
                "publicacion_distinta": int((c.loc[comunes, "fecha_publicacion"]
                                             != f.loc[comunes, "fecha_publicacion"]).sum()),
            }
            quitar.append(sid)
        info[sid] = medida
        nuevas.append(filas)
    if not nuevas:
        return base, info
    resto = base[~base["serie"].isin(quitar)]
    return pd.concat([resto, *nuevas], ignore_index=True)[list(COLUMNAS_SERIES)], info


def _iso(valor: Any) -> str | None:
    return None if pd.isna(valor) else pd.Timestamp(valor).date().isoformat()


def _anotar(df: pd.DataFrame, mascara: pd.Series, texto: str | pd.Series) -> None:
    """Añade ``texto`` a la ``nota`` de las filas de ``mascara`` (in situ; idempotente por texto)."""
    if not mascara.any():
        return
    t = texto[mascara] if isinstance(texto, pd.Series) else pd.Series(texto, index=df.index[mascara])
    previa = df.loc[mascara, "nota"]
    ya = previa.fillna("").astype(str).str.contains(t.iloc[0].split(":")[0], regex=False)
    nueva = t.where(previa.isna(), previa.astype(str) + "; " + t)
    df.loc[mascara, "nota"] = nueva.where(~ya, previa)


def empalmar_pib_real_usa(base: pd.DataFrame, catalogo: dict[str, dict]) -> tuple[pd.DataFrame, int]:
    """Añade ``EMPALME_PIB_USA`` (si está en el catálogo y están sus dos tramos): RTDSM antes del primer
    vintage de ALFRED de ``fred:GDPC1`` y ALFRED desde él (solo filas ``vintage_exacto``; los periodos
    previos a ALFRED fechados con la regla quedan fuera: los cubre el RTDSM). Devuelve (base, filas)."""
    e = catalogo.get(EMPALME_PIB_USA)
    if e is None:
        return base, 0
    base = base[base["serie"] != EMPALME_PIB_USA]
    rt = base[(base["serie"] == "philly:ROUTPUT") & (base["metodo_publicacion"] == "vintage_exacto")]
    al = base[(base["serie"] == "fred:GDPC1") & (base["metodo_publicacion"] == "vintage_exacto")]
    if rt.empty or al.empty:
        return base, 0
    previa = (catalogo.get("fred:GDPC1") or {}).get("regla_previa_alfred") or {}
    corte = pd.Timestamp(previa.get("primer_vintage") or al["fecha_publicacion"].min())
    rt = rt[rt["fecha_publicacion"] <= corte].assign(_tramo=TRAMO_PNB)
    # Un solo formato de vintage por serie (contrato): AAAA-MM del RTDSM -> último día de ese mes (su fecha).
    rt["vintage"] = pd.PeriodIndex(rt["vintage"], freq="M").end_time.strftime("%Y-%m-%d")
    al = al[al["fecha_publicacion"] >= corte].assign(_tramo=TRAMO_PIB)
    emp = pd.concat([rt, al], ignore_index=True)
    previa_nota = emp["nota"].where(emp["nota"].notna(), None)
    emp["nota"] = [f"{NOTA_EMPALME_PIB_USA}; {t}" + (f"; {n}" if isinstance(n, str) and n else "")
                   for t, n in zip(emp["_tramo"], previa_nota)]
    emp = emp.drop(columns="_tramo").assign(serie=EMPALME_PIB_USA, unidad=e["unidad"], divisa=e["divisa"],
                                            fuente=e["fuente"])
    return pd.concat([base, emp[list(COLUMNAS_SERIES)]], ignore_index=True), int(len(emp))


def anotar_anomalias(base: pd.DataFrame, catalogo: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anota en ``nota`` las anomalías de las diarias de mercado sin tocar valores ni fechas.

    - ``NOTA_PICO`` (solo ``cierre_mercado``): valor que sube o baja más de ``UMBRAL_PICO`` en log
      respecto al anterior y vuelve al día siguiente (en sentido contrario y a menos de ``UMBRAL_REVERSION``
      del anterior): VXDCLS 2,71 entre 14,86 y 15,51 (2021-07-13).
    - ``NOTA_FIN_DE_SEMANA``: fila de una serie ``cierre_mercado`` en sábado o domingo.
    - ``NOTA_RACHA``: ``MIN_RACHA`` o más observaciones diarias seguidas con el mismo valor
      (DED3 0,28 durante 184 días de 2013), salvo tipos administrados (``SIN_RACHAS``) y tipos de
      cambio de H.10 (paridades fijas anteriores a 1973).

    Devuelve ``(base anotada, anomalías)``; ``anomalías``: ``serie, fecha_referencia, tipo``
    (solo fechas, sin valores).
    """
    out = base.copy()
    filas = []
    diarias = [sid for sid, e in catalogo.items() if e["frecuencia"] == "diaria"
               and e["metodo"] in ("cierre_mercado", "calendario")]
    unico = out["vintage"] == "unico"
    for sid in sorted(set(diarias) & set(out["serie"])):
        m = (out["serie"] == sid) & unico & out["valor"].notna()
        g = out.loc[m].sort_values("fecha_referencia")
        if len(g) < 3:
            continue
        v = g["valor"]
        tipos: dict[str, pd.Series] = {}
        mercado = catalogo[sid]["metodo"] == "cierre_mercado"
        # Picos solo en cierres de mercado (índices de volatilidad de Cboe...): en los tipos cerca de cero
        # o en los saltos de fin de mes de la financiación un |Δlog| grande es normal.
        if mercado and (v > 0).all():
            lv = np.log(v.to_numpy(float))
            ida = np.r_[np.nan, np.diff(lv)]
            vuelta = np.r_[np.diff(lv), np.nan]
            nivel = np.r_[np.nan, lv[2:] - lv[:-2], np.nan]
            pico = ((np.abs(ida) > UMBRAL_PICO) & (np.abs(vuelta) > UMBRAL_PICO) & (np.sign(ida) != np.sign(vuelta))
                    & (np.abs(nivel) < UMBRAL_REVERSION))
            tipos["pico"] = pd.Series(pico, index=g.index)
        if catalogo[sid]["metodo"] == "cierre_mercado":
            tipos["fin_de_semana"] = g["fecha_referencia"].dt.weekday >= 5
        if sid not in SIN_RACHAS and catalogo[sid].get("publicador") not in ("fed_h10", "ecb_fx"):   # paridades fijas
            grupo = (v != v.shift()).cumsum()
            largo = grupo.map(grupo.value_counts())
            tipos["racha"] = largo >= MIN_RACHA
            racha_txt = NOTA_RACHA + " " + largo.astype(str) + " observaciones"
        for tipo, mask in tipos.items():
            mask = mask.fillna(False).astype(bool)
            if not mask.any():
                continue
            idx = mask[mask].index
            texto = {"pico": NOTA_PICO, "fin_de_semana": NOTA_FIN_DE_SEMANA}.get(tipo)
            sel = pd.Series(False, index=out.index)
            sel[idx] = True
            _anotar(out, sel, texto if texto else racha_txt.reindex(out.index))
            filas.append(pd.DataFrame({"serie": sid, "fecha_referencia": g.loc[idx, "fecha_referencia"].to_numpy(),
                                       "tipo": tipo}))
    anom = (pd.concat(filas, ignore_index=True) if filas
            else pd.DataFrame({"serie": pd.Series(dtype=object), "fecha_referencia": pd.Series(dtype="datetime64[ns]"),
                               "tipo": pd.Series(dtype=object)}))
    return out, anom


def catalogo_efectivo(catalogo: dict[str, dict]) -> dict[str, dict]:
    """Catálogo para validar la base: ``pendiente_alfred`` con ``vintage_exacto`` cuenta como cargable.

    Es lo que pide ese estado (WEI cargado con todas sus vintages de ALFRED), igual que en
    ``fuentes.fred.tabla_fred_sustitutos``.
    """
    return {sid: (dict(e, estado="activa") if e["estado"] == "pendiente_alfred"
                  and e["metodo"] == "vintage_exacto" else e) for sid, e in catalogo.items()}


# --- Construcción -------------------------------------------------------------------------------
def construir_base(*, sin_red: bool = False, escribir: bool = True, ruta: Path | None = None,
                   permitir_perdidas: bool = False, purgar: bool = False, catalogo: dict[str, dict] | None = None,
                   tablas: Mapping[str, pd.DataFrame] | None = None, raiz_fase1: Path | None = None,
                   raiz_crudo_academicos: Path | None = None, cliente_fred: Any = None,
                   excepciones: pd.DataFrame | None = None, cliente_alfred: Any = None,
                   desde_cero: bool = False, respaldo: str = RESPALDO,
                   registro: Callable[[str], None] | None = None) -> tuple[ResumenBase, pd.DataFrame]:
    """Une ALFRED + FRED vivo + académicos + fase1, resuelve conflictos, valida y (si ``escribir``) guarda.

    ``tablas``: tablas ya cargadas por cargador (tests; sin ellas, ``tablas_fuentes``).
    ``ruta``: parquet de destino (por defecto ``almacen.ruta_series()``, es decir
    ``<corpus>/mercados_series/series/base_series.parquet``). Las revisables con vintage ``unico``
    pueden cambiar de valor (``almacen.escribir_series(revisables=...)``: ediciones académicas
    nuevas); ``permitir_perdidas`` (correcciones propias) deja cambiar cualquier fila. Series
    guardadas que ya no se construyen: ``ErrorContrato`` salvo ``purgar=True`` (se quitan y se
    enumeran en ``resumen.purgadas``). ``excepciones``: tabla de ``catalogo.cargar_excepciones``
    (por defecto, el CSV de config). ``desde_cero`` (reconstrucción tras cambiar reglas o fuentes):
    antes de escribir copia la base guardada a ``<carpeta>/<respaldo>`` (si esa copia ya existe, se
    conserva y la base vigente va a ``<stem>_previa.parquet``, que rota) y la sustituye entera, sin
    comprobar pérdidas ni huérfanas; se niega (``ErrorContrato``) si alguna serie de ALFRED no tiene
    descarga completa archivada, salvo ``permitir_perdidas``.
    Devuelve ``(resumen, tabla validada)``. ``ErrorContrato`` si la unión no cumple el contrato o el
    catálogo, o si la escritura perdería o cambiaría filas guardadas.
    """
    t0 = time.monotonic()
    log = registro or (lambda _m: None)
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    informe_alfred: dict[str, dict] = {}
    if tablas is None:
        tablas = tablas_fuentes(sin_red=sin_red, catalogo=cat, raiz_fase1=raiz_fase1,
                                raiz_crudo_academicos=raiz_crudo_academicos, cliente_fred=cliente_fred,
                                cliente_alfred=cliente_alfred, informe_alfred=informe_alfred, registro=log)
    union, elegido, conflictos = unir(tablas)
    respaldo_fechas = fechas_solo_respaldo(tablas, elegido)
    union, n_exc = cat_mod.aplicar_excepciones(union, excepciones, catalogo=cat)
    log(f"excepciones de publicación (ALFRED) aplicadas: {n_exc} filas")
    union, derivadas = derivar_diferencias(union, cat)
    union, n_emp = empalmar_pib_real_usa(union, cat)
    if n_emp:
        log(f"empalme del PIB real de EE. UU. por vintage: {n_emp} filas")
        elegido[EMPALME_PIB_USA] = CARGADOR_DERIVADAS  # sin esto, el resumen falla tras escribir la base
    union = cat_mod.anotar_reconstruidas(union, cat)
    union, anomalias = anotar_anomalias(union, cat)
    for sid, d in derivadas.items():
        if d.get("recalculada"):
            elegido[sid] = CARGADOR_DERIVADAS
    tabla = validar_series(union)
    efectivo = catalogo_efectivo(cat)
    cat_mod.validar_contra_catalogo(tabla, efectivo)
    duplicadas = tabla.duplicated(["serie", "fecha_referencia", "vintage"]).sum()
    if duplicadas:   # validar_series ya lo impide; se deja explícito por la unión
        raise ErrorContrato("series", [f"{int(duplicadas)} claves repetidas tras la unión"])
    destino = Path(ruta) if ruta is not None else None
    servidas = set(tabla["serie"])
    # En un ensayo sin ruta no se toca el corpus (ruta_series crea la carpeta): sin huérfanas que mirar.
    ruta_guardada = destino if destino is not None else (almacen.ruta_series() if escribir else None)
    guardadas = (set(almacen.leer_series(ruta=ruta_guardada)["serie"])
                 if ruta_guardada is not None and ruta_guardada.exists() else set())
    huerfanas = sorted(guardadas - servidas)
    from ingesta.mercados_series.fuentes.alfred import AVISO_SIN_ARCHIVO

    sin_archivo = sorted(sid for sid, i in informe_alfred.items()
                         if AVISO_SIN_ARCHIVO in (i.get("avisos") or []))
    if desde_cero and escribir and sin_archivo and not permitir_perdidas:
        # Con un crudo parcial (disco desmontado, descarga a medias) desde cero perdería esas series.
        raise ErrorContrato("series", [f"{len(sin_archivo)} series de ALFRED sin descarga completa archivada: "
                                       f"{sin_archivo[:10]}; desde cero las quitaría de la base. Descárgalas "
                                       f"(descargar-alfred) o fuerza con permitir_perdidas (--reemplazar)"])
    copia_previa = None
    if desde_cero and escribir and ruta_guardada is not None and ruta_guardada.exists():
        copia = ruta_guardada.with_name(respaldo)
        if copia.exists():
            # El respaldo fijo (bloque 2) se conserva; la base vigente va a una copia que rota.
            copia = ruta_guardada.with_name(f"{ruta_guardada.stem}{SUFIJO_PREVIA}")
            log(f"respaldo {respaldo} ya existente, se conserva; la base vigente se copia a {copia}")
        else:
            log(f"respaldo de la base guardada: {copia}")
        shutil.copy2(ruta_guardada, copia)
        copia_previa = str(copia)
    if huerfanas and escribir and not purgar and copia_previa is None:
        raise ErrorContrato("series", [f"{len(huerfanas)} series guardadas ya no se construyen (huérfanas): "
                                       f"{huerfanas[:10]}; get_serie las seguiría sirviendo. Revisa el catálogo "
                                       f"o quítalas con purgar=True (--purgar)"])
    revisables = sorted(sid for sid in servidas if cat.get(sid, {}).get("revisable")
                        and (tabla.loc[tabla["serie"] == sid, "vintage"] == "unico").all())
    escrita = None
    if escribir:
        # Desde cero: se quitan todas las series guardadas y se escribe la tabla nueva entera.
        quitar = sorted(guardadas) if copia_previa else (huerfanas if purgar else ())
        escrita = almacen.escribir_series(tabla, destino, permitir_perdidas=permitir_perdidas or bool(copia_previa),
                                          revisables=revisables, purgar=quitar)
        log(f"base escrita en {escrita}" + (f"; purgadas {huerfanas}" if purgar and huerfanas else ""))
    cargables = [sid for sid, e in efectivo.items() if e["estado"] in cat_mod.ESTADOS_CARGABLES]
    por_serie = {sid: cat[sid] for sid in servidas if sid in cat}
    resumen = ResumenBase(
        ruta=str(escrita) if escrita is not None else (str(destino) if destino else None),
        escrita=escrita is not None, n_filas=int(len(tabla)), n_series=len(servidas),
        n_valores=int(tabla["valor"].notna().sum()), n_huecos=int(tabla["valor"].isna().sum()),
        segundos=round(time.monotonic() - t0, 1),
        por_cargador=dict(sorted(Counter(elegido[s] for s in servidas).items())),
        filas_por_cargador={k: int(v) for k, v in sorted(
            tabla["serie"].map(elegido).value_counts().items())},
        conflictos=conflictos, derivadas=derivadas,
        sin_datos=sorted(set(cargables) - servidas),
        excepciones_aplicadas=n_exc, huerfanas=huerfanas,
        purgadas=huerfanas if (escribir and (purgar or copia_previa)) else [],
        revisables=len(revisables),
        anomalias={k: int(v) for k, v in anomalias["tipo"].value_counts().sort_index().items()},
        detalle_anomalias=[{"serie": r.serie, "fecha_referencia": _iso(r.fecha_referencia), "tipo": r.tipo}
                           for r in anomalias.itertuples(index=False)],
        por_estado=dict(sorted(Counter(e["estado"] for e in por_serie.values()).items())),
        por_metodo=dict(sorted(Counter(e["metodo"] for e in por_serie.values()).items())),
        por_fuente=dict(sorted(Counter(e["fuente"] for e in por_serie.values()).items())),
        respaldo=respaldo_fechas, alfred=_resumen_alfred(informe_alfred), desde_cero=copia_previa is not None,
        copia_previa=copia_previa, bytes_disco=Path(escrita).stat().st_size if escrita is not None else None,
    )
    return resumen, tabla


def _resumen_alfred(informe: dict[str, dict]) -> dict[str, Any]:
    """Recuentos del informe por serie de ``fuentes.alfred.tabla_alfred`` (sin valores) y sus avisos
    relevantes (sin archivo, primer vintage distinto del catálogo, filas publicadas antes de su periodo)."""
    if not informe:
        return {}
    suma = {k: int(sum(int(i.get(k) or 0) for i in informe.values()))
            for k in ("filas_alfred", "filas", "previas", "puntos", "retiradas", "fundidas",
                      "proyecciones_visibles", "proyecciones_descartadas")}
    claves = ("sin ", "primer vintage medido")
    avisos = {sid: [a for a in i.get("avisos", []) if a.startswith(claves) or "antes de su fecha" in a]
              for sid, i in sorted(informe.items())}
    return {"series": len(informe), **suma, "avisos": {k: v for k, v in avisos.items() if v}}


# --- Cobertura (solo metadatos) -----------------------------------------------------------------
COLUMNAS_COBERTURA = ("id", "alias", "fuente", "frecuencia", "metodo", "estado", "duplicado_de", "sucesor",
                      "revisable", "inicio", "fin", "n_obs", "n_huecos", "n_vintages", "lag_mediano_dias",
                      "lag_min_dias", "lag_max_dias")


def cobertura(base: pd.DataFrame, catalogo: dict[str, dict] | None = None) -> pd.DataFrame:
    """Una fila por serie de ``base`` con metadatos del catálogo y cobertura medida (sin valores).

    ``inicio``/``fin``: primera y última ``fecha_referencia`` con valor; ``n_obs``: periodos con
    valor; ``n_huecos``: periodos con valor nulo; ``lag_*_dias``: ``fecha_publicacion −
    fecha_referencia`` en días naturales sobre la primera publicación de cada periodo.
    """
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    if base.empty:
        return pd.DataFrame(columns=list(COLUMNAS_COBERTURA))
    orden = base.sort_values(["serie", "fecha_referencia", "fecha_publicacion"], kind="mergesort")
    primera = orden.drop_duplicates(["serie", "fecha_referencia"], keep="first")
    lag = (primera["fecha_publicacion"] - primera["fecha_referencia"]).dt.days
    con_valor = primera["valor"].notna()
    g_lag = lag.groupby(primera["serie"])
    g_val = primera.loc[con_valor].groupby("serie")["fecha_referencia"]
    medidas = pd.DataFrame({
        "inicio": g_val.min(), "fin": g_val.max(), "n_obs": g_val.size(),
        "n_huecos": (~con_valor).groupby(primera["serie"]).sum(),
        "n_vintages": base.groupby("serie")["vintage"].nunique(),
        "lag_mediano_dias": g_lag.median(), "lag_min_dias": g_lag.min(), "lag_max_dias": g_lag.max(),
    })
    filas = []
    for sid, m in medidas.iterrows():
        e = cat.get(sid, {})
        filas.append({
            "id": sid, "alias": " | ".join(e.get("alias") or []), "fuente": e.get("fuente"),
            "frecuencia": e.get("frecuencia"), "metodo": e.get("metodo"), "estado": e.get("estado"),
            "duplicado_de": e.get("duplicado_de"), "sucesor": e.get("sucesor"), "revisable": e.get("revisable"),
            "inicio": _iso(m["inicio"]), "fin": _iso(m["fin"]),
            "n_obs": int(m["n_obs"]) if pd.notna(m["n_obs"]) else 0, "n_huecos": int(m["n_huecos"]),
            "n_vintages": int(m["n_vintages"]), "lag_mediano_dias": float(m["lag_mediano_dias"]),
            "lag_min_dias": int(m["lag_min_dias"]), "lag_max_dias": int(m["lag_max_dias"]),
        })
    out = pd.DataFrame(filas, columns=list(COLUMNAS_COBERTURA))
    return out.sort_values("id", kind="mergesort").reset_index(drop=True)


def escribir_anomalias(detalle: list[dict[str, str]], ruta: Path | None = None) -> Path:
    """``tablas/anomalias_series.csv``: serie, fecha y tipo de cada anomalía anotada (sin valores)."""
    if ruta is None:
        carpeta = resultados_dir(TEMA) / "tablas"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / "anomalias_series.csv"
    pd.DataFrame(detalle, columns=["serie", "fecha_referencia", "tipo"]).to_csv(ruta, index=False, encoding="utf-8")
    return Path(ruta)


def escribir_cobertura(cob: pd.DataFrame, ruta: Path | None = None) -> Path:
    """``results/ingesta/mercados_series/tablas/cobertura_series.csv`` (o ``ruta``)."""
    if ruta is None:
        carpeta = resultados_dir(TEMA) / "tablas"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / "cobertura_series.csv"
    cob.to_csv(ruta, index=False, encoding="utf-8")
    return Path(ruta)


def resumen_base(base: pd.DataFrame | None = None, catalogo: dict[str, dict] | None = None) -> dict[str, Any]:
    """Recuentos de la base guardada (o de ``base``): filas, series, y series por fuente, método y estado."""
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    t = base if base is not None else almacen.leer_series()
    ids = sorted(set(t["serie"]))
    conteo = {campo: dict(sorted(Counter(str(cat.get(s, {}).get(campo)) for s in ids).items()))
              for campo in ("fuente", "frecuencia", "metodo", "estado")}
    return {"n_filas": int(len(t)), "n_series": len(ids), "n_valores": int(t["valor"].notna().sum()),
            "n_huecos": int(t["valor"].isna().sum()),
            "inicio": _iso(t["fecha_referencia"].min()) if len(t) else None,
            "fin": _iso(t["fecha_referencia"].max()) if len(t) else None,
            "no_catalogadas": [s for s in ids if s not in cat], **conteo}


def _np_a_py(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _np_a_py(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_np_a_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v
