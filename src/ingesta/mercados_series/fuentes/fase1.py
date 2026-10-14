"""Cargador de ``data/raw`` de la Fase 1 (solo lectura) → tabla larga del contrato de series.

Una serie por entrada del catálogo con ``fichero_fase1`` (``catalogo_series.yaml``), fechada con la
regla de su entrada:

- ``cierre_mercado``: cierre del ``mercado`` (``config/mercados.yaml``) + 30 min
  (``visibilidad.disponible_cierre_serie``); ``disponible_utc`` conocido.
- ``calendario``: hora del ``publicador`` (``visibilidad.disponible_publicador``: H.15 el hábil
  siguiente, H.10 el lunes siguiente...); ``disponible_utc`` conocido.
- ``lag_fijo`` y ``fichero``: fin del periodo + ``lag`` de ``config/lags.yaml``
  (``catalogo.fecha_publicacion_lag``); solo se conoce el día: ``disponible_utc`` nulo.
- ``derivada``: la máxima de sus componentes (``contrato.fecha_publicacion_derivada``). Un
  componente que no está en ``data/raw`` (DBAA y DAAA de Moody's) se fecha con la regla de su
  propia entrada sobre las fechas de la derivada, sin valores: da la misma fecha que tendrá con
  sus datos. Las fechas que no están en todos los componentes se descartan (la vol. realizada de
  ``data/raw`` llega más allá que el ``^GSPC`` de ``data/raw``).

Reglas de carga: solo estados cargables (``catalogo.ESTADOS_CARGABLES``) y ``fase1_utilizable``;
las ETF ``excluida_ajustada`` quedan fuera (``get_precios``, bloque 6). La primera
``fichero_fase1`` es la serie; las demás son copias (columnas alias: se comprueba que coinciden).
En ``yfinance/`` se descarta la barra con fecha >= día de descarga (instante declarado en
``config/descargas_fase1.yaml``, contrastado con el mtime del parquet: la descarga del 2026-07-20
fue a las 15:37, intradía; si el mtime no cuadra se avisa y manda lo declarado). En los ficheros
académicos (método ``fichero``) se descartan los periodos no cerrados el día de la descarga (la
última fila de Shiller es el mes en curso, provisional). Fechas normalizadas: mensuales,
trimestrales y anuales al día 1 del periodo; semanales al día de cierre de su semana (ICSA sábado;
NFCI, sus subíndices y STLFSI4 viernes). Huecos permanentes (periodos sin dato dentro de una
serie mensual, trimestral, semanal o anual): fila con valor nulo y ``NOTA_HUECO``. Diarias: un
día sin dato no es un hueco (festivos); una interrupción de más de ``UMBRAL_HUECO_DIARIO`` (DGS20
de 1987 a 1993, DTB1YR de 2002 a 2008) deja UNA fila nula con ``NOTA_HUECO_DIARIO`` el primer día
laborable tras el último dato. Vintage ``unico``; las revisables llevan ``catalogo.NOTA_REVISABLE``
en todas sus filas.

``tabla_fase1`` devuelve la tabla validada (``validar_series`` + ``validar_contra_catalogo``);
``resumen_fase1`` la cobertura y el retraso medidos por serie. CLI:
``python -m ingesta.mercados_series.fuentes.fase1 [--escribir]`` escribe la ficha ``fase1_raw`` y
``tablas/fase1_resumen.csv`` (metadatos, sin valores); con ``--escribir`` vuelca además la tabla
en el almacén (``almacen.escribir_series``).
This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ingesta.comun.rutas import ROOT, resultados_dir
from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series import visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    FECHA,
    UTC,
    fecha_publicacion_derivada,
    primera_decision_vec,
    vacio_series,
    validar_series,
)

TEMA = "mercados_series"
FUENTE_FICHA = "fase1_raw"
RAIZ_FASE1 = ROOT / "data" / "raw"
VINTAGE = "unico"
NOTA_HUECO = "hueco permanente: la fuente no tiene dato de este periodo (sin valor en data/raw de la Fase 1)"
NOTA_HUECO_DIARIO = "hueco permanente: la fuente interrumpió la serie diaria"
# Interrupción de una diaria que se declara como hueco (más larga que cualquier cierre de mercado).
UMBRAL_HUECO_DIARIO = pd.Timedelta(days=31)
HUSO_DESCARGA = "America/New_York"
RUTA_DESCARGAS = Path(cat_mod.__file__).resolve().parent / "config" / "descargas_fase1.yaml"
CARPETA_YFINANCE = "yfinance"
METODOS_FASE1 = ("cierre_mercado", "calendario", "lag_fijo", "fichero", "derivada")
# Día de cierre de la semana de referencia (0 = lunes) de las semanales de FRED.
DIA_SEMANAL = {"fred:ICSA": 5, "fred:NFCI": 4, "fred:ANFCI": 4, "fred:NFCIRISK": 4, "fred:NFCICREDIT": 4,
               "fred:NFCILEVERAGE": 4, "fred:STLFSI4": 4}
AVISO_FRED = ("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve "
              "Bank of St. Louis.")

_PERIODO = {"mensual": "M", "trimestral": "Q", "anual": "Y"}
_FRECUENCIA_HUECOS = {"mensual": "MS", "trimestral": "QS", "anual": "YS", "semanal": "7D"}
_COLUMNAS_RESUMEN = (
    "id", "alias", "fichero_fase1", "frecuencia", "metodo", "regla", "estado", "revisable", "incluida",
    "motivo", "n_filas", "n_obs", "n_huecos", "inicio", "fin", "lag_mediano_dias", "lag_min_dias",
    "lag_max_dias", "con_hora", "barras_descartadas", "filas_sin_componentes", "componentes_por_regla",
    "copias_distintas", "provisionales", "dia_descarga", "desfase_descarga_dias")

__all__ = ["RAIZ_FASE1", "NOTA_HUECO", "NOTA_HUECO_DIARIO", "UMBRAL_HUECO_DIARIO", "DIA_SEMANAL", "tabla_fase1",
           "resumen_fase1", "ficha_fase1", "dia_descarga", "instante_descarga", "descargas_declaradas",
           "normalizar_fechas"]


@dataclass
class _Info:
    """Lo medido al cargar una serie (para el resumen y la ficha)."""

    incluida: bool = False
    motivo: str | None = None
    barras_descartadas: int = 0
    filas_sin_componentes: int = 0
    componentes_por_regla: list[str] = field(default_factory=list)
    copias_distintas: int | None = None
    dia_descarga: pd.Timestamp | None = None
    provisionales: int = 0
    notas_hueco: dict[pd.Timestamp, str] = field(default_factory=dict)


# --- Lectura ------------------------------------------------------------------------------------
def descargas_declaradas(ruta: Path | None = None) -> dict[str, Any]:
    """``config/descargas_fase1.yaml``: ``{'por_defecto': UTC, 'ficheros': {rel: UTC}, 'tolerancia': Timedelta}``."""
    r = Path(ruta) if ruta is not None else RUTA_DESCARGAS
    with open(r, encoding="utf-8") as f:
        crudo = yaml.safe_load(f)
    huso = crudo["huso"]

    def _utc(texto: str) -> pd.Timestamp:
        return pd.Timestamp(str(texto)).tz_localize(huso).tz_convert("UTC")

    return {"por_defecto": _utc(crudo["por_defecto"]),
            "ficheros": {str(k): _utc(v) for k, v in (crudo.get("ficheros") or {}).items()},
            "tolerancia": pd.Timedelta(minutes=int(crudo.get("tolerancia_min", 10)))}


def _declaradas_por_defecto(raiz: Path | None) -> dict[str, Any] | None:
    """Las de ``config/descargas_fase1.yaml`` solo para el ``data/raw`` del repositorio."""
    if raiz is None or Path(raiz).resolve() == RAIZ_FASE1.resolve():
        return descargas_declaradas()
    return None


def instante_descarga(ruta: Path, raiz: Path | None = None, declaradas: dict[str, Any] | None = None) -> pd.Timestamp:
    """Instante UTC de la descarga de ``ruta`` (un parquet de ``raiz``).

    Con ``declaradas`` (por defecto ``config/descargas_fase1.yaml`` si ``raiz`` es el ``data/raw``
    del repositorio): el instante declarado del fichero (o ``por_defecto``); si el mtime cuadra
    con él (``tolerancia``) se devuelve el mtime exacto y si no, se avisa (``UserWarning``) y
    manda lo declarado. Sin declaración (otra carpeta): el mtime.
    """
    ruta = Path(ruta)
    mtime = pd.Timestamp(ruta.stat().st_mtime, unit="s", tz="UTC")
    if declaradas is None:
        declaradas = _declaradas_por_defecto(raiz if raiz is not None else ruta.parent.parent)
    if not declaradas:
        return mtime
    base = Path(raiz) if raiz is not None else ruta.parent.parent
    try:
        rel = ruta.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        rel = f"{ruta.parent.name}/{ruta.name}"
    t = declaradas["ficheros"].get(rel, declaradas["por_defecto"])
    if abs(mtime - t) <= declaradas["tolerancia"]:
        return mtime
    warnings.warn(f"{rel}: el mtime ({mtime.tz_convert(HUSO_DESCARGA):%Y-%m-%d %H:%M} NY) no coincide con la "
                  f"descarga declarada ({t.tz_convert(HUSO_DESCARGA):%Y-%m-%d %H:%M} NY): se usa la declarada "
                  f"(¿data/raw copiado o restaurado?)", UserWarning, stacklevel=2)
    return t


def dia_descarga(ruta: Path, raiz: Path | None = None, declaradas: dict[str, Any] | None = None) -> pd.Timestamp:
    """Día (Nueva York, naive) de la descarga de ``ruta`` (``instante_descarga``)."""
    t = instante_descarga(ruta, raiz, declaradas).tz_convert(HUSO_DESCARGA)
    return t.tz_localize(None).normalize()


def _leer_columna(raiz: Path, ref: str, cache: dict[str, pd.DataFrame]) -> tuple[pd.Series, Path]:
    """Columna ``'<carpeta>/<fichero>.parquet:<columna>'`` como Series float con índice de fechas."""
    rel, col = ref.split(":", 1)
    ruta = raiz / rel
    if rel not in cache:
        cache[rel] = pd.read_parquet(ruta)
    df = cache[rel]
    if col not in df.columns:
        raise KeyError(f"{ref}: la columna {col!r} no está en {rel} (columnas: {list(df.columns)})")
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(pd.to_numeric(df[col], errors="raise").astype(float).to_numpy(), index=idx.normalize())
    return s.sort_index(kind="mergesort"), ruta


def normalizar_fechas(fechas: pd.DatetimeIndex, frecuencia: str, dia_semana: int | None = None) -> pd.DatetimeIndex:
    """Fechas de referencia con el convenio del tema.

    Mensual, trimestral y anual → día 1 del periodo. Semanal → día de cierre de la semana
    (``dia_semana``, 0 = lunes; la fecha se lleva hacia delante hasta ese día); sin ``dia_semana``
    todas deben caer el mismo día de la semana (``ValueError`` si no). Diaria → tal cual.
    """
    idx = pd.DatetimeIndex(fechas).normalize()
    if frecuencia in _PERIODO:
        return idx.to_period(_PERIODO[frecuencia]).start_time.normalize()
    if frecuencia == "semanal":
        if dia_semana is not None:
            return idx + pd.to_timedelta((dia_semana - idx.weekday) % 7, unit="D")
        if len(set(idx.weekday)) > 1:
            raise ValueError(f"semanal con varios días de la semana {sorted(set(idx.weekday))}: declara su "
                             f"día de cierre en DIA_SEMANAL")
    return idx


def _huecos_diarios(s: pd.Series) -> tuple[pd.Series, dict[pd.Timestamp, str]]:
    """Marca las interrupciones largas de una diaria (más de ``UMBRAL_HUECO_DIARIO`` naturales).

    Añade UNA fila nula en el primer día laborable (lunes a viernes) tras el último dato antes del
    hueco, con nota propia: sin ella, ``get_serie`` devolvería durante años el último valor anterior
    (DGS20 de 1987 a 1993, DTB1YR de 2002 a 2008). Los días sueltos sin dato (festivos) no se marcan.
    """
    if len(s) < 2:
        return s, {}
    idx = s.index
    saltos = np.flatnonzero((idx[1:] - idx[:-1]) > UMBRAL_HUECO_DIARIO)
    notas: dict[pd.Timestamp, str] = {}
    for i in saltos:
        antes, despues = idx[i], idx[i + 1]
        dia = antes + pd.offsets.BDay(1)
        notas[pd.Timestamp(dia)] = (f"{NOTA_HUECO_DIARIO} (sin datos de {(antes + pd.Timedelta(days=1)).date()} a "
                                    f"{(despues - pd.Timedelta(days=1)).date()})")
    if not notas:
        return s, {}
    extra = pd.Series(np.nan, index=pd.DatetimeIndex(list(notas)))
    return pd.concat([s, extra]).sort_index(kind="mergesort"), notas


def _con_huecos(s: pd.Series, frecuencia: str) -> pd.Series:
    """Reindexa una serie periódica a todos sus periodos entre la primera y la última fecha."""
    regla = _FRECUENCIA_HUECOS.get(frecuencia)
    if regla is None or len(s) < 2:
        return s
    completo = pd.date_range(s.index[0], s.index[-1], freq=regla)
    if not s.index.isin(completo).all():
        raise ValueError(f"fechas fuera de la rejilla {regla} (p. ej. {s.index[~s.index.isin(completo)][:3].tolist()})")
    return s.reindex(completo)


# --- Fechado ------------------------------------------------------------------------------------
def _disponible_calendario(entrada: dict[str, Any], fechas: pd.Series,
                           cache: dict[tuple[str, pd.Timestamp], pd.Timestamp]) -> pd.Series:
    pub = entrada.get("publicador")
    if pub is None:
        raise ValueError("metodo calendario sin 'publicador': no se puede fechar desde data/raw")
    unicas = pd.unique(fechas.dropna())
    cfg = None
    for d in unicas:
        clave = (pub, pd.Timestamp(d))
        if clave not in cache:
            # La configuración por defecto, una vez por serie (no una lectura de su firma por día).
            cfg = cfg if cfg is not None else visibilidad._publicaciones_defecto()
            cache[clave] = visibilidad.disponible_publicador(pub, clave[1], cfg)
    out = fechas.map(lambda d: cache[(pub, pd.Timestamp(d))])
    return pd.Series(pd.to_datetime(out, utc=True).to_numpy(), index=fechas.index, dtype=UTC)


def _fechar(entrada: dict[str, Any], fechas: pd.Series,
            cache: dict[tuple[str, pd.Timestamp], pd.Timestamp]) -> tuple[pd.Series, pd.Series]:
    """``(fecha_publicacion, disponible_utc)`` de ``fechas`` (Series de fechas de referencia)."""
    metodo = entrada["metodo"]
    if metodo == "cierre_mercado":
        disp = visibilidad.disponible_cierre_serie(fechas, entrada["mercado"])
    elif metodo == "calendario":
        disp = _disponible_calendario(entrada, fechas, cache)
    elif metodo in cat_mod.METODOS_CON_LAG:
        pub = cat_mod.fecha_publicacion_regla(fechas, entrada)
        return pub.astype(FECHA), pd.Series(pd.NaT, index=fechas.index, dtype=UTC)
    else:
        raise ValueError(f"metodo {metodo!r} no se fecha por regla desde data/raw")
    return primera_decision_vec(disp).astype(FECHA), disp.astype(UTC)


def _filas(sid: str, entrada: dict[str, Any], s: pd.Series, pub: pd.Series, disp: pd.Series,
           notas_hueco: dict[pd.Timestamp, str] | None = None) -> pd.DataFrame:
    """Filas del contrato de una serie (``s`` con índice de fechas; ``pub``/``disp`` posicionales).

    Valor nulo → nota de hueco (``notas_hueco`` por fecha o ``NOTA_HUECO``); las revisables
    añaden ``NOTA_REVISABLE`` a todas sus filas.
    """
    nota_base = cat_mod.nota_revisable(entrada) if entrada["revisable"] else None
    valor = s.to_numpy(dtype=float)
    nulo = np.isnan(valor)
    notas_hueco = notas_hueco or {}
    nota = []
    for f, es_nulo in zip(s.index, nulo):
        if es_nulo:
            hueco = notas_hueco.get(pd.Timestamp(f), NOTA_HUECO)
            nota.append(f"{hueco}; {nota_base}" if nota_base else hueco)
        else:
            nota.append(nota_base)
    nota = pd.Series(nota, dtype=object)
    return pd.DataFrame({
        "serie": sid,
        "fecha_referencia": pd.Series(s.index, dtype=FECHA).to_numpy(),
        "fecha_publicacion": pub.to_numpy(),
        "disponible_utc": pd.Series(disp, dtype=UTC).array,
        "vintage": VINTAGE,
        "valor": valor,
        "unidad": entrada["unidad"],
        "divisa": entrada["divisa"],
        "fuente": entrada["fuente"],
        "metodo_publicacion": entrada["metodo"],
        "nota": nota.to_numpy(),
    })


# --- Carga --------------------------------------------------------------------------------------
def _motivo_exclusion(entrada: dict[str, Any]) -> str | None:
    if entrada["estado"] not in cat_mod.ESTADOS_CARGABLES:
        return f"estado {entrada['estado']}"
    if not entrada.get("fase1_utilizable", True):
        return "fase1_utilizable: false (parquet de data/raw no fiable)"
    if entrada["metodo"] not in METODOS_FASE1:
        return f"metodo {entrada['metodo']} (no se fecha desde data/raw)"
    return None


def _serie_cruda(sid: str, entrada: dict[str, Any], raiz: Path, cache: dict, info: _Info,
                 declaradas: dict[str, Any] | None = None) -> pd.Series:
    """Primera ``fichero_fase1`` limpia: sin nulos, sin barra intradía ni periodos provisionales,
    fechas normalizadas, huecos."""
    ref = entrada["fichero_fase1"][0]
    s, ruta = _leer_columna(raiz, ref, cache)
    s = s.dropna()
    if ref.split("/", 1)[0] == CARPETA_YFINANCE:
        info.dia_descarga = dia_descarga(ruta, raiz, declaradas)
        intradia = s.index >= info.dia_descarga
        info.barras_descartadas = int(intradia.sum())
        s = s[~intradia]
    elif entrada["metodo"] == "fichero" and entrada.get("fin_periodo") not in (None, "fecha"):
        # Periodo no cerrado el día de la descarga: fila provisional (Shiller: el mes en curso).
        dia = dia_descarga(ruta, raiz, declaradas)
        abierto = (cat_mod.fin_de_periodo(pd.Series(s.index), entrada["fin_periodo"]) >= dia).to_numpy()
        info.provisionales = int(abierto.sum())
        s = s[~abierto]
    # Copias (columnas alias o ficheros de validación): deben coincidir con la primera.
    if len(entrada["fichero_fase1"]) > 1:
        distintas = 0
        for otra in entrada["fichero_fase1"][1:]:
            c, _ = _leer_columna(raiz, otra, cache)
            comun = s.index.intersection(c.dropna().index)
            distintas += int((~np.isclose(s.loc[comun].to_numpy(), c.loc[comun].to_numpy(),
                                          rtol=1e-12, atol=0.0)).sum())
        info.copias_distintas = distintas
    idx = normalizar_fechas(s.index, entrada["frecuencia"], DIA_SEMANAL.get(sid))
    if idx.has_duplicates:
        raise ValueError(f"{sid}: fechas repetidas tras normalizar a {entrada['frecuencia']} "
                         f"(p. ej. {idx[idx.duplicated()][:3].date.tolist()})")
    s = pd.Series(s.to_numpy(), index=idx).sort_index(kind="mergesort")
    if entrada["frecuencia"] == "diaria":
        s, info.notas_hueco = _huecos_diarios(s)
        return s
    return _con_huecos(s, entrada["frecuencia"])


def _derivada(sid: str, entrada: dict[str, Any], s: pd.Series, tablas: dict[str, pd.DataFrame],
              cat: dict[str, dict], cache_pub: dict, info: _Info) -> pd.DataFrame:
    """Filas de una derivada: fecha = máxima de sus componentes por fecha de referencia."""
    base = []
    for c in entrada["componentes"]:
        if c in tablas:
            base.append(tablas[c][["serie", "fecha_referencia", "fecha_publicacion", "disponible_utc"]])
            continue
        ec = cat.get(c)
        if ec is None or ec["metodo"] not in ("cierre_mercado", "calendario", *cat_mod.METODOS_CON_LAG):
            raise ValueError(f"{sid}: el componente {c!r} no está en data/raw ni se puede fechar por regla")
        fechas = pd.Series(s.index, dtype=FECHA)
        pub, disp = _fechar(ec, fechas, cache_pub)
        base.append(pd.DataFrame({"serie": c, "fecha_referencia": fechas, "fecha_publicacion": pub,
                                  "disponible_utc": disp}))
        info.componentes_por_regla.append(c)
    fechado = fecha_publicacion_derivada(pd.concat(base, ignore_index=True), entrada["componentes"])
    fechado = fechado.set_index("fecha_referencia")
    comunes = s.index.intersection(fechado.index)
    info.filas_sin_componentes = int(len(s) - len(comunes))
    s = s.loc[comunes]
    return _filas(sid, entrada, s, fechado.loc[comunes, "fecha_publicacion"].reset_index(drop=True),
                  fechado.loc[comunes, "disponible_utc"].reset_index(drop=True), info.notas_hueco)


def _cargar(raiz: Path | None, catalogo: dict[str, dict] | None,
            descargas: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, _Info], dict]:
    """Tabla sin validar, lo medido por serie y el catálogo usado."""
    raiz = Path(raiz) if raiz is not None else RAIZ_FASE1
    declaradas = descargas if descargas is not None else _declaradas_por_defecto(raiz)
    cat = catalogo if catalogo is not None else cat_mod.cargar_catalogo()
    infos: dict[str, _Info] = {}
    tablas: dict[str, pd.DataFrame] = {}
    crudas: dict[str, pd.Series] = {}
    cache_parquet: dict[str, pd.DataFrame] = {}
    cache_pub: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    for sid in sorted(cat):
        e = cat[sid]
        if not e["fichero_fase1"]:
            continue
        info = infos[sid] = _Info(motivo=_motivo_exclusion(e))
        if info.motivo is not None:
            continue
        crudas[sid] = _serie_cruda(sid, e, raiz, cache_parquet, info, declaradas)
        if e["metodo"] == "derivada":
            continue
        fechas = pd.Series(crudas[sid].index, dtype=FECHA)
        pub, disp = _fechar(e, fechas, cache_pub)
        tablas[sid] = _filas(sid, e, crudas[sid], pub, disp, info.notas_hueco)
        info.incluida = True
    for sid in sorted(crudas):
        e = cat[sid]
        if e["metodo"] == "derivada":
            tablas[sid] = _derivada(sid, e, crudas[sid], tablas, cat, cache_pub, infos[sid])
            infos[sid].incluida = True
    if not tablas:
        return vacio_series(), infos, cat
    tabla = pd.concat([tablas[k] for k in sorted(tablas)], ignore_index=True)
    return tabla[list(COLUMNAS_SERIES)], infos, cat


def tabla_fase1(raiz: Path | None = None, catalogo: dict[str, dict] | None = None, *,
                descargas: dict[str, Any] | None = None) -> pd.DataFrame:
    """Tabla larga del contrato con todas las series del catálogo que tienen ``fichero_fase1``.

    ``raiz``: carpeta ``data/raw`` (por defecto la del repositorio; solo se lee). ``catalogo``:
    ``catalogo.cargar_catalogo()`` o un subconjunto (por defecto el catálogo del tema). Validada
    con ``contrato.validar_series`` y ``catalogo.validar_contra_catalogo``; ``ErrorContrato`` si
    algo no cumple. ``descargas``: instantes de descarga declarados (``descargas_declaradas``; por
    defecto los de config para el ``data/raw`` del repositorio, el mtime en otra carpeta). Reglas
    de carga: docstring del módulo.
    """
    tabla, _, cat = _cargar(raiz, catalogo, descargas)
    out = validar_series(tabla)
    cat_mod.validar_contra_catalogo(out, cat)
    return out


def _resumen(tabla: pd.DataFrame, infos: dict[str, _Info], cat: dict[str, dict]) -> pd.DataFrame:
    filas = []
    grupos = dict(tuple(tabla.groupby("serie", sort=False))) if len(tabla) else {}
    for sid in sorted(infos):
        e, info = cat[sid], infos[sid]
        regla = e.get("publicador") or e.get("mercado") or e.get("lag") or " + ".join(e.get("componentes") or [])
        if e["metodo"] in cat_mod.METODOS_CON_LAG:
            regla = f"{e['lag']} desde el fin de {e['fin_periodo']}"
        fila: dict[str, Any] = {
            "id": sid, "alias": " | ".join(e["alias"]), "fichero_fase1": e["fichero_fase1"][0],
            "frecuencia": e["frecuencia"], "metodo": e["metodo"], "regla": regla, "estado": e["estado"],
            "revisable": e["revisable"], "incluida": info.incluida, "motivo": info.motivo,
            "barras_descartadas": info.barras_descartadas, "filas_sin_componentes": info.filas_sin_componentes,
            "componentes_por_regla": " | ".join(info.componentes_por_regla) or None,
            "copias_distintas": info.copias_distintas, "provisionales": info.provisionales,
            "dia_descarga": info.dia_descarga.date().isoformat() if info.dia_descarga is not None else None,
        }
        g = grupos.get(sid)
        if g is not None and len(g):
            obs = g[g["valor"].notna()]
            lag = (obs["fecha_publicacion"] - obs["fecha_referencia"]).dt.days
            fin = obs["fecha_referencia"].max()
            fila |= {
                "n_filas": len(g), "n_obs": len(obs), "n_huecos": int(g["valor"].isna().sum()),
                "inicio": obs["fecha_referencia"].min().date().isoformat(), "fin": fin.date().isoformat(),
                "lag_mediano_dias": float(lag.median()), "lag_min_dias": int(lag.min()),
                "lag_max_dias": int(lag.max()), "con_hora": bool(g["disponible_utc"].notna().all()),
            }
            if info.dia_descarga is not None and e["metodo"] != "derivada":
                fila["desfase_descarga_dias"] = int((info.dia_descarga - fin).days)
        filas.append(fila)
    out = pd.DataFrame(filas).reindex(columns=list(_COLUMNAS_RESUMEN))
    for col in ("n_filas", "n_obs", "n_huecos"):
        out[col] = out[col].fillna(0).astype(int)
    return out


def resumen_fase1(raiz: Path | None = None, catalogo: dict[str, dict] | None = None,
                  tabla: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cobertura y retraso medidos por serie (una fila por entrada con ``fichero_fase1``).

    Columnas: ``id``, ``metodo``, ``regla``, ``incluida`` / ``motivo`` (exclusiones: ETF
    ajustadas, parquet no fiable), ``n_obs`` (valores no nulos), ``n_huecos`` (filas nulas con
    nota), ``inicio`` / ``fin``, ``lag_mediano_dias`` / ``lag_min_dias`` / ``lag_max_dias``
    (``fecha_publicacion - fecha_referencia`` en días naturales), ``con_hora`` (``disponible_utc``
    conocido), ``barras_descartadas`` (barra intradía de yfinance), ``filas_sin_componentes``
    (derivadas), ``copias_distintas`` (valores distintos en las copias alias), ``dia_descarga`` y
    ``desfase_descarga_dias`` (yfinance salvo derivadas: días entre el último dato y la
    descarga). ``tabla``: la de ``tabla_fase1`` si ya se calculó (las mediciones de carga se
    repiten igualmente).
    """
    t, infos, cat = _cargar(raiz, catalogo)
    if tabla is not None:
        t = tabla
    return _resumen(t, infos, cat)


# --- Ficha --------------------------------------------------------------------------------------
def ficha_fase1(resumen: pd.DataFrame, n_filas: int, raiz: Path | None = None):
    """``FichaFuente`` de ``fase1_raw`` (fuente agregada ``data/raw``) con lo medido en ``resumen``."""
    from ingesta.comun.esquema import FichaFuente

    raiz = Path(raiz) if raiz is not None else RAIZ_FASE1
    inc = resumen[resumen["incluida"]]
    ficheros = sorted(raiz.glob("*/*.parquet"))
    mtimes = pd.Series([dia_descarga(p, raiz) for p in ficheros]) if ficheros else pd.Series(dtype=FECHA)
    lag_metodo = {m: {"series": int(len(g)), "lag_mediano_dias": float(g["lag_mediano_dias"].median()),
                      "lag_min_dias": int(g["lag_min_dias"].min()), "lag_max_dias": int(g["lag_max_dias"].max())}
                  for m, g in inc.groupby("metodo")}
    fin_frec = {f: {"series": int(len(g)), "fin_mediano": sorted(g["fin"])[(len(g) - 1) // 2]}
                for f, g in inc.groupby("frecuencia")}
    retraso = "; ".join(f"{m}: mediana {v['lag_mediano_dias']:g} d (de {v['lag_min_dias']} a {v['lag_max_dias']})"
                        for m, v in lag_metodo.items())
    excluidas = resumen[~resumen["incluida"]]
    medida_alfred = _medida_alfred()
    return FichaFuente(
        fuente=FUENTE_FICHA,
        nombre="data/raw de la Fase 1 (instantánea agregada de FRED, yfinance, Ken French, Goyal-Welch, "
               "Shiller, datahub y OFR)",
        tema=TEMA,
        agentes=["macro", "risk", "equity", "portfolio"],
        rag="serie",
        url="data/raw (repositorio, gitignored; procedencia en data/raw/provenance.json)",
        acceso="descarga",
        autenticacion="ninguna (local; FRED_API_KEY solo en la descarga original de la Fase 1)",
        coste="0 €",
        licencia_y_terminos=("mixta: FRED solo local, sin redistribuir; ICE, S&P, Moody's, Cboe, FTSE Russell, "
                             "Nasdaq, Bloomberg y Wilshire con copyright (marca 'licencia' del catálogo); Ken "
                             "French, Goyal-Welch y Shiller de uso académico. " + AVISO_FRED),
        cobertura_inicio=inc["inicio"].min() if len(inc) else None,
        cobertura_fin=inc["fin"].max() if len(inc) else None,
        frecuencia="diaria, semanal, mensual y trimestral (según serie)",
        retraso_publicacion=(("regla de cada serie (no es una medición): fecha_publicacion - fecha_referencia "
                              "= " + retraso + ". " + medida_alfred["texto"]) if retraso else "sin series cargadas"),
        formato="parquet (una columna por serie; FF y Goyal-Welch multicolumna)",
        volumen_estimado=(f"{len(ficheros)} parquet, {sum(p.stat().st_size for p in ficheros) / 1e6:.1f} MB; "
                          f"{n_filas} filas en la tabla larga"),
        limites=("instantánea estática (sin actualización): la sustituyen los extractores del bloque 2-3; las "
                 "revisables llevan el último vintage (fuga de revisiones posible)"),
        estado_sonda="ok" if len(inc) else "fallo",
        mediciones={
            "series_con_fichero": int(len(resumen)), "series_cargadas": int(len(inc)),
            "excluidas": {str(m): int(n) for m, n in excluidas["motivo"].value_counts().items()},
            "filas": int(n_filas), "valores": int(inc["n_obs"].sum()), "huecos_declarados": int(inc["n_huecos"].sum()),
            "series_con_huecos": sorted(inc.loc[inc["n_huecos"] > 0, "id"].tolist()),
            "barras_intradia_descartadas": int(resumen["barras_descartadas"].sum()),
            "periodos_provisionales_descartados": {r.id: int(r.provisionales) for r in resumen.itertuples()
                                                   if getattr(r, "provisionales", 0)},
            "filas_derivadas_sin_componentes": {r.id: int(r.filas_sin_componentes)
                                                for r in inc.itertuples() if r.filas_sin_componentes},
            "copias_alias_distintas": int(resumen["copias_distintas"].fillna(0).sum()),
            "retraso_por_metodo_regla": lag_metodo,
            "publicacion_medida_alfred": medida_alfred["mediciones"],
            "fin_por_frecuencia": fin_frec,
            "dias_de_descarga_ny": {str(d.date()): int(n) for d, n in mtimes.value_counts().sort_index().items()},
            "desfase_descarga_yfinance_dias": {
                "mediana": float(inc["desfase_descarga_dias"].median()) if inc["desfase_descarga_dias"].notna().any()
                else None,
                "max": int(inc["desfase_descarga_dias"].max()) if inc["desfase_descarga_dias"].notna().any() else None},
        },
        notas=[
            "Cargador: ingesta.mercados_series.fuentes.fase1 (tabla_fase1, resumen_fase1); detalle por serie en "
            "results/ingesta/mercados_series/tablas/fase1_resumen.csv (metadatos, sin valores)",
            "Fechas de publicación por la regla del catálogo (cierre de mercado, publicador, lag de lags.yaml, "
            "máxima de componentes); disponible_utc solo donde se conoce la hora (cierres y publicadores)",
            "yfinance: se descarta la barra con fecha >= día de descarga; la descarga del 2026-07-20 fue a las "
            "15:37 NY (intradía). El instante está declarado en config/descargas_fase1.yaml (medido con el mtime): "
            "si data/raw se copia o restaura y el mtime cambia, se avisa y manda lo declarado",
            "Ficheros académicos: se descartan los periodos no cerrados el día de la descarga (Shiller trae el "
            "mes en curso, provisional)",
            "ETF de yfinance con auto_adjust=True fuera de la base (excluida_ajustada; get_precios en el bloque 6)",
            "Huecos permanentes: en periódicas (mensual, trimestral, semanal), una fila nula con nota por periodo "
            "sin dato; en diarias, una fila nula con nota tras cada interrupción de más de 31 días naturales",
            AVISO_FRED,
        ],
    )


def _medida_alfred() -> dict[str, Any]:
    """Resumen de ``tablas/publicacion_alfred_medida.csv`` (regla frente a la primera vintage de ALFRED)."""
    ruta = resultados_dir(TEMA) / "tablas" / "publicacion_alfred_medida.csv"
    if not ruta.exists():
        return {"texto": "Sin medir contra publicaciones reales (falta excepciones-alfred).", "mediciones": None}
    m = pd.read_csv(ruta)
    ok = m[m["error"].isna()] if "error" in m else m
    ok = ok[pd.to_numeric(ok["n_cubiertas"], errors="coerce").fillna(0) > 0]
    fugas = pd.to_numeric(ok["n_fugas"], errors="coerce").fillna(0)
    cubiertas = pd.to_numeric(ok["n_cubiertas"], errors="coerce").fillna(0)
    holgura = pd.to_numeric(ok["holgura_mediana_dias"], errors="coerce")
    med = {"series_medidas": int(len(ok)), "periodos_cubiertos": int(cubiertas.sum()),
           "periodos_con_fuga_de_la_regla": int(fugas.sum()),
           "series_con_fugas": int((fugas > 0).sum()),
           "fuga_max_dias": int(pd.to_numeric(ok["fuga_max_dias"], errors="coerce").max()) if len(ok) else None,
           "holgura_mediana_dias_mediana": float(holgura.median()) if holgura.notna().any() else None,
           "tabla": ruta.relative_to(ROOT).as_posix()}
    texto = (f"Medido en ALFRED (primera vintage de cada observación, output_type=4) en {med['series_medidas']} "
             f"series de FRED: {med['periodos_con_fuga_de_la_regla']} de {med['periodos_cubiertos']} periodos se "
             f"publicaron después de la regla (hasta {med['fuga_max_dias']} días: cierres del Gobierno, Moody's "
             f"2016-17, medias de H.15 de 1997-2016) y se corrigen con config/excepciones_publicacion.csv; holgura "
             f"mediana de la regla {med['holgura_mediana_dias_mediana']} días")
    return {"texto": texto, "mediciones": med}


# --- CLI ----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from ingesta.comun.fichas import guardar_ficha

    p = argparse.ArgumentParser(description="data/raw de la Fase 1 -> tabla larga del contrato de series")
    p.add_argument("--raiz", type=Path, default=None, help="carpeta data/raw (por defecto la del repositorio)")
    p.add_argument("--escribir", action="store_true", help="vuelca la tabla en el almacén del tema")
    a = p.parse_args(argv)

    tabla, infos, cat = _cargar(a.raiz, None)
    tabla = validar_series(tabla)
    cat_mod.validar_contra_catalogo(tabla, cat)
    res = _resumen(tabla, infos, cat)
    carpeta = resultados_dir(TEMA) / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / "fase1_resumen.csv"
    res.to_csv(ruta_csv, index=False, encoding="utf-8")
    ruta_ficha = guardar_ficha(ficha_fase1(res, len(tabla), a.raiz))
    print(f"{int(res['incluida'].sum())} series cargadas de {len(res)} con fichero_fase1; {len(tabla)} filas")
    print(f"resumen -> {ruta_csv}\nficha   -> {ruta_ficha}")
    if a.escribir:
        from ingesta.mercados_series import almacen

        # Mismas fechas que la base: excepciones medidas en ALFRED (la vía completa es construir-base).
        tabla, _ = cat_mod.aplicar_excepciones(tabla, catalogo=cat)
        print(f"almacén -> {almacen.escribir_series(tabla)}")


if __name__ == "__main__":
    main()
