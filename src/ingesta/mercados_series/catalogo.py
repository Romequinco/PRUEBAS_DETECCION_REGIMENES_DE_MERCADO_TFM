"""Catálogo de series del Tema 1 (bloques 2 y 3): una entrada por serie de la base.

``config/catalogo_series.yaml`` es la ÚNICA fuente de verdad de las series del tema: metadatos
(nombre, unidad, divisa, licencia, estado, duplicados...) y su parte de publicación (método,
publicador, mercado, componentes...). ``visibilidad.cargar_publicaciones`` fusiona esa parte con
``publicaciones.yaml`` y la valida con las mismas reglas (un id o alias en los dos ficheros es un
error). Los retrasos de ``lag_fijo`` y ``fichero`` salen de ``config/lags.yaml`` por ``lag_clave``
(la entrada no repite el ``lag``).

Ids canónicos con prefijo de fuente (``fred:DGS10``, ``yf:^GSPC``, ``ff:F3D.SMB``, ``gw:M.tbl``,
``shiller:CAPE``, ``datahub:US10Y_MENSUAL``, ``ofr:FSI``, ``deriv:DBAA-DAAA``); ``alias`` = nombres
internos de la Fase 1 (``VIX``, ``SP500``...). CLI: ``python -m ingesta.mercados_series.catalogo``
escribe la tabla de metadatos con la cobertura medida en ``data/raw`` (sin valores).

Bloque 3 (ALFRED y FRED como fuente viva):

- ``vintage_exacto`` de ALFRED: cada fila es un valor nuevo o revisado con su ``realtime_start``.
  ``regla_previa_alfred`` (``{lag_clave, primer_vintage}``) fecha las observaciones anteriores al
  primer vintage (cargadas con el valor de ese vintage, ``lag_fijo`` y ``nota_previa_alfred``); sin
  ella, esas observaciones solo son visibles desde el primer vintage (WEI, GDPNow).
- ``fuente_viva``: ``fred`` (por defecto en las ``fred:`` de ``fred_alfred``; ``null`` si FRED ya
  no la sirve). Orígenes por prioridad (``origenes``, ``PRIORIDAD_ORIGENES``): ``alfred``
  (vintage_exacto) > ``academicos`` > ``derivadas`` > ``fred`` vivo > ``fase1`` (``data/raw``: lo
  que FRED no sirve e historia de las capadas).
- Metadatos de la lista de ALFRED: ``grupo``, ``agentes``, ``release``, ``proyecciones``
  (``recortar``: quitar las observaciones posteriores al vintage; ``conservar``: la serie es una
  previsión), ``reestimada`` (índice re-estimado entero en cada vintage), ``cobertura_variable``
  (el inicio cambia según el vintage), ``unidad_por_vintage`` (año base o dólares encadenados de
  cada vintage; ``unidad`` es la del vintage vigente) y ``coherente_con`` (series de las que FRED
  la deriva: test de coherencia dentro del mismo vintage).
"""
from __future__ import annotations

import copy
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ingesta.mercados_series import visibilidad
from ingesta.mercados_series.contrato import _PATRON_DIVISA, METODOS_PUBLICACION, ErrorContrato

RUTA_CATALOGO = Path(__file__).resolve().parent / "config" / "catalogo_series.yaml"
RUTA_LAGS = Path(__file__).resolve().parent / "config" / "lags.yaml"

NOTA_REVISABLE = "valores del último vintage: posible fuga de revisiones (ALFRED en el bloque 3)"
# Fuentes fuera de ALFRED (Ken French, Shiller, Goyal-Welch, OFR, JST): su historia de ediciones solo
# la da el archivo propio de crudos (cada descarga se conserva desde el bloque 2).
NOTA_REVISABLE_EDICION = ("valores de la última edición del fichero: posible fuga de revisiones (archivo de "
                          "ediciones propio desde 2026-10-12; ALFRED no tiene esta fuente)")
# Bloque 3: primera edición archivada de French, Shiller y Goyal-Welch (vintage por edición ``AAAA-MM``).
NOTA_PRIMERA_EDICION = ("valor de la primera edición archivada: posible fuga de revisiones anteriores a ella "
                        "(el archivo de ediciones empieza ahí)")
NOTAS_REVISABLE = (NOTA_REVISABLE, NOTA_REVISABLE_EDICION, NOTA_PRIMERA_EDICION)
# Fuentes del catálogo cuyas revisables se resolverán con ALFRED (las demás, con el archivo propio).
FUENTES_ALFRED = ("fred_alfred", "calculo_propio")
NOTA_RECONSTRUIDA = "serie reconstruida hacia atrás: el índice no se publicaba antes de"
NOTA_EXCEPCION = "fecha de publicación medida (primera vintage de ALFRED); la regla daba"
ESTADOS = ("activa", "retirada", "capada", "excluida_ajustada", "pendiente_alfred", "duplicada")
# Bloque 3: grupos y agentes de la lista de ALFRED (results/.../tablas/alfred_lista_propuesta.csv).
GRUPOS = ("actividad", "comercio", "consumo", "dinero_credito", "empleo", "encuestas", "estres_financiero",
          "otros", "pib", "precios", "recesion", "vivienda")
AGENTES = ("macro", "risk", "equity", "portfolio")
PROYECCIONES = ("recortar", "conservar")
FUENTES_VIVAS = ("fred",)
# Orígenes (cargadores) de una serie por prioridad (bloque 3): vintages de ALFRED > ediciones académicas >
# derivadas recalculadas > FRED vivo > data/raw de la Fase 1. Una serie solo tiene algunos (``origenes``).
PRIORIDAD_ORIGENES = ("alfred", "academicos", "derivadas", "fred", "fase1")
# Fuentes con archivo de ediciones propio (fuentes.academicos): la copia de data/raw es su edición previa.
FUENTES_EDICION = ("ken_french", "shiller", "goyal_welch", "ofr_fsi")
NOTA_PREVIA_ALFRED = "anterior a ALFRED: valor del primer vintage"
# Estados que entran en la base de series (con datos de data/raw o de los extractores).
ESTADOS_CARGABLES = ("activa", "retirada", "capada", "duplicada")
FRECUENCIAS = ("diaria", "semanal", "mensual", "trimestral", "anual")
FINES_PERIODO = ("fecha", "mes", "trimestre", "anio")
PREFIJOS = ("fred", "yf", "ff", "gw", "shiller", "datahub", "ofr", "deriv")
ROLES_FASE1 = ("spine", "core", "enricher", "fallback", "validation")
METODOS_CON_LAG = ("lag_fijo", "fichero")

CAMPOS_OBLIGATORIOS = frozenset({
    "alias", "nombre", "fuente", "id_origen", "frecuencia", "unidad", "divisa", "metodo",
    "revisable", "rol_fase1", "licencia", "estado", "verificado"})
CAMPOS_OPCIONALES = frozenset({
    "fichero_fase1", "fase1_utilizable", "sucesor", "duplicado_de", "notas", "lag_clave",
    "publicador", "mercado", "componentes", "hora_local", "huso", "margen_min", "calendario",
    "publicado_desde", "grupo", "agentes", "release", "regla_previa_alfred", "proyecciones", "fuente_viva",
    "reestimada", "cobertura_variable", "unidad_por_vintage", "coherente_con"})
_CLAVES_PREVIA = {"lag_clave", "primer_vintage"}
# Claves que pasan a la sección 'series' de publicaciones (el resto son metadatos del catálogo).
CLAVES_PUBLICACION = frozenset(visibilidad._CLAVES_SERIE | visibilidad._CLAVES_SERIE_OPC)

_PATRON_ID = re.compile(rf"(?:{'|'.join(PREFIJOS)}):\S+")
_PATRON_FICHERO = re.compile(r"(?:fred|yfinance|academico|github|ofr)/[^/:]+\.parquet:.+")
_PATRON_LAG_PARTES = re.compile(r"(?:(\d+)M)?(?:(\d+)D)?")
_CLAVES_LAG = {"lag", "fin_periodo", "verificado"}
_CLAVES_LAG_OPC = {"meses", "dias", "tipo", "fuente", "justificacion", "habiles", "n_habiles", "correccion"}
CALENDARIOS_LAG = ("semana", "us_federal")
RUTA_EXCEPCIONES = Path(__file__).resolve().parent / "config" / "excepciones_publicacion.csv"
# Columnas del CSV de config (solo fechas). La medición completa (regla, origen, holguras) va a
# results/ingesta/mercados_series/tablas/publicacion_alfred_medida.csv.
COLUMNAS_EXCEPCIONES = ("serie", "fecha_referencia", "fecha_publicacion")
COLUMNAS_EXCEPCIONES_DETALLE = (*COLUMNAS_EXCEPCIONES, "fecha_regla", "origen")

__all__ = [
    "RUTA_CATALOGO", "RUTA_LAGS", "RUTA_EXCEPCIONES", "NOTA_REVISABLE", "NOTA_REVISABLE_EDICION",
    "NOTA_PRIMERA_EDICION",
    "NOTAS_REVISABLE", "NOTA_RECONSTRUIDA", "NOTA_EXCEPCION", "NOTA_PREVIA_ALFRED", "ESTADOS", "ESTADOS_CARGABLES",
    "FRECUENCIAS", "GRUPOS", "AGENTES", "PROYECCIONES", "FUENTES_VIVAS", "FUENTES_EDICION", "PRIORIDAD_ORIGENES",
    "origenes", "metodos_admitidos", "nota_previa_alfred", "entrada_previa", "vista_regla_previa",
    "FINES_PERIODO", "PREFIJOS", "cargar_lags", "fin_de_periodo", "fecha_publicacion_lag",
    "fecha_publicacion_regla", "nota_revisable", "es_nota_revisable", "cargar_excepciones",
    "aplicar_excepciones", "anotar_reconstruidas", "resolver_caja",
    "cargar_catalogo", "parte_publicacion", "por_id", "alias", "resolver", "filtrar", "tabla",
    "resumen", "validar_contra_catalogo",
]


# --- Lags ---------------------------------------------------------------------------------------
def _partes_lag(lag: str) -> tuple[int, int]:
    """``"1M1D"`` → ``(1, 1)``; ``"45D"`` → ``(0, 45)``; ``"24M"`` → ``(24, 0)``."""
    if not isinstance(lag, str) or not visibilidad._PATRON_LAG.fullmatch(lag):
        raise ValueError(f"lag {lag!r} no es \"<n>D\", \"<n>M\" ni \"<n>M<m>D\"")
    m = _PATRON_LAG_PARTES.fullmatch(lag)
    return int(m.group(1) or 0), int(m.group(2) or 0)


def _validar_lags(crudo: Any, origen: str) -> dict[str, dict[str, Any]]:
    if not isinstance(crudo, dict):
        raise ValueError(f"{origen}: se esperaba un diccionario")
    visibilidad._claves(crudo, {"version", "lag_publicacion_fase1", "adicionales"}, set(), origen)
    out: dict[str, dict[str, Any]] = {}
    for seccion in ("lag_publicacion_fase1", "adicionales"):
        for clave, e in (crudo[seccion] or {}).items():
            donde = f"{origen}: {seccion} {clave!r}"
            visibilidad._claves(e, _CLAVES_LAG, _CLAVES_LAG_OPC, donde)
            if clave in out:
                raise ValueError(f"{donde}: clave repetida en las dos secciones")
            if e["fin_periodo"] not in FINES_PERIODO:
                raise ValueError(f"{donde}: fin_periodo {e['fin_periodo']!r} no es uno de {list(FINES_PERIODO)}")
            try:
                meses, _ = _partes_lag(e["lag"])
            except ValueError as err:
                raise ValueError(f"{donde}: {err}") from None
            if meses and e["fin_periodo"] == "fecha":
                raise ValueError(f"{donde}: un lag en meses necesita fin_periodo mes, trimestre o anio")
            if not isinstance(e["verificado"], bool):
                raise ValueError(f"{donde}: 'verificado' debe ser true o false")
            hab, n_hab = e.get("habiles"), e.get("n_habiles")
            if hab is not None and hab not in CALENDARIOS_LAG:
                raise ValueError(f"{donde}: habiles {hab!r} no es uno de {list(CALENDARIOS_LAG)}")
            if n_hab is not None and (hab is None or not isinstance(n_hab, int) or isinstance(n_hab, bool)
                                      or n_hab < 1):
                raise ValueError(f"{donde}: 'n_habiles' debe ser un entero >= 1 y necesita 'habiles'")
            if seccion == "lag_publicacion_fase1" and not {"meses", "dias", "tipo", "fuente"} <= set(e):
                raise ValueError(f"{donde}: la copia de LAG_PUBLICACION necesita meses, dias, tipo y fuente")
            if seccion == "adicionales" and not str(e.get("justificacion") or "").strip():
                raise ValueError(f"{donde}: falta 'justificacion'")
            out[str(clave)] = {**e, "seccion": seccion}
    return out


@lru_cache(maxsize=4)
def _lags_cache(ruta: str, _mtime_ns: int) -> dict[str, dict[str, Any]]:
    return _validar_lags(visibilidad._leer_yaml(Path(ruta)), Path(ruta).name)


def cargar_lags(ruta: Path | None = None) -> dict[str, dict[str, Any]]:
    """``config/lags.yaml`` validado: ``{clave: {lag, fin_periodo, verificado, seccion, ...}}``.

    Une las dos secciones (copia congelada de LAG_PUBLICACION y adicionales); una clave repetida,
    un ``lag`` mal formado o un ``fin_periodo`` desconocido son ``ValueError``.
    """
    r = Path(ruta) if ruta is not None else RUTA_LAGS
    return copy.deepcopy(_lags_cache(str(r.resolve()), r.stat().st_mtime_ns))


def fin_de_periodo(fechas: Any, fin_periodo: str) -> pd.Series:
    """Último día natural del periodo (``mes``, ``trimestre``, ``anio``) o la propia fecha (``fecha``)."""
    s = pd.Series(pd.to_datetime(fechas)).dt.normalize()
    if fin_periodo == "fecha":
        return s
    regla = {"mes": "M", "trimestre": "Q", "anio": "Y"}.get(fin_periodo)
    if regla is None:
        raise ValueError(f"fin_periodo {fin_periodo!r} no es uno de {list(FINES_PERIODO)}")
    return s.dt.to_period(regla).dt.end_time.dt.normalize()


def _primer_habil(d: pd.Timestamp, calendario: str, cache: dict) -> pd.Timestamp:
    """Primer hábil >= ``d`` (``visibilidad.es_habil``, con los cierres extraordinarios)."""
    if d not in cache:
        x = d
        while not visibilidad.es_habil(x, calendario):
            x += pd.Timedelta(days=1)
        cache[d] = x
    return cache[d]


def fecha_publicacion_lag(fechas: Any, lag: str, fin_periodo: str, *, habiles: str | None = None,
                          n_habiles: int | None = None) -> pd.Series:
    """Fecha de decisión desde la que se ve el dato: fin del periodo + ``lag`` (lags.yaml).

    ``"nM"`` = n fines de mes (``MonthEnd``) tras el fin; ``"nD"`` = n días; ``"nMmD"`` = primero
    los meses. Ejemplos (mensual, referencia 2024-01-01): ``"1D"`` → 2024-02-01; ``"1M1D"`` →
    2024-03-01; ``"24M"`` → 2026-01-31. Nulos → NaT. Devuelve ``datetime64[ns]`` a medianoche.

    ``habiles`` (``semana`` o ``us_federal``): la fecha se lleva al primer hábil >= ella (el dato
    no sale en fin de semana ni festivo). ``n_habiles``: además, nunca antes del n-ésimo hábil
    posterior al fin del periodo (NFCI: tercer hábil tras el viernes de referencia, es decir, el
    miércoles, o el jueves si la semana trae un festivo). Solo retrasan, nunca adelantan.
    """
    meses, dias = _partes_lag(lag)
    if meses and fin_periodo == "fecha":
        raise ValueError("un lag en meses necesita fin_periodo mes, trimestre o anio")
    if n_habiles is not None and habiles is None:
        raise ValueError("'n_habiles' necesita 'habiles'")
    fin = fin_de_periodo(fechas, fin_periodo)
    pub = fin + pd.offsets.MonthEnd(meses) if meses else fin
    pub = (pub + pd.Timedelta(days=dias)).astype("datetime64[ns]")
    if habiles is None:
        return pub
    cache: dict = {}
    out = []
    for p, f in zip(pub, fin):
        if pd.isna(p):
            out.append(pd.NaT)
            continue
        if n_habiles:
            x = pd.Timestamp(f)
            for _ in range(n_habiles):
                x = _primer_habil(x + pd.Timedelta(days=1), habiles, cache)
            p = max(pd.Timestamp(p), x)
        out.append(_primer_habil(pd.Timestamp(p), habiles, cache))
    return pd.Series(out, index=pub.index, dtype="datetime64[ns]")


def fecha_publicacion_regla(fechas: Any, entrada: dict[str, Any]) -> pd.Series:
    """``fecha_publicacion_lag`` con la regla completa de una entrada del catálogo (lag, fin, hábiles)."""
    return fecha_publicacion_lag(fechas, entrada["lag"], entrada["fin_periodo"],
                                 habiles=entrada.get("lag_habiles"), n_habiles=entrada.get("lag_n_habiles"))


# --- Notas de revisables ------------------------------------------------------------------------
def nota_revisable(entrada: dict[str, Any]) -> str:
    """Nota obligatoria de una revisable cargada con el último vintage: ALFRED o archivo de ediciones."""
    return NOTA_REVISABLE if entrada["fuente"] in FUENTES_ALFRED else NOTA_REVISABLE_EDICION


def es_nota_revisable(nota: Any) -> Any:
    """True si ``nota`` (texto o Series) lleva alguna de las ``NOTAS_REVISABLE``."""
    if isinstance(nota, pd.Series):
        t = nota.fillna("").astype(str)
        out = pd.Series(False, index=nota.index)
        for n in NOTAS_REVISABLE:
            out |= t.str.contains(n, regex=False)
        return out
    return isinstance(nota, str) and any(n in nota for n in NOTAS_REVISABLE)


# --- Excepciones de publicación (fechas medidas) -------------------------------------------------
def cargar_excepciones(ruta: Path | None = None) -> pd.DataFrame:
    """``config/excepciones_publicacion.csv``: primera publicación medida por (serie, periodo).

    Solo fechas (sin valores): ``serie, fecha_referencia, fecha_publicacion`` (primera vintage de
    ALFRED, ``output_type=4``, posterior a la regla al generarla). La genera ``python -m
    ingesta.mercados_series excepciones-alfred`` (origen y fecha de descarga por serie en
    ``tablas/publicacion_alfred_medida.csv``). Columnas de más se ignoran. Sin fichero: tabla vacía.
    """
    r = Path(ruta) if ruta is not None else RUTA_EXCEPCIONES
    if not r.exists():
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c.startswith("fecha") else object)
                             for c in COLUMNAS_EXCEPCIONES})
    df = pd.read_csv(r, dtype=str, keep_default_na=False)
    faltan = sorted(set(COLUMNAS_EXCEPCIONES) - set(df.columns))
    if faltan:
        raise ValueError(f"{r.name}: faltan columnas {faltan}")
    for c in ("fecha_referencia", "fecha_publicacion"):
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d").astype("datetime64[ns]")
    if df.duplicated(["serie", "fecha_referencia"]).any():
        raise ValueError(f"{r.name}: (serie, fecha_referencia) repetidos")
    return df[list(COLUMNAS_EXCEPCIONES)]


def aplicar_excepciones(df: pd.DataFrame, excepciones: pd.DataFrame | None = None, *,
                        publicaciones: dict[str, Any] | None = None,
                        catalogo: dict[str, dict] | None = None) -> tuple[pd.DataFrame, int]:
    """``fecha_publicacion = max(regla, excepción)`` en las filas vintage ``unico`` de ``df``.

    Solo retrasa (nunca adelanta). Una fila que cambia: si tenía ``disponible_utc`` (hora de un
    publicador), pasa a ese día a la hora del publicador de su entrada y su fecha es la primera
    decisión tras ella; sin hora, ``fecha_publicacion`` = el día de la excepción y
    ``disponible_utc`` nulo. Se anota ``NOTA_EXCEPCION`` con la fecha de la regla. Idempotente.
    Devuelve ``(tabla, filas cambiadas)``.
    """
    exc = cargar_excepciones() if excepciones is None else excepciones
    if df.empty or exc.empty:
        return df, 0
    cat = _cat(catalogo)
    out = df.reset_index(drop=True).copy()
    cruce = out[["serie", "fecha_referencia"]].merge(
        exc[["serie", "fecha_referencia", "fecha_publicacion"]].rename(columns={"fecha_publicacion": "_exc"}),
        on=["serie", "fecha_referencia"], how="left")
    exc_fecha = pd.Series(cruce["_exc"].to_numpy(), index=out.index).astype("datetime64[ns]")
    candidata = (out["vintage"] == "unico") & exc_fecha.notna() & (exc_fecha >= out["fecha_publicacion"]
                                                                   - pd.Timedelta(days=1))
    if not candidata.any():
        return df, 0
    pubs = (publicaciones or visibilidad.cargar_publicaciones(catalogo=False))["publicadores"]
    actual_disp = pd.to_datetime(out["disponible_utc"], utc=True)
    nueva_pub = out["fecha_publicacion"].astype("datetime64[ns]").copy()
    nueva_disp = actual_disp.copy()
    cambia = pd.Series(False, index=out.index)
    for i in out.index[candidata]:
        dia = exc_fecha[i]
        e = cat.get(out.at[i, "serie"], {})
        pub = pubs.get(e.get("publicador")) if pd.notna(actual_disp[i]) else None
        # Fecha efectiva: el día de ALFRED a la hora del publicador (17:05 → decisión del día siguiente).
        if pub is not None:
            disp = visibilidad.disponible_publicacion(dia, pub["hora"], pub["huso"],
                                                      pd.Timedelta(minutes=pub["margen_min"]))
            efectiva = visibilidad.primera_decision(disp)
        else:
            disp, efectiva = pd.NaT, dia
        if efectiva > out.at[i, "fecha_publicacion"]:
            cambia[i] = True
            nueva_disp[i] = disp
            nueva_pub[i] = efectiva
    if not cambia.any():
        return df, 0
    texto = NOTA_EXCEPCION + " " + out.loc[cambia, "fecha_publicacion"].dt.strftime("%Y-%m-%d")
    previa = out.loc[cambia, "nota"]
    out.loc[cambia, "nota"] = texto.where(previa.isna(), previa.astype(str) + "; " + texto)
    out["fecha_publicacion"] = nueva_pub
    out["disponible_utc"] = nueva_disp
    return out, int(cambia.sum())


def anotar_reconstruidas(df: pd.DataFrame, catalogo: dict[str, dict] | None = None) -> pd.DataFrame:
    """Nota ``NOTA_RECONSTRUIDA`` en las filas publicadas antes del ``publicado_desde`` de su serie.

    Índices construidos hacia atrás (NFCI desde 1971, publicado desde 2011): sus valores antiguos
    no existían en su fecha. No cambia valores ni fechas (``get_serie`` avisa). Idempotente.
    """
    cat = _cat(catalogo)
    out = df.copy()
    for sid, e in cat.items():
        desde = e.get("publicado_desde")
        if not desde:
            continue
        m = (out["serie"] == sid) & (out["fecha_publicacion"] < pd.Timestamp(desde))
        m &= ~out["nota"].fillna("").astype(str).str.contains(NOTA_RECONSTRUIDA, regex=False)
        if not m.any():
            continue
        texto = f"{NOTA_RECONSTRUIDA} {desde}"
        previa = out.loc[m, "nota"]
        out.loc[m, "nota"] = previa.where(previa.isna(), previa.astype(str) + "; " + texto).fillna(texto)
    return out


# --- Catálogo -----------------------------------------------------------------------------------
def _lista(valor: Any, donde: str, campo: str) -> list[str]:
    if valor is None:
        return []
    if isinstance(valor, str):
        valor = [valor]
    if not isinstance(valor, list) or not all(isinstance(v, str) and v for v in valor):
        raise ValueError(f"{donde}: '{campo}' debe ser un texto o una lista de textos")
    return list(valor)


def _texto(entrada: dict, campo: str, donde: str) -> str:
    v = entrada.get(campo)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{donde}: '{campo}' debe ser un texto no vacío")
    return v


def _validar_entrada(sid: str, e: Any, lags: dict[str, dict], publicadores: dict, origen: str) -> dict[str, Any]:
    donde = f"{origen}: serie {sid!r}"
    visibilidad._claves(e, set(CAMPOS_OBLIGATORIOS), set(CAMPOS_OPCIONALES), donde)
    if not _PATRON_ID.fullmatch(sid):
        raise ValueError(f"{donde}: id sin prefijo de fuente válido ({'|'.join(PREFIJOS)}:...)")
    prefijo = sid.split(":", 1)[0]
    for campo in ("nombre", "fuente", "id_origen", "unidad", "licencia"):
        _texto(e, campo, donde)
    if prefijo in ("fred", "yf") and sid != f"{prefijo}:{e['id_origen']}":
        raise ValueError(f"{donde}: el id debe ser '{prefijo}:<id_origen>' ({e['id_origen']!r})")
    if e["frecuencia"] not in FRECUENCIAS:
        raise ValueError(f"{donde}: frecuencia {e['frecuencia']!r} no es una de {list(FRECUENCIAS)}")
    if e["estado"] not in ESTADOS:
        raise ValueError(f"{donde}: estado {e['estado']!r} no es uno de {list(ESTADOS)}")
    if e["metodo"] not in METODOS_PUBLICACION:
        raise ValueError(f"{donde}: metodo {e['metodo']!r} no es uno de {list(METODOS_PUBLICACION)}")
    if e["divisa"] is not None and not (isinstance(e["divisa"], str) and _PATRON_DIVISA.fullmatch(e["divisa"])):
        raise ValueError(f"{donde}: divisa {e['divisa']!r} no es ISO de 3 letras (o null si no es monetaria)")
    if e["rol_fase1"] is not None and e["rol_fase1"] not in ROLES_FASE1:
        raise ValueError(f"{donde}: rol_fase1 {e['rol_fase1']!r} no es uno de {list(ROLES_FASE1)} (o null)")
    for campo in ("revisable", "verificado"):
        if not isinstance(e[campo], bool):
            raise ValueError(f"{donde}: '{campo}' debe ser true o false")
    if "fase1_utilizable" in e and not isinstance(e["fase1_utilizable"], bool):
        raise ValueError(f"{donde}: 'fase1_utilizable' debe ser true o false")

    out = dict(e)
    out["alias"] = _lista(e["alias"], donde, "alias")
    out["fichero_fase1"] = _lista(e.get("fichero_fase1"), donde, "fichero_fase1")
    out["notas"] = _lista(e.get("notas"), donde, "notas")
    for f in out["fichero_fase1"]:
        if not _PATRON_FICHERO.fullmatch(f):
            raise ValueError(f"{donde}: fichero_fase1 {f!r} no es '<carpeta de data/raw>/<fichero>.parquet:<columna>'")
    if out["fichero_fase1"]:
        out.setdefault("fase1_utilizable", True)
    elif "fase1_utilizable" in e:
        raise ValueError(f"{donde}: 'fase1_utilizable' sin 'fichero_fase1'")
    else:
        out["fase1_utilizable"] = None
    out.setdefault("sucesor", None)
    out.setdefault("duplicado_de", None)

    # Retraso: de lags.yaml por lag_clave (nunca repetido en la entrada).
    metodo = e["metodo"]
    clave_lag = e.get("lag_clave")
    if metodo in METODOS_CON_LAG:
        if clave_lag not in lags:
            raise ValueError(f"{donde}: {metodo} necesita 'lag_clave' de lags.yaml (encontrado {clave_lag!r})")
        out["lag"] = lags[clave_lag]["lag"]
        out["fin_periodo"] = lags[clave_lag]["fin_periodo"]
        out["lag_habiles"] = lags[clave_lag].get("habiles")
        out["lag_n_habiles"] = lags[clave_lag].get("n_habiles")
    elif clave_lag is not None:
        raise ValueError(f"{donde}: 'lag_clave' solo vale con metodo {list(METODOS_CON_LAG)}")
    else:
        out["lag"] = out["fin_periodo"] = out["lag_habiles"] = out["lag_n_habiles"] = None
    if e.get("publicado_desde") is not None:
        try:
            out["publicado_desde"] = pd.Timestamp(str(e["publicado_desde"])).date().isoformat()
        except ValueError:
            raise ValueError(f"{donde}: publicado_desde {e['publicado_desde']!r} no es AAAA-MM-DD") from None
    else:
        out["publicado_desde"] = None
    if metodo != "derivada" and e.get("componentes"):
        raise ValueError(f"{donde}: 'componentes' solo vale con metodo derivada")
    _validar_bloque3(sid, e, out, lags, donde)

    # Revisables: vintages exactos, salvo la excepción declarada del bloque 2 (último vintage + nota).
    if e["revisable"] and metodo != "vintage_exacto":
        if metodo not in METODOS_CON_LAG:
            raise ValueError(f"{donde}: revisable con metodo {metodo!r}: solo vintage_exacto, o lag_fijo/fichero "
                             f"con la nota obligatoria")
        obligatoria = nota_revisable(out)
        if not any(obligatoria in n for n in out["notas"]):
            raise ValueError(f"{donde}: revisable cargada con el último vintage ({metodo}): falta la nota "
                             f"obligatoria {obligatoria!r}")
    if e["estado"] == "excluida_ajustada" and not (e["fuente"] == "yfinance" and metodo == "cierre_mercado"):
        raise ValueError(f"{donde}: excluida_ajustada es para cierres de yfinance ajustados (auto_adjust=True)")
    if e["estado"] == "duplicada" and not out["duplicado_de"]:
        raise ValueError(f"{donde}: estado duplicada necesita 'duplicado_de'")

    pub = {k: v for k, v in out.items() if k in CLAVES_PUBLICACION and v is not None}
    pub.pop("notas", None)
    if metodo != "lag_fijo":
        pub.pop("lag", None)
    visibilidad._validar_serie_cfg(pub, donde, publicadores)
    return out


def _coherencia(cat: dict[str, dict], origen: str) -> None:
    """Referencias entre entradas (componentes, sucesor, duplicado_de), alias y ficheros únicos."""
    vistos_alias: dict[str, str] = {}
    vistos_fichero: dict[str, str] = {}
    for sid, e in cat.items():
        donde = f"{origen}: serie {sid!r}"
        for campo in ("sucesor", "duplicado_de"):
            ref = e.get(campo)
            if ref is not None and (ref not in cat or ref == sid):
                raise ValueError(f"{donde}: {campo} {ref!r} no es otra serie del catálogo")
        for c in e.get("componentes") or []:
            if c not in cat or c == sid:
                raise ValueError(f"{donde}: componente {c!r} no es otra serie del catálogo")
        for c in e.get("coherente_con") or []:
            if c not in cat or c == sid:
                raise ValueError(f"{donde}: coherente_con {c!r} no es otra serie del catálogo")
        for a in e["alias"]:
            if a in cat:
                raise ValueError(f"{donde}: el alias {a!r} coincide con un id")
            if a in vistos_alias:
                raise ValueError(f"{donde}: alias {a!r} repetido (también en {vistos_alias[a]!r})")
            vistos_alias[a] = sid
        for f in e["fichero_fase1"]:
            if f in vistos_fichero:
                raise ValueError(f"{donde}: fichero_fase1 {f!r} repetido (también en {vistos_fichero[f]!r})")
            vistos_fichero[f] = sid


def _fecha_iso(valor: Any, donde: str, campo: str) -> str:
    try:
        return pd.Timestamp(str(valor)).date().isoformat()
    except ValueError:
        raise ValueError(f"{donde}: {campo} {valor!r} no es AAAA-MM-DD") from None


def _validar_bloque3(sid: str, e: dict[str, Any], out: dict[str, Any], lags: dict[str, dict], donde: str) -> None:
    """Campos del bloque 3 (ALFRED, fuente viva y metadatos de la lista); normaliza ``out`` in situ."""
    metodo = e["metodo"]
    grupo = e.get("grupo")
    if grupo is not None and grupo not in GRUPOS:
        raise ValueError(f"{donde}: grupo {grupo!r} no es uno de {list(GRUPOS)}")
    out["grupo"] = grupo
    out["agentes"] = _lista(e.get("agentes"), donde, "agentes")
    malos = [a for a in out["agentes"] if a not in AGENTES]
    if malos or len(set(out["agentes"])) != len(out["agentes"]):
        raise ValueError(f"{donde}: agentes {out['agentes']} deben ser distintos y de {list(AGENTES)}")
    if e.get("release") is not None:
        _texto(e, "release", donde)
    out["release"] = e.get("release")
    for campo in ("reestimada", "cobertura_variable", "unidad_por_vintage"):
        if campo in e and not isinstance(e[campo], bool):
            raise ValueError(f"{donde}: '{campo}' debe ser true o false")
        out[campo] = bool(e.get(campo, False))
    out["coherente_con"] = _lista(e.get("coherente_con"), donde, "coherente_con")

    # vintage_exacto: hora del publicador en cada vintage; regla previa y proyecciones solo aquí.
    if metodo == "vintage_exacto" and not e.get("publicador"):
        raise ValueError(f"{donde}: vintage_exacto necesita 'publicador' (hora de cada vintage)")
    for campo in ("regla_previa_alfred", "proyecciones"):
        if e.get(campo) is not None and metodo != "vintage_exacto":
            raise ValueError(f"{donde}: '{campo}' solo vale con metodo vintage_exacto")
    if e.get("proyecciones") is not None and e["proyecciones"] not in PROYECCIONES:
        raise ValueError(f"{donde}: proyecciones {e['proyecciones']!r} no es una de {list(PROYECCIONES)}")
    out["proyecciones"] = e.get("proyecciones")
    previa = e.get("regla_previa_alfred")
    if previa is not None:
        visibilidad._claves(previa, _CLAVES_PREVIA, set(), f"{donde}, regla_previa_alfred")
        clave = previa["lag_clave"]
        if clave not in lags:
            raise ValueError(f"{donde}: regla_previa_alfred necesita 'lag_clave' de lags.yaml (encontrado {clave!r})")
        out["regla_previa_alfred"] = {
            "metodo": "lag_fijo", "lag_clave": clave, "lag": lags[clave]["lag"],
            "fin_periodo": lags[clave]["fin_periodo"], "lag_habiles": lags[clave].get("habiles"),
            "lag_n_habiles": lags[clave].get("n_habiles"),
            "primer_vintage": _fecha_iso(previa["primer_vintage"], donde, "regla_previa_alfred.primer_vintage")}
    else:
        out["regla_previa_alfred"] = None

    # Fuente viva: por defecto FRED en las fred: de fred_alfred; null explícito si FRED ya no la sirve.
    es_fred = sid.split(":", 1)[0] == "fred" and e["fuente"] == "fred_alfred"
    viva = e["fuente_viva"] if "fuente_viva" in e else ("fred" if es_fred else None)
    if viva is not None and (viva not in FUENTES_VIVAS or not es_fred):
        raise ValueError(f"{donde}: fuente_viva {viva!r}: solo {list(FUENTES_VIVAS)} en series fred: de fred_alfred")
    if metodo == "vintage_exacto" and e["fuente"] == "fred_alfred" and viva != "fred":
        raise ValueError(f"{donde}: vintage_exacto de ALFRED necesita fuente_viva 'fred'")
    out["fuente_viva"] = viva


def _validar_catalogo(crudo: Any, origen: str, lags: dict, publicadores: dict) -> dict[str, dict[str, Any]]:
    if not isinstance(crudo, dict):
        raise ValueError(f"{origen}: se esperaba un diccionario")
    visibilidad._claves(crudo, {"version", "series"}, set(), origen)
    series = crudo["series"]
    if not isinstance(series, dict) or not series:
        raise ValueError(f"{origen}: 'series' debe ser un diccionario no vacío {{id: entrada}}")
    cat = {str(sid): _validar_entrada(str(sid), e, lags, publicadores, origen) for sid, e in series.items()}
    _coherencia(cat, origen)
    return cat


@lru_cache(maxsize=4)
def _catalogo_cache(ruta: str, _mtime: int, ruta_lags: str, _mtime_lags: int) -> dict[str, dict[str, Any]]:
    pub = visibilidad.cargar_publicaciones(visibilidad.RUTA_PUBLICACIONES, catalogo=False)
    return _validar_catalogo(visibilidad._leer_yaml(Path(ruta)), Path(ruta).name,
                             cargar_lags(Path(ruta_lags)), pub["publicadores"])


def cargar_catalogo(ruta: Path | None = None, *, lags: Path | None = None,
                    publicaciones: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """``config/catalogo_series.yaml`` validado: ``{id: entrada}``.

    Cada entrada lleva sus campos normalizados (``alias``, ``fichero_fase1`` y ``notas`` como
    listas; ``sucesor``, ``duplicado_de``, ``lag`` y ``fin_periodo`` presentes aunque sean None).
    Comprueba campos y tipos, prefijo del id, estado, frecuencia, divisa ISO, ``lag_clave`` en
    ``lags.yaml``, la regla de revisables (vintage_exacto, o lag_fijo/fichero con
    ``nota_revisable``: ALFRED o archivo de ediciones), referencias (componentes, sucesor, duplicado_de), alias y ficheros
    únicos, y la parte de publicación con los validadores de ``visibilidad`` (publicador de
    ``publicaciones.yaml``, mercado de ``mercados.yaml``). Bloque 3: ``vintage_exacto`` con
    publicador, ``regla_previa_alfred`` (``lag_clave`` de ``lags.yaml`` y ``primer_vintage``) y
    ``proyecciones`` solo en vintage_exacto, ``fuente_viva`` (``fred`` por defecto en las ``fred:``
    de ``fred_alfred``), ``grupo``, ``agentes``, avisos booleanos y ``coherente_con`` (ids del
    catálogo); se normalizan aunque falten. ``ValueError`` con el id culpable.
    ``publicaciones``: configuración cuyos ``publicadores`` se usan (por defecto
    ``publicaciones.yaml`` sin fusionar el catálogo). Devuelve una copia.
    """
    r = Path(ruta) if ruta is not None else RUTA_CATALOGO
    rl = Path(lags) if lags is not None else RUTA_LAGS
    if publicaciones is not None:
        return _validar_catalogo(visibilidad._leer_yaml(r), r.name, cargar_lags(rl), publicaciones["publicadores"])
    return copy.deepcopy(_catalogo_cache(str(r.resolve()), r.stat().st_mtime_ns,
                                         str(rl.resolve()), rl.stat().st_mtime_ns))


def _cat(catalogo: dict[str, dict] | None) -> dict[str, dict]:
    if catalogo is not None:
        return catalogo
    return _catalogo_cache(str(RUTA_CATALOGO), RUTA_CATALOGO.stat().st_mtime_ns,
                           str(RUTA_LAGS), RUTA_LAGS.stat().st_mtime_ns)


def origenes(entrada: dict[str, Any]) -> tuple[str, ...]:
    """Cargadores que sirven la serie, en el orden de ``PRIORIDAD_ORIGENES`` (bloque 3).

    ``vintage_exacto`` con fuente viva: solo ``alfred`` (los periodos anteriores al primer vintage
    salen del propio ALFRED con ``regla_previa_alfred``). Otra con ``fuente_viva`` ``fred``: ``fred``
    y, si tiene ``fichero_fase1`` utilizable, ``fase1`` (historia que FRED ya no sirve: capadas).
    ``FUENTES_EDICION``: ``academicos`` (+ ``fase1``, su edición previa). ``derivada``: ``derivadas``
    (+ la copia de ``fase1`` si la hay, que se contrasta). El resto: ``fase1`` si tiene fichero.
    """
    fase1 = ("fase1",) if entrada.get("fichero_fase1") and entrada.get("fase1_utilizable") is not False else ()
    if entrada.get("fuente_viva") == "fred":
        return ("alfred",) if entrada["metodo"] == "vintage_exacto" else ("fred", *fase1)
    if entrada["fuente"] in FUENTES_EDICION:
        return ("academicos", *fase1)
    if entrada["metodo"] == "derivada":
        return ("derivadas", *fase1)
    return fase1


def metodos_admitidos(entrada: dict[str, Any]) -> frozenset[str]:
    """``metodo_publicacion`` válidos en las filas de la serie: el suyo y el de ``regla_previa_alfred``."""
    previa = entrada.get("regla_previa_alfred")
    return frozenset({entrada["metodo"], *([previa["metodo"]] if previa else [])})


def entrada_previa(entrada: dict[str, Any]) -> dict[str, Any]:
    """Copia de la entrada con su ``regla_previa_alfred`` como método (``lag_fijo`` con su lag).

    Sirve para fechar con ``fecha_publicacion_regla`` las observaciones anteriores al primer
    vintage (y para reproducir la carga del bloque 2, con el último vintage y ``nota_revisable``).
    ``ValueError`` si la entrada no tiene regla previa.
    """
    previa = entrada.get("regla_previa_alfred")
    if not previa:
        raise ValueError("la serie no tiene regla_previa_alfred")
    out = copy.deepcopy(entrada)
    out.update(metodo=previa["metodo"], lag_clave=previa["lag_clave"], lag=previa["lag"],
               fin_periodo=previa["fin_periodo"], lag_habiles=previa["lag_habiles"],
               lag_n_habiles=previa["lag_n_habiles"], regla_previa_alfred=None)
    return out


def vista_regla_previa(catalogo: dict[str, dict] | None = None, ids: Any = None) -> dict[str, dict]:
    """Copia del catálogo con las series ``ids`` (por defecto, todas las que tienen regla previa) en
    la forma del bloque 2: ``entrada_previa`` (``lag_fijo`` con el último vintage y ``nota_revisable``).

    Para fechar los periodos anteriores a ALFRED en bloque y para los tests de las reglas de
    publicación (``lag_fijo``, excepciones, ``publicado_desde``) con entradas reales.
    """
    cat = copy.deepcopy(_cat(catalogo))
    elegidas = [s for s, e in cat.items() if e.get("regla_previa_alfred")] if ids is None else list(ids)
    for sid in elegidas:
        cat[sid] = entrada_previa(cat[sid])
    return cat


def nota_previa_alfred(entrada: dict[str, Any]) -> str:
    """Nota de las observaciones anteriores al primer vintage: ``NOTA_PREVIA_ALFRED (AAAA-MM-DD)``."""
    previa = entrada.get("regla_previa_alfred")
    if not previa:
        raise ValueError("la serie no tiene regla_previa_alfred")
    return f"{NOTA_PREVIA_ALFRED} ({previa['primer_vintage']})"


def parte_publicacion(entrada: dict[str, Any]) -> dict[str, Any]:
    """Claves de ``series`` de ``publicaciones.yaml`` de una entrada (lo que valida y usa visibilidad)."""
    pub = {k: copy.deepcopy(v) for k, v in entrada.items() if k in CLAVES_PUBLICACION and v not in (None, [])}
    if entrada["metodo"] != "lag_fijo":
        pub.pop("lag", None)
    return pub


def por_id(id: str, catalogo: dict[str, dict] | None = None) -> dict[str, Any]:
    """Entrada de ``id`` (canónico o alias de la Fase 1). ``KeyError`` si no está catalogado."""
    cat = _cat(catalogo)
    sid = resolver(id, cat)
    try:
        return copy.deepcopy(cat[sid])
    except KeyError:
        raise KeyError(f"serie {id!r} no está en el catálogo") from None


def alias(catalogo: dict[str, dict] | None = None) -> dict[str, str]:
    """``{alias de la Fase 1: id canónico}`` del catálogo."""
    return {a: sid for sid, e in _cat(catalogo).items() for a in e["alias"]}


def resolver(id: str, catalogo: dict[str, dict] | None = None) -> str:
    """Id canónico de ``id`` (alias → id; un id o un desconocido se devuelven tal cual)."""
    return alias(catalogo).get(id, id)


def resolver_caja(id: str, catalogo: dict[str, dict] | None = None) -> str | None:
    """Id canónico que coincide con ``id`` sin distinguir mayúsculas (``'gw:M.TBL'`` → ``'gw:M.tbl'``).

    Solo si la coincidencia es única entre ids y alias; si no, None (no se adivina).
    """
    cat = _cat(catalogo)
    clave = id.casefold()
    candidatos = {sid for sid in cat if sid.casefold() == clave}
    candidatos |= {sid for a, sid in alias(cat).items() if a.casefold() == clave}
    return candidatos.pop() if len(candidatos) == 1 else None


def filtrar(catalogo: dict[str, dict] | None = None, **criterios: Any) -> list[str]:
    """Ids cuyas entradas cumplen ``campo=valor`` (o ``campo=(v1, v2)``), ordenados.

    Ejemplo: ``filtrar(estado=ESTADOS_CARGABLES, fuente='fred_alfred')``.
    """
    out = []
    for sid, e in _cat(catalogo).items():
        ok = True
        for campo, valor in criterios.items():
            v = e.get(campo)
            ok &= v in valor if isinstance(valor, (tuple, list, set, frozenset)) else v == valor
        if ok:
            out.append(sid)
    return sorted(out)


_COLUMNAS_TABLA = ("id", "alias", "nombre", "fuente", "id_origen", "fichero_fase1", "fase1_utilizable",
                   "frecuencia", "unidad", "divisa", "metodo", "publicador", "mercado", "lag", "fin_periodo",
                   "componentes", "revisable", "rol_fase1", "licencia", "estado", "sucesor", "duplicado_de",
                   "verificado", "fuente_viva", "grupo", "agentes", "release", "proyecciones", "reestimada",
                   "lag_previo_alfred", "primer_vintage_alfred", "notas")


def tabla(catalogo: dict[str, dict] | None = None) -> pd.DataFrame:
    """Catálogo como tabla de metadatos (listas unidas con ' | '), una fila por id."""
    filas = []
    for sid, e in sorted(_cat(catalogo).items()):
        fila = {"id": sid}
        previa = e.get("regla_previa_alfred") or {}
        extra = {"lag_previo_alfred": previa.get("lag"), "primer_vintage_alfred": previa.get("primer_vintage")}
        for c in _COLUMNAS_TABLA[1:]:
            v = extra[c] if c in extra else e.get(c)
            fila[c] = " | ".join(v) if isinstance(v, list) else v
        filas.append(fila)
    return pd.DataFrame(filas, columns=list(_COLUMNAS_TABLA))


def resumen(catalogo: dict[str, dict] | None = None) -> dict[str, dict[str, int]]:
    """Recuentos por estado, método, fuente, revisable, frecuencia y fuente viva (para informes)."""
    cat = _cat(catalogo).values()
    return {campo: dict(sorted(Counter(str(e[campo]) for e in cat).items()))
            for campo in ("estado", "metodo", "fuente", "revisable", "frecuencia", "fuente_viva")}


def validar_contra_catalogo(df: pd.DataFrame, catalogo: dict[str, dict] | None = None, *,
                            solo_catalogadas: bool = True) -> None:
    """Coherencia de una tabla del contrato de series con el catálogo; ``ErrorContrato`` si falla.

    Por serie: que esté catalogada (salvo ``solo_catalogadas=False``) y en un estado cargable,
    que ``unidad``, ``divisa`` y ``fuente`` coincidan con el catálogo, que ``metodo_publicacion``
    sea el suyo (o el de su ``regla_previa_alfred``: filas anteriores al primer vintage), y
    que una serie revisable con vintage ``unico`` lleve ``nota_revisable`` en la nota de todas
    sus filas (paso de los cargadores antes de ``almacen.escribir_series``).
    """
    cat = _cat(catalogo)
    problemas: list[str] = []
    for sid, g in df.groupby("serie", sort=True):
        e = cat.get(sid)
        if e is None:
            if solo_catalogadas:
                problemas.append(f"{sid}: no está en el catálogo")
            continue
        if e["estado"] not in ESTADOS_CARGABLES:
            problemas.append(f"{sid}: estado {e['estado']!r} no entra en la base de series")
        for col, esperado in (("unidad", e["unidad"]), ("divisa", e["divisa"]), ("fuente", e["fuente"])):
            valores = set(g[col].where(g[col].notna(), None))
            if valores != {esperado}:
                problemas.append(f"{sid}: {col} {sorted(map(str, valores))} distinta de la del catálogo {esperado!r}")
        metodos = set(g["metodo_publicacion"].where(g["metodo_publicacion"].notna(), None))
        if not metodos <= metodos_admitidos(e):
            previa = " (o el de su regla_previa_alfred)" if e.get("regla_previa_alfred") else ""
            problemas.append(f"{sid}: metodo_publicacion {sorted(map(str, metodos))} distinta de la del catálogo "
                             f"{e['metodo']!r}{previa}")
        if e["revisable"]:
            unico = g["vintage"] == "unico"
            sin_nota = unico & ~g["nota"].fillna("").str.contains(nota_revisable(e), regex=False)
            if sin_nota.any():
                problemas.append(f"{sid}: revisable con vintage 'unico' sin la nota obligatoria en "
                                 f"{int(sin_nota.sum())} filas")
    if problemas:
        raise ErrorContrato("catalogo", problemas)


# --- CLI: tabla de metadatos con cobertura medida en data/raw ------------------------------------
def _cobertura_fase1(entrada: dict[str, Any], raiz: Path) -> dict[str, Any]:
    """Primera/última fecha, observaciones y mayor hueco de la primera columna de data/raw (sin valores)."""
    if not entrada["fichero_fase1"]:
        return {}
    ruta, col = entrada["fichero_fase1"][0].split(":", 1)
    s = pd.read_parquet(raiz / ruta, columns=[col])[col].dropna()
    s = s[~s.index.duplicated()]
    if s.empty:
        return {"cobertura_inicio": None, "cobertura_fin": None, "n_obs": 0}
    idx = pd.DatetimeIndex(s.index).sort_values()
    return {"cobertura_inicio": idx[0].date().isoformat(), "cobertura_fin": idx[-1].date().isoformat(),
            "n_obs": int(len(idx))}


def main() -> None:
    from ingesta.comun.rutas import ROOT, resultados_dir

    t = tabla()
    raw = ROOT / "data" / "raw"
    if raw.exists():
        cat = _cat(None)
        cob = pd.DataFrame([_cobertura_fase1(cat[sid], raw) for sid in t["id"]])
        t = pd.concat([t, cob.reindex(columns=["cobertura_inicio", "cobertura_fin", "n_obs"])], axis=1)
    carpeta = resultados_dir("mercados_series") / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "catalogo_series.csv"
    t.to_csv(ruta, index=False, encoding="utf-8")
    print(f"{len(t)} series -> {ruta}")
    for campo, cuenta in resumen().items():
        print(campo, cuenta)


if __name__ == "__main__":
    main()
