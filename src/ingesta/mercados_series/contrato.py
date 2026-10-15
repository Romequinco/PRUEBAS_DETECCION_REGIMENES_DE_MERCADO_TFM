"""Contrato de datos del Tema 1: tabla larga de series y tabla de precios as-traded.

Todo lo que se guarda en la base del tema (y todo lo que leen ``get_serie`` y ``get_precios``)
pasa por ``validar_series`` o ``validar_precios``: devuelven el DataFrame normalizado (dtypes,
orden de columnas y de filas) o lanzan ``ErrorContrato`` con la lista de problemas.

Clave del TimeGate: ``fecha_publicacion`` es la PRIMERA fecha de decisión ``d`` cuyo momento de
decisión (16:30 America/New_York del día ``d``) es posterior o igual al instante en que el dato es
público (``disponible_utc``). ``primera_decision_vec`` es la regla vectorizada de referencia; los
validadores comprueban con ella la coherencia entre ambas columnas cuando ``disponible_utc`` existe.

Series (una fila = un valor tal como se publicó; clave ``(serie, fecha_referencia, vintage)``)::

    serie | fecha_referencia | fecha_publicacion | disponible_utc | vintage | valor | unidad
    | divisa | fuente | metodo_publicacion | nota

Precios (una fila por ticker y sesión, as-traded; clave ``(ticker, sesion)``)::

    ticker | sesion | mercado | open | high | low | close | volume | dividendo | split | divisa
    | fecha_publicacion | disponible_utc | fuente | descargado

Nunca se guardan precios ajustados: ninguna columna cuyo nombre contenga «adj» (en cualquier
caja) entra en el contrato. El ajuste se reconstruye a fecha en ``timegate.ajustar_a_fecha``.
"""
from __future__ import annotations

import re
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

# --- Hora de decisión (la usan visibilidad, sinteticos y los validadores) ----------------------
HORA_DECISION = time(16, 30)
HUSO_DECISION = "America/New_York"
_CORTE_DECISION = pd.Timedelta(hours=HORA_DECISION.hour, minutes=HORA_DECISION.minute)

# --- Columnas y dtypes --------------------------------------------------------------------------
TEXTO = "object"            # str o nulo (None)
FECHA = "datetime64[ns]"    # naive, a medianoche
UTC = "datetime64[ns, UTC]"
REAL = "float64"

COLUMNAS_SERIES: dict[str, str] = {
    "serie": TEXTO,
    "fecha_referencia": FECHA,
    "fecha_publicacion": FECHA,
    "disponible_utc": UTC,
    "vintage": TEXTO,
    "valor": REAL,
    "unidad": TEXTO,
    "divisa": TEXTO,
    "fuente": TEXTO,
    "metodo_publicacion": TEXTO,
    "nota": TEXTO,
}
CLAVE_SERIES = ("serie", "fecha_referencia", "vintage")

COLUMNAS_PRECIOS: dict[str, str] = {
    "ticker": TEXTO,
    "sesion": FECHA,
    "mercado": TEXTO,
    "open": REAL,
    "high": REAL,
    "low": REAL,
    "close": REAL,
    "volume": REAL,
    "dividendo": REAL,
    "split": REAL,
    "divisa": TEXTO,
    "fecha_publicacion": FECHA,
    "disponible_utc": UTC,
    "fuente": TEXTO,
    "descargado": UTC,
}
CLAVE_PRECIOS = ("ticker", "sesion")

METODOS_PUBLICACION = (
    "vintage_exacto",   # fecha real de cada vintage (ALFRED, ediciones con fecha)
    "calendario",       # calendario oficial de publicaciones (BCE, BLS...)
    "lag_fijo",         # fecha de referencia + retraso declarado (conservador)
    "cierre_mercado",   # cierre de la sesión + MARGEN_CIERRE
    "derivada",         # máxima fecha_publicacion de sus componentes
    "fichero",          # fecha del fichero publicado (Last-Modified, nombre de edición)
)

# Columnas que nunca pueden entrar en el contrato. La regla real es más amplia: cualquier
# nombre que contenga «adj» sin distinguir mayúsculas (``es_columna_prohibida``).
COLUMNAS_PROHIBIDAS = ("adj_close", "Adj Close", "adjclose", "adjusted_close", "Adj_Close")
_PATRON_PROHIBIDO = re.compile("adj", re.IGNORECASE)

# Vintage: fecha ALFRED (AAAA-MM-DD), edición (AAAA-MM) o 'unico' (serie sin revisiones).
# Los patrones se aplican con ``fullmatch``: un salto de línea final no pasa.
_PATRON_VINTAGE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|unico")
# Divisa: código ISO de 3 letras en mayúsculas o subunidad de cotización (las de
# ``divisas.SUBUNIDADES``: GBp, GBX, ZAc, ILA; GBX e ILA ya encajan como ISO).
_PATRON_DIVISA = re.compile(r"[A-Z]{3}|GBp|ZAc")

_MAX_EJEMPLOS = 5


class ErrorContrato(ValueError):
    """Un DataFrame no cumple el contrato. ``problemas`` lista cada incumplimiento."""

    def __init__(self, tabla: str, problemas: list[str]):
        self.tabla = tabla
        self.problemas = list(problemas)
        super().__init__(f"contrato de {tabla}: " + "; ".join(self.problemas))


# --- Regla del TimeGate -------------------------------------------------------------------------
def primera_decision_vec(instantes_utc: pd.Series) -> pd.Series:
    """Primera fecha de decisión (naive, medianoche) cuyo momento de decisión es >= cada instante.

    Un instante exactamente a las 16:30 de Nueva York es visible ese mismo día; a las 16:31, el
    siguiente. El huso lleva DST. Los nulos se conservan (NaT).
    """
    s = pd.Series(instantes_utc)
    if not isinstance(s.dtype, pd.DatetimeTZDtype):
        raise TypeError("primera_decision_vec necesita instantes con huso (UTC)")
    local = s.dt.tz_convert(HUSO_DECISION).dt.tz_localize(None)
    dia = local.dt.normalize()
    res = dia.where(local - dia <= _CORTE_DECISION, dia + pd.Timedelta(days=1))
    return res.astype(FECHA)


# --- Columnas prohibidas ------------------------------------------------------------------------
def es_columna_prohibida(nombre: Any) -> bool:
    """True si el nombre contiene «adj» (en cualquier caja): precios ajustados, nunca se guardan."""
    return bool(_PATRON_PROHIBIDO.search(str(nombre)))


# --- Normalización por tipo ---------------------------------------------------------------------
def _texto(s: pd.Series, col: str) -> pd.Series:
    s = s.astype(object)
    s = s.where(s.notna(), None)
    # Vectorizado: infer_dtype recorre la columna en C; el bucle Python solo corre si hay error.
    if pd.api.types.infer_dtype(s, skipna=True) not in ("string", "empty"):
        no_texto = s.dropna().map(lambda v: not isinstance(v, str))
        ejemplo = s.dropna()[no_texto].iloc[0]
        raise ErrorContrato("columnas", [f"{col!r} debe ser texto (encontrado {type(ejemplo).__name__})"])
    return s


def _fecha(s: pd.Series, col: str) -> pd.Series:
    try:
        out = pd.to_datetime(s)
    except (ValueError, TypeError) as e:
        raise ErrorContrato("columnas", [f"{col!r} no es una fecha válida ({e})"]) from None
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        raise ErrorContrato("columnas", [f"{col!r} debe ser una fecha sin huso (naive)"])
    out = out.astype(FECHA)
    nn = out.dropna()
    if (nn != nn.dt.normalize()).any():
        raise ErrorContrato("columnas", [f"{col!r} debe estar a medianoche (sin hora)"])
    return out


def _utc(s: pd.Series, col: str) -> pd.Series:
    if s.isna().all():
        return pd.Series(pd.NaT, index=s.index, dtype=UTC)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC").astype(UTC)
    if pd.api.types.is_datetime64_dtype(s):
        raise ErrorContrato("columnas", [f"{col!r} debe llevar huso (instante UTC), no fecha naive"])
    try:
        sellos = s.dropna().map(pd.Timestamp)
    except (ValueError, TypeError) as e:
        raise ErrorContrato("columnas", [f"{col!r} no es un instante válido ({e})"]) from None
    if sellos.map(lambda t: t.tzinfo is None).any():
        raise ErrorContrato("columnas", [f"{col!r} debe llevar huso (instante UTC), no fecha naive"])
    return pd.to_datetime(s, utc=True).astype(UTC)


def _real(s: pd.Series, col: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        raise ErrorContrato("columnas", [f"{col!r} debe ser numérica, no booleana"])
    try:
        return pd.to_numeric(s, errors="raise").astype(REAL)
    except (ValueError, TypeError) as e:
        raise ErrorContrato("columnas", [f"{col!r} debe ser numérica ({e})"]) from None


_NORMALIZADORES = {TEXTO: _texto, FECHA: _fecha, UTC: _utc, REAL: _real}


def _normalizar(df: pd.DataFrame, columnas: dict[str, str], tabla: str) -> pd.DataFrame:
    """Columnas prohibidas, que faltan o sobran; después dtypes y orden de columnas."""
    if not isinstance(df, pd.DataFrame):
        raise ErrorContrato(tabla, [f"se esperaba un DataFrame, no {type(df).__name__}"])
    problemas: list[str] = []
    prohibidas = [c for c in df.columns if es_columna_prohibida(c)]
    if prohibidas:
        problemas.append(f"columnas prohibidas (precio ajustado, nunca se guarda): {prohibidas}")
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        problemas.append(f"faltan columnas: {faltan}")
    sobran = [c for c in df.columns if c not in columnas and c not in prohibidas]
    if sobran:
        problemas.append(f"columnas fuera del contrato: {sobran}")
    if df.columns.duplicated().any():
        problemas.append(f"columnas repetidas: {sorted(set(df.columns[df.columns.duplicated()]))}")
    if problemas:
        raise ErrorContrato(tabla, problemas)
    out = pd.DataFrame(index=df.index)
    for col, dtype in columnas.items():
        try:
            out[col] = _NORMALIZADORES[dtype](df[col], col)
        except ErrorContrato as e:
            raise ErrorContrato(tabla, e.problemas) from None
    return out


# --- Comprobaciones comunes ---------------------------------------------------------------------
def _ejemplos(df: pd.DataFrame, mascara: pd.Series, cols: tuple[str, ...] | list[str]) -> str:
    filas = df.loc[mascara, list(cols)].head(_MAX_EJEMPLOS)
    return ", ".join(str(tuple(str(v) for v in fila)) for fila in filas.itertuples(index=False))


def _nulos(df: pd.DataFrame, cols: tuple[str, ...], clave: tuple[str, ...]) -> list[str]:
    problemas = []
    for c in cols:
        m = df[c].isna()
        if m.any():
            problemas.append(f"{int(m.sum())} nulos en {c!r} (p. ej. {_ejemplos(df, m, clave)})")
    return problemas


def _duplicados(df: pd.DataFrame, clave: tuple[str, ...]) -> list[str]:
    m = df.duplicated(list(clave), keep=False)
    if m.any():
        return [f"{int(m.sum())} filas con clave {clave} repetida (p. ej. {_ejemplos(df, m, clave)})"]
    return []


def _patron(df: pd.DataFrame, col: str, patron: re.Pattern, clave: tuple[str, ...]) -> list[str]:
    # La regex solo se aplica a los valores distintos (pocos: vintages, divisas), no fila a fila.
    malos = [v for v in pd.unique(df[col].dropna()) if not patron.fullmatch(v)]
    m = df[col].isin(malos)
    if m.any():
        return [f"{col!r} con formato inválido: {sorted(set(df.loc[m, col]))[:_MAX_EJEMPLOS]}"]
    return []


def _no_finitos(df: pd.DataFrame, cols: tuple[str, ...], clave: tuple[str, ...]) -> list[str]:
    problemas = []
    for c in cols:
        m = np.isinf(df[c].to_numpy())
        if m.any():
            problemas.append(f"valores infinitos en {c!r} (p. ej. {_ejemplos(df, pd.Series(m, df.index), clave)})")
    return problemas


def _constante_por(df: pd.DataFrame, grupo: str, cols: tuple[str, ...]) -> list[str]:
    problemas = []
    for c in cols:
        n = df.groupby(grupo, sort=True)[c].nunique(dropna=False)
        malas = n[n > 1].index.tolist()
        if malas:
            problemas.append(f"{c!r} cambia dentro de {grupo} {malas[:_MAX_EJEMPLOS]}")
    return problemas


def _coherencia_disponible(df: pd.DataFrame, clave: tuple[str, ...]) -> list[str]:
    """fecha_publicacion debe ser exactamente la primera decisión tras disponible_utc."""
    nn = df["disponible_utc"].notna() & df["fecha_publicacion"].notna()
    if not nn.any():
        return []
    esperada = primera_decision_vec(df.loc[nn, "disponible_utc"])
    m = pd.Series(False, index=df.index)
    m.loc[nn] = df.loc[nn, "fecha_publicacion"] != esperada
    if m.any():
        return [f"{int(m.sum())} filas con fecha_publicacion incoherente con disponible_utc "
                f"(debe ser la primera decisión de las {HORA_DECISION:%H:%M} {HUSO_DECISION}; "
                f"p. ej. {_ejemplos(df, m, clave)})"]
    return []


def _coherencia_mercado(df: pd.DataFrame, clave: tuple[str, ...]) -> list[str]:
    """``disponible_utc`` no puede ser anterior al cierre del mercado + margen (``config/mercados.yaml``).

    Se admite un instante posterior (conservador). Un mercado desconocido o una sesión sin tramo
    de cierre es un problema. Import perezoso: ``visibilidad`` importa este módulo.
    """
    nn = df["sesion"].notna() & df["mercado"].notna() & df["disponible_utc"].notna()
    if not nn.any():
        return []
    from ingesta.mercados_series import visibilidad

    sub = df.loc[nn]
    try:
        esperado = visibilidad.disponible_cierre_serie(sub["sesion"], sub["mercado"])
    except (KeyError, ValueError) as e:
        return [f"mercado o sesión sin hora de cierre en config/mercados.yaml: {e.args[0]}"]
    m = pd.Series(False, index=df.index)
    m.loc[nn] = sub["disponible_utc"] < esperado
    if m.any():
        minutos = int(visibilidad.MARGEN_CIERRE.total_seconds() // 60)
        return [f"{int(m.sum())} velas visibles antes del cierre de su mercado + {minutos} min "
                f"(disponible_utc adelantado; p. ej. {_ejemplos(df, m, clave)})"]
    return []


# --- Validadores --------------------------------------------------------------------------------
def validar_series(df: pd.DataFrame, *, comprobar_disponible: bool = True) -> pd.DataFrame:
    """Valida y normaliza una tabla larga de series; devuelve una copia ordenada por clave.

    Lanza ``ErrorContrato`` si faltan o sobran columnas, hay columnas prohibidas, nulos en campos
    obligatorios (todo salvo ``disponible_utc``, ``divisa``, ``nota`` y ``valor``), un ``valor``
    nulo sin ``nota`` (los huecos permanentes se declaran), método o vintage inválidos, una serie
    que mezcla formatos de vintage, unidad o divisa, claves repetidas, ``fecha_publicacion``
    anterior a ``fecha_referencia`` (un dato no se publica antes de que empiece su periodo; el
    contrato no admite previsiones) o ``fecha_publicacion`` incoherente con ``disponible_utc``
    (esto último desactivable con ``comprobar_disponible=False``).
    """
    tabla = "series"
    out = _normalizar(df, COLUMNAS_SERIES, tabla)
    clave = CLAVE_SERIES
    problemas = _nulos(out, ("serie", "fecha_referencia", "fecha_publicacion", "vintage", "unidad",
                             "fuente", "metodo_publicacion"), clave)
    m = out["valor"].isna() & out["nota"].isna()
    if m.any():
        problemas.append(f"{int(m.sum())} valores nulos sin nota (un hueco se declara en 'nota'; "
                         f"p. ej. {_ejemplos(out, m, clave)})")
    problemas += _no_finitos(out, ("valor",), clave)
    m = out["fecha_publicacion"] < out["fecha_referencia"]
    if m.any():
        problemas.append(f"{int(m.sum())} filas con fecha_publicacion anterior a fecha_referencia "
                         f"(un dato no se publica antes de su periodo; p. ej. {_ejemplos(out, m, clave)})")
    metodo_malo = out["metodo_publicacion"].notna() & ~out["metodo_publicacion"].isin(METODOS_PUBLICACION)
    if metodo_malo.any():
        problemas.append(f"metodo_publicacion inválido {sorted(set(out.loc[metodo_malo, 'metodo_publicacion']))}"
                         f"; válidos: {list(METODOS_PUBLICACION)}")
    problemas += _patron(out, "vintage", _PATRON_VINTAGE, clave)
    problemas += _patron(out, "divisa", _PATRON_DIVISA, clave)
    if out["serie"].notna().all() and out["vintage"].notna().all() and len(out):
        formato = out["vintage"].str.len().where(out["vintage"] != "unico", 0)
        n = formato.groupby(out["serie"]).nunique()
        if (n > 1).any():
            problemas.append(f"series que mezclan formatos de vintage: {n[n > 1].index.tolist()[:_MAX_EJEMPLOS]}")
        problemas += _constante_por(out, "serie", ("unidad", "divisa"))
    problemas += _duplicados(out, clave)
    if comprobar_disponible:
        problemas += _coherencia_disponible(out, clave)
    if problemas:
        raise ErrorContrato(tabla, problemas)
    orden = ["serie", "fecha_referencia", "fecha_publicacion", "vintage"]
    return out.sort_values(orden, kind="mergesort").reset_index(drop=True)


def validar_precios(df: pd.DataFrame, *, comprobar_disponible: bool = True,
                    comprobar_mercado: bool = True) -> pd.DataFrame:
    """Valida y normaliza una tabla de precios as-traded; devuelve una copia ordenada por clave.

    Lanza ``ErrorContrato`` si faltan o sobran columnas, hay columnas prohibidas (``Adj Close`` y
    cualquier nombre con «adj»), nulos en campos obligatorios (todo salvo ``open``, ``high``,
    ``low`` y ``volume``), ``split <= 0``, ``dividendo < 0``, ``fecha_publicacion < sesion``,
    velas incompletas (``disponible_utc > descargado``), mercado o divisa que cambian dentro de un
    ticker, claves repetidas, ``fecha_publicacion`` incoherente con ``disponible_utc``
    (desactivable con ``comprobar_disponible=False``) o ``disponible_utc`` anterior al cierre del
    ``mercado`` + 30 min de ``config/mercados.yaml``, con mercado desconocido incluido
    (desactivable con ``comprobar_mercado=False``, solo para tablas de prueba).
    Los precios pueden ser negativos (futuros de petróleo en abril de 2020): no se comprueba signo.
    """
    tabla = "precios"
    out = _normalizar(df, COLUMNAS_PRECIOS, tabla)
    clave = CLAVE_PRECIOS
    problemas = _nulos(out, ("ticker", "sesion", "mercado", "close", "dividendo", "split", "divisa",
                             "fecha_publicacion", "disponible_utc", "fuente", "descargado"), clave)
    problemas += _no_finitos(out, ("open", "high", "low", "close", "volume", "dividendo", "split"), clave)
    for col, cond, texto in (("split", out["split"] <= 0, "split <= 0 (ratio nuevo/antiguo; 1.0 sin evento)"),
                             ("dividendo", out["dividendo"] < 0, "dividendo < 0 (0 sin evento)"),
                             ("volume", out["volume"] < 0, "volume < 0")):
        if cond.any():
            problemas.append(f"{int(cond.sum())} filas con {texto} (p. ej. {_ejemplos(out, cond, clave)})")
    m = out["fecha_publicacion"] < out["sesion"]
    if m.any():
        problemas.append(f"{int(m.sum())} filas con fecha_publicacion anterior a la sesión "
                         f"(p. ej. {_ejemplos(out, m, clave)})")
    m = out["disponible_utc"] > out["descargado"]
    if m.any():
        problemas.append(f"{int(m.sum())} velas incompletas (disponible_utc posterior a descargado; se "
                         f"descartan antes de guardar; p. ej. {_ejemplos(out, m, clave)})")
    problemas += _patron(out, "divisa", _PATRON_DIVISA, clave)
    if out["ticker"].notna().all() and len(out):
        problemas += _constante_por(out, "ticker", ("mercado", "divisa"))
    problemas += _duplicados(out, clave)
    if comprobar_disponible:
        problemas += _coherencia_disponible(out, clave)
    if comprobar_mercado:
        problemas += _coherencia_mercado(out, clave)
    if problemas:
        raise ErrorContrato(tabla, problemas)
    return out.sort_values(list(clave), kind="mergesort").reset_index(drop=True)


def descartar_velas_incompletas(df: pd.DataFrame) -> pd.DataFrame:
    """Copia de ``df`` sin las velas incompletas (``disponible_utc > descargado``).

    Paso de los extractores antes de ``validar_precios``: una vela descargada antes de ser
    definitiva (sesión en curso o cierre provisional) no se guarda y se recoge en la próxima
    descarga. Las filas con ``disponible_utc`` o ``descargado`` nulos se conservan (las rechaza
    el validador). No modifica ``df``.
    """
    disp = pd.to_datetime(df["disponible_utc"], utc=True)
    desc = pd.to_datetime(df["descargado"], utc=True)
    return df.loc[~(disp > desc)].copy()


def fecha_publicacion_derivada(base: pd.DataFrame, componentes: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """``fecha_publicacion`` de una serie derivada: la máxima de sus componentes, por periodo.

    Para cada ``fecha_referencia`` presente en TODOS los ``componentes`` (tabla del contrato de
    series) toma la primera publicación de cada componente (mínima ``fecha_publicacion`` entre sus
    vintages) y devuelve la máxima; ``disponible_utc`` es la máxima de esas primeras publicaciones
    (nula si alguna lo es). Columnas: ``fecha_referencia``, ``fecha_publicacion``,
    ``disponible_utc``. Las revisiones de un componente dan vintages nuevas de la derivada: las
    calcula el extractor con esta misma regla sobre la base truncada a cada vintage.
    ``KeyError`` si algún componente no tiene filas en ``base``.
    """
    if not componentes:
        raise ValueError("una serie derivada necesita al menos un componente")
    falta = [c for c in componentes if not (base["serie"] == c).any()]
    if falta:
        raise KeyError(f"componentes sin datos en la base: {falta}")
    primeras = []
    for c in componentes:
        orden = base[base["serie"] == c].sort_values(["fecha_referencia", "fecha_publicacion"], kind="mergesort")
        primeras.append(orden.drop_duplicates("fecha_referencia", keep="first")
                        .set_index("fecha_referencia")[["fecha_publicacion", "disponible_utc"]])
    comunes = primeras[0].index
    for p in primeras[1:]:
        comunes = comunes.intersection(p.index)
    comunes = comunes.sort_values()
    pub = pd.concat([p.loc[comunes, "fecha_publicacion"] for p in primeras], axis=1)
    disp = pd.concat([p.loc[comunes, "disponible_utc"] for p in primeras], axis=1)
    disp_max = disp.max(axis=1).where(disp.notna().all(axis=1))
    return pd.DataFrame({
        "fecha_referencia": pd.Series(comunes, dtype=FECHA),
        "fecha_publicacion": pub.max(axis=1).astype(FECHA).to_numpy(),
        "disponible_utc": pd.Series(disp_max.to_numpy(), dtype=UTC).to_numpy(),
    })


def vacio_series() -> pd.DataFrame:
    """Tabla de series vacía con las columnas y dtypes del contrato."""
    return _normalizar(pd.DataFrame({c: [] for c in COLUMNAS_SERIES}), COLUMNAS_SERIES, "series")


def vacio_precios() -> pd.DataFrame:
    """Tabla de precios vacía con las columnas y dtypes del contrato."""
    return _normalizar(pd.DataFrame({c: [] for c in COLUMNAS_PRECIOS}), COLUMNAS_PRECIOS, "precios")


# --- Esquemas pandera (opcionales) --------------------------------------------------------------
def _pandera():
    try:
        import pandera.pandas as pa
    except ImportError as e:  # pandera es dependencia del tema, no de la Fase 1 ni de la CI
        raise ImportError("pandera no está instalado: pip install -r requirements/ingesta-mercados_series.txt") from e
    return pa


def _dtype_pandera(dtype: str):
    return pd.DatetimeTZDtype("ns", "UTC") if dtype == UTC else dtype


def esquema_pandera_series():
    """Esquema ``pandera.pandas`` equivalente (parcial) a ``validar_series``; import perezoso.

    Cubre columnas, dtypes, nulos, métodos, unicidad de clave y hueco con nota. La coherencia
    ``disponible_utc``/``fecha_publicacion`` y los formatos por serie solo los mira ``validar_series``.
    """
    pa = _pandera()
    nulables = {"disponible_utc", "valor", "divisa", "nota"}
    checks_col = {
        "metodo_publicacion": [pa.Check.isin(list(METODOS_PUBLICACION))],
        "vintage": [pa.Check.str_matches(rf"(?:{_PATRON_VINTAGE.pattern})\Z")],
    }
    columnas = {c: pa.Column(_dtype_pandera(t), nullable=c in nulables, checks=checks_col.get(c))
                for c, t in COLUMNAS_SERIES.items()}
    return pa.DataFrameSchema(
        columnas, strict=True, ordered=True, unique=list(CLAVE_SERIES),
        checks=[pa.Check(lambda d: d["valor"].notna() | d["nota"].notna(), name="valor_nulo_con_nota")],
        name="series_t1")


def esquema_pandera_precios():
    """Esquema ``pandera.pandas`` equivalente (parcial) a ``validar_precios``; import perezoso."""
    pa = _pandera()
    nulables = {"open", "high", "low", "volume"}
    checks_col = {"split": [pa.Check.gt(0)], "dividendo": [pa.Check.ge(0)]}
    columnas = {c: pa.Column(_dtype_pandera(t), nullable=c in nulables, checks=checks_col.get(c))
                for c, t in COLUMNAS_PRECIOS.items()}
    return pa.DataFrameSchema(
        columnas, strict=True, ordered=True, unique=list(CLAVE_PRECIOS),
        checks=[pa.Check(lambda d: d["fecha_publicacion"] >= d["sesion"], name="publicacion_tras_sesion"),
                pa.Check(lambda d: d["disponible_utc"] <= d["descargado"], name="vela_completa")],
        name="precios_t1")
