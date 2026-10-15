"""Tools del Tema 1 para los agentes: ``get_serie`` y ``get_precios``.

Ambas aplican el TimeGate (nada con ``fecha_publicacion > fecha``) y devuelven una respuesta con
los datos, sus metadatos y una lista de ``avisos`` legibles (huecos, fechas de publicación
aproximadas, conversiones sin FX, tickers sin datos...). Los errores de uso lanzan
``ValueError`` (ajuste o divisa desconocidos, fecha inválida o ambigua).

Truncado también en los mensajes: la respuesta en ``fecha`` (datos, avisos, metadatos y tipo de
error) es la misma con la base completa que con la base truncada a lo publicado hasta ``fecha``.
Por eso una serie o un ticker sin nada publicado hasta ``fecha`` se trata igual que uno que no
existe, sin revelar si existirá ni cuándo: ``get_serie`` lanza ``KeyError`` y ``get_precios``
añade un aviso neutro; los metadatos salen solo de filas visibles.

``ultimo()`` devuelve un diccionario citable (valores Python, fechas ISO): lo que un agente pone
en su respuesta junto con la fuente y la fecha de publicación.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from ingesta.mercados_series import almacen, divisas, timegate, visibilidad
from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series.catalogo import (
    NOTA_EXCEPCION,
    NOTA_PREVIA_ALFRED,
    NOTA_PRIMERA_EDICION,
    NOTA_RECONSTRUIDA,
    es_nota_revisable,
)
from ingesta.mercados_series.contrato import FECHA, REAL, TEXTO, vacio_series
from ingesta.mercados_series.fuentes.alfred import NOTA_REESTIMADA, NOTA_UNIDAD_VINTAGE

FechaLike = date | str | pd.Timestamp
Ajuste = Literal["ninguno", "splits", "total"]

AJUSTES: tuple[str, ...] = ("ninguno", "splits", "total")
DIVISAS_SALIDA: tuple[str, ...] = ("original", "USD")

# Columnas de ``RespuestaSerie.datos`` (más ``divisa_original``, ``fx_usd`` y ``fx_fecha`` si se
# pide USD).
COLUMNAS_DATOS_SERIE: dict[str, str] = {
    "fecha_referencia": FECHA, "valor": REAL, "fecha_publicacion": FECHA, "vintage": TEXTO,
    "metodo_publicacion": TEXTO, "nota": TEXTO,
}
# Columnas de ``RespuestaPrecios.datos`` (ídem). Precios y dividendo con el ajuste pedido; el
# volumen, en acciones de hoy-en-fecha si el ajuste incluye splits.
COLUMNAS_DATOS_PRECIOS: dict[str, str] = {
    "ticker": TEXTO, "sesion": FECHA, "mercado": TEXTO, "open": REAL, "high": REAL, "low": REAL,
    "close": REAL, "volume": REAL, "dividendo": REAL, "split": REAL, "divisa": TEXTO,
    "fecha_publicacion": FECHA, "fuente": TEXTO,
}
_COLUMNAS_FX: dict[str, str] = {"divisa_original": TEXTO, "fx_usd": REAL, "fx_fecha": FECHA}

# Columna de ``timegate.ajustar_a_fecha`` que se publica como cada columna, por ajuste.
_ORIGEN_AJUSTE: dict[str, dict[str, str]] = {
    "ninguno": {},
    "splits": {"open": "open_aj_splits", "high": "high_aj_splits", "low": "low_aj_splits",
               "close": "close_aj_splits", "volume": "volume_aj_splits",
               "dividendo": "dividendo_aj_splits"},
    "total": {"open": "open_aj_total", "high": "high_aj_total", "low": "low_aj_total",
              "close": "close_aj_total", "volume": "volume_aj_splits",
              "dividendo": "dividendo_aj_splits"},
}
_MONETARIAS_PRECIOS = ("open", "high", "low", "close", "dividendo")


# --- Utilidades ---------------------------------------------------------------------------------
def _vacio(columnas: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in columnas.items()})


def _iso(valor: Any) -> str | None:
    return None if pd.isna(valor) else pd.Timestamp(valor).date().isoformat()


def _real(valor: Any) -> float | None:
    return None if pd.isna(valor) else float(valor)


def _texto(valor: Any) -> str | None:
    return None if valor is None or (not isinstance(valor, str) and pd.isna(valor)) else str(valor)


def _validar_opcion(valor: str, validas: tuple[str, ...], nombre: str) -> None:
    if valor not in validas:
        raise ValueError(f"{nombre} desconocido {valor!r}; válidos: {list(validas)}")


def alias_por_defecto() -> dict[str, str]:
    """Alias del catálogo (``catalogo.alias``: nombres de la Fase 1) más los de ``publicaciones.yaml``.

    Un alias no puede estar en los dos ficheros (lo impide ``visibilidad.cargar_publicaciones``).
    """
    return {**visibilidad.cargar_publicaciones(catalogo=False)["alias"], **cat_mod.alias()}


def resolver_id(id: str, alias: dict[str, str] | None = None) -> str:
    """Id canónico de ``id``: ``alias[id]`` si es un alias; si no, ``id`` tal cual.

    ``alias`` por defecto: ``alias_por_defecto()`` (catálogo de series + ``publicaciones.yaml``).
    """
    if not isinstance(id, str) or not id:
        raise ValueError(f"id de serie vacío o no textual: {id!r}")
    tabla = alias if alias is not None else alias_por_defecto()
    if id in tabla:
        return tabla[id]
    if alias is None and id not in cat_mod._cat(None):
        # Ids y alias sin distinguir mayúsculas ('gw:M.TBL' → 'gw:M.tbl'), solo si es único.
        return cat_mod.resolver_caja(id) or id
    return id


def _comprobar_excluida(canonico: str, id: str) -> None:
    """``KeyError`` si el catálogo excluye ``canonico`` de la base de series (ETF ajustadas).

    También un ticker suelto (``'SPY'``) cuyo ``yf:<ticker>`` está excluido. Es un metadato
    estático (no depende de la fecha): no revela nada del futuro.
    """
    cat = cat_mod._cat(None)
    if canonico not in cat and f"yf:{canonico}" in cat:
        canonico = f"yf:{canonico}"
    e = cat.get(canonico)
    if e is None or e["estado"] != "excluida_ajustada":
        return
    nombre = repr(canonico) if canonico == id else f"{canonico!r} (alias {id!r})"
    raise KeyError(f"serie {nombre} excluida de la base de series: ETF de yfinance con niveles ajustados a "
                   f"posteriori (auto_adjust=True). Usa get_precios({e['id_origen']!r}, fecha), que reconstruye "
                   f"el ajuste con lo conocido en la fecha (bloque 6)")


# --- Respuestas ---------------------------------------------------------------------------------
@dataclass
class RespuestaSerie:
    """Respuesta de ``get_serie``.

    ``datos``: ``COLUMNAS_DATOS_SERIE`` (una fila por periodo, la vintage visible en ``fecha``;
    con ``divisa_original``, ``fx_usd`` y ``fx_fecha`` si se pidió ``divisa='USD'``).
    ``metodo_publicacion``: el de la última fila. ``divisa``: la de ``valor`` tras la conversión.
    """

    serie: str
    fecha: pd.Timestamp
    datos: pd.DataFrame
    unidad: str | None
    divisa: str | None
    fuente: str | None
    metodo_publicacion: str | None
    avisos: list[str] = field(default_factory=list)

    def ultimo(self) -> dict[str, Any] | None:
        """Última fila (mayor ``fecha_referencia``) como diccionario citable; None sin datos.

        Claves: ``serie, valor, fecha_referencia, fecha_publicacion, vintage, unidad, divisa,
        fuente, metodo_publicacion, nota, fecha_consulta`` (fechas en ISO; ``valor`` None en un
        hueco declarado, con su ``nota``).
        """
        if self.datos.empty:
            return None
        fila = self.datos.iloc[-1]
        return {
            "serie": self.serie, "valor": _real(fila["valor"]),
            "fecha_referencia": _iso(fila["fecha_referencia"]),
            "fecha_publicacion": _iso(fila["fecha_publicacion"]), "vintage": _texto(fila["vintage"]),
            "unidad": self.unidad, "divisa": self.divisa, "fuente": self.fuente,
            "metodo_publicacion": _texto(fila["metodo_publicacion"]), "nota": _texto(fila["nota"]),
            "fecha_consulta": _iso(self.fecha),
        }


@dataclass
class RespuestaPrecios:
    """Respuesta de ``get_precios``.

    ``datos``: ``COLUMNAS_DATOS_PRECIOS`` (tabla larga, ordenada por ticker y sesión) con el
    ajuste pedido ya aplicado (y ``divisa_original``, ``fx_usd`` y ``fx_fecha`` si se convirtió).
    ``divisa``: 'USD' u 'original'.
    """

    tickers: list[str]
    fecha: pd.Timestamp
    datos: pd.DataFrame
    ajuste: Ajuste
    divisa: str
    avisos: list[str] = field(default_factory=list)

    def ultimo(self, ticker: str | None = None) -> dict[str, Any] | None:
        """Última sesión visible de ``ticker`` como diccionario citable; None sin datos.

        Sin ``ticker`` solo vale si se pidió uno (``ValueError`` con varios: usar ``ultimos``).
        Claves: ``ticker, sesion, valor`` (cierre con el ajuste pedido, que dice ``ajuste``),
        ``fecha_publicacion, fuente, mercado, ajuste, divisa, fx_usd`` (None sin conversión) y
        ``fecha_consulta``.
        """
        if ticker is None:
            if len(self.tickers) != 1:
                raise ValueError(f"varios tickers {self.tickers}: usa ultimo(ticker) o ultimos()")
            ticker = self.tickers[0]
        filas = self.datos[self.datos["ticker"] == ticker]
        if filas.empty:
            return None
        fila = filas.iloc[-1]
        return {
            "ticker": ticker, "sesion": _iso(fila["sesion"]), "valor": _real(fila["close"]),
            "fecha_publicacion": _iso(fila["fecha_publicacion"]), "fuente": _texto(fila["fuente"]),
            "mercado": _texto(fila["mercado"]), "ajuste": self.ajuste, "divisa": _texto(fila["divisa"]),
            "fx_usd": _real(fila["fx_usd"]) if "fx_usd" in filas.columns else None,
            "fecha_consulta": _iso(self.fecha),
        }

    def ultimos(self) -> dict[str, dict[str, Any] | None]:
        """``{ticker: ultimo(ticker)}`` para todos los tickers pedidos (None si no hay datos)."""
        return {tk: self.ultimo(tk) for tk in self.tickers}


# --- get_serie ----------------------------------------------------------------------------------
# Huecos de una respuesta que se detallan uno a uno; con más, un solo aviso con el recuento.
MAX_HUECOS_DETALLADOS = 3


# Notas que tienen su propio aviso (no se repiten en el de huecos).
_NOTAS_CON_AVISO = (NOTA_EXCEPCION, NOTA_RECONSTRUIDA, NOTA_PREVIA_ALFRED, NOTA_REESTIMADA, NOTA_UNIDAD_VINTAGE)


def _sin_notas_comunes(nota: Any) -> str:
    """Nota de un hueco sin las notas que ya avisan aparte (revisables, excepción, ALFRED...)."""
    partes = [p for p in str(nota or "").split("; ")
              if p and not es_nota_revisable(p) and not p.startswith(_NOTAS_CON_AVISO)]
    return "; ".join(partes) or "sin nota"


def _contiene(nota: pd.Series, texto: str) -> pd.Series:
    return nota.str.contains(texto, regex=False)


def _avisos_serie(datos: pd.DataFrame) -> list[str]:
    """Avisos de una respuesta, agregados: como mucho uno por tipo (huecos, fechas, revisiones...).

    Bloque 3: una fila ``vintage_exacto`` (ALFRED) no avisa; sí las filas anteriores a ALFRED
    (valor del primer vintage con la fecha de la regla previa), los índices re-estimados enteros,
    la unidad por vintage y las ediciones de los ficheros académicos (vintage ``AAAA-MM``).
    """
    avisos = []
    nulos = datos[datos["valor"].isna()]
    if len(nulos) <= MAX_HUECOS_DETALLADOS:
        for fila in nulos.itertuples(index=False):
            avisos.append(f"{_iso(fila.fecha_referencia)}: valor nulo (hueco declarado): "
                          f"{_sin_notas_comunes(fila.nota)}")
    else:
        notas = sorted({_sin_notas_comunes(n) for n in nulos["nota"]})
        detalle = "; ".join(notas[:3]) + (f" (y {len(notas) - 3} notas más)" if len(notas) > 3 else "")
        avisos.append(f"{len(nulos)} periodos con valor nulo (huecos declarados) entre "
                      f"{_iso(nulos['fecha_referencia'].min())} y {_iso(nulos['fecha_referencia'].max())}: {detalle}")
    nota = datos["nota"].fillna("").astype(str)
    previas = _contiene(nota, NOTA_PREVIA_ALFRED)
    lag = (datos["metodo_publicacion"] == "lag_fijo") & ~previas
    medidas = _contiene(nota, NOTA_EXCEPCION)
    if previas.any():
        primera = nota[previas].iloc[0]
        desde = primera[primera.index(NOTA_PREVIA_ALFRED) + len(NOTA_PREVIA_ALFRED):].split(";")[0].strip(" ()")
        avisos.append(f"{int(previas.sum())} periodos anteriores a ALFRED (primer vintage {desde}): llevan el valor "
                      f"de ese primer vintage, ya revisado, con la fecha de la regla previa (lag_fijo): posible "
                      f"fuga de revisiones y fecha de publicación aproximada")
    if lag.any():
        avisos.append(f"fecha de publicación por regla en {int(lag.sum())} periodos (método lag_fijo: retraso "
                      f"declarado en lags.yaml, no la fecha real de cada publicación; corregido con la primera "
                      f"vintage de ALFRED en {int((lag & medidas).sum())} periodos)")
    elif medidas.any():
        avisos.append(f"fecha de publicación retrasada a la primera vintage de ALFRED en {int(medidas.sum())} "
                      f"periodos (la regla de su publicador la adelantaba)")
    reconstruidas = _contiene(nota, NOTA_RECONSTRUIDA)
    if reconstruidas.any():
        primera = nota[reconstruidas].iloc[0]
        desde = primera[primera.index(NOTA_RECONSTRUIDA) + len(NOTA_RECONSTRUIDA):].split(";")[0].strip()
        avisos.append(f"{int(reconstruidas.sum())} periodos con fecha de publicación anterior a la creación del "
                      f"índice ({desde}): son valores reconstruidos hacia atrás que nadie conocía en su fecha")
    reestimada = _contiene(nota, NOTA_REESTIMADA)
    if reestimada.any():
        exactas = datos.loc[reestimada & (datos["metodo_publicacion"] == "vintage_exacto"), "vintage"]
        ultimo = f" (último vintage visible: {exactas.max()})" if len(exactas) else ""
        avisos.append(f"índice re-estimado entero en cada publicación: los {int(reestimada.sum())} periodos son los "
                      f"del vintage vigente en la fecha{ultimo}; en otra fecha cambian también los valores antiguos")
    if _contiene(nota, NOTA_UNIDAD_VINTAGE).any():
        avisos.append("unidad del vintage vigente (año base o dólares encadenados): los niveles de vintages "
                      "distintos no son comparables entre sí")
    # Ediciones de los ficheros académicos (vintage AAAA-MM): una por descarga archivada.
    vint = datos["vintage"].astype(str)
    edicion = vint.str.fullmatch(r"\d{4}-\d{2}")
    if edicion.any():
        primera_ed = _contiene(nota, NOTA_PRIMERA_EDICION) & edicion
        ediciones = sorted(set(vint[edicion]))
        avisos.append(f"valores por edición del fichero (vintage AAAA-MM): {int(edicion.sum())} periodos de "
                      f"{len(ediciones)} ediciones ({', '.join(ediciones)}); la última visible en la fecha es "
                      f"{ediciones[-1]}" + (f"; {int(primera_ed.sum())} periodos con el valor de la primera edición "
                                            f"archivada: posible fuga de revisiones anteriores a ella"
                                            if primera_ed.any() else ""))
    # Series revisables cargadas con el último vintage o edición (bloque 2): lo dice la nota de la fila.
    revisadas = es_nota_revisable(datos["nota"]) & ~_contiene(nota, NOTA_PRIMERA_EDICION)
    if revisadas.any():
        avisos.append(f"serie revisable con los valores del último vintage o edición en {int(revisadas.sum())} "
                      f"periodos: posible fuga de revisiones (no son los valores que se conocían en su fecha)")
    return avisos


_MESES_PERIODO = (1, 3, 6, 12)


def _meses_periodo(referencias: pd.Series) -> int:
    """Meses del periodo de una serie (1, 3, 6 o 12) o 0 si es diaria o no se puede inferir.

    Periódica = todas las ``fecha_referencia`` son día 1 y hay al menos dos; el periodo es la
    separación mínima en meses entre referencias consecutivas.
    """
    u = pd.DatetimeIndex(referencias.dropna().unique()).sort_values()
    if len(u) < 2 or (u.day != 1).any():
        return 0
    paso = int(np.diff(u.year * 12 + u.month).min())
    return paso if paso in _MESES_PERIODO else 0


def get_serie(id: str, fecha: FechaLike, *, desde: FechaLike | None = None, divisa: str = "original",
              base: pd.DataFrame | None = None, alias: dict[str, str] | None = None,
              base_fx: pd.DataFrame | None = None) -> RespuestaSerie:
    """Serie ``id`` tal como se conocía en ``fecha`` (``timegate.as_of_series``).

    ``id``: canónico con prefijo de fuente (``'fred:DGS10'``) o alias de la Fase 1 (``'VIX'``),
    resuelto con ``alias`` o, por defecto, con ``alias_por_defecto()`` (catálogo de series y
    ``publicaciones.yaml``). Una ETF que el catálogo excluye de la base (``excluida_ajustada``)
    lanza ``KeyError`` con un mensaje que remite a ``get_precios``.
    ``desde``: recorta por ``fecha_referencia >= desde`` (con un aviso si no queda nada).
    ``base``: tabla del contrato (por defecto ``almacen.leer_series``); ``base_fx``: tabla con el
    FX (por defecto ``base`` si se pasó, si no el almacén).

    ``divisa``: ``'original'`` o ``'USD'``. Con ``'USD'`` los datos llevan siempre
    ``divisa_original``, ``fx_usd`` y ``fx_fecha`` (como ``get_precios``): una serie en USD lleva
    ``fx_usd = 1``; una no monetaria, ``fx_usd`` nulo y un aviso. El FX es el del BCE visible en
    ``fecha`` en el fin del periodo de cada valor: la fecha de referencia en una serie diaria; el
    último día del mes, trimestre, semestre o año en una periódica (fechas de referencia en día
    1, periodo inferido de las visibles; con un solo periodo visible, su día 1). Si ese día aún no
    tenía fixing visible en ``fecha`` (periodo en curso), se usa el último visible y se avisa.

    ``KeyError`` si no hay nada publicado del id hasta ``fecha``, tanto si no existe como si se
    publicará después (no se revela cuál). Unidad, divisa y fuente salen de filas visibles. Un
    periodo con valor nulo (hueco permanente) se devuelve nulo y añade un aviso con su nota (con
    más de ``MAX_HUECOS_DETALLADOS``, uno solo con el recuento, el primero, el último y las notas
    distintas); el método ``lag_fijo`` añade el aviso «fecha de publicación por regla»; una fecha
    retrasada a la primera vintage de ALFRED, una serie reconstruida hacia atrás y una revisable
    cargada con el último vintage o edición (``catalogo.NOTAS_REVISABLE``) añaden el suyo. Bloque 3:
    las filas ``vintage_exacto`` de ALFRED no avisan; sí las anteriores a ALFRED (valor del primer
    vintage con la fecha de la regla previa), los índices re-estimados enteros, la unidad por
    vintage y las ediciones de los ficheros académicos (vintage ``AAAA-MM``). Como mucho un aviso
    por tipo. Un id sin alias se resuelve sin distinguir mayúsculas si es único.
    """
    f = timegate.fecha_decision(fecha)
    _validar_opcion(divisa, DIVISAS_SALIDA, "divisa")
    canonico = resolver_id(id, alias)
    _comprobar_excluida(canonico, id)
    corte = timegate.fecha_decision(desde) if desde is not None else None
    # Del almacén: solo la serie, lo publicado hasta f y la última fila por periodo (resuelto en Arrow).
    tabla = base if base is not None else almacen.leer_ultimas(canonico, f)
    visibles = timegate.as_of_series(tabla, canonico, f)
    if visibles.empty:
        nombre = repr(canonico) if canonico == id else f"{canonico!r} (alias {id!r})"
        raise KeyError(f"serie {nombre} sin datos publicados hasta {_iso(f)} "
                       f"(desconocida o aún no publicada)")
    # Unidad, divisa y fuente son constantes por serie (contrato); se toman de lo visible.
    meta = visibles.iloc[-1]
    unidad, divisa_serie, fuente = _texto(meta["unidad"]), _texto(meta["divisa"]), _texto(meta["fuente"])

    avisos: list[str] = []
    meses = _meses_periodo(visibles["fecha_referencia"])
    datos = visibles[list(COLUMNAS_DATOS_SERIE)]
    if corte is not None:
        datos = datos[datos["fecha_referencia"] >= corte]
        if datos.empty:
            avisos.append(f"{canonico}: ningún periodo visible en {_iso(f)} desde {_iso(corte)}")
    datos = datos.reset_index(drop=True)
    avisos += _avisos_serie(datos)

    columnas = dict(COLUMNAS_DATOS_SERIE)
    divisa_salida = divisa_serie
    if divisa == "USD":
        columnas |= _COLUMNAS_FX
        if divisa_serie is None:
            avisos.append(f"{canonico} no es monetaria (sin divisa): se devuelve sin convertir")
            datos = datos.assign(divisa_original=None, fx_usd=np.nan, fx_fecha=pd.NaT)
        elif _es_tipo_de_cambio(canonico):
            # Las cotizaciones del BCE (X por 1 EUR) llevan divisa X, pero son tipos de cambio, no importes:
            # convertirlas daría USD por EUR con la unidad «X por 1 EUR». Mismo criterio que fred:DEX*.
            avisos.append(f"{canonico} es un tipo de cambio (cotización, no un importe en {divisa_serie}): "
                          "se devuelve sin convertir")
            datos = datos.assign(divisa_original=divisa_serie, fx_usd=np.nan, fx_fecha=pd.NaT)
        else:
            fx = base_fx if base_fx is not None else (base if base is not None else _base_fx([divisa_serie]))
            ref = datos["fecha_referencia"]
            fecha_fx = ref + pd.offsets.MonthEnd(meses) if meses else ref
            convertidos = divisas.a_usd(datos.assign(divisa=divisa_serie, fecha_fx=fecha_fx), ["valor"],
                                        "divisa", fx, f, "fecha_fx")
            avisos += convertidos.attrs.get("avisos", [])
            datos = convertidos.drop(columns=["divisa", "fecha_fx"])
            divisa_salida = "USD"
    datos = datos[list(columnas)].astype(columnas).reset_index(drop=True)
    datos.attrs = {}
    metodo = _texto(datos["metodo_publicacion"].iloc[-1]) if len(datos) else None
    return RespuestaSerie(serie=canonico, fecha=f, datos=datos, unidad=unidad, divisa=divisa_salida,
                          fuente=fuente, metodo_publicacion=metodo, avisos=avisos)


def _es_tipo_de_cambio(serie: str) -> bool:
    """True en las cotizaciones de divisas del BCE (``divisas.SERIE_FX``: ``ecb:EXR.D.<X>.EUR``)."""
    prefijo, _, sufijo = divisas.SERIE_FX.partition("{divisa}")
    return serie.startswith(prefijo) and serie.endswith(sufijo)


def _base_fx(divisas_necesarias: Sequence[str]) -> pd.DataFrame:
    """Series del BCE que hacen falta para convertir ``divisas_necesarias`` (leídas del almacén)."""
    isos = {divisas.normalizar_subunidad(1.0, d)[1] for d in divisas_necesarias} - {"USD"}
    if not isos:
        return vacio_series()
    ids = {divisas.SERIE_USD_EUR} | {divisas.SERIE_FX.format(divisa=d) for d in isos if d != "EUR"}
    return almacen.leer_series(sorted(ids))


# --- get_precios --------------------------------------------------------------------------------
def _precios_ticker(tk: str, base: pd.DataFrame | None, proveedor: str) -> pd.DataFrame:
    if base is not None:
        return base[base["ticker"] == tk]
    return almacen.leer_precios(tk, proveedor)


def _publicar_ajuste(ajustado: pd.DataFrame, ajuste: str) -> pd.DataFrame:
    """Columnas de salida con el ajuste pedido (las ajustadas ocupan el sitio de las crudas)."""
    out = ajustado.copy()
    for destino, origen in _ORIGEN_AJUSTE[ajuste].items():
        out[destino] = ajustado[origen]
    out = out[list(COLUMNAS_DATOS_PRECIOS)]
    out.attrs = {}
    return out


def get_precios(tk: str | Sequence[str], fecha: FechaLike, *, desde: FechaLike | None = None,
                ajuste: Ajuste = "total", divisa: str = "USD", base: pd.DataFrame | None = None,
                base_fx: pd.DataFrame | None = None, proveedor: str = "yahoo") -> RespuestaPrecios:
    """Precios de ``tk`` (uno o varios) con el ajuste que se conocía en ``fecha``.

    ``ajuste``: ``'ninguno'`` (as-traded), ``'splits'`` o ``'total'`` (splits y dividendos), vía
    ``timegate.ajustar_a_fecha`` POR TICKER y con ``fecha`` (nunca con eventos posteriores).
    ``divisa``: ``'USD'`` (subunidades normalizadas y FX del BCE visible en ``fecha``, por sesión)
    u ``'original'``. ``base``: tabla de precios del contrato (por defecto ``almacen.leer_precios``
    de ``proveedor``); ``base_fx``: tabla de series con el FX (por defecto ``almacen.leer_series``).
    ``desde`` recorta por ``sesion >= desde`` DESPUÉS de ajustar (el ajuste no depende de ``desde``).

    Un ticker sin nada publicado hasta ``fecha`` no lanza: no aporta filas y añade el mismo aviso
    neutro tanto si no existe como si cotizará después (no se revela cuál ni desde cuándo). Un
    ticker que ``desde`` deja sin filas, los dividendos que no se pueden ajustar y las sesiones
    sin FX también se avisan.
    """
    f = timegate.fecha_decision(fecha)
    _validar_opcion(ajuste, AJUSTES, "ajuste")
    _validar_opcion(divisa, DIVISAS_SALIDA, "divisa")
    lista = [tk] if isinstance(tk, str) else list(tk)
    if not lista or not all(isinstance(t, str) and t for t in lista):
        raise ValueError(f"tickers vacíos o no textuales: {tk!r}")
    tickers = list(dict.fromkeys(lista))
    corte = timegate.fecha_decision(desde) if desde is not None else None

    avisos: list[str] = []
    tablas = []
    for t in tickers:
        crudo = _precios_ticker(t, base, proveedor)
        # avisar=False: los avisos vuelven en attrs, sin tocar el estado global de warnings.
        ajustado = timegate.ajustar_a_fecha(crudo, f, avisar=False)
        if ajustado.empty:
            avisos.append(f"{t}: sin sesiones publicadas hasta {_iso(f)} (ticker desconocido para "
                          f"{proveedor!r} o aún sin datos)")
            continue
        avisos += ajustado.attrs.get("avisos", [])
        salida = _publicar_ajuste(ajustado, ajuste)
        if corte is not None:
            salida = salida[salida["sesion"] >= corte]
            if salida.empty:
                avisos.append(f"{t}: ninguna sesión visible en {_iso(f)} desde {_iso(corte)}")
                continue
        tablas.append(salida)

    columnas = dict(COLUMNAS_DATOS_PRECIOS)
    datos = pd.concat(tablas, ignore_index=True) if tablas else _vacio(columnas)
    if divisa == "USD":
        columnas |= _COLUMNAS_FX
        if len(datos):
            necesarias = sorted(set(datos["divisa"]))
            fx = base_fx if base_fx is not None else _base_fx(necesarias)
            datos = divisas.a_usd(datos, list(_MONETARIAS_PRECIOS), "divisa", fx, f, "sesion")
            avisos += datos.attrs.get("avisos", [])
            datos.attrs = {}
        else:
            datos = _vacio(columnas)
    datos = datos[list(columnas)].astype(columnas).reset_index(drop=True)
    return RespuestaPrecios(tickers=tickers, fecha=f, datos=datos, ajuste=ajuste, divisa=divisa,
                            avisos=avisos)


__all__ = [
    "AJUSTES", "COLUMNAS_DATOS_PRECIOS", "COLUMNAS_DATOS_SERIE", "DIVISAS_SALIDA", "RespuestaPrecios",
    "RespuestaSerie", "alias_por_defecto", "get_precios", "get_serie", "resolver_id",
]
