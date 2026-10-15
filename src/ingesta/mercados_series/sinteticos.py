"""Datos sintéticos deterministas del Tema 1 (sin red) para tests y notebooks.

Cada generador recibe una ``semilla`` y devuelve una tabla que ya cumple el contrato
(``validar_series`` o ``validar_precios``). Los ids llevan el prefijo de fuente ``sint:`` y los
tickers son ficticios (prefijo ``ZZ`` o ``9999``).

- ``serie_con_vintages``    paro mensual: primera publicación y revisión anual de enero (vintages),
                            un retraso por cierre del Gobierno y un hueco permanente.
- ``serie_mercado_diaria``  índice diario que cierra con la bolsa de Nueva York (``cierre_mercado``,
                            borde de las 16:30).
- ``accion_con_eventos``    acción as-traded con un split 2:1 y 4 dividendos trimestrales.
- ``accion_eventos_dificiles`` contrasplit, split 3:2 y split con dividendo en la misma sesión.
- ``vista_yahoo_hoy``       lo que mostraría Yahoo HOY de esa acción (SOLO para contrastar;
                            lleva ``Adj Close`` y nunca se guarda).
- ``fx_ecb_sintetico``      tipos de referencia del BCE (USD, GBP y JPY por 1 EUR).
- ``precios_multimercado``  dos semanas de XNYS, XLON, XETR, XTKS, CME y CRYPTO alrededor del cambio
                            de hora de EE. UU. (2024-03-10), con el de Europa aún pendiente.

Las horas de cierre salen de ``config/mercados.yaml`` (``visibilidad.disponible_cierre_serie``),
la misma fuente de verdad que usan los extractores.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from dateutil.easter import easter

from ingesta.mercados_series import visibilidad
from ingesta.mercados_series.contrato import (
    HUSO_DECISION,
    primera_decision_vec,
    validar_precios,
    validar_series,
)

FUENTE = "sintetico"

# --- accion_con_eventos: eventos conocidos (los tests los usan) -------------------------------
TICKER_ACCION = "ZZAC"
SESION_SPLIT = pd.Timestamp("2023-06-12")
RATIO_SPLIT = 2.0
# Fecha ex → dividendo por acción en términos de HOY (acciones posteriores al split).
DIVIDENDOS_ECONOMICOS = {
    pd.Timestamp("2022-11-15"): 0.40,
    pd.Timestamp("2023-02-15"): 0.40,
    pd.Timestamp("2023-08-15"): 0.40,
    pd.Timestamp("2023-11-15"): 0.40,
}

# --- accion_eventos_dificiles: contrasplit, split fraccionario y split + dividendo el mismo día -
TICKER_DIFICIL = "ZZRV"
SPLITS_DIFICILES = {pd.Timestamp("2024-03-15"): 0.1, pd.Timestamp("2024-04-15"): 1.5}
DIVIDENDOS_DIFICILES = {   # por acción de HOY
    pd.Timestamp("2024-02-15"): 0.30,
    pd.Timestamp("2024-04-15"): 0.50,   # misma sesión que el split 3:2
    pd.Timestamp("2024-05-15"): 0.20,
}

# --- serie_con_vintages ----------------------------------------------------------------------
SERIE_VINTAGES = "sint:PARO"
HORA_INFORME_EMPLEO = time(8, 30)          # America/New_York
# Mes de referencia → fecha real del informe (None = nunca se publicó). Lo demás: primer viernes.
_INFORMES_EXCEPCIONALES: dict[pd.Timestamp, pd.Timestamp | None] = {
    pd.Timestamp("2024-12-01"): pd.Timestamp("2025-01-10"),   # el primer viernes caía el 3 de enero
    pd.Timestamp("2025-09-01"): pd.Timestamp("2025-11-20"),   # retrasado por el cierre del Gobierno
    pd.Timestamp("2025-10-01"): None,                          # hueco permanente
    pd.Timestamp("2025-11-01"): pd.Timestamp("2025-12-16"),
    pd.Timestamp("2025-12-01"): pd.Timestamp("2026-01-09"),
    pd.Timestamp("2026-01-01"): pd.Timestamp("2026-02-11"),   # previsto el 6, retrasado por el cierre parcial
}
# La revisión anual de factores estacionales (con el informe de enero) reescribe los datos
# desestacionalizados de los últimos años de referencia completos.
ANIOS_REVISION_ANUAL = 5
MES_RETRASADO = pd.Timestamp("2025-09-01")
MES_HUECO = pd.Timestamp("2025-10-01")
FECHA_PREVISTA_RETRASADO = pd.Timestamp("2025-10-03")
INFORME_HUECO = pd.Timestamp("2025-12-16")   # informe en que se supo que el hueco no se publicaría
NOTA_RETRASO = "publicación retrasada por el cierre del Gobierno federal (prevista 2025-10-03)"
NOTA_HUECO = ("hueco permanente: no publicado por el cierre del Gobierno federal "
              "(1-oct a 12-nov-2025); la encuesta de hogares de octubre no se recogió")

SERIE_DIARIA = "sint:INDICE_NY"
SERIES_FX = ("ecb:EXR.D.USD.EUR", "ecb:EXR.D.GBP.EUR", "ecb:EXR.D.JPY.EUR")
HORA_FX_BCE = time(16, 0)                  # Europe/Berlin


# --- Utilidades ----------------------------------------------------------------------------------
def _disponible_cierre(sesiones: pd.DatetimeIndex, mercado: str) -> pd.Series:
    """Cierre local de cada sesión + ``MARGEN_CIERRE``, en UTC (``config/mercados.yaml``)."""
    return visibilidad.disponible_cierre_serie(pd.Series(pd.DatetimeIndex(sesiones)), mercado)


def _instante_local(fechas: pd.DatetimeIndex, hora: time, huso: str) -> pd.Series:
    """Fecha local + hora → instante UTC."""
    local = (pd.DatetimeIndex(fechas) + pd.Timedelta(hours=hora.hour, minutes=hora.minute)).tz_localize(huso)
    return pd.Series(local.tz_convert("UTC")).astype("datetime64[ns, UTC]")


def _dias_target(desde: str, hasta: str) -> pd.DatetimeIndex:
    """Días TARGET2: laborables sin 1-ene, Viernes Santo, Lunes de Pascua, 1-may, 25 y 26-dic."""
    dias = pd.bdate_range(desde, hasta)
    festivos = set()
    for a in range(dias.min().year, dias.max().year + 1):
        pascua = pd.Timestamp(easter(a))
        festivos |= {pd.Timestamp(a, 1, 1), pascua - pd.Timedelta(days=2), pascua + pd.Timedelta(days=1),
                     pd.Timestamp(a, 5, 1), pd.Timestamp(a, 12, 25), pd.Timestamp(a, 12, 26)}
    return dias[~dias.isin(list(festivos))]


def _primer_viernes(mes: pd.Timestamp) -> pd.Timestamp:
    d = mes.replace(day=1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def _tabla_precios(ticker: str, mercado: str, divisa: str, sesiones: pd.DatetimeIndex,
                   o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray,
                   dividendo: np.ndarray, split: np.ndarray, descargado: pd.Timestamp) -> pd.DataFrame:
    disponible = _disponible_cierre(sesiones, mercado)
    return pd.DataFrame({
        "ticker": ticker, "sesion": sesiones, "mercado": mercado,
        "open": o, "high": h, "low": lo, "close": c, "volume": v,
        "dividendo": dividendo, "split": split, "divisa": divisa,
        "fecha_publicacion": primera_decision_vec(disponible).to_numpy(),
        "disponible_utc": disponible.to_numpy(),
        "fuente": FUENTE, "descargado": descargado,
    })


def _ohlc(rng: np.random.Generator, apertura: np.ndarray, cierre: np.ndarray, decimales: int = 2):
    """High y low envolventes de apertura y cierre, con ruido; todo redondeado."""
    sube = np.abs(rng.normal(0.0, 0.006, len(cierre)))
    baja = np.abs(rng.normal(0.0, 0.006, len(cierre)))
    o = np.round(apertura, decimales)
    c = np.round(cierre, decimales)
    h = np.round(np.maximum(o, c) * (1 + sube), decimales)
    lo = np.round(np.minimum(o, c) * (1 - baja), decimales)
    return o, np.maximum(h, np.maximum(o, c)), np.minimum(lo, np.minimum(o, c)), c


# --- Series ------------------------------------------------------------------------------------
def serie_con_vintages(semilla: int = 11) -> pd.DataFrame:
    """Tasa de paro mensual sintética (``sint:PARO``) con vintages; tabla larga del contrato.

    Imita la tasa de paro de EE. UU. (encuesta de hogares, ``fred:UNRATE``), que NO se revisa mes a
    mes: meses de referencia 2023-01 a 2026-02; el informe del mes ``m`` sale el primer viernes de
    ``m+1`` a las 08:30 Nueva York (salvo ``_INFORMES_EXCEPCIONALES``) con la primera estimación de
    ``m``. La única revisión es la anual de factores estacionales, con el informe de enero: reescribe
    los meses de los ``ANIOS_REVISION_ANUAL`` años anteriores. Así un mes tiene de 1 a 4 vintages
    (2023: primera más las revisiones de 2024, 2025 y 2026). Nivel de 3,5 % a ~4,4 %, como el real.
    Septiembre de 2025 se publica tarde (2025-11-20, con nota); octubre de 2025 nunca se publica:
    una fila con valor nulo y nota en el informe de 2025-12-16, que ninguna revisión rellena.
    ``vintage`` = fecha del informe (estilo ALFRED).
    """
    rng = np.random.default_rng(semilla)
    meses = pd.date_range("2023-01-01", "2026-02-01", freq="MS")
    informe = {m: _INFORMES_EXCEPCIONALES.get(m, _primer_viernes(m + pd.offsets.MonthBegin(1)))
               for m in meses}
    tendencia = np.linspace(3.5, 4.4, len(meses))
    real = np.clip(tendencia + np.cumsum(rng.normal(0.0, 0.04, len(meses))), 3.0, 6.0)
    ruido_primera, ruido_revision = 0.08, 0.03
    # Informes de enero (revisión anual) dentro del rango: año → fecha del informe.
    anuales = {m.year: f for m, f in informe.items() if m.month == 1 and f is not None}

    filas: list[dict] = []
    for i, m in enumerate(meses):
        primera = informe[m]
        if primera is None:
            filas.append({"fecha_referencia": m, "pub_local": INFORME_HUECO, "valor": np.nan,
                          "nota": NOTA_HUECO})
            continue
        nota = NOTA_RETRASO if m == MES_RETRASADO else None
        filas.append({"fecha_referencia": m, "pub_local": primera, "nota": nota,
                      "valor": round(float(real[i] + rng.normal(0.0, ruido_primera)), 1)})
        for anio, f in sorted(anuales.items()):
            if anio - ANIOS_REVISION_ANUAL <= m.year < anio and f > primera:
                filas.append({"fecha_referencia": m, "pub_local": f, "nota": None,
                              "valor": round(float(real[i] + rng.normal(0.0, ruido_revision)), 1)})

    df = pd.DataFrame(filas)
    disponible = _instante_local(pd.DatetimeIndex(df["pub_local"]), HORA_INFORME_EMPLEO, HUSO_DECISION)
    df = df.assign(
        serie=SERIE_VINTAGES,
        fecha_publicacion=primera_decision_vec(disponible).to_numpy(),
        disponible_utc=disponible.to_numpy(),
        vintage=df["pub_local"].dt.strftime("%Y-%m-%d"),
        unidad="porcentaje", divisa=None, fuente=FUENTE, metodo_publicacion="vintage_exacto",
    ).drop(columns="pub_local")
    return validar_series(df)


def serie_mercado_diaria(semilla: int = 3, desde: str = "2023-01-02", hasta: str = "2025-12-31") -> pd.DataFrame:
    """Índice diario sintético (``sint:INDICE_NY``) que cierra con la bolsa de Nueva York (XNYS).

    Sirve para probar el borde: visible a las 16:00 + 30 min de Nueva York = 16:30, justo el
    momento de decisión, así que ``fecha_publicacion == fecha_referencia`` (borde incluido). No
    imita el VIX: Cboe lo calcula hasta las 16:15 (mercado ``CBOE_IDX``), luego el VIX de la sesión
    t se ve en t+1. Sesiones de lunes a viernes; log-proceso de Ornstein-Uhlenbeck alrededor de 18
    con saltos; vintage ``unico``.
    """
    rng = np.random.default_rng(semilla)
    sesiones = pd.bdate_range(desde, hasta)
    x = np.empty(len(sesiones))
    x[0] = np.log(18.0)
    choques = rng.normal(0.0, 0.07, len(sesiones))
    saltos = (rng.random(len(sesiones)) < 0.01) * 0.4
    for t in range(1, len(sesiones)):
        x[t] = x[t - 1] + 0.05 * (np.log(18.0) - x[t - 1]) + choques[t] + saltos[t]
    disponible = _disponible_cierre(sesiones, "XNYS")
    df = pd.DataFrame({
        "serie": SERIE_DIARIA, "fecha_referencia": sesiones,
        "fecha_publicacion": primera_decision_vec(disponible).to_numpy(),
        "disponible_utc": disponible.to_numpy(), "vintage": "unico",
        "valor": np.round(np.exp(x), 2), "unidad": "puntos de índice", "divisa": None,
        "fuente": FUENTE, "metodo_publicacion": "cierre_mercado", "nota": None,
    })
    return validar_series(df)


def fx_ecb_sintetico(semilla: int = 13, desde: str = "2022-01-03", hasta: str = "2025-12-31") -> pd.DataFrame:
    """Tipos de referencia del BCE sintéticos: unidades de X por 1 EUR, X en USD, GBP y JPY.

    Ids ``ecb:EXR.D.<X>.EUR``; días TARGET2; publicados a las 16:00 Europe/Berlin (visibles el
    mismo día de decisión); ``metodo_publicacion='calendario'``; ``divisa`` = X (la del valor).
    """
    rng = np.random.default_rng(semilla)
    dias = _dias_target(desde, hasta)
    disponible = _instante_local(dias, HORA_FX_BCE, "Europe/Berlin")
    fecha_pub = primera_decision_vec(disponible).to_numpy()
    tablas = []
    for serie, (inicio, sigma, decimales) in zip(
            SERIES_FX, ((1.10, 0.004, 4), (0.86, 0.003, 4), (140.0, 0.005, 2)), strict=True):
        divisa = serie.split(".")[2]
        valor = np.round(inicio * np.exp(np.cumsum(rng.normal(0.0, sigma, len(dias)))), decimales)
        tablas.append(pd.DataFrame({
            "serie": serie, "fecha_referencia": dias, "fecha_publicacion": fecha_pub,
            "disponible_utc": disponible.to_numpy(), "vintage": "unico", "valor": valor,
            "unidad": f"{divisa} por 1 EUR", "divisa": divisa, "fuente": FUENTE,
            "metodo_publicacion": "calendario", "nota": None,
        }))
    return validar_series(pd.concat(tablas, ignore_index=True))


# --- Precios -----------------------------------------------------------------------------------
def accion_con_eventos(semilla: int = 5, descargado: pd.Timestamp | None = None) -> pd.DataFrame:
    """Acción ficticia ``ZZAC`` (XNYS, USD), as-traded, de 2022-01-03 a 2024-12-31.

    Split 2:1 en ``SESION_SPLIT`` (``split=2.0`` en esa fila; el precio as-traded se parte por
    la mitad) y 4 dividendos trimestrales en ``DIVIDENDOS_ECONOMICOS``: 0,40 por acción de hoy, que
    as-traded son 0,80 antes del split. El cierre cae lo que el dividendo en su fecha ex. El
    volumen está en acciones de su día (la mitad antes del split). Sin columnas ajustadas.
    """
    rng = np.random.default_rng(semilla)
    sesiones = pd.bdate_range("2022-01-03", "2024-12-31")
    n = len(sesiones)
    dividendo_hoy = np.array([DIVIDENDOS_ECONOMICOS.get(s, 0.0) for s in sesiones])
    retornos = rng.normal(0.0003, 0.015, n)
    cierre_hoy = np.empty(n)
    cierre_hoy[0] = 50.0
    for t in range(1, n):
        cierre_hoy[t] = cierre_hoy[t - 1] * np.exp(retornos[t]) - dividendo_hoy[t]
    previo = np.r_[cierre_hoy[0], cierre_hoy[:-1]]
    apertura_hoy = (previo - dividendo_hoy) * np.exp(rng.normal(0.0, 0.001, n))

    acciones_por_hoy = np.where(sesiones < SESION_SPLIT, RATIO_SPLIT, 1.0)  # precio as-traded = hoy × esto
    o, h, lo, c = _ohlc(rng, apertura_hoy * acciones_por_hoy, cierre_hoy * acciones_por_hoy)
    volumen = np.round(2e6 * rng.lognormal(0.0, 0.3, n) / acciones_por_hoy)
    split = np.where(sesiones == SESION_SPLIT, RATIO_SPLIT, 1.0)
    descargado = descargado if descargado is not None else pd.Timestamp("2025-01-02 12:00", tz="UTC")
    df = _tabla_precios(TICKER_ACCION, "XNYS", "USD", sesiones, o, h, lo, c, volumen,
                        np.round(dividendo_hoy * acciones_por_hoy, 2), split, descargado)
    return validar_precios(df)


def accion_eventos_dificiles(semilla: int = 23, descargado: pd.Timestamp | None = None) -> pd.DataFrame:
    """Acción ficticia ``ZZRV`` (XNYS, USD), as-traded, primer semestre de 2024, con casos difíciles.

    Contrasplit 1:10 (``split=0.1``) en 2024-03-15, split 3:2 (``split=1.5``) con dividendo en la
    MISMA sesión (2024-04-15) y otros dos dividendos (``SPLITS_DIFICILES``,
    ``DIVIDENDOS_DIFICILES``: por acción de hoy; as-traded, multiplicados por los splits
    posteriores). El dividendo del día del split va en acciones nuevas, como dice el contrato.
    """
    rng = np.random.default_rng(semilla)
    sesiones = pd.bdate_range("2024-01-02", "2024-06-28")
    n = len(sesiones)
    split = np.array([SPLITS_DIFICILES.get(s, 1.0) for s in sesiones])
    dividendo_hoy = np.array([DIVIDENDOS_DIFICILES.get(s, 0.0) for s in sesiones])
    cierre_hoy = np.empty(n)
    cierre_hoy[0] = 40.0
    retornos = rng.normal(0.0, 0.02, n)
    for t in range(1, n):
        cierre_hoy[t] = cierre_hoy[t - 1] * np.exp(retornos[t]) - dividendo_hoy[t]
    previo = np.r_[cierre_hoy[0], cierre_hoy[:-1]]
    apertura_hoy = (previo - dividendo_hoy) * np.exp(rng.normal(0.0, 0.002, n))
    acciones_por_hoy = np.r_[np.cumprod(split[::-1])[::-1][1:], 1.0]   # splits posteriores a cada sesión
    o, h, lo, c = _ohlc(rng, apertura_hoy * acciones_por_hoy, cierre_hoy * acciones_por_hoy, decimales=4)
    volumen = np.round(5e5 * rng.lognormal(0.0, 0.3, n) / acciones_por_hoy)
    descargado = descargado if descargado is not None else pd.Timestamp("2024-07-01 12:00", tz="UTC")
    df = _tabla_precios(TICKER_DIFICIL, "XNYS", "USD", sesiones, o, h, lo, c, volumen,
                        np.round(dividendo_hoy * acciones_por_hoy, 4), split, descargado)
    return validar_precios(df)


def vista_yahoo_hoy(precios: pd.DataFrame) -> pd.DataFrame:
    """Lo que mostraría Yahoo HOY (``auto_adjust=False``) de un ticker: SOLO para contrastar.

    ``Open/High/Low/Close`` y ``Dividends`` ajustados por TODOS los splits (también los futuros
    respecto a cada fila), ``Volume`` multiplicado por ellos, ``Adj Close`` además por todos los
    dividendos (método multiplicativo: antes de cada fecha ex ``i``, factor ``1 - D_i / C_{i-1}``)
    y ``Stock Splits`` = 0 sin evento (convención de Yahoo). Índice ``Date``. Contiene
    ``Adj Close``: nunca pasa ``validar_precios`` ni se guarda.
    """
    if precios["ticker"].nunique() != 1:
        raise ValueError("vista_yahoo_hoy espera un único ticker")
    p = precios.sort_values("sesion").reset_index(drop=True)
    split = p["split"].to_numpy(float)
    posteriores = np.r_[np.cumprod(split[::-1])[::-1][1:], 1.0]   # producto de splits de j > i
    ajusta = 1.0 / posteriores
    cierre = p["close"].to_numpy(float) * ajusta
    dividendo = p["dividendo"].to_numpy(float) * ajusta
    factor_div = np.ones(len(p))
    for i in np.flatnonzero(dividendo > 0):
        if i > 0:
            factor_div[:i] *= 1.0 - dividendo[i] / cierre[i - 1]
    return pd.DataFrame({
        "Open": p["open"].to_numpy(float) * ajusta,
        "High": p["high"].to_numpy(float) * ajusta,
        "Low": p["low"].to_numpy(float) * ajusta,
        "Close": cierre,
        "Adj Close": cierre * factor_div,
        "Volume": p["volume"].to_numpy(float) * posteriores,
        "Dividends": dividendo,
        "Stock Splits": np.where(split == 1.0, 0.0, split),
    }, index=pd.DatetimeIndex(p["sesion"], name="Date"))


# Ticker, divisa de cotización y precio inicial por mercado.
_MULTIMERCADO: dict[str, tuple[str, str, float]] = {
    "XNYS": ("ZZNY", "USD", 100.0),
    "XLON": ("ZZLN.L", "GBp", 2500.0),        # peniques: subunidad
    "XETR": ("ZZDE.DE", "EUR", 80.0),
    "XTKS": ("9999.T", "JPY", 3000.0),
    "CME": ("ZZ=F", "USD", 5000.0),
    "CRYPTO": ("ZZ-USD", "USD", 60000.0),
}


def precios_multimercado(semilla: int = 17, desde: str = "2024-03-04", hasta: str = "2024-03-15",
                         descargado: pd.Timestamp | None = None) -> pd.DataFrame:
    """Un ticker ficticio por mercado (XNYS, XLON, XETR, XTKS, CME, CRYPTO), sin eventos.

    Bolsas y CME de lunes a viernes (sin festivos); CRYPTO los 7 días. Las fechas por defecto
    cruzan el cambio de hora de EE. UU. (2024-03-10) con Europa aún en horario de invierno.
    ``fecha_publicacion``: XNYS, XLON, XETR y XTKS el mismo día; CME y CRYPTO el siguiente.
    XLON cotiza en peniques (``GBp``).
    """
    rng = np.random.default_rng(semilla)
    descargado = descargado if descargado is not None else pd.Timestamp("2024-03-18 12:00", tz="UTC")
    tablas = []
    for mercado, (ticker, divisa, inicio) in _MULTIMERCADO.items():
        sesiones = (pd.date_range(desde, hasta, freq="D") if mercado == "CRYPTO"
                    else pd.bdate_range(desde, hasta))
        n = len(sesiones)
        cierre = inicio * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
        apertura = np.r_[inicio, cierre[:-1]] * np.exp(rng.normal(0.0, 0.003, n))
        o, h, lo, c = _ohlc(rng, apertura, cierre)
        volumen = np.round(1e5 * rng.lognormal(0.0, 0.3, n))
        tablas.append(_tabla_precios(ticker, mercado, divisa, sesiones, o, h, lo, c, volumen,
                                     np.zeros(n), np.ones(n), descargado))
    return validar_precios(pd.concat(tablas, ignore_index=True))
