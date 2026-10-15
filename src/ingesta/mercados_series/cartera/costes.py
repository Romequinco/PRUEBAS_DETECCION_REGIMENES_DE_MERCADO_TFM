"""Modelo de costes de transacción del Tema 1 (bloque 8).

``coste = comisión + FX + spread``, en USD por operación:

- **Comisión**: tarifa pública del bróker por grupo de mercado. DEGIRO (inversor español con
  10.000 € de patrimonio) es el caso base e IBKR Pro (tarifa fija) la sensibilidad. Las cifras
  son la tarifa pública consultada de memoria y van con ``verificado = False`` en
  ``costes_modelo.csv``: hay que contrastarlas con la web del bróker antes de citarlas.
- **FX**: ``fx_pct`` del bróker sobre el importe si la divisa del activo no es EUR (DEGIRO AutoFX
  0,25 %).
- **Spread**: medio spread base por clase de activo × multiplicador de crisis
  ``m_t = clip(CHL_21d / mediana en calma, 1, 10)``. ``CHL_21d`` es el estimador de Abdi-Ranaldo
  (cierre, máximo y mínimo) medio de 21 sesiones sobre OHLC AS-TRADED (los pares que cruzan un
  split se descartan); Corwin-Schultz se calcula igual como contraste. La «calma» son las
  sesiones con volatilidad realizada de 21 días por debajo de su mediana expansiva, y la mediana
  en calma es la mediana expansiva del CHL en esas sesiones: todo con datos hasta la sesión.

``m_t`` se indexa por ``fecha_publicacion`` de la sesión (lo que se sabía en esa decisión).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VENTANA_CHL = 21
MIN_SESIONES_CALMA = 252
LIMITES_MULTIPLICADOR = (1.0, 10.0)

# Grupo de comisión por mercado (MIC o código de config/mercados.yaml).
GRUPO_MERCADO: dict[str, str] = {
    "ARCX": "US", "XNAS": "US", "XNYS": "US", "BATS": "US", "NAV_US": "FONDO_US",
    "XMAD": "ES", "XETR": "EU", "XPAR": "EU", "XAMS": "EU", "XMIL": "EU", "XBRU": "EU", "XLIS": "EU",
    "XLON": "UK", "XSWX": "CH", "XTKS": "ASIA", "XHKG": "ASIA", "XTAI": "ASIA", "XKRX": "ASIA",
    "XSHG": "ASIA", "XTSE": "CA", "XASX": "ASIA", "CRYPTO": "CRIPTO",
}

# Spread base COMPLETO (fracción del precio) por clase; por operación se cobra la mitad.
SPREAD_BASE: dict[str, float] = {
    "ETF": 0.0005, "ETF_UCITS": 0.0015, "EQUITY": 0.0005, "MUTUALFUND": 0.0, "CRYPTO": 0.0010,
    "FUTURE": 0.0002, "INDEX": 0.0, "CURRENCY": 0.0001,
}


@dataclass(frozen=True)
class Comision:
    """Comisión de una operación: ``fija_eur + pct·importe`` (+ ``por_accion_usd·acciones``).

    Con mínimo (``minimo_eur``) y tope (``maximo_pct`` del importe, si no es None).
    """

    fija_eur: float = 0.0
    pct: float = 0.0
    por_accion_usd: float = 0.0
    minimo_eur: float = 0.0
    maximo_pct: float | None = None

    def importe_usd(self, importe_usd: float, eur_usd: float, precio: float | None = None) -> float:
        """Comisión en USD de una orden de ``importe_usd`` (``eur_usd`` = USD por 1 EUR)."""
        if importe_usd <= 0:
            return 0.0
        c = self.fija_eur * eur_usd + self.pct * importe_usd
        if self.por_accion_usd:
            acciones = importe_usd / precio if precio and precio > 0 else 0.0
            c += self.por_accion_usd * max(np.ceil(acciones), 1.0)
        c = max(c, self.minimo_eur * eur_usd)
        if self.maximo_pct is not None:
            c = min(c, self.maximo_pct * importe_usd)
        return float(c)


@dataclass(frozen=True)
class Broker:
    nombre: str
    comisiones: dict[str, Comision]
    fx_pct: float
    url: str
    verificado: bool = False
    notas: tuple[str, ...] = field(default_factory=tuple)

    def comision(self, grupo: str) -> Comision:
        return self.comisiones.get(grupo, self.comisiones["OTROS"])


DEGIRO = Broker(
    nombre="degiro", fx_pct=0.0025, url="https://www.degiro.es/tarifas", verificado=False,
    comisiones={
        "US": Comision(fija_eur=2.00), "FONDO_US": Comision(fija_eur=2.00), "ES": Comision(fija_eur=3.90),
        "EU": Comision(fija_eur=3.90), "UK": Comision(fija_eur=3.90), "CH": Comision(fija_eur=3.90),
        "CA": Comision(fija_eur=2.00), "ASIA": Comision(fija_eur=5.00, pct=0.0005),
        "ETF_SELECCION": Comision(fija_eur=1.00), "CRIPTO": Comision(), "OTROS": Comision(fija_eur=5.00, pct=0.0005),
    },
    notas=("tarifa pública de cuenta Básica (España), incluidos los gastos de tramitación; pendiente de verificar",
           "AutoFX 0,25 % si la divisa no es EUR; gasto anual de conectividad por bolsa no incluido",
           "ETF de EE. UU. no comprables por un minorista de la UE (PRIIPs/KID): solo como referencia"))

IBKR = Broker(
    nombre="ibkr", fx_pct=0.0003, url="https://www.interactivebrokers.ie/es/pricing/commissions-stocks.php",
    verificado=False,
    comisiones={
        "US": Comision(por_accion_usd=0.005, minimo_eur=0.90, maximo_pct=0.01),
        "FONDO_US": Comision(fija_eur=0.0), "ES": Comision(pct=0.0005, minimo_eur=3.0),
        "EU": Comision(pct=0.0005, minimo_eur=3.0), "UK": Comision(pct=0.0005, minimo_eur=3.0),
        "CH": Comision(pct=0.0005, minimo_eur=9.0), "CA": Comision(por_accion_usd=0.01, minimo_eur=0.9),
        "ASIA": Comision(pct=0.0008, minimo_eur=5.0), "ETF_SELECCION": Comision(pct=0.0005, minimo_eur=3.0),
        "CRIPTO": Comision(pct=0.0018, minimo_eur=1.6), "OTROS": Comision(pct=0.0008, minimo_eur=5.0),
    },
    notas=("tarifa fija IBKR Pro (0,005 USD/acción, mín. 1 USD, máx. 1 %; Europa 0,05 % mín. 3 €); pendiente de verificar",
           "FX: conversión manual (≈0,002 % + horquilla); se toma 0,03 % como supuesto"))

BROKERS: dict[str, Broker] = {"degiro": DEGIRO, "ibkr": IBKR}


# --- Estimadores de spread ----------------------------------------------------------------------
def _ohlc_sin_splits(ohlc: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """OHLC ordenado por sesión y máscara de pares (t-1, t) válidos (sin split en t, precios > 0)."""
    o = ohlc.sort_values("sesion").reset_index(drop=True)
    sin_split = (o["split"].fillna(1.0) == 1.0).to_numpy() if "split" in o else np.ones(len(o), bool)
    positivos = (o[["high", "low", "close"]] > 0).all(axis=1).to_numpy()
    ok = np.r_[False, sin_split[1:] & positivos[1:] & positivos[:-1]]
    return o, ok


def spread_abdi_ranaldo(ohlc: pd.DataFrame) -> pd.Series:
    """``s²_t = 4·(c_{t-1} − η_{t-1})·(c_{t-1} − η_t)`` (logaritmos; η = punto medio de máx. y mín.).

    Usa solo las sesiones ``t-1`` y ``t`` (estimable al cierre de ``t``). Índice: ``sesion``.
    Devuelve ``s²`` (puede ser negativo: se promedia antes de la raíz).
    """
    o, ok = _ohlc_sin_splits(ohlc)
    c = np.log(o["close"].to_numpy(float))
    eta = (np.log(o["high"].to_numpy(float)) + np.log(o["low"].to_numpy(float))) / 2.0
    s2 = np.full(len(o), np.nan)
    s2[1:] = 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    s2[~ok] = np.nan
    return pd.Series(s2, index=pd.DatetimeIndex(o["sesion"]), name="s2_abdi_ranaldo")


def spread_corwin_schultz(ohlc: pd.DataFrame) -> pd.Series:
    """Spread de Corwin y Schultz (2012) de los pares (t-1, t), negativos a 0. Índice: ``sesion``."""
    o, ok = _ohlc_sin_splits(ohlc)
    h, lo = np.log(o["high"].to_numpy(float)), np.log(o["low"].to_numpy(float))
    beta = np.full(len(o), np.nan)
    gamma = np.full(len(o), np.nan)
    beta[1:] = (h[1:] - lo[1:]) ** 2 + (h[:-1] - lo[:-1]) ** 2
    gamma[1:] = (np.maximum(h[1:], h[:-1]) - np.minimum(lo[1:], lo[:-1])) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    with np.errstate(invalid="ignore"):
        alfa = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alfa) - 1.0) / (1.0 + np.exp(alfa))
    s = np.where(ok, np.maximum(s, 0.0), np.nan)
    return pd.Series(s, index=pd.DatetimeIndex(o["sesion"]), name="spread_corwin_schultz")


def chl_21d(ohlc: pd.DataFrame, ventana: int = VENTANA_CHL) -> pd.Series:
    """Spread CHL (Abdi-Ranaldo) de ventana móvil: ``sqrt(max(media(s²), 0))``. Índice: ``sesion``."""
    s2 = spread_abdi_ranaldo(ohlc)
    media = s2.rolling(ventana, min_periods=max(ventana // 2, 2)).mean()
    return np.sqrt(media.clip(lower=0.0)).rename("chl_21d")


def multiplicador_crisis(ohlc: pd.DataFrame, *, ventana: int = VENTANA_CHL,
                         min_calma: int = MIN_SESIONES_CALMA,
                         limites: tuple[float, float] = LIMITES_MULTIPLICADOR) -> pd.DataFrame:
    """Multiplicador de crisis ``m_t`` de un activo (normalmente SPY) a partir de su OHLC as-traded.

    Devuelve un DataFrame indexado por ``sesion`` con ``fecha_publicacion``, ``chl_21d``,
    ``cs_21d`` (Corwin-Schultz medio, contraste), ``vol_21d``, ``calma``, ``mediana_calma`` y
    ``m``. Todo es expansivo o de ventana pasada: ``m`` en la sesión ``t`` solo usa sesiones
    ``<= t``. Antes de ``min_calma`` sesiones en calma, ``m = 1``.
    """
    o = ohlc.sort_values("sesion").reset_index(drop=True)
    chl = chl_21d(o, ventana)
    cs = spread_corwin_schultz(o).rolling(ventana, min_periods=max(ventana // 2, 2)).mean()
    _, ok = _ohlc_sin_splits(o)
    r = np.log(o["close"].to_numpy(float))
    ret = np.full(len(o), np.nan)
    ret[1:] = np.diff(r)
    ret[~ok] = np.nan
    vol = pd.Series(ret, index=chl.index).rolling(ventana, min_periods=max(ventana // 2, 2)).std() * np.sqrt(252)
    umbral = vol.expanding(min_periods=min_calma).median()
    calma = (vol <= umbral) & chl.notna()
    mediana = chl.where(calma).expanding(min_periods=1).median()
    n_calma = calma.cumsum()
    mediana = mediana.where(n_calma >= min_calma)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (chl / mediana).clip(*limites)
    # Sin referencia de calma (o con mediana 0: CHL nulo en calma) no hay multiplicador: 1.
    m = m.where(mediana.gt(0) & chl.notna(), 1.0).fillna(1.0)
    out = pd.DataFrame({"chl_21d": chl, "cs_21d": cs, "vol_21d": vol, "calma": calma,
                        "mediana_calma": mediana, "m": m})
    if "fecha_publicacion" in o:
        out.insert(0, "fecha_publicacion", pd.DatetimeIndex(o["fecha_publicacion"]))
    return out


def valor_en(serie: pd.Series | float | None, fecha: pd.Timestamp, defecto: float) -> float:
    """Último valor de ``serie`` (índice = fecha de publicación) con índice ``<= fecha``."""
    if serie is None:
        return defecto
    if not isinstance(serie, pd.Series):
        return float(serie)
    s = serie.dropna()
    i = s.index.searchsorted(fecha, side="right") - 1
    return float(s.iloc[i]) if i >= 0 else defecto


# --- Modelo de costes ---------------------------------------------------------------------------
@dataclass
class ModeloCostes:
    """Coste de una operación en USD: comisión + FX + medio spread base × ``m_t``.

    ``info``: ``{ticker: {'mercado', 'divisa', 'clase'}}`` (de ``yahoo_nucleo.csv``).
    ``multiplicador``: serie ``m_t`` indexada por fecha de publicación (``None`` → 1).
    ``eur_usd``: USD por EUR visible (serie por fecha de publicación o número), para pasar las
    comisiones fijas en EUR. ``escala_spread`` multiplica el spread base (sensibilidad).
    """

    broker: Broker = DEGIRO
    info: dict[str, dict[str, str]] = field(default_factory=dict)
    multiplicador: pd.Series | float | None = None
    eur_usd: pd.Series | float = 1.10
    escala_spread: float = 1.0
    divisa_inversor: str = "EUR"

    def desglose(self, ticker: str, importe_usd: float, fecha: pd.Timestamp,
                 precio: float | None = None) -> dict[str, float]:
        """Comisión, FX, spread y total (USD) de una orden de ``importe_usd`` en ``fecha``."""
        meta = self.info.get(ticker, {})
        grupo = GRUPO_MERCADO.get(meta.get("mercado", ""), "OTROS")
        clase = meta.get("clase", "ETF")
        divisa = meta.get("divisa", "USD")
        fx_eur = valor_en(self.eur_usd, fecha, 1.10)
        importe = abs(float(importe_usd))
        comision = self.broker.comision(grupo).importe_usd(importe, fx_eur, precio)
        fx = self.broker.fx_pct * importe if divisa != self.divisa_inversor else 0.0
        m = valor_en(self.multiplicador, fecha, 1.0)
        spread = 0.5 * SPREAD_BASE.get(clase, SPREAD_BASE["ETF"]) * self.escala_spread * m * importe
        return {"comision": comision, "fx": fx, "spread": spread, "total": comision + fx + spread, "m": m}

    def __call__(self, ticker: str, importe_usd: float, fecha: pd.Timestamp, precio: float | None = None) -> float:
        return self.desglose(ticker, importe_usd, fecha, precio)["total"]


def tabla_parametros() -> pd.DataFrame:
    """Parámetros del modelo (sin precios) para ``results/.../tablas/costes_modelo.csv``."""
    filas = []
    for b in BROKERS.values():
        for grupo, c in b.comisiones.items():
            filas.append({"tipo": "comision", "broker": b.nombre, "clave": grupo, "fija_eur": c.fija_eur,
                          "pct": c.pct, "por_accion_usd": c.por_accion_usd, "minimo_eur": c.minimo_eur,
                          "maximo_pct": c.maximo_pct, "valor": None, "verificado": b.verificado, "fuente": b.url,
                          "nota": "; ".join(b.notas)})
        filas.append({"tipo": "fx", "broker": b.nombre, "clave": "divisa != EUR", "valor": b.fx_pct,
                      "verificado": b.verificado, "fuente": b.url, "nota": "fracción del importe"})
    for clase, s in SPREAD_BASE.items():
        filas.append({"tipo": "spread_base", "broker": "todos", "clave": clase, "valor": s, "verificado": False,
                      "fuente": "supuesto del bloque 8", "nota": "spread completo; se cobra la mitad por operación"})
    filas += [
        {"tipo": "multiplicador", "broker": "todos", "clave": "ventana_chl", "valor": VENTANA_CHL, "verificado": True,
         "fuente": "Abdi y Ranaldo (2017), RFS 30(12)", "nota": "media de s² en 21 sesiones y raíz"},
        {"tipo": "multiplicador", "broker": "todos", "clave": "limites", "valor": f"{LIMITES_MULTIPLICADOR}",
         "verificado": True, "fuente": "decisión cerrada del Tema 1", "nota": "m = clip(CHL/mediana en calma, 1, 10)"},
        {"tipo": "multiplicador", "broker": "todos", "clave": "min_sesiones_calma", "valor": MIN_SESIONES_CALMA,
         "verificado": True, "fuente": "supuesto del bloque 8", "nota": "calma = vol 21d <= su mediana expansiva"},
        {"tipo": "patrimonio", "broker": "degiro", "clave": "capital_inicial_eur", "valor": 10_000.0,
         "verificado": True, "fuente": "decisión cerrada del Tema 1", "nota": "IBKR como sensibilidad"},
    ]
    for g, mic in sorted((g, m) for m, g in GRUPO_MERCADO.items()):
        filas.append({"tipo": "grupo_mercado", "broker": "todos", "clave": mic, "valor": g, "verificado": True,
                      "fuente": "config/mercados.yaml", "nota": "grupo de comisión"})
    return pd.DataFrame(filas)


__all__ = [
    "BROKERS", "Broker", "Comision", "DEGIRO", "GRUPO_MERCADO", "IBKR", "LIMITES_MULTIPLICADOR", "ModeloCostes",
    "SPREAD_BASE", "chl_21d", "multiplicador_crisis", "spread_abdi_ranaldo", "spread_corwin_schultz",
    "tabla_parametros", "valor_en",
]
