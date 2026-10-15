"""Muestra fija de la sonda del Tema 4: días testigo, consultas y crisis de la Fase 1.

Como las 10 empresas del Tema 3, pero en el tiempo: todas las fuentes de texto se miden en los
mismos días, así el mapa de cobertura por épocas sale de comparar fuentes en las mismas fechas.
Las series de alternativos (GPR, EPU, put/call, tono...) se bajan enteras: son pequeñas y la
validez necesita toda su historia.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DiaTestigo:
    clave: str
    fecha: date
    motivo: str


DIAS_TESTIGO: tuple[DiaTestigo, ...] = (
    DiaTestigo("lehman", date(2008, 9, 15), "Quiebra de Lehman: solo fuentes antiguas (GDELT 1.0, Wikipedia, Form 4)"),
    DiaTestigo("elecciones_2016", date(2016, 11, 9), "Elecciones de EE. UU.: arranca CC-NEWS"),
    DiaTestigo("covid", date(2020, 3, 16), "COVID: peor sesión del S&P 500 desde 1987"),
    DiaTestigo("svb", date(2023, 3, 10), "Quiebra de Silicon Valley Bank"),
    DiaTestigo("aranceles", date(2025, 4, 7), "Aranceles: mercados de predicción ya líquidos"),
)


def dias_testigo(hoy: date | None = None, incluir_hoy: bool = True) -> list[DiaTestigo]:
    """Días testigo; el último es «hoy» (fuentes solo en vivo: RSS, EWS, PredictIt)."""
    dias = list(DIAS_TESTIGO)
    if incluir_hoy:
        dias.append(DiaTestigo("hoy", hoy or date.today(), "Fuentes en vivo"))
    return dias


# Consultas fijas para buscadores (GDELT DOC, Bing News): pocas y en inglés.
CONSULTAS: tuple[str, ...] = ("stock market", "recession", "federal reserve", "bank failure", "tariffs")

# Artículos de Wikipedia para las visitas (atención del público como indicador de estrés).
ARTICULOS_WIKI: tuple[str, ...] = (
    "Recession", "Bank_run", "Stock_market_crash", "Inflation", "Financial_crisis", "Tariff",
)

# Empresas para Form 4 (insiders) y para la rentabilidad anormal de la validez.
TICKERS_MUESTRA: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "XOM", "JNJ", "NVDA")


def crisis_fase1(pista: str = "A") -> dict[str, tuple[str, str]]:
    """Ventanas de crisis ``{nombre: (pico, suelo)}`` del juez de la Fase 1 (configs/benchmark_spec.yaml).

    Solo lectura: se importa ``regimenes`` dentro para no cargarlo si no hace falta.
    """
    from regimenes.evaluacion import ranking as rk

    return rk.track_crisis_windows(pista)
