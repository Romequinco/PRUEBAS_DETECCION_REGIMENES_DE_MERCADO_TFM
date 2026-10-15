"""A5 · Mercados de predicción: Polymarket (Gamma, CLOB, Data API), Kalshi y PredictIt.

Uso (tool ``get_polymarket(evento, fecha)``, agentes Risk y Macro): probabilidad implícita de eventos de
cola (recesión, decisiones de la Fed, aranceles, elecciones) y **saltos** de precio (≥ 15 pp en un día,
la «divergencia» del plan) para cruzarlos con la primera noticia.

TimeGate (medido en la sonda):

- El precio es el hecho y se publica en tiempo real: ``precio_a_fecha(token, día)`` = último punto de
  ``prices-history`` con marca ≤ 16:30 de Nueva York de ese día (cierre de la sesión de EE. UU.).
- ``prices-history`` con ``interval=max`` solo admite ``fidelity`` ≥ 720 (un punto cada 12 h); con
  ``startTs/endTs`` la ventana máxima es de ~14 días (más da 400). Se usa 12 h para todo.
- Los mercados de la época AMM (2020-2022, ``enableOrderBook`` ≠ True) no tienen historial en el CLOB.
- El resultado de un mercado se conoce en ``closedTime`` (resolución UMA), no en ``endDate``; Kalshi lo
  publica en ``settlement_ts``.
- Kalshi: los mercados liquidados antes del corte (``/historical/cutoff``) están en ``/historical/markets``
  (desde 2021-07, con los tickers antiguos FED-, CPI-, GDP- dentro de las series KX*); los posteriores en
  ``/markets?status=settled``.
- PredictIt: solo instantánea (sin histórico por API); máx. 1 petición por minuto.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from ingesta.comun.http import get, robots_permite  # noqa: F401  (APIs documentadas: no se rastrea)
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_utc

MODULO = "prediccion"
F_PM, F_KA, F_PI = "polymarket", "kalshi", "predictit"

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
PREDICTIT = "https://www.predictit.org/api/marketdata/all/"

NY = ZoneInfo("America/New_York")
CORTE_NY = time(16, 30)
FIDELIDAD = 720                     # minutos; mínimo que acepta interval=max
UMBRAL_SALTO = 0.15
VOLUMEN_MIN_CALIB = 100_000         # USD (Polymarket)
CONTRATOS_MIN_KALSHI = 1_000
HORQUILLA_MAX = 0.25                # bid/ask más abierto que esto no da un precio fiable

LIC_PM = ("Polymarket Terms of Use: datos públicos de API sin clave; uso no comercial/académico; "
          "no se opera (las restricciones geográficas afectan a operar, no a leer)")
LIC_KA = "Kalshi API / Market Data Terms: datos de mercado públicos sin autenticación; uso no comercial"
LIC_PI = "PredictIt API: uso no comercial con atribución a PredictIt; solo instantánea"

#: Mercados emblemáticos de economía (alias -> slug de Gamma). Metadatos públicos.
EMBLEMATICOS: dict[str, str] = {
    "recesion_eeuu_2025": "us-recession-in-2025",
    "fed_recorte25_sep2025": "fed-decreases-interest-rates-by-25-bps-after-september-2025-meeting",
    "recesion_eeuu_2026": "us-recession-by-end-of-2026",
    "recesion_eeuu_2024": "us-recession-in-2024-1",
    "fed_sin_recortes_2026": "will-no-fed-rate-cuts-happen-in-2026",
}

#: Series macro de Kalshi (las antiguas FED-, CPI-, GDP-... están dentro de las KX*).
SERIES_KALSHI: tuple[str, ...] = ("KXFED", "KXFEDDECISION", "KXCPIYOY", "KXCPI", "KXGDP", "KXRECSSNBER", "KXINXY")

_CATEGORIAS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fed", re.compile(r"\bfed\b|fomc|interest rates?|powell|rate (cut|hike)|bps", re.I)),
    ("economia", re.compile(r"recession|inflation|\bcpi\b|\bgdp\b|unemploy|payroll|\bjobs\b|tariff|s&p|nasdaq|"
                            r"\bdow\b|\boil\b|\bgold\b|treasury|\byields?\b|debt ceiling|shutdown|default|\bbanks?\b",
                            re.I)),
    ("cripto", re.compile(r"bitcoin|\bbtc\b|ethereum|\$?eth\b|solana|crypto|doge|airdrop|\bnft|coinbase|binance|"
                          r"tether|usdt|usdc|token|fdv|memecoin|microstrategy", re.I)),
    ("deportes", re.compile(r"\bvs\.?\b|\bv\.\s|super ?bowl|\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|ufc|premier league|"
                            r"champions league|world cup|grand prix|\bf1\b|stanley cup|world series|wimbledon", re.I)),
    ("politica", re.compile(r"elect|president|senate|house|congress|trump|biden|harris|governor|mayor|minister|"
                            r"parliament|vote|nominee|primary|impeach|indict|supreme court|putin|zelensky|ukraine|"
                            r"russia|israel|netanyahu|iran|china|\bwar\b|ceasefire|pope|inaugurat|democrat|republican|"
                            r"\bstate\b", re.I)),
)


# --------------------------------------------------------------------------- cliente con caché y registro

class Cliente:
    """GET JSON con caché en ``<corpus>/noticias_alt/crudo/prediccion`` y registro opcional en un ``Resultado``.

    Las tools lo usan sin ``Resultado`` (``Cliente()``); la sonda, con él.
    """

    def __init__(self, res: base.Resultado | None = None, usar_cache: bool = True):
        self.res = res
        self.usar_cache = usar_cache

    def _registrar(self, *a: Any, **kw: Any) -> None:
        if self.res is not None:
            self.res.registrar(*a, **kw)

    def json(self, fuente: str, item: str, url: str, params: dict[str, Any] | None = None, *, dia: str = "",
             por_segundo: float = 4.0, cache: bool = True, timeout: int = 60) -> Any:
        params = params or {}
        clave = hashlib.sha1((url + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()[:20]
        ruta = base.corpus("crudo", MODULO, fuente) / f"{clave}.json"
        if cache and self.usar_cache and ruta.exists() and ruta.stat().st_size > 0:
            crudo = ruta.read_bytes()
            self._registrar(fuente, item, url, True, dia=dia, bytes=len(crudo), segundos=0.0)
            return json.loads(crudo)
        t0 = base.cronometro()
        try:
            r = get(url, params=params or None, por_segundo=por_segundo, timeout=timeout)
            r.raise_for_status()
            crudo = r.content
            datos = json.loads(crudo)
        except Exception as e:  # noqa: BLE001
            self._registrar(fuente, item, url, False, dia=dia, segundos=base.segundos(t0),
                            error=f"{type(e).__name__}: {e}")
            raise
        self._registrar(fuente, item, url, True, dia=dia, bytes=len(crudo), segundos=base.segundos(t0))
        if cache:
            base.escribir_atomico(ruta, crudo)
        return datos


_CLI_DEFECTO = Cliente()


def _cli(cli: Cliente | None) -> Cliente:
    return cli or _CLI_DEFECTO


# --------------------------------------------------------------------------- fechas y TimeGate

def corte_ny(dia: date | str) -> datetime:
    """Instante TimeGate de un día: 16:30 de Nueva York (con su horario de verano), en UTC."""
    d = dia.date() if isinstance(dia, datetime) else (dia if isinstance(dia, date) else pd.Timestamp(dia).date())
    return datetime.combine(d, CORTE_NY, tzinfo=NY).astimezone(UTC)


def _marcas(historial: list[dict[str, Any]]) -> list[float]:
    return [float(h["t"]) for h in historial]


def precio_en(historial: list[dict[str, Any]], momento: datetime) -> tuple[float | None, float | None]:
    """(precio, marca) del último punto con ``t ≤ momento`` (historial ordenado por ``t``)."""
    if not historial:
        return None, None
    i = bisect.bisect_right(_marcas(historial), momento.timestamp())
    if i == 0:
        return None, None
    h = historial[i - 1]
    return float(h["p"]), float(h["t"])


def precio_a_fecha(token_o_historial: str | list[dict[str, Any]], fecha: date | datetime | str,
                   cli: Cliente | None = None) -> float | None:
    """TimeGate de ``get_polymarket``: precio disponible el día ``fecha`` (corte 16:30 NY).

    ``fecha`` puede ser un día (se aplica el corte) o un instante con zona (se usa tal cual).
    """
    hist = (historial_precio(token_o_historial, cli) if isinstance(token_o_historial, str)
            else token_o_historial)
    if isinstance(fecha, datetime):
        momento = fecha if fecha.tzinfo else fecha.replace(tzinfo=UTC)
    else:
        momento = corte_ny(fecha)
    return precio_en(sorted(hist, key=lambda h: h["t"]), momento)[0]


def serie_diaria(historial: list[dict[str, Any]], hasta: date | None = None) -> pd.DataFrame:
    """Un valor por día de NY (precio al corte 16:30 NY): columnas ``fecha, valor, t``."""
    hist = sorted(historial, key=lambda h: h["t"])
    if not hist:
        return pd.DataFrame(columns=["fecha", "valor", "t"])
    marcas = _marcas(hist)
    d0 = datetime.fromtimestamp(marcas[0], tz=UTC).astimezone(NY).date()
    d1 = datetime.fromtimestamp(marcas[-1], tz=UTC).astimezone(NY).date()
    if hasta is not None:
        d1 = min(d1, hasta)
    filas = []
    d = d0
    while d <= d1:
        i = bisect.bisect_right(marcas, corte_ny(d).timestamp())
        if i:
            filas.append(dict(fecha=d, valor=float(hist[i - 1]["p"]), t=marcas[i - 1]))
        d += timedelta(days=1)
    return pd.DataFrame(filas, columns=["fecha", "valor", "t"])


def saltos(serie: pd.DataFrame | list[dict[str, Any]], umbral: float = UMBRAL_SALTO,
           nombre: str = "") -> pd.DataFrame:
    """Saltos ≥ ``umbral`` (en probabilidad, 0,15 = 15 pp) entre días consecutivos de una serie diaria.

    Acepta la salida de ``serie_diaria`` (``fecha, valor``) o una serie larga (``fecha_referencia, valor``).
    """
    df = pd.DataFrame(serie)
    cols = ["serie", "fecha", "de", "a", "cambio_pp"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    fcol = "fecha" if "fecha" in df.columns else "fecha_referencia"
    df = df.sort_values(fcol)
    prev = df["valor"].shift(1)
    m = (df["valor"] - prev).abs() >= umbral - 1e-12
    out = pd.DataFrame(dict(serie=nombre or (df["serie"].iloc[0] if "serie" in df.columns else ""),
                            fecha=pd.to_datetime(df.loc[m, fcol]).dt.date, de=prev[m].round(4),
                            a=df.loc[m, "valor"].round(4)))
    out["cambio_pp"] = ((out["a"] - out["de"]) * 100).round(1)
    return out[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- Polymarket: listar / descargar / parsear

def gamma(ruta: str, cli: Cliente | None = None, *, item: str = "", dia: str = "", cache: bool = True,
          **params: Any) -> Any:
    return _cli(cli).json(F_PM, item or ruta, f"{GAMMA}/{ruta}", params, dia=dia, cache=cache)


def paginar(ruta: str, cli: Cliente | None = None, *, max_paginas: int = 5, limite: int = 100, item: str = "",
            dia: str = "", **params: Any) -> list[dict[str, Any]]:
    """``/events`` o ``/markets`` de Gamma paginando con ``limit/offset`` (máx. 100 por página)."""
    salida: list[dict[str, Any]] = []
    for k in range(max_paginas):
        pag = gamma(ruta, cli, item=f"{item or ruta} p{k}", dia=dia, limit=limite, offset=k * limite, **params)
        if not pag:
            break
        salida += pag
        if len(pag) < limite:
            break
    return salida


def mercado_por_slug(slug: str, cli: Cliente | None = None) -> dict[str, Any]:
    return gamma(f"markets/slug/{slug}", cli, item=f"mercado {slug}")


def _lista(v: Any) -> list[Any]:
    if v is None or v == "":
        return []
    return json.loads(v) if isinstance(v, str) else list(v)


def tokens_mercado(mercado: dict[str, Any]) -> list[str]:
    return [str(t) for t in _lista(mercado.get("clobTokenIds"))]


def token_yes(mercado: dict[str, Any]) -> str | None:
    """Token CLOB de la opción 'Yes' (None si el mercado no es binario Yes/No o no tiene CLOB)."""
    out, toks = [str(o).strip().lower() for o in _lista(mercado.get("outcomes"))], tokens_mercado(mercado)
    if out != ["yes", "no"] or len(toks) != 2:
        return None
    return toks[0]


def resultado_binario(mercado: dict[str, Any]) -> int | None:
    """1 si ganó 'Yes', 0 si 'No', None si no está resuelto o se repartió (precios finales no extremos)."""
    precios = _lista(mercado.get("outcomePrices"))
    if not mercado.get("closed") or len(precios) != 2:
        return None
    try:
        p = float(precios[0])
    except (TypeError, ValueError):
        return None
    return 1 if p >= 0.99 else (0 if p <= 0.01 else None)


def fecha_cierre(mercado: dict[str, Any]) -> datetime | None:
    """Cierre efectivo: el primero de ``endDate`` (programado) y ``closedTime`` (resolución)."""
    fs = [f for f in (a_utc(mercado.get("endDate")), a_utc(mercado.get("closedTime"))) if f is not None]
    return min(fs) if fs else None


def volumen(mercado: dict[str, Any]) -> float:
    for k in ("volumeNum", "volume", "volume_fp"):
        v = mercado.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def categoria(pregunta: str) -> str:
    for nombre, rx in _CATEGORIAS:
        if rx.search(pregunta or ""):
            return nombre
    return "otros"


def historial_precio(token: str, cli: Cliente | None = None, *, fidelidad: int = FIDELIDAD,
                     inicio: datetime | None = None, fin: datetime | None = None, cache: bool = True,
                     item: str = "") -> list[dict[str, Any]]:
    """Serie ``[{t, p}]`` de un token. Sin fechas, ``interval=max`` (fidelidad ≥ 720 min).

    Con ``inicio/fin`` (ventana ≤ 14 días) admite fidelidad de 1-60 min.
    """
    params: dict[str, Any] = {"market": token, "fidelity": fidelidad}
    if inicio and fin:
        params.update(startTs=int(inicio.timestamp()), endTs=int(fin.timestamp()))
    else:
        params["interval"] = "max"
    j = _cli(cli).json(F_PM, item or f"historial {token[:10]}", f"{CLOB}/prices-history", params, cache=cache)
    return sorted(j.get("history", []) if isinstance(j, dict) else [], key=lambda h: h["t"])


def operaciones(cli: Cliente | None = None, *, mercado: str | None = None, limite: int = 20,
                cache: bool = False) -> list[dict[str, Any]]:
    """Data API ``/trades`` (marca en segundos; ``mercado`` = conditionId)."""
    params: dict[str, Any] = {"limit": limite}
    if mercado:
        params["market"] = mercado
    return _cli(cli).json(F_PM, "trades", f"{DATA}/trades", params, cache=cache) or []


# --------------------------------------------------------------------------- Kalshi

def kalshi_mercados(serie: str, cli: Cliente | None = None, *, historico: bool = True, estado: str = "settled",
                    max_paginas: int = 5, limite: int = 1000) -> list[dict[str, Any]]:
    """Mercados de una serie: ``/historical/markets`` (liquidados antes del corte) o ``/markets?status=<estado>``."""
    ruta = "historical/markets" if historico else "markets"
    salida: list[dict[str, Any]] = []
    cursor = None
    for k in range(max_paginas):
        params: dict[str, Any] = {"series_ticker": serie, "limit": limite}
        if not historico:
            params["status"] = estado
        if cursor:
            params["cursor"] = cursor
        etiqueta = ruta if historico else f"{ruta}[{estado}]"
        j = _cli(cli).json(F_KA, f"{etiqueta} {serie} p{k}", f"{KALSHI}/{ruta}", params, cache=estado != "open")
        ms = j.get("markets") or []
        for m in ms:
            m["_historico"] = historico
            m["_serie"] = serie
        salida += ms
        cursor = j.get("cursor")
        if not cursor or not ms:
            break
    return salida


def kalshi_velas(mercado: dict[str, Any], inicio: datetime, fin: datetime, cli: Cliente | None = None,
                 periodo_min: int = 1440) -> list[dict[str, Any]]:
    t = mercado["ticker"]
    url = (f"{KALSHI}/historical/markets/{t}/candlesticks" if mercado.get("_historico", True)
           else f"{KALSHI}/series/{mercado.get('_serie', '')}/markets/{t}/candlesticks")
    j = _cli(cli).json(F_KA, f"velas {t}", url, dict(start_ts=int(inicio.timestamp()), end_ts=int(fin.timestamp()),
                                                     period_interval=periodo_min))
    return sorted(j.get("candlesticks") or [], key=lambda v: v.get("end_period_ts", 0))


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v / 100 if v > 1 else v      # formato antiguo en centavos


def precio_vela(vela: dict[str, Any]) -> float | None:
    """Último precio negociado de la vela; si no hubo operaciones, el punto medio bid/ask (con horquilla ≤ 25 pp);
    si no, el previo."""
    precio = vela.get("price") or {}
    p = _num(precio.get("close"))
    if p is not None:
        return p
    bid, ask = _num((vela.get("yes_bid") or {}).get("close")), _num((vela.get("yes_ask") or {}).get("close"))
    if bid is not None and ask is not None and 0 < ask and ask - bid <= HORQUILLA_MAX:
        return round((bid + ask) / 2, 4)
    return _num(precio.get("previous"))


def velas_a_historial(velas: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Velas de Kalshi -> ``[{t, p}]`` (``t`` = fin del periodo: el precio se conoce entonces)."""
    out = []
    for v in velas:
        p = precio_vela(v)
        if p is not None and v.get("end_period_ts"):
            out.append({"t": float(v["end_period_ts"]), "p": p})
    return out


def resultado_kalshi(m: dict[str, Any]) -> int | None:
    r = str(m.get("result") or "").lower()
    return 1 if r == "yes" else (0 if r == "no" else None)


# --------------------------------------------------------------------------- PredictIt

def predictit_instantanea(cli: Cliente | None = None) -> tuple[dict[str, Any], int]:
    """Instantánea completa (sin caché: es en vivo). Devuelve (json, bytes)."""
    t0 = base.cronometro()
    c = _cli(cli)
    try:
        r = get(PREDICTIT, por_segundo=1 / 60, timeout=60)
        r.raise_for_status()
        crudo = r.content
        j = json.loads(crudo)
    except Exception as e:  # noqa: BLE001
        c._registrar(F_PI, "instantanea", PREDICTIT, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
        raise
    c._registrar(F_PI, "instantanea", PREDICTIT, True, bytes=len(crudo), segundos=base.segundos(t0))
    ruta = base.corpus("crudo", MODULO, F_PI) / f"instantanea_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    base.escribir_atomico(ruta, crudo)
    return j, len(crudo)


def resumen_predictit(j: dict[str, Any], ahora: datetime | None = None) -> dict[str, Any]:
    ms = j.get("markets") or []
    ahora = ahora or datetime.now(UTC)
    marcas = [a_utc_ny(m.get("timeStamp")) for m in ms]
    marcas = [t for t in marcas if t is not None]
    ultima = max(marcas) if marcas else None
    return dict(ts=ahora.isoformat(timespec="seconds"), ts_fuente=ultima.isoformat(timespec="seconds") if ultima else None,
                n_mercados=len(ms), n_contratos=sum(len(m.get("contracts") or []) for m in ms),
                lag_h=base.lag_horas(ultima, ahora) if ultima else None,
                mercado_ejemplo=(ms[0].get("shortName") or ms[0].get("name")) if ms else None)


def a_utc_ny(valor: Any) -> datetime | None:
    """Marca sin zona de PredictIt (hora de Nueva York) -> UTC."""
    if not valor:
        return None
    try:
        ts = pd.Timestamp(str(valor))
    except (ValueError, TypeError):
        return None
    ts = ts.tz_localize(NY) if ts.tzinfo is None else ts
    return ts.tz_convert("UTC").floor("us").to_pydatetime()


# --------------------------------------------------------------------------- muestreo

def repartir(candidatos: list[dict[str, Any]], n: int, grupo: Callable[[dict[str, Any]], Any],
             peso: Callable[[dict[str, Any]], float]) -> list[dict[str, Any]]:
    """Muestra variada: turno rotatorio entre grupos, cada grupo ordenado por ``peso`` descendente."""
    grupos: dict[Any, list[dict[str, Any]]] = {}
    for c in candidatos:
        grupos.setdefault(grupo(c), []).append(c)
    colas = [sorted(v, key=peso, reverse=True) for _, v in sorted(grupos.items(), key=lambda kv: str(kv[0]))]
    out: list[dict[str, Any]] = []
    while len(out) < n and any(colas):
        for cola in colas:
            if cola and len(out) < n:
                out.append(cola.pop(0))
    return out


def fila_calibracion(plataforma: str, mercado_id: str, pregunta: str, cat: str, cierre: datetime,
                     vol: float, historial: list[dict[str, Any]], resultado: int | None) -> dict[str, Any]:
    """Precios 1, 7 y 30 días antes del cierre (último punto ≤ cierre − k días) y resultado."""
    fila: dict[str, Any] = dict(plataforma=plataforma, mercado_id=mercado_id, pregunta=pregunta[:200], categoria=cat,
                                fecha_cierre=cierre.date().isoformat(), anio=cierre.year, volumen_usd=round(vol, 2))
    for k in (1, 7, 30):
        fila[f"p_{k}d"] = precio_en(historial, cierre - timedelta(days=k))[0]
    fila["resultado"] = resultado
    fila["n_puntos"] = len(historial)
    fila["inicio_historial"] = (datetime.fromtimestamp(historial[0]["t"], tz=UTC).date().isoformat()
                                if historial else None)
    return fila


def resumen_calibracion(df: pd.DataFrame, tramos: int = 5) -> pd.DataFrame:
    """Tabla de fiabilidad por plataforma y horizonte: tramos de probabilidad, frecuencia observada y Brier."""
    filas = []
    if df.empty:
        return pd.DataFrame(columns=["plataforma", "horizonte", "tramo", "n", "p_media", "frec_yes", "brier", "n_total"])
    bordes = [i / tramos for i in range(tramos + 1)]
    for plat, dp in df.groupby("plataforma"):
        for h in ("p_1d", "p_7d", "p_30d"):
            d = dp[[h, "resultado"]].dropna()
            if d.empty:
                continue
            brier = round(float(((d[h] - d["resultado"]) ** 2).mean()), 4)
            corte = pd.cut(d[h], bordes, include_lowest=True)
            for tramo, g in d.groupby(corte, observed=True):
                filas.append(dict(plataforma=plat, horizonte=h[2:], tramo=str(tramo), n=len(g),
                                  p_media=round(float(g[h].mean()), 4), frec_yes=round(float(g["resultado"].mean()), 4),
                                  brier=brier, n_total=len(d)))
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- sonda: Polymarket

def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def _pm_liquidez(cfg: base.ConfigSonda, cli: Cliente, acum: dict[str, Any]) -> None:
    hoy = date.today()
    paginas = cfg.extra.get("paginas_liquidez", 1 if cfg.modo_rapido else 10)
    filas = []
    for y in range(2020, hoy.year + 1):
        try:
            ev = paginar("events", cli, max_paginas=paginas, item=f"eventos {y}", start_date_min=f"{y}-01-01",
                         start_date_max=f"{y}-12-31", order="volume", ascending="false", include_markets="false")
        except Exception as e:  # noqa: BLE001
            cfg.log(f"  polymarket liquidez {y}: {_err(e)}")
            continue
        vols = sorted((volumen(e) for e in ev), reverse=True)
        filas.append(dict(anio=y, n_eventos=len(vols), volumen_usd=round(sum(vols), 2),
                          n_mayor_1m=sum(v >= 1e6 for v in vols),
                          mediana_volumen_usd=round(float(pd.Series(vols).median()), 2) if vols else None,
                          techo_muestra=paginas * 100, truncado=len(vols) >= paginas * 100, criterio="startDate"))
    acum["liquidez"] = pd.DataFrame(filas)


def _pm_dias(cfg: base.ConfigSonda, cli: Cliente, acum: dict[str, Any]) -> None:
    filas = []
    lim = 20 if cfg.modo_rapido else 50
    for d in cfg.dias:
        f = d.fecha.isoformat()
        try:
            ev = gamma("events", cli, item=f"activos {d.clave}", dia=f, limit=lim, start_date_max=f, end_date_min=f,
                       order="volume", ascending="false", include_markets="false")
        except Exception as e:  # noqa: BLE001
            cfg.log(f"  polymarket día {d.clave}: {_err(e)}")
            continue
        ev = [e for e in ev if (a_utc(e.get("startDate")) or datetime.max.replace(tzinfo=UTC)).date() <= d.fecha]
        vols = [volumen(e) for e in ev]
        top = max(ev, key=volumen) if ev else {}
        filas.append(dict(plataforma=F_PM, dia=d.clave, fecha=f, n_activos=len(ev), techo=lim,
                          volumen_usd=round(sum(vols), 2), max_volumen_usd=round(max(vols), 2) if vols else 0.0,
                          titular_top=(top.get("title") or "")[:120], cobertura="ok" if ev else "sin cobertura"))
    acum["dias"] = filas


def _pm_calibracion(cfg: base.ConfigSonda, cli: Cliente, acum: dict[str, Any]) -> None:
    hoy = date.today()
    n_obj = int(cfg.extra.get("n_calibracion", 40 if cfg.modo_rapido else 150))
    paginas = 1 if cfg.modo_rapido else 2
    vistos: dict[str, dict[str, Any]] = {}
    por_anio: dict[int, dict[str, int]] = {}
    for y in range(2021, hoy.year + 1):
        try:
            ms = paginar("markets", cli, max_paginas=paginas, item=f"resueltos {y}", closed="true",
                         volume_num_min=VOLUMEN_MIN_CALIB, end_date_min=f"{y}-01-01", end_date_max=f"{y}-12-31",
                         order="volumeNum", ascending="false")
        except Exception as e:  # noqa: BLE001
            cfg.log(f"  polymarket resueltos {y}: {_err(e)}")
            continue
        cnt = por_anio.setdefault(y, dict(listados=0, binarios=0, sin_clob=0, candidatos=0))
        for m in ms:
            cnt["listados"] += 1
            if token_yes(m) is None:
                continue
            cnt["binarios"] += 1
            if m.get("enableOrderBook") is not True:
                cnt["sin_clob"] += 1
                continue
            if resultado_binario(m) is None or volumen(m) < VOLUMEN_MIN_CALIB or fecha_cierre(m) is None:
                continue
            if str(m.get("id")) not in vistos:
                cnt["candidatos"] += 1
                vistos[str(m.get("id"))] = m
    acum["calib_por_anio"] = por_anio
    elegidos = repartir(list(vistos.values()), n_obj,
                        grupo=lambda m: (fecha_cierre(m).year, categoria(m.get("question", ""))), peso=volumen)
    filas, resol = [], []
    for m in elegidos:
        cierre = fecha_cierre(m)
        try:
            hist = historial_precio(token_yes(m), cli, item=f"historial {m.get('slug', '')[:50]}")
        except Exception as e:  # noqa: BLE001
            cfg.log(f"  polymarket historial {m.get('slug')}: {_err(e)}")
            continue
        filas.append(fila_calibracion(F_PM, str(m.get("id")), m.get("question", ""), categoria(m.get("question", "")),
                                      cierre, volumen(m), hist, resultado_binario(m)))
        fin, cerrado = a_utc(m.get("endDate")), a_utc(m.get("closedTime"))
        if fin and cerrado:
            resol.append(dict(item=m.get("question", "")[:120], fin=fin, cerrado=cerrado,
                              lag_h=base.lag_horas(fin, cerrado)))
    acum["calibracion"] = filas
    acum["resolucion"] = resol


def _pm_emblematicos(cfg: base.ConfigSonda, cli: Cliente, res: base.Resultado, acum: dict[str, Any]) -> None:
    filas, tablas_saltos, tg = [], [], []
    alias = list(EMBLEMATICOS.items())[: (3 if cfg.modo_rapido else len(EMBLEMATICOS))]
    for nombre, slug in alias:
        try:
            m = mercado_por_slug(slug, cli)
            tok = token_yes(m)
            if not tok:
                raise ValueError("mercado no binario o sin token CLOB")
            # Los mercados abiertos cambian: sin caché para tener el último punto.
            hist = historial_precio(tok, cli, cache=bool(m.get("closed")), item=f"historial {nombre}")
            d = serie_diaria(hist)
            if d.empty:
                raise ValueError("historial vacío")
            serie = f"polymarket_{nombre}"
            larga = pd.DataFrame(dict(
                serie=serie, fecha_referencia=pd.to_datetime(d["fecha"]), fecha_publicacion=pd.to_datetime(d["fecha"]),
                valor=d["valor"], unidad="probabilidad (precio Yes, 0-1)", fuente=F_PM, vintage=None,
                disponible_utc=[datetime.fromtimestamp(t, tz=UTC).isoformat() for t in d["t"]],
                metodo_publicacion="tiempo real; último punto CLOB ≤ 16:30 NY del día (resolución 12 h)",
                nota=(m.get("question") or "")[:200]))
            res.series.append(base.guardar_serie(larga, serie, rapido=cfg.modo_rapido).stem)
            s = saltos(d, UMBRAL_SALTO, nombre=serie)
            tablas_saltos.append(s)
            filas.append(dict(serie=serie, slug=slug, pregunta=m.get("question"), categoria=categoria(m.get("question", "")),
                              inicio=d["fecha"].iloc[0].isoformat(), fin=d["fecha"].iloc[-1].isoformat(), n_dias=len(d),
                              volumen_usd=round(volumen(m), 2), cerrado=bool(m.get("closed")),
                              resultado=resultado_binario(m), p_min=round(float(d["valor"].min()), 4),
                              p_max=round(float(d["valor"].max()), 4), p_ultimo=round(float(d["valor"].iloc[-1]), 4),
                              n_saltos_15pp=len(s)))
            t_ult = datetime.fromtimestamp(float(d["t"].iloc[-1]), tz=UTC)
            tg.append(dict(fuente=F_PM, item=f"{nombre} precio {d['fecha'].iloc[-1]}", fecha_hecho=t_ult.isoformat(),
                           fecha_publicacion=t_ult.isoformat(), campo_fecha="prices-history.t (≤ 16:30 NY)", lag_h=0.0))
        except Exception as e:  # noqa: BLE001
            res.registrar(F_PM, f"serie {nombre}", f"{GAMMA}/markets/slug/{slug}", False, error=_err(e))
    acum["emblematicos"] = filas
    acum["saltos"] = pd.concat(tablas_saltos, ignore_index=True) if tablas_saltos else saltos([])
    acum["timegate"] = tg


def _pm_extras(cfg: base.ConfigSonda, cli: Cliente, acum: dict[str, Any]) -> None:
    viejos = gamma("markets", cli, item="primer mercado", limit=1, order="createdAt", ascending="true", closed="true")
    acum["primer_mercado"] = (viejos[0].get("createdAt") if viejos else None)
    tr = operaciones(cli, limite=20)
    ahora = datetime.now(UTC)
    marcas = [datetime.fromtimestamp(float(t["timestamp"]), tz=UTC) for t in tr if t.get("timestamp")]
    acum["trades_campos"] = sorted(tr[0].keys()) if tr else []
    acum["frescura_h"] = [base.lag_horas(t, ahora) for t in marcas]
    if marcas:
        t = max(marcas)
        acum.setdefault("timegate", []).append(dict(
            fuente=F_PM, item="última operación (Data API)", fecha_hecho=t.isoformat(),
            fecha_publicacion=ahora.isoformat(timespec="seconds"), campo_fecha="trades.timestamp (s) vs hora de consulta",
            lag_h=base.lag_horas(t, ahora)))


def _sondear_polymarket(cfg: base.ConfigSonda, res: base.Resultado, cli: Cliente, acum: dict[str, Any]) -> None:
    pasos = (("liquidez", lambda: _pm_liquidez(cfg, cli, acum)), ("días testigo", lambda: _pm_dias(cfg, cli, acum)),
             ("calibración", lambda: _pm_calibracion(cfg, cli, acum)),
             ("emblemáticos", lambda: _pm_emblematicos(cfg, cli, res, acum)), ("extras", lambda: _pm_extras(cfg, cli, acum)))
    for nombre, f in pasos:
        t0 = base.cronometro()
        try:
            f()
        except Exception as e:  # noqa: BLE001
            res.registrar(F_PM, f"paso {nombre}", "", False, error=_err(e))
        cfg.log(f"  polymarket · {nombre}: {base.segundos(t0)} s")


# --------------------------------------------------------------------------- sonda: Kalshi

def _kalshi_tabla(ms: list[dict[str, Any]]) -> pd.DataFrame:
    filas = []
    for m in ms:
        c = a_utc(m.get("close_time"))
        if c is not None:
            filas.append(dict(anio=c.year, serie=m.get("_serie"), vol=volumen(m),
                              abierto=a_utc(m.get("open_time")) or c, cierre=c))
    df = pd.DataFrame(filas, columns=["anio", "serie", "vol", "abierto", "cierre"])
    for c in ("abierto", "cierre"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def _sondear_kalshi(cfg: base.ConfigSonda, res: base.Resultado, cli: Cliente, acum: dict[str, Any]) -> None:
    series = SERIES_KALSHI[:3] if cfg.modo_rapido else SERIES_KALSHI
    paginas = 1 if cfg.modo_rapido else 5
    try:
        acum["corte"] = cli.json(F_KA, "historical/cutoff", f"{KALSHI}/historical/cutoff", cache=False)
    except Exception:  # noqa: BLE001
        acum["corte"] = {}
    todos: dict[str, dict[str, Any]] = {}
    abiertos: list[dict[str, Any]] = []
    for s in series:
        for historico, est in ((True, "settled"), (False, "settled"), (False, "open")):
            try:
                ms_ = kalshi_mercados(s, cli, historico=historico, estado=est, max_paginas=1 if est == "open" else paginas)
            except Exception as e:  # noqa: BLE001
                cfg.log(f"  kalshi {s} ({'hist' if historico else est}): {_err(e)}")
                continue
            if est == "open":
                abiertos += ms_
            else:
                for m in ms_:
                    todos.setdefault(m["ticker"], m)
    ms = list(todos.values())
    acum["n_mercados"], acum["n_abiertos"] = len(ms), len(abiertos)
    # liquidez por año de cierre (volumen en contratos: 1 contrato paga 1 USD)
    df = _kalshi_tabla(ms)
    if not df.empty:
        acum["liquidez"] = (df.groupby("anio").agg(n_mercados=("vol", "size"), volumen_contratos=("vol", "sum"),
                                                   n_mayor_100k=("vol", lambda v: int((v >= 1e5).sum())),
                                                   n_series=("serie", "nunique")).reset_index())
        acum["liquidez"]["volumen_contratos"] = acum["liquidez"]["volumen_contratos"].round(0)
        acum["inicio"] = df["abierto"].min().date().isoformat()
        acum["fin"] = df["cierre"].max().date().isoformat()
    # actividad en los días testigo: liquidados + abiertos (para «hoy»)
    act = _kalshi_tabla(ms + abiertos)
    filas_dias = []
    for d in cfg.dias:
        vivos = act[(act["abierto"].dt.date <= d.fecha) & (act["cierre"].dt.date >= d.fecha)] if not act.empty else act
        filas_dias.append(dict(plataforma=F_KA, dia=d.clave, fecha=d.fecha.isoformat(), n_activos=len(vivos), techo=None,
                               volumen_usd=round(float(vivos["vol"].sum()), 0) if len(vivos) else 0.0,
                               max_volumen_usd=round(float(vivos["vol"].max()), 0) if len(vivos) else 0.0,
                               titular_top="series macro " + ",".join(series),
                               cobertura="ok" if len(vivos) else "sin cobertura"))
    acum["dias"] = filas_dias
    # retraso de liquidación: cierre -> settlement_ts
    resol = []
    for m in ms:
        c, s_ = a_utc(m.get("close_time")), a_utc(m.get("settlement_ts"))
        if c and s_:
            resol.append(dict(item=m["ticker"], fin=c, cerrado=s_, lag_h=base.lag_horas(c, s_)))
    acum["resolucion"] = resol
    # calibración: precio 1/7/30 días antes del cierre desde velas diarias
    n_obj = int(cfg.extra.get("n_calibracion_kalshi", 10 if cfg.modo_rapido else 40))
    cand = [m for m in ms if resultado_kalshi(m) is not None and volumen(m) >= CONTRATOS_MIN_KALSHI
            and a_utc(m.get("close_time"))]
    elegidos = repartir(cand, n_obj, grupo=lambda m: (a_utc(m["close_time"]).year, m["_serie"]), peso=volumen)
    calib = []
    for m in elegidos:
        cierre = a_utc(m["close_time"])
        try:
            velas = kalshi_velas(m, cierre - timedelta(days=32), cierre, cli)
        except Exception as e:  # noqa: BLE001
            cfg.log(f"  kalshi velas {m['ticker']}: {_err(e)}")
            continue
        calib.append(fila_calibracion(F_KA, m["ticker"], m.get("title") or m["ticker"], "macro:" + m["_serie"], cierre,
                                      volumen(m), velas_a_historial(velas), resultado_kalshi(m)))
    acum["calibracion"] = calib
    # frescura: últimas operaciones públicas
    try:
        tr = (cli.json(F_KA, "trades", f"{KALSHI}/markets/trades", {"limit": 20}, cache=False).get("trades") or [])
        ahora = datetime.now(UTC)
        marcas = [a_utc(t.get("created_time")) for t in tr]
        marcas = [t for t in marcas if t]
        acum["frescura_h"] = [max(0.0, base.lag_horas(t, ahora) or 0.0) for t in marcas]
        if marcas:
            acum["timegate_trade"] = dict(fuente=F_KA, item="última operación", fecha_hecho=max(marcas).isoformat(),
                                          fecha_publicacion=ahora.isoformat(timespec="seconds"),
                                          campo_fecha="trades.created_time vs hora de consulta",
                                          lag_h=max(0.0, base.lag_horas(max(marcas), ahora) or 0.0))
    except Exception as e:  # noqa: BLE001
        cfg.log(f"  kalshi trades: {_err(e)}")


# --------------------------------------------------------------------------- sonda: PredictIt

def _sondear_predictit(cfg: base.ConfigSonda, res: base.Resultado, cli: Cliente, acum: dict[str, Any]) -> None:
    j, nbytes = predictit_instantanea(cli)
    r = resumen_predictit(j)
    r["bytes"] = nbytes
    acum["instantanea"] = r


# --------------------------------------------------------------------------- tablas, timegate, cobertura y fichas

def _mediana(xs: Iterable[float | None]) -> float | None:
    return base.resumen_lag(xs)["mediana_h"]


def _timegate_resolucion(fuente: str, resol: list[dict[str, Any]], campo: str, n: int = 5) -> list[dict[str, Any]]:
    filas = [r for r in resol if r["lag_h"] is not None and r["lag_h"] >= 0][:n]
    return [dict(fuente=fuente, item=r["item"], fecha_hecho=r["fin"].isoformat(), fecha_publicacion=r["cerrado"].isoformat(),
                 campo_fecha=campo, lag_h=r["lag_h"]) for r in filas]


def _cerrar(cfg: base.ConfigSonda, res: base.Resultado, acum: dict[str, dict[str, Any]]) -> None:
    pm, ka, pi = acum[F_PM], acum[F_KA], acum[F_PI]
    hoy = date.today().isoformat()

    # ---- tablas
    if isinstance(pm.get("liquidez"), pd.DataFrame) and not pm["liquidez"].empty:
        res.tablas["polymarket_liquidez_anio"] = pm["liquidez"]
    if isinstance(ka.get("liquidez"), pd.DataFrame) and not ka["liquidez"].empty:
        res.tablas["kalshi_liquidez_anio"] = ka["liquidez"]
    calib = pd.DataFrame(pm.get("calibracion", []) + ka.get("calibracion", []))
    if not calib.empty:
        res.tablas["prediccion_calibracion"] = calib
        res.tablas["prediccion_calibracion_resumen"] = resumen_calibracion(calib)
    dias = pd.DataFrame(pm.get("dias", []) + ka.get("dias", []))
    if not dias.empty:
        res.tablas["prediccion_dias_testigo"] = dias
    if pm.get("emblematicos"):
        res.tablas["polymarket_emblematicos"] = pd.DataFrame(pm["emblematicos"])
    if "saltos" in pm:
        res.tablas["polymarket_saltos"] = pm["saltos"]
    if pi.get("instantanea"):
        res.tablas["predictit_instantanea"] = pd.DataFrame([pi["instantanea"]])

    # ---- timegate
    res.timegate += pm.get("timegate", [])
    res.timegate += _timegate_resolucion(F_PM, pm.get("resolucion", []), "gamma endDate -> closedTime (resolución UMA)")
    res.timegate += _timegate_resolucion(F_KA, ka.get("resolucion", []), "close_time -> settlement_ts (liquidación)")
    if ka.get("timegate_trade"):
        res.timegate.append(ka["timegate_trade"])
    if pi.get("instantanea") and pi["instantanea"].get("ts_fuente"):
        r = pi["instantanea"]
        res.timegate.append(dict(fuente=F_PI, item="instantánea", fecha_hecho=r["ts_fuente"], fecha_publicacion=r["ts"],
                                 campo_fecha="markets[].timeStamp (hora NY) vs hora de consulta", lag_h=r["lag_h"]))

    # ---- mediciones calculadas
    cal_pm = calib[calib["plataforma"] == F_PM] if not calib.empty else pd.DataFrame()
    cal_ka = calib[calib["plataforma"] == F_KA] if not calib.empty else pd.DataFrame()

    def _med_cal(d: pd.DataFrame) -> dict[str, Any]:
        if d.empty:
            return dict(n=0)
        con = d.dropna(subset=["p_1d"])
        out: dict[str, Any] = dict(n=len(d), n_con_precio=len(con), inicio_historial_min=d["inicio_historial"].dropna().min()
                                   if d["inicio_historial"].notna().any() else None,
                                   por_categoria=d["categoria"].value_counts().to_dict(),
                                   por_anio=d["anio"].value_counts().sort_index().to_dict())
        for h in ("p_1d", "p_7d", "p_30d"):
            x = d[[h, "resultado"]].dropna()
            out[f"brier_{h[2:]}"] = round(float(((x[h] - x["resultado"]) ** 2).mean()), 4) if len(x) else None
        return out

    lag_res_pm = base.resumen_lag(r["lag_h"] for r in pm.get("resolucion", []) if r["lag_h"] is not None and r["lag_h"] >= 0)
    lag_res_ka = base.resumen_lag(r["lag_h"] for r in ka.get("resolucion", []) if r["lag_h"] is not None and r["lag_h"] >= 0)
    fres_pm, fres_ka = _mediana(pm.get("frescura_h", [])), _mediana(ka.get("frescura_h", []))

    def _bytes(f: str) -> int:
        return sum(r["bytes"] for r in res.registro if r["fuente"] == f)

    def _kb_mediana(f: str, prefijo: str) -> float | None:
        kb = [r["bytes"] / 1024 for r in res.registro if r["fuente"] == f and r["item"].startswith(prefijo) and r["bytes"]]
        return round(float(pd.Series(kb).median()), 1) if kb else None

    def _seg(f: str) -> float:
        return round(sum(r["segundos"] for r in res.registro if r["fuente"] == f), 1)

    # ---- Polymarket
    liq = pm.get("liquidez", pd.DataFrame())
    liq_txt = ("; ".join(f"{int(r.anio)}: {r.volumen_usd / 1e6:,.0f} M USD ({int(r.n_mayor_1m)} ev. >1M)"
                         for r in liq.itertuples()) if isinstance(liq, pd.DataFrame) and not liq.empty else "sin medir")
    med_pm = _med_cal(cal_pm)
    ini_pm = (pm.get("primer_mercado") or "")[:10] or None
    ini_precios = med_pm.get("inicio_historial_min")
    kb_med = _kb_mediana(F_PM, "historial")
    n_listados = sum(v.get("listados", 0) for v in pm.get("calib_por_anio", {}).values())
    emb = pm.get("emblematicos", [])
    fin_pm = max([e["fin"] for e in emb], default=None) or hoy
    est_pm = base.estado(res.registro, F_PM)
    n_saltos = len(pm.get("saltos", [])) if "saltos" in pm else 0
    res.cobertura.append(dict(
        fuente=F_PM, familia="prediccion", tipo="serie", inicio=ini_pm, fin=fin_pm, hora=True, idiomas="en",
        lag_mediana_h=fres_pm, destino="serie", backtest=bool(ini_precios and ini_precios < "2024-01-01"), estado=est_pm,
        nota=(f"precios CLOB desde {ini_precios} (AMM 2020-22 sin historial); liquidez real desde 2024; "
              f"resolución {lag_res_pm['mediana_h']} h tras endDate (mediana); {len(emb)} series, {n_saltos} saltos ≥15 pp")))
    res.fichas.append(base.ficha(
        fuente=F_PM, nombre="Polymarket (Gamma + CLOB prices-history + Data API)", agentes=["risk", "macro"], rag="serie",
        url="https://docs.polymarket.com", acceso="api", licencia_y_terminos=LIC_PM,
        cobertura_inicio=ini_pm, cobertura_fin=fin_pm,
        frecuencia="continua (operaciones al segundo; historial 12 h con interval=max, 1 min en ventanas ≤ 14 días)",
        retraso_publicacion=(f"tiempo real (última operación hace {fres_pm} h, mediana); resultado {lag_res_pm['mediana_h']} h "
                             "después del fin programado (mediana, closedTime)"),
        formato="json",
        volumen_estimado=(f"historial a 12 h ≈ {kb_med} KB/mercado (mediana); ≥ {n_listados} mercados resueltos > 100 k USD "
                          f"listados (tope de la muestra) -> ≈ {round(n_listados * (kb_med or 0) / 1024, 1)} MB; página de "
                          f"100 eventos sin mercados ≈ {_kb_mediana(F_PM, 'eventos')} KB"),
        limites=("sin clave; /events y /markets máx. 100 por página (limit/offset); prices-history: interval=max solo con "
                 "fidelity ≥ 720, startTs/endTs ≤ 14 días (400 si no); límites por IP según docs"),
        estado_sonda=est_pm,
        mediciones=dict(liquidez_por_anio=liq.to_dict("records") if isinstance(liq, pd.DataFrame) else [],
                        calibracion=med_pm, calib_por_anio=pm.get("calib_por_anio", {}), primer_mercado=pm.get("primer_mercado"),
                        lag_resolucion=lag_res_pm, frescura_h=fres_pm, series=[e["serie"] for e in emb], n_saltos_15pp=n_saltos,
                        trades_campos=pm.get("trades_campos", []), bytes=_bytes(F_PM), segundos=_seg(F_PM),
                        historial_kb_mediana=kb_med),
        notas=[f"Liquidez por año (top por volumen, año de startDate): {liq_txt}.",
               "Mercados AMM (2020-2022, enableOrderBook ≠ True) sin historial en el CLOB: calibración y series desde 2023.",
               "TimeGate: precio a una fecha = último punto ≤ 16:30 NY; el resultado se conoce en closedTime, no en endDate.",
               "Divergencia = salto ≥ 15 pp en un día (o compras grandes en Data API) antes de la primera noticia -> alerta Risk."]))

    # ---- Kalshi
    est_ka = base.estado(res.registro, F_KA)
    med_ka = _med_cal(cal_ka)
    liq_ka = ka.get("liquidez", pd.DataFrame())
    series_k = list(SERIES_KALSHI[:3] if cfg.modo_rapido else SERIES_KALSHI)
    kb_velas = _kb_mediana(F_KA, "velas")
    ini_ka_prec = med_ka.get("inicio_historial_min")
    res.cobertura.append(dict(
        fuente=F_KA, familia="prediccion", tipo="serie", inicio=ka.get("inicio"), fin=ka.get("fin") or hoy, hora=True,
        idiomas="en", lag_mediana_h=fres_ka, destino="serie",
        backtest=bool(ini_ka_prec and ini_ka_prec < "2024-01-01"), estado=est_ka,
        nota=(f"series macro {','.join(series_k)}: {ka.get('n_mercados', 0)} mercados liquidados; velas diarias desde "
              f"{ini_ka_prec}; liquidación {lag_res_ka['mediana_h']} h tras el cierre (mediana); volumen en contratos")))
    res.fichas.append(base.ficha(
        fuente=F_KA, nombre="Kalshi (mercado de eventos regulado por la CFTC)", agentes=["risk", "macro"], rag="serie",
        url="https://docs.kalshi.com", acceso="api", licencia_y_terminos=LIC_KA,
        cobertura_inicio=ka.get("inicio"), cobertura_fin=ka.get("fin") or hoy,
        frecuencia="continua (velas de 1 min, 1 h y 1 día)",
        retraso_publicacion=(f"tiempo real (última operación hace {fres_ka} h); liquidación {lag_res_ka['mediana_h']} h "
                             "después de close_time (mediana, settlement_ts)"),
        formato="json",
        volumen_estimado=(f"{ka.get('n_mercados', 0)} mercados en {len(series_k)} series macro; velas diarias de 32 días ≈ "
                          f"{kb_velas} KB por mercado (mediana) -> ≈ "
                          f"{round(ka.get('n_mercados', 0) * (kb_velas or 0) * 12 / 1024, 1)} MB con un año de velas por mercado"),
        limites=("lectura sin clave; /historical/markets hasta el corte "
                 f"{(ka.get('corte') or {}).get('market_settled_ts')}, después /markets?status=settled; cursor, 1000 por página"),
        estado_sonda=est_ka,
        mediciones=dict(liquidez_por_anio=liq_ka.to_dict("records") if isinstance(liq_ka, pd.DataFrame) else [],
                        n_liquidados=ka.get("n_mercados", 0), n_abiertos=ka.get("n_abiertos", 0),
                        calibracion=med_ka, lag_liquidacion=lag_res_ka, frescura_h=fres_ka, corte=ka.get("corte"),
                        series=series_k, bytes=_bytes(F_KA), segundos=_seg(F_KA)),
        notas=["El histórico 2021-2024 está en /historical/markets bajo las series KX* (tickers antiguos FED-, CPI-, GDP-...).",
               "volume_fp = contratos (cada uno paga 1 USD): en prediccion_calibracion, volumen_usd de Kalshi = contratos.",
               "Precio de vela: último negociado; sin operaciones, punto medio bid/ask; si no, el previo."]))

    # ---- PredictIt
    est_pi = base.estado(res.registro, F_PI)
    inst = pi.get("instantanea") or {}
    res.cobertura.append(dict(
        fuente=F_PI, familia="prediccion", tipo="instantanea", inicio=hoy if inst else None, fin=hoy if inst else None,
        hora=True, idiomas="en", lag_mediana_h=inst.get("lag_h"), destino="vivo", backtest=False, estado=est_pi,
        nota=(f"{inst.get('n_mercados')} mercados / {inst.get('n_contratos')} contratos; solo instantánea"
              if inst else "sin instantánea")))
    res.fichas.append(base.ficha(
        fuente=F_PI, nombre="PredictIt (mercados políticos de EE. UU.)", agentes=["risk"], rag="ninguno", url=PREDICTIT,
        acceso="api", licencia_y_terminos=LIC_PI, cobertura_inicio=hoy if inst else None, cobertura_fin=hoy if inst else None,
        frecuencia="instantánea (actualiza ~1 min)", retraso_publicacion=f"tiempo real (timeStamp hace {inst.get('lag_h')} h)",
        formato="json", volumen_estimado=f"≈ {round(inst.get('bytes', 0) / 1e6, 2)} MB por instantánea",
        limites="1 petición/min; sin histórico por API (descarga manual de 90 días por contrato)", estado_sonda=est_pi,
        mediciones=dict(**{k: v for k, v in inst.items()}, segundos=_seg(F_PI)),
        notas=["Solo en vivo: guardar instantáneas propias desde hoy si se quiere serie."]))


def sondear(cfg: base.ConfigSonda | None = None) -> base.Resultado:
    """Sonda A5: liquidez, calibración, series emblemáticas y saltos (Polymarket), Kalshi y PredictIt."""
    cfg = cfg or base.ConfigSonda()
    res = base.Resultado(MODULO)
    cli = Cliente(res, cfg.usar_cache)
    acum: dict[str, dict[str, Any]] = {F_PM: {}, F_KA: {}, F_PI: {}}
    for fuente, f in ((F_PM, _sondear_polymarket), (F_KA, _sondear_kalshi), (F_PI, _sondear_predictit)):
        t0 = base.cronometro()
        try:
            f(cfg, res, cli, acum[fuente])
        except Exception as e:  # noqa: BLE001
            res.registrar(fuente, "sonda", "", False, error=_err(e))
        cfg.log(f"{fuente}: {base.segundos(t0)} s")
    try:
        _cerrar(cfg, res, acum)
    except Exception as e:  # noqa: BLE001
        res.registrar(MODULO, "cierre", "", False, error=_err(e))
    return res
