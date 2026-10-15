"""Constituyentes de índices (bloque 5): S&P 500 point-in-time y fotos actuales de otros índices.

S&P 500 (``sp500``): repositorio ``fja05680/sp500`` de GitHub (licencia MIT), fichero
``S&P 500 Historical Components & Changes (Updated).csv``: una fila por fecha de cambio con la
lista COMPLETA de miembros vigente desde esa fecha (``date,tickers``; desde 1996-01-02). Se
convierte en tramos de pertenencia: una fila por (ticker, alta, baja) con ``baja`` = primera fecha
del fichero en la que ya no está (excluida; null si sigue). ``fecha_publicacion`` = fecha del
cambio (fecha efectiva: S&P lo anuncia unos días antes, luego es conservadora: un alta o una baja
solo se ven cuando ya han ocurrido, nunca antes de su anuncio; no hay fuga, solo se pierde la
información del anuncio). ``fecha_publicacion_baja`` = ``baja``. Notas:

- los miembros de 1996-01-02 estaban antes de esa fecha (inicio del fichero, alta truncada);
- un ticker con varios tramos es una reentrada o un TICKER REUTILIZADO por otra empresa (el
  fichero no lleva identificador permanente: CIK/PERMNO pendiente de WRDS);
- los tickers son los de cada momento según el autor (cambios de ticker = baja + alta), salvo los
  28 tramos de quiebra con sufijo Q/QQ (``NOTA_TICKER_POSTERIOR``): el fichero sin actualizar del
  mismo repositorio (``S&P 500 Historical Components & Changes.csv``) los lleva igual, con el mes
  de exclusión de cotización (``AAMRQ-201312``), así que fja05680 no da el ticker de la época.

Fotos actuales (``indices``): tablas de composición de Wikipedia (inglés), si ``robots_permite``:
DAX, CAC 40, IBEX 35, FTSE MIB, AEX, FTSE 100, SMI, Nikkei 225, Ibovespa, NIFTY 50, Hang Seng y
KOSPI 200. Se elige la tabla con columna de ticker/código cuyo número de filas más se acerca al
número de miembros del índice. Cada fila: ``indice, ticker, nombre, foto_actual=True,
fecha_foto`` (día de descarga, NY) y ``fecha_publicacion`` = esa fecha: SOLO VALE PARA HOY. Usarla
en el pasado es sesgo de supervivencia (``AVISO_SUPERVIVENCIA``).

Salidas en ``<corpus>/mercados_series/constituyentes/``: ``sp500.parquet`` y ``<indice>.parquet``;
crudos en ``crudo/constituyentes/`` con su ``.meta.json``. CLI:
``python -m ingesta.mercados_series.fuentes.constituyentes [--sp500] [--indices] [--fichas]``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.comun.rutas import corpus_dir

TEMA = "mercados_series"
URL_SP500 = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
             + quote("S&P 500 Historical Components & Changes (Updated).csv"))
REPO_SP500 = "https://github.com/fja05680/sp500"
COLUMNAS_SP500 = ("indice", "ticker", "alta", "baja", "fecha_publicacion", "fecha_publicacion_baja", "nota")
COLUMNAS_FOTO = ("indice", "ticker", "nombre", "foto_actual", "fecha_foto", "fecha_publicacion", "url", "nota")
NOTA_TRUNCADA = "en el índice antes de {inicio} (inicio del fichero): alta truncada"
NOTA_TICKER_POSTERIOR = ("identificador posterior al tramo: ticker con sufijo Q/QQ de quiebra (adelanta un evento "
                         "futuro y no casa con los precios as-traded de su época; mapear con CIK/PERMNO)")
PATRON_TICKER_POSTERIOR = re.compile(r"^[A-Z]{3,}Q{1,2}$")
TICKERS_Q_DE_EPOCA = frozenset({"MERQ", "NDAQ"})  # la Q es parte del ticker original (Mercury Interactive, Nasdaq)
NOTA_REUTILIZADO = "ticker con {n} tramos: reentrada o ticker reutilizado por otra empresa (sin CIK/PERMNO)"
AVISO_SUPERVIVENCIA = ("foto ACTUAL de la composición: no vale para fechas pasadas (sesgo de supervivencia); "
                       "solo el S&P 500 es point-in-time")

# índice → (página de Wikipedia, nº de miembros esperado)
INDICES: dict[str, tuple[str, int]] = {
    "dax": ("DAX", 40),
    "cac40": ("CAC_40", 40),
    "ibex35": ("IBEX_35", 35),
    "ftse_mib": ("FTSE_MIB", 40),
    "aex": ("AEX_index", 25),
    "ftse100": ("FTSE_100_Index", 100),
    "smi": ("Swiss_Market_Index", 20),
    "nikkei225": ("Nikkei_225", 225),
    "ibovespa": ("Ibovespa", 85),
    "nifty50": ("NIFTY_50", 50),
    "hang_seng": ("Hang_Seng_Index", 85),
    "kospi200": ("KOSPI_200", 200),
}
COLUMNAS_TICKER = ("ticker", "symbol", "code", "epic", "stock code", "ticker symbol", "sehk", "tse code")
COLUMNAS_NOMBRE = ("company", "name", "constituent", "company name", "stock name")


def destino_dir(raiz: Path | None = None) -> Path:
    p = (raiz or corpus_dir(TEMA)) / "constituyentes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _archivar(contenido: bytes, nombre: str, url: str, estado: int, raiz: Path | None) -> Path:
    carpeta = (raiz or corpus_dir(TEMA)) / "crudo" / "constituyentes"
    carpeta.mkdir(parents=True, exist_ok=True)
    dia = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d")
    ruta = carpeta / f"{dia}_{nombre}"
    ruta.write_bytes(contenido)
    meta = {"url": url, "estado": estado, "bytes": len(contenido), "sha256": hashlib.sha256(contenido).hexdigest(),
            "descargado_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    ruta.with_name(ruta.name + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ruta


# --- S&P 500 point-in-time ---------------------------------------------------------------------------
def leer_sp500_historico(texto: str | bytes) -> pd.DataFrame:
    """CSV ``date,tickers`` → DataFrame (``fecha``, ``miembros``: frozenset), ordenado y sin repetidos."""
    if isinstance(texto, bytes):
        texto = texto.decode("utf-8")
    crudo = pd.read_csv(io.StringIO(texto), dtype=str)
    if list(crudo.columns[:2]) != ["date", "tickers"]:
        raise ValueError(f"cabecera inesperada {list(crudo.columns)}: se esperaba date,tickers")
    df = pd.DataFrame({
        "fecha": pd.to_datetime(crudo["date"], format="%Y-%m-%d"),
        "miembros": [frozenset(t.strip() for t in str(s).split(",") if t.strip()) for s in crudo["tickers"]],
    })
    return df.sort_values("fecha").drop_duplicates("fecha", keep="last").reset_index(drop=True)


def es_ticker_posterior(ticker: str, baja: Any) -> bool:
    """Ticker de un tramo cerrado con sufijo de quiebra (``ENRNQ``, ``MTLQQ``): identificador posterior.

    Solo tramos con baja y tickers de 4+ letras (``NDAQ`` sigue en el índice; ``CPQ`` es de la época).
    """
    t = str(ticker)
    return bool(pd.notna(baja) and t not in TICKERS_Q_DE_EPOCA and PATRON_TICKER_POSTERIOR.match(t))


def tramos_sp500(historico: pd.DataFrame) -> pd.DataFrame:
    """Instantáneas → tramos (ticker, alta, baja) con ``fecha_publicacion`` y notas."""
    abiertos: dict[str, pd.Timestamp] = {}
    filas: list[dict[str, Any]] = []
    previos: frozenset[str] = frozenset()
    for fecha, miembros in zip(historico["fecha"], historico["miembros"], strict=True):
        for t in miembros - previos:
            abiertos[t] = fecha
        for t in previos - miembros:
            filas.append({"ticker": t, "alta": abiertos.pop(t), "baja": fecha})
        previos = miembros
    filas += [{"ticker": t, "alta": a, "baja": pd.NaT} for t, a in abiertos.items()]
    df = pd.DataFrame(filas, columns=["ticker", "alta", "baja"])
    df["baja"] = pd.to_datetime(df["baja"])
    inicio = historico["fecha"].min()
    n_tramos = df.groupby("ticker")["alta"].transform("size")
    notas = []
    for t, alta, baja, n in zip(df["ticker"], df["alta"], df["baja"], n_tramos, strict=True):
        partes = []
        if alta == inicio:
            partes.append(NOTA_TRUNCADA.format(inicio=inicio.date()))
        if n > 1:
            partes.append(NOTA_REUTILIZADO.format(n=n))
        if es_ticker_posterior(t, baja):
            partes.append(NOTA_TICKER_POSTERIOR)
        notas.append("; ".join(partes) or None)
    df["nota"] = notas
    df["indice"] = "sp500"
    df["fecha_publicacion"] = df["alta"]
    df["fecha_publicacion_baja"] = df["baja"]
    return df[list(COLUMNAS_SP500)].sort_values(["alta", "ticker"]).reset_index(drop=True)


def miembros_en(tramos: pd.DataFrame, fecha: Any) -> list[str]:
    """Miembros vigentes y PUBLICADOS en ``fecha`` (alta <= fecha < baja; fecha_publicacion <= fecha)."""
    f = pd.Timestamp(fecha)
    vivo = (tramos["alta"] <= f) & (tramos["fecha_publicacion"] <= f) & (tramos["baja"].isna() | (tramos["baja"] > f))
    return sorted(tramos.loc[vivo, "ticker"])


def descargar_sp500(raiz: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descarga, archiva y convierte; escribe ``constituyentes/sp500.parquet``."""
    from ingesta.comun.http import get, robots_permite

    if not robots_permite(URL_SP500):
        raise RuntimeError("robots.txt no permite descargar el CSV del S&P 500")
    r = get(URL_SP500, timeout=120)
    crudo = _archivar(r.content, "sp500_historico.csv", URL_SP500, r.status_code, raiz)
    hist = leer_sp500_historico(r.content)
    df = tramos_sp500(hist)
    df.to_parquet(destino_dir(raiz) / "sp500.parquet", index=False)
    tamanos = hist["miembros"].map(len)
    med = {"bytes": len(r.content), "crudo": crudo.name, "n_instantaneas": int(len(hist)),
           "n_tramos": int(len(df)), "n_tickers": int(df["ticker"].nunique()),
           "tickers_varios_tramos": int((df.groupby("ticker").size() > 1).sum()),
           "tramos_ticker_posterior": int(df["nota"].fillna("").str.contains("identificador posterior").sum()),
           "miembros_min": int(tamanos.min()), "miembros_max": int(tamanos.max()),
           "miembros_ultimo": int(tamanos.iloc[-1]),
           "inicio": hist["fecha"].min().date().isoformat(), "fin": hist["fecha"].max().date().isoformat(),
           "dias_desde_ultimo_cambio": int((pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
                                            - hist["fecha"].max()).days)}
    return df, med


# --- Fotos actuales (Wikipedia) ----------------------------------------------------------------------
def _norm(c: Any) -> str:
    if isinstance(c, tuple):
        c = c[-1]
    return str(c).strip().lower()


def elegir_tabla(tablas: list[pd.DataFrame], esperado: int) -> pd.DataFrame | None:
    """Tabla con columna de ticker/código cuyo nº de filas más se acerca a ``esperado`` (±40 %)."""
    candidatas = []
    for t in tablas:
        cols = [_norm(c) for c in t.columns]
        if any(c in COLUMNAS_TICKER or c.startswith(("ticker", "symbol", "code")) for c in cols):
            if 0.6 * esperado <= len(t) <= 1.4 * esperado:
                candidatas.append((abs(len(t) - esperado), t))
    return min(candidatas, key=lambda x: x[0])[1] if candidatas else None


def limpiar_ticker(s: pd.Series) -> pd.Series:
    """Quita notas ``[1]``, espacios duros y el prefijo de bolsa (``SEHK: 5`` → ``5``)."""
    t = s.astype(str).str.replace(r"\[.*?\]", "", regex=True).str.replace("\xa0", " ", regex=False)
    return t.str.replace(r"^[A-Za-z]+:\s*", "", regex=True).str.strip()


def normalizar_foto(tabla: pd.DataFrame, indice: str, fecha_foto: pd.Timestamp, url: str) -> pd.DataFrame:
    """Tabla de Wikipedia → columnas ``COLUMNAS_FOTO``."""
    cols = {_norm(c): c for c in tabla.columns}
    c_t = next(cols[c] for c in cols if c in COLUMNAS_TICKER or c.startswith(("ticker", "symbol", "code")))
    c_n = next((cols[c] for c in cols if c in COLUMNAS_NOMBRE or c.startswith(("company", "name"))), None)
    df = pd.DataFrame({
        "indice": indice,
        "ticker": limpiar_ticker(tabla[c_t]),
        "nombre": (tabla[c_n].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
                   if c_n is not None else None),
        "foto_actual": True, "fecha_foto": fecha_foto, "fecha_publicacion": fecha_foto, "url": url,
        "nota": AVISO_SUPERVIVENCIA,
    })
    return df[df["ticker"].ne("") & df["ticker"].ne("nan")].reset_index(drop=True)[list(COLUMNAS_FOTO)]


def descargar_indice(indice: str, raiz: Path | None = None) -> dict[str, Any]:
    """Foto actual de un índice desde Wikipedia; escribe ``constituyentes/<indice>.parquet``."""
    from ingesta.comun.http import get, robots_permite

    pagina, esperado = INDICES[indice]
    url = f"https://en.wikipedia.org/wiki/{pagina}"
    med: dict[str, Any] = {"indice": indice, "url": url, "esperado": esperado, "estado": None, "n_filas": None,
                           "error": None}
    if not robots_permite(url):
        med["error"] = "robots.txt no lo permite"
        return med
    try:
        r = get(url, timeout=60)
    except Exception as e:  # noqa: BLE001 - se documenta y se sigue
        med["error"] = str(e)[:200]
        return med
    med["estado"] = r.status_code
    _archivar(r.content, f"wikipedia_{indice}.html", url, r.status_code, raiz)
    try:
        tablas = pd.read_html(io.StringIO(r.text))
    except ValueError as e:
        med["error"] = f"sin tablas: {e}"
        return med
    t = elegir_tabla(tablas, esperado)
    if t is None:
        med["error"] = f"ninguna tabla con columna de ticker y ~{esperado} filas ({len(tablas)} tablas)"
        return med
    fecha = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    df = normalizar_foto(t, indice, fecha, url)
    df.to_parquet(destino_dir(raiz) / f"{indice}.parquet", index=False)
    med["n_filas"] = int(len(df))
    med["fecha_foto"] = fecha.date().isoformat()
    return med


# --- Fichas ------------------------------------------------------------------------------------------
def ficha_sp500(med: dict[str, Any]) -> Path:
    return guardar_ficha(FichaFuente(
        fuente="sp500_constituyentes", nombre="S&P 500: composición histórica point-in-time (fja05680/sp500)",
        tema=TEMA, agentes=["equity", "portfolio"], rag="serie", url=REPO_SP500, acceso="descarga",
        autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="Código y datos del repositorio con licencia MIT; la composición es de S&P Dow Jones Indices (solo local)",
        cobertura_inicio=med.get("inicio"), cobertura_fin=med.get("fin"), frecuencia="por evento (cambios del índice)",
        retraso_publicacion=(f"fecha_publicacion = fecha efectiva del cambio (S&P lo anuncia antes: conservador, "
                             "sin fuga: solo se pierde la información del anuncio); "
                             f"medido: el último cambio del fichero es de hace {med.get('dias_desde_ultimo_cambio')} días"),
        formato="csv → parquet", volumen_estimado=f"{med.get('bytes', 0) / 1e6:.1f} MB de CSV; {med.get('n_tramos')} tramos",
        limites="raw.githubusercontent.com, 1 petición/s; actualización manual del autor (irregular)",
        estado_sonda="ok" if med.get("n_tramos") else "fallo", mediciones=med,
        notas=["Cargador: ingesta.mercados_series.fuentes.constituyentes (descargar_sp500, miembros_en)",
               "Un ticker con varios tramos puede ser reentrada o ticker reutilizado por otra empresa: sin CIK/PERMNO no se distingue (WRDS pendiente de MIAX)",
               "Los miembros a 1996-01-02 tienen el alta truncada al inicio del fichero",
               f"{med.get('tramos_ticker_posterior', 'sin medir')} tramos con ticker posterior al tramo (sufijo Q/QQ "
               "de quiebra: ENRNQ, AAMRQ...): nota en el tramo. Cotejado (2026-10-13) con el fichero sin actualizar "
               "del repositorio (S&P 500 Historical Components & Changes.csv): lleva los mismos tickers Q con el mes "
               "de exclusión (AAMRQ-201312), no el de la época; el mapeo exige CIK/PERMNO (WRDS)",
               "Fechas de anuncio: sin fuente pública histórica y reutilizable (las notas de S&P DJI no se pueden "
               "archivar en bloque); la fecha efectiva es conservadora y sin fuga: deuda cerrada",
               "Sin datos de precios de los miembros: solo pertenencia (supervivencia de la sección cruzada medida aquí)"],
    ))


def ficha_indices(meds: list[dict[str, Any]]) -> Path:
    ok = [m for m in meds if m.get("n_filas")]
    fechas = [m["fecha_foto"] for m in ok]
    return guardar_ficha(FichaFuente(
        fuente="indices_constituyentes", nombre="Composición ACTUAL de índices no estadounidenses (Wikipedia)",
        tema=TEMA, agentes=["equity"], rag="serie", url="https://en.wikipedia.org/", acceso="scraping",
        autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="Wikipedia CC BY-SA 4.0 (texto); la composición oficial es de cada proveedor de índices",
        cobertura_inicio=min(fechas) if fechas else None, cobertura_fin=max(fechas) if fechas else None,
        frecuencia="foto puntual (día de descarga)",
        retraso_publicacion="desconocido: Wikipedia se edita a mano tras cada revisión del índice; fecha_publicacion = día de la foto",
        formato="html → parquet", volumen_estimado=f"{sum(m['n_filas'] for m in ok)} filas en {len(ok)} índices",
        limites="1 petición/s; robots.txt consultado",
        estado_sonda="ok" if len(ok) == len(meds) else ("parcial" if ok else "fallo"),
        mediciones={"indices": meds},
        notas=[AVISO_SUPERVIVENCIA, "Nº de filas medido frente al nº de miembros esperado; tabla elegida por columna de ticker",
               "Sin historia point-in-time gratuita para estos índices: solo contexto, nunca para backtests de selección"],
    ))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sp500", action="store_true")
    p.add_argument("--indices", nargs="*", default=None, help="índices (todos si se omite la lista)")
    p.add_argument("--fichas", action="store_true")
    a = p.parse_args(argv)
    if a.sp500:
        _, med = descargar_sp500()
        print(json.dumps(med, indent=1))
        if a.fichas:
            print(ficha_sp500(med))
    if a.indices is not None:
        meds = [descargar_indice(i) for i in (a.indices or list(INDICES))]
        for m in meds:
            print(m)
        if a.fichas:
            print(ficha_indices(meds))


if __name__ == "__main__":
    main()
