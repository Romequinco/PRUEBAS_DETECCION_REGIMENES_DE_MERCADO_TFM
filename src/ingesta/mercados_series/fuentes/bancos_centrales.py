"""Bancos centrales del Tema 1 (bloque 4): tipos oficiales y bonos a 10 años de Canadá, Japón,
Alemania y Australia; el Banco de Inglaterra queda documentado como vetado por robots.txt.

- Banco de Canadá (Valet, JSON): ``V39079`` (objetivo del tipo a un día) y ``BD.CDN.10YR.DQ.YLD``
  (bono a 10 años). ``https://www.bankofcanada.ca/valet/observations/<serie>/json``.
- Banco de Japón (API de stat-search, 2026): ``getDataCode`` con ``db=FM01`` y
  ``code=STRDCLUCON`` (tipo call sin garantía a un día). El IPC de Japón NO está en el BoJ (lo
  publica el MIC): se toma de la OCDE (``oecd:CP.JPN``, con ediciones).
- Bundesbank (API REST SDMX, CSV en alemán: separador ``;`` y coma decimal): curva Svensson a 10
  años (``BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A``). ``'.'`` = sin valor.
- RBA (tablas CSV F1 y F2 diarias): ``FIRMMCRTD`` (cash rate target) y ``FCMYGBAG10D`` (bono a 10).
- Banco de Inglaterra (IADB): ``robots.txt`` prohíbe ``/boeapps/iadb`` y
  ``/boeapps/database/_iadb-FromShowColumns.asp`` (la ruta de descarga CSV): no se descarga ni se
  busca otra ruta; se documenta en la ficha ``boe`` (alternativa: series del Reino Unido en FRED/OCDE).

Publicación (conservadora, publicadores del fragmento): el dato del día D se da por público el hábil
siguiente a las 12:00 locales (``siguiente_habil``) → ``vintage`` ``unico`` (no se revisan),
``metodo_publicacion`` ``calendario``. Un hueco de la fuente (``'.'``, ``null``) se descarta (son
festivos sin dato; no se inventan filas).

CLI: ``python -m ingesta.mercados_series.fuentes.bancos_centrales [--fuentes boc boj bundesbank rba boe]
[--descargar] [--fichas]``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from ingesta.comun.http import get, robots_permite
from ingesta.mercados_series.fuentes.b4_comun import TEMA, guardar_crudo, ruta_crudo, tabla_contrato

SERIES: dict[str, dict[str, Any]] = {
    "boc:V39079": {"fuente": "boc", "publicador": "boc_valet", "codigo": "V39079",
                   "url": "https://www.bankofcanada.ca/valet/observations/V39079/json"},
    "boc:BD.CDN.10YR.DQ.YLD": {"fuente": "boc", "publicador": "boc_valet", "codigo": "BD.CDN.10YR.DQ.YLD",
                               "url": "https://www.bankofcanada.ca/valet/observations/BD.CDN.10YR.DQ.YLD/json"},
    "boj:FM01.STRDCLUCON": {"fuente": "boj", "publicador": "boj_diario", "codigo": "STRDCLUCON",
                            "url": "https://www.stat-search.boj.or.jp/api/v1/getDataCode",
                            "params": {"format": "json", "lang": "en", "db": "FM01", "code": "STRDCLUCON",
                                       "startDate": "198501"}},
    "bbk:BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A": {
        "fuente": "bundesbank", "publicador": "bundesbank_diario",
        "codigo": "BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A",
        "url": "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A",
        "params": {"format": "csv", "lang": "de"}},
    "rba:FIRMMCRTD": {"fuente": "rba", "publicador": "rba_diario", "codigo": "FIRMMCRTD",
                      "url": "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"},
    "rba:FCMYGBAG10D": {"fuente": "rba", "publicador": "rba_diario", "codigo": "FCMYGBAG10D",
                        "url": "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"},
}
URL_BOE = "https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns.asp?csv.x=yes&SeriesCodes=IUDBEDR"
UNIDAD = "% anual"

__all__ = ["SERIES", "URL_BOE", "leer_boc", "leer_boj", "leer_bundesbank", "leer_rba", "leer", "tabla_banco",
           "descargar", "estado_boe", "fichas_bancos"]


# --- Lectores (bytes crudos → serie fecha → valor, sin nulos) ------------------------------------
def _serie(fechas: Any, valores: Any) -> pd.Series:
    s = pd.Series(pd.to_numeric(pd.Series(list(valores)), errors="coerce").to_numpy(),
                  index=pd.to_datetime(pd.Series(list(fechas))).to_numpy())
    s = s[~s.index.duplicated(keep="last")].dropna().sort_index()
    s.index.name = "fecha"
    return s


def leer_boc(contenido: bytes, codigo: str) -> pd.Series:
    obs = json.loads(contenido)["observations"]
    return _serie([o["d"] for o in obs], [(o.get(codigo) or {}).get("v") for o in obs])


def leer_boj(contenido: bytes, codigo: str) -> pd.Series:
    doc = json.loads(contenido)
    for rs in doc.get("RESULTSET") or []:
        if rs.get("SERIES_CODE") == codigo:
            v = rs["VALUES"]
            fechas = pd.to_datetime(pd.Series(v["SURVEY_DATES"]).astype(str), format="%Y%m%d")
            return _serie(fechas, v["VALUES"])
    raise ValueError(f"BoJ: serie {codigo} ausente de la respuesta")


def leer_bundesbank(contenido: bytes, codigo: str | None = None) -> pd.Series:
    texto = contenido.decode("utf-8-sig")
    filas = [ln.split(";") for ln in texto.splitlines() if ln[:4].isdigit() and ln[4:5] == "-"]
    fechas = [f[0] for f in filas]
    valores = [f[1].replace(".", "").replace(",", ".") if f[1].strip() not in (".", "") else None for f in filas]
    return _serie(fechas, valores)


def leer_rba(contenido: bytes, codigo: str) -> pd.Series:
    texto = contenido.decode("utf-8-sig", errors="replace")
    lineas = texto.splitlines()
    i = next(k for k, ln in enumerate(lineas) if ln.startswith("Series ID"))
    filas = list(csv.reader(io.StringIO("\n".join(lineas[i:]))))
    col = [c.strip() for c in filas[0]].index(codigo)
    datos = [(f[0], f[col] if col < len(f) else None) for f in filas[1:] if f]
    fechas = pd.to_datetime(pd.Series([d for d, _ in datos]), format="%d-%b-%Y", errors="coerce")
    ok = fechas.notna().to_numpy()
    valores = [v for (_, v), b in zip(datos, ok) if b]
    return _serie(fechas[ok], valores)


_LECTORES = {"boc": leer_boc, "boj": leer_boj, "bundesbank": leer_bundesbank, "rba": leer_rba}


def leer(sid: str, contenido: bytes) -> pd.Series:
    info = SERIES[sid]
    return _LECTORES[info["fuente"]](contenido, info["codigo"])


def tabla_banco(sid: str, serie: pd.Series, *, publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Filas del contrato (vintage ``unico``, ``calendario``) de una serie diaria de banco central."""
    info = SERIES[sid]
    return tabla_contrato(sid, serie.index, serie.index, serie.to_numpy(), vintage="unico", unidad=UNIDAD,
                          divisa=None, fuente=info["fuente"], metodo="calendario", publicador=info["publicador"],
                          publicaciones=publicaciones)


def _nombre_crudo(sid: str) -> str:
    return sid.replace(":", "_").replace("/", "_") + (".json" if SERIES[sid]["fuente"] in ("boc", "boj") else ".csv")


def descargar(sids: Iterable[str] = tuple(SERIES)) -> dict[str, dict[str, Any]]:
    """Descarga cada serie (una petición, reintento único si falla) y guarda el crudo en el corpus."""
    informe: dict[str, dict[str, Any]] = {}
    for sid in sids:
        info = SERIES[sid]
        if not robots_permite(info["url"]):
            informe[sid] = {"estado": "vetado_robots", "url": info["url"]}
            continue
        t = time.monotonic()
        try:
            r = get(info["url"], params=info.get("params"), timeout=90, reintentos=2)
        except Exception as e:  # noqa: BLE001 — bloqueo o error: se documenta y se sigue
            informe[sid] = {"estado": "fallo", "error": str(e)[:200], "url": info["url"]}
            continue
        ruta = guardar_crudo(info["fuente"], _nombre_crudo(sid), r.content)
        informe[sid] = {"estado": "ok", "bytes": len(r.content), "segundos": round(time.monotonic() - t, 2),
                        "ruta": str(ruta)}
    return informe


def estado_boe() -> dict[str, Any]:
    """Comprueba robots.txt del Banco de Inglaterra para la descarga CSV de la IADB (no descarga)."""
    return {"url": URL_BOE, "robots_permite": robots_permite(URL_BOE),
            "robots_permite_iadb": robots_permite("https://www.bankofengland.co.uk/boeapps/iadb/")}


# --- Fichas -------------------------------------------------------------------------------------
_META = {
    "boc": ("Banco de Canadá (Valet)", "https://www.bankofcanada.ca/valet/docs", "json",
            "Términos del Banco de Canadá (https://www.bankofcanada.ca/terms/): uso libre citando la fuente"),
    "boj": ("Banco de Japón (API de stat-search)", "https://www.stat-search.boj.or.jp/info/api_manual_en.pdf", "json",
            "Banco de Japón: uso libre citando la fuente"),
    "bundesbank": ("Deutsche Bundesbank (API de la base estadística)", "https://api.statistiken.bundesbank.de/doc/index.html",
                   "csv", "Deutsche Bundesbank: uso libre citando la fuente"),
    "rba": ("Reserve Bank of Australia (tablas estadísticas)", "https://www.rba.gov.au/statistics/tables/", "csv",
            "RBA: CC BY 4.0"),
}


def fichas_bancos(informe: dict[str, dict] | None = None, *, guardar: bool = True,
                  hoy: Any = None) -> dict[str, Any]:
    """Fichas ``boc``, ``boj``, ``bundesbank``, ``rba`` (medidas con los crudos) y ``boe`` (robots)."""
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    hoy = pd.Timestamp(hoy or pd.Timestamp.now(tz="America/New_York").date())
    rutas: dict[str, Any] = {}
    for fuente, (nombre, url, formato, lic) in _META.items():
        med: dict[str, Any] = {}
        for sid, info in SERIES.items():
            if info["fuente"] != fuente:
                continue
            ruta = ruta_crudo(fuente) / _nombre_crudo(sid)
            if not ruta.exists():
                med[sid] = {"estado": (informe or {}).get(sid, {}).get("estado", "sin_descargar"),
                            "error": (informe or {}).get(sid, {}).get("error")}
                continue
            s = leer(sid, ruta.read_bytes())
            t = tabla_banco(sid, s)
            ultimo = s.index.max()
            med[sid] = {"inicio": s.index.min().date().isoformat(), "fin": ultimo.date().isoformat(),
                        "n_obs": int(len(s)), "filas_contrato": int(len(t)), "bytes": ruta.stat().st_size,
                        "dias_naturales_ultimo_dato_a_sonda": int((hoy - ultimo).days),
                        "fecha_publicacion_mediana_menos_fecha_dias": float(
                            (t["fecha_publicacion"] - t["fecha_referencia"]).dt.days.median())}
        ok = [m for m in med.values() if "n_obs" in m]
        ficha = FichaFuente(
            fuente=fuente, nombre=nombre, tema=TEMA, agentes=["macro", "risk"], rag="serie", url=url,
            acceso="api" if fuente != "rba" else "descarga", autenticacion="ninguna", coste="0 €",
            licencia_y_terminos=lic,
            cobertura_inicio=min((m["inicio"] for m in ok), default=None),
            cobertura_fin=max((m["fin"] for m in ok), default=None), frecuencia="diaria",
            retraso_publicacion=("medido en la sonda: días naturales del último dato a la fecha de la sonda: "
                                 + "; ".join(f"{s.split(':', 1)[1]} {m['dias_naturales_ultimo_dato_a_sonda']}"
                                             for s, m in med.items() if "n_obs" in m)
                                 + ". Regla conservadora: hábil siguiente a las 12:00 locales"),
            formato=formato, volumen_estimado=f"{sum(m.get('bytes', 0) for m in ok) / 1e6:.2f} MB ({len(ok)} series)",
            limites="sin clave; 1 petición/s; reintento único ante error",
            estado_sonda="ok" if ok and len(ok) == len(med) else ("parcial" if ok else "fallo"),
            mediciones={"series": med, "descarga": {k: v for k, v in (informe or {}).items() if k in med}},
            notas=["vintage 'unico': series de mercado/política monetaria sin revisiones",
                   "huecos de la fuente (festivos, '.') descartados: no se inventan filas"]
                  + (["IPC de Japón: no lo publica el BoJ (MIC/e-Stat); se usa oecd:CP.JPN"] if fuente == "boj" else []),
            fecha_sonda=hoy.date().isoformat())
        rutas[fuente] = guardar_ficha(ficha) if guardar else ficha
    boe = estado_boe()
    ficha = FichaFuente(
        fuente="boe", nombre="Banco de Inglaterra (Interactive Database, IADB)", tema=TEMA, agentes=["macro", "risk"],
        rag="serie", url="https://www.bankofengland.co.uk/boeapps/database/", acceso="descarga",
        autenticacion="ninguna", coste="0 €", licencia_y_terminos="Banco de Inglaterra: términos de uso del sitio",
        cobertura_inicio=None, cobertura_fin=None, frecuencia="diaria",
        retraso_publicacion="no medido (fuente vetada por robots.txt)", formato="csv",
        volumen_estimado="0 (no descargado)",
        limites=("robots.txt: Disallow /boeapps/iadb y /boeapps/database/_iadb-FromShowColumns.asp (la ruta CSV) "
                 "y ShowChart.asp: no se rastrea ni se buscan rutas alternativas"),
        estado_sonda="fallo", mediciones={"robots": boe},
        notas=["Vetado por robots.txt (comprobado en la sonda); no se esquiva",
               "Alternativas: Bank Rate y gilts del Reino Unido vía FRED (p. ej. IRLTLT01GBM156N, mensual, OCDE) "
               "o la OCDE; decisión del integrador"],
        fecha_sonda=hoy.date().isoformat())
    rutas["boe"] = guardar_ficha(ficha) if guardar else ficha
    return rutas


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--series", nargs="*", default=list(SERIES))
    p.add_argument("--descargar", action="store_true")
    p.add_argument("--fichas", action="store_true")
    a = p.parse_args(argv)
    informe = descargar(a.series) if a.descargar else None
    if informe:
        for k, v in informe.items():
            print(k, v.get("estado"), v.get("bytes"), v.get("error") or "")
    if a.fichas:
        for k, r in fichas_bancos(informe).items():
            print(k, r)


if __name__ == "__main__":
    main()
