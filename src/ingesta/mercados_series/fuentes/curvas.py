"""Curvas oficiales de tipos y bonos de referencia (bloque 6 del Tema 1).

Cinco publicadores, una tabla larga del contrato de series (``validar_series``) con la hora real
de publicación de cada dato:

- ``ust_curva`` (Tesoro de EE. UU.): curva par diaria, un CSV por año desde 1990
  (``daily-treasury-rates.csv/<año>/all?type=daily_treasury_yield_curve``). Los plazos cambian con
  los años: el CSV de 1990 empieza en ``3 Mo``; ``1 Mo`` llega en 2001, ``2 Mo`` en 2018, ``4 Mo``
  en 2022 y ``1.5 Month`` (6 semanas) en 2025; el 30 años falta de 2002-02 a 2006-02. Cada columna
  se lee por su cabecera, nunca por posición. Ids ``ust:PAR.<plazo>`` (``1M``, ``6W``, ``3M``...).
  Publicación: publicador ``treasury_curva`` de ``config/publicaciones.yaml`` (18:00 NY del mismo
  día → visible en la decisión de D+1).
- ``bce_curvas`` (BCE, SDMX ``YC``): curva Svensson de la zona euro (composición cambiante, U2),
  bonos AAA (``G_N_A``) y todos los emisores (``G_N_C``), tipos al contado ``SR_<plazo>``, desde
  2004-09-06. Ids ``ecb:YC.AAA.<plazo>`` y ``ecb:YC.ALL.<plazo>``. Sale a las 12:00 Europe/Berlin
  del hábil TARGET siguiente.
- ``mof_jgb`` (Ministerio de Finanzas de Japón): ``jgbcme_all.csv`` (histórico desde 1974-09-24,
  se rehace a primeros de mes) + ``jgbcme.csv`` (mes en curso). Cabecera con título y unidad, ``-``
  como faltante y nota al pie (aviso de caché) que se descarta. Ids ``mof:JGB.<plazo>``. Dato del
  día D publicado el hábil japonés siguiente (08:30 JST medido con ``Last-Modified``).
- ``boc_curva`` (Banco de Canadá, Valet): grupo ``bond_yields_benchmark`` (bonos de referencia 2,
  3, 5, 7, 10 años, largo y real); CSV con bloques de metadatos y los datos tras el marcador
  ``"OBSERVATIONS"``. Ids ``boc:BOND.<plazo>``. Solo curva (los tipos oficiales los baja otro bloque).
- ``boe_curvas`` (Banco de Inglaterra): ZIP con xlsx de curvas (hoja ``4. spot curve``, plazos en
  años en la fila ``years:``). ``latest-yield-curve-data.zip`` trae el mes en curso; el histórico
  (``glcnominalddata.zip``) se baja con la CLI (``--historico-boe``). Ids ``boe:NOM.<plazo>``.

Hábiles: el dato de D sale el HÁBIL SIGUIENTE DEL PROPIO PUBLICADOR. La siguiente fecha con dato en
la propia serie es ese hábil (las fuentes solo publican en sus días hábiles: festivos japoneses,
canadienses, británicos o TARGET incluidos); para la última fila, el siguiente día de lunes a
viernes (puede adelantar un día si el siguiente es festivo: se anota en la ficha).
Huecos dentro del tramo publicado de un plazo (celda vacía o ``-``): valor nulo con
``NOTA_HUECO``. Vintage ``unico``: ninguna de estas curvas se revisa tras publicarse.

Red solo con ``ingesta.comun.http.get`` tras ``robots_permite``; un 403 o un reto no se esquivan
(quedan en el ``.meta.json``). Crudos en ``<corpus>/mercados_series/crudo/<fuente>/`` con nombre
``<AAAAMMDD>_<fichero>`` (día de descarga en Nueva York); tabla en
``<corpus>/mercados_series/series/curvas_b6.parquet`` (``almacen.escribir_series``).

CLI: ``python -m ingesta.mercados_series.fuentes.curvas [--descargar] [--desde-anio N]
[--historico-boe] [--fichas] [--escribir]``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun import http
from ingesta.comun.rutas import corpus_dir
from ingesta.mercados_series import visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    HUSO_DECISION,
    primera_decision_vec,
    vacio_series,
    validar_series,
)

TEMA = "mercados_series"
VINTAGE = "unico"
UNIDAD = "% anual"
METODO = "calendario"
NOTA_HUECO = "hueco permanente: la fuente no publica este plazo ese día"
FUENTES = ("ust_curva", "bce_curvas", "mof_jgb", "boc_curva", "boe_curvas")
FICHERO_TABLA = "curvas_b6.parquet"
CABECERAS = ("Last-Modified", "ETag", "Content-Type", "Content-Length", "Date")

# --- URLs ---------------------------------------------------------------------------------------
URL_UST = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           "daily-treasury-rates.csv/{anio}/all?type=daily_treasury_yield_curve"
           "&field_tdr_date_value={anio}&page&_format=csv")
UST_DESDE = 1990
PLAZOS_BCE = ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
CURVAS_BCE = {"G_N_A": "AAA", "G_N_C": "ALL"}
URL_BCE = ("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.{curvas}.SV_C_YM.{plazos}"
           "?format=csvdata&detail=dataonly&startPeriod={desde}")
BCE_DESDE = "2004-09-06"
URL_MOF_TODO = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
URL_MOF_MES = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
URL_BOC = "https://www.bankofcanada.ca/valet/observations/group/bond_yields_benchmark/csv?start_date=1990-01-01"
URL_BOE_ULTIMO = ("https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
                  "latest-yield-curve-data.zip")
URL_BOE_HISTORICO = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalddata.zip"
HOJA_BOE = "4. spot curve"
PLAZOS_BOE = (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40)
# Bundesbank y RBA: fuera del bloque (Alemania va con la curva del BCE; Australia no está en el
# universo de países). Una sonda por cada una para documentar el acceso en la ficha del BCE.
URL_SONDAS = {
    "bundesbank": ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
                   "D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A?format=csv&lang=en&startPeriod=2026-09-01"),
    "rba": "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv",
}

# --- Publicadores (copia de config/fragmentos/b6_curvas.yaml; ust usa treasury_curva) -----------
# hora local y huso de la salida del dato de D en el hábil siguiente del publicador.
PUBLICADORES: dict[str, dict[str, Any]] = {
    "bce_curvas": {"clave": "ecb_yc", "hora": "12:00", "huso": "Europe/Berlin"},
    "mof_jgb": {"clave": "mof_jgb", "hora": "08:30", "huso": "Asia/Tokyo"},
    "boc_curva": {"clave": "boc_bonos", "hora": "18:00", "huso": "America/Toronto"},
    "boe_curvas": {"clave": "boe_yc", "hora": "12:00", "huso": "Europe/London"},
}
PUBLICADOR_UST = "treasury_curva"


# --- Plazos ------------------------------------------------------------------------------------
_PATRON_UST = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(Mo|Month|Months|Yr|Year|Years|Wk|Week|Weeks)\s*$", re.IGNORECASE)


def plazo_ust(cabecera: str) -> str | None:
    """Cabecera del CSV del Tesoro → plazo canónico (``"1 Mo"`` → ``1M``, ``"1.5 Month"`` → ``6W``).

    ``None`` si la columna no es un plazo (``Date`` u otra que añada el Tesoro).
    """
    m = _PATRON_UST.match(str(cabecera))
    if not m:
        return None
    n, u = float(m.group(1)), m.group(2).lower()
    if u.startswith("w"):
        return f"{int(n)}W"
    if u.startswith("m"):
        return f"{int(n)}M" if n == int(n) else f"{int(round(n * 52 / 12))}W"
    return f"{int(n)}Y" if n == int(n) else f"{int(round(n * 12))}M"


def plazo_anios(anios: float) -> str:
    """Plazo en años (``0.5``, ``10``) → ``6M``, ``10Y``."""
    a = float(anios)
    return f"{int(a)}Y" if a == int(a) else f"{int(round(a * 12))}M"


# --- Descargas ----------------------------------------------------------------------------------
def ruta_crudo(fuente: str, raiz: Path | None = None) -> Path:
    """``<corpus>/mercados_series/crudo/<fuente>/`` (fuera del repositorio)."""
    if fuente not in FUENTES:
        raise ValueError(f"fuente {fuente!r} no es una de {list(FUENTES)}")
    base = Path(raiz) if raiz is not None else corpus_dir(TEMA) / "crudo"
    p = base / fuente
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hoy_ny() -> str:
    return pd.Timestamp.now(tz=HUSO_DECISION).strftime("%Y%m%d")


def _respuesta(url: str, timeout: int = 90) -> tuple[Any, str | None]:
    if not http.robots_permite(url):
        return None, "robots.txt no permite la descarga con nuestro User-Agent"
    try:
        r = http.get(url, timeout=timeout, reintentos=2)
    except Exception as e:  # noqa: BLE001 - se documenta en el .meta.json
        resp = getattr(e, "response", None)
        return resp, (f"{type(e).__name__}: {http.sin_clave(str(e))[:300]} "
                      f"(estado {getattr(resp, 'status_code', None)}; no se esquiva)")
    if "html" in r.headers.get("Content-Type", "").lower():
        return r, "respuesta HTML en lugar del fichero (posible reto o bloqueo; no se esquiva)"
    return r, None


def descargar_url(fuente: str, nombre: str, url: str, *, forzar: bool = False, raiz: Path | None = None,
                  timeout: int = 90) -> dict[str, Any]:
    """Descarga ``url`` como ``<AAAAMMDD>_<nombre>`` en el crudo de ``fuente`` y devuelve su meta.

    El crudo de hoy (Nueva York) se reutiliza salvo ``forzar``. Un fallo no lanza: queda en el meta.
    """
    ruta = ruta_crudo(fuente, raiz) / f"{_hoy_ny()}_{nombre}"
    ruta_meta = ruta.with_name(ruta.name + ".meta.json")
    if not forzar and ruta.exists() and ruta_meta.exists():
        return json.loads(ruta_meta.read_text(encoding="utf-8"))
    r, error = _respuesta(url, timeout)
    meta: dict[str, Any] = {"fuente": fuente, "clave": nombre, "url": url,
                            "descargado_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "estado": getattr(r, "status_code", None)}
    if r is not None:
        meta["cabeceras"] = {k: r.headers[k] for k in CABECERAS if k in r.headers}
    if error is None:
        meta |= {"bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()}
        ruta.write_bytes(r.content)
    else:
        meta["error"] = error
    ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def descargar(fuente: str, *, desde_anio: int = UST_DESDE, historico_boe: bool = False, forzar: bool = False,
              raiz: Path | None = None) -> list[dict[str, Any]]:
    """Descarga los crudos de ``fuente``; reanudable (los de hoy se reutilizan)."""
    if fuente == "ust_curva":
        hoy = pd.Timestamp.now(tz=HUSO_DECISION).year
        # El año en curso siempre se vuelve a bajar si se fuerza; los cerrados no cambian.
        return [descargar_url(fuente, f"ust_{a}.csv", URL_UST.format(anio=a), forzar=forzar and a == hoy, raiz=raiz)
                for a in range(desde_anio, hoy + 1)]
    if fuente == "bce_curvas":
        url = URL_BCE.format(curvas="+".join(CURVAS_BCE), plazos="+".join(f"SR_{p}" for p in PLAZOS_BCE),
                             desde=BCE_DESDE)
        return [descargar_url(fuente, "yc_svensson.csv", url, forzar=forzar, raiz=raiz, timeout=300)]
    if fuente == "mof_jgb":
        return [descargar_url(fuente, "jgbcme_all.csv", URL_MOF_TODO, forzar=forzar, raiz=raiz),
                descargar_url(fuente, "jgbcme.csv", URL_MOF_MES, forzar=forzar, raiz=raiz)]
    if fuente == "boc_curva":
        return [descargar_url(fuente, "bond_yields_benchmark.csv", URL_BOC, forzar=forzar, raiz=raiz)]
    if fuente == "boe_curvas":
        out = [descargar_url(fuente, "latest-yield-curve-data.zip", URL_BOE_ULTIMO, forzar=forzar, raiz=raiz)]
        if historico_boe:
            out.append(descargar_url(fuente, "glcnominalddata.zip", URL_BOE_HISTORICO, forzar=forzar, raiz=raiz,
                                     timeout=600))
        return out
    raise ValueError(f"fuente {fuente!r} no es una de {list(FUENTES)}")


def _ultimos(fuente: str, patron: str, raiz: Path | None = None) -> list[Path]:
    """Crudos de ``fuente`` cuyo nombre (sin el prefijo de fecha) casa con ``patron``; el más
    reciente de cada nombre."""
    carpeta = ruta_crudo(fuente, raiz)
    por_nombre: dict[str, Path] = {}
    for p in sorted(carpeta.glob(f"*_{patron}")):
        if p.name.endswith(".meta.json"):
            continue
        por_nombre[p.name.split("_", 1)[1]] = p
    return [por_nombre[k] for k in sorted(por_nombre)]


# --- Lectores (devuelven DataFrame ancho: índice fecha, columnas plazo canónico) ----------------
def _numeros(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: pd.to_numeric(c.replace({"-": np.nan, "": np.nan, "ND": np.nan}), errors="coerce"))


def leer_ust(fuente: Path | bytes | str) -> pd.DataFrame:
    """CSV anual del Tesoro → ancho por plazo. Columnas por cabecera; filas en orden ascendente."""
    texto = fuente.decode("utf-8-sig") if isinstance(fuente, bytes) else (
        Path(fuente).read_text(encoding="utf-8-sig") if isinstance(fuente, Path) else fuente)
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False)
    if "Date" not in df.columns:
        raise ValueError("CSV del Tesoro sin columna 'Date'")
    plazos = {c: plazo_ust(c) for c in df.columns if c != "Date"}
    plazos = {c: p for c, p in plazos.items() if p is not None}
    out = _numeros(df[list(plazos)]).rename(columns=plazos)
    out.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    out.index.name = "fecha"
    return out.sort_index()


def leer_bce(fuente: Path | bytes | str) -> pd.DataFrame:
    """CSV SDMX del BCE (``csvdata``, con o sin ``detail=dataonly``) → ancho ``<CURVA>.<plazo>``."""
    texto = fuente.decode("utf-8-sig") if isinstance(fuente, bytes) else (
        Path(fuente).read_text(encoding="utf-8-sig") if isinstance(fuente, Path) else fuente)
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False)
    faltan = {"INSTRUMENT_FM", "DATA_TYPE_FM", "TIME_PERIOD", "OBS_VALUE"} - set(df.columns)
    if faltan:
        raise ValueError(f"CSV del BCE sin columnas {sorted(faltan)}")
    df = df[df["INSTRUMENT_FM"].isin(CURVAS_BCE) & df["DATA_TYPE_FM"].str.startswith("SR_")]
    col = df["INSTRUMENT_FM"].map(CURVAS_BCE) + "." + df["DATA_TYPE_FM"].str.removeprefix("SR_")
    largo = pd.DataFrame({"fecha": pd.to_datetime(df["TIME_PERIOD"]), "col": col.to_numpy(),
                          "valor": pd.to_numeric(df["OBS_VALUE"].replace("", np.nan), errors="coerce")})
    ancho = largo.pivot_table(index="fecha", columns="col", values="valor", aggfunc="last", dropna=False)
    ancho.columns.name = None
    return ancho.sort_index()


def leer_mof(fuente: Path | bytes | str) -> pd.DataFrame:
    """CSV del MoF (``jgbcme_all.csv`` o ``jgbcme.csv``) → ancho ``<n>Y``.

    Primera línea: título y ``(Unit : %)``; segunda: cabecera ``Date,1Y,...``. Fechas ``AAAA/M/D``;
    ``-`` es faltante; las líneas finales sin fecha (vacías o la nota sobre la caché del navegador)
    se descartan.
    """
    crudo = fuente if isinstance(fuente, bytes) else (Path(fuente).read_bytes() if isinstance(fuente, Path)
                                                      else fuente.encode("utf-8"))
    try:
        texto = crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = crudo.decode("cp932", errors="replace")
    filas = list(csv.reader(io.StringIO(texto)))
    i_cab = next((i for i, f in enumerate(filas) if f and f[0].strip() == "Date"), None)
    if i_cab is None:
        raise ValueError("CSV del MoF sin fila de cabecera 'Date'")
    cab = [c.strip() for c in filas[i_cab]]
    patron_fecha = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")
    datos = [f for f in filas[i_cab + 1:] if f and patron_fecha.match(f[0].strip())]
    df = pd.DataFrame([f[:len(cab)] + [""] * (len(cab) - len(f)) for f in datos], columns=cab)
    plazos = [c for c in cab[1:] if re.fullmatch(r"\d+Y", c)]
    out = _numeros(df[plazos])
    out.index = pd.to_datetime(df["Date"].str.strip(), format="%Y/%m/%d")
    out.index.name = "fecha"
    return out[~out.index.duplicated(keep="last")].sort_index()


_PLAZO_BOC = re.compile(r"BD\.CDN\.(\w+)\.DQ\.YLD")


def leer_boc(fuente: Path | bytes | str) -> pd.DataFrame:
    """CSV de Valet → ancho ``2Y``, ``3Y``, ``5Y``, ``7Y``, ``10Y``, ``LONG``, ``RRB``.

    Los datos empiezan tras la línea ``"OBSERVATIONS"`` (antes: términos, nombre, descripción y la
    lista de series). Celdas vacías → nulo.
    """
    texto = fuente.decode("utf-8-sig") if isinstance(fuente, bytes) else (
        Path(fuente).read_text(encoding="utf-8-sig") if isinstance(fuente, Path) else fuente)
    lineas = texto.splitlines()
    i = next((k for k, s in enumerate(lineas) if s.strip().strip('"') == "OBSERVATIONS"), None)
    if i is None:
        raise ValueError("CSV de Valet sin el marcador OBSERVATIONS")
    df = pd.read_csv(io.StringIO("\n".join(lineas[i + 1:])), dtype=str, keep_default_na=False)
    if "date" not in df.columns:
        raise ValueError("CSV de Valet sin columna 'date' tras OBSERVATIONS")
    nombres = {}
    for c in df.columns:
        m = _PLAZO_BOC.fullmatch(c)
        if m:
            p = m.group(1)
            nombres[c] = p.replace("YR", "Y") if p.endswith("YR") else p
    out = _numeros(df[list(nombres)]).rename(columns=nombres)
    out.index = pd.to_datetime(df["date"])
    out.index.name = "fecha"
    return out.sort_index()


def leer_boe_xlsx(fuente: Path | bytes, hoja: str = HOJA_BOE, plazos: Iterable[float] = PLAZOS_BOE) -> pd.DataFrame:
    """Una hoja de curva del BoE (fila ``years:`` con los plazos; fechas en la primera columna)."""
    datos = Path(fuente).read_bytes() if isinstance(fuente, Path) else fuente
    libro = pd.ExcelFile(io.BytesIO(datos))
    if hoja not in libro.sheet_names:
        # Los ficheros antiguos llaman a la hoja '4. nominal spot curve' (o 'real', 'inflation').
        candidatas = [h for h in libro.sheet_names if "spot curve" in h.lower() and "short" not in h.lower()]
        if not candidatas:
            raise ValueError(f"xlsx del BoE sin hoja de curva spot ({libro.sheet_names})")
        hoja = candidatas[0]
    crudo = pd.read_excel(libro, sheet_name=hoja, header=None)
    i_anios = next((k for k in range(min(len(crudo), 10)) if str(crudo.iat[k, 0]).strip().lower() == "years:"), None)
    if i_anios is None:
        raise ValueError(f"hoja {hoja!r} del BoE sin fila 'years:'")
    anios = pd.to_numeric(crudo.iloc[i_anios, 1:], errors="coerce")
    fechas = pd.to_datetime(crudo.iloc[:, 0].map(lambda v: v if isinstance(v, (pd.Timestamp, datetime)) else None))
    filas = fechas.notna() & (crudo.index > i_anios)
    quiero = {float(p) for p in plazos}
    cols = {j: plazo_anios(a) for j, a in anios.items() if pd.notna(a) and float(a) in quiero}
    out = crudo.loc[filas, list(cols)].apply(pd.to_numeric, errors="coerce").rename(columns=cols)
    out.index = pd.DatetimeIndex(fechas[filas]).normalize()
    out.index.name = "fecha"
    # Filas sin ningún valor (festivos que el BoE lista en blanco): no son días publicados.
    return out[~out.index.duplicated(keep="last")].dropna(how="all").sort_index()


def leer_boe_zip(fuente: Path | bytes, hoja: str = HOJA_BOE) -> pd.DataFrame:
    """ZIP del BoE → ancho de la curva nominal (todos los xlsx nominales del zip, unidos)."""
    datos = Path(fuente).read_bytes() if isinstance(fuente, Path) else fuente
    partes = []
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        for nombre in sorted(z.namelist()):
            n = nombre.lower()
            if n.endswith((".xlsx", ".xls")) and "nominal" in n:
                try:
                    partes.append(leer_boe_xlsx(z.read(nombre), hoja))
                except ValueError:
                    continue
    if not partes:
        raise ValueError("ZIP del BoE sin xlsx nominal legible")
    out = pd.concat(partes)
    return out[~out.index.duplicated(keep="last")].sort_index()


# --- Fechado ------------------------------------------------------------------------------------
def disponible_habil_siguiente(fechas: pd.DatetimeIndex, hora: str, huso: str) -> pd.Series:
    """Instante UTC de publicación: ``hora`` (``huso``) del hábil del publicador siguiente a cada fecha.

    El hábil siguiente es la siguiente fecha de ``fechas`` (las fechas con dato son los hábiles de
    la fuente); para la última, el siguiente día de lunes a viernes.
    """
    f = pd.DatetimeIndex(pd.to_datetime(fechas)).normalize()
    unicas = f.unique().sort_values()
    sig = pd.Series(unicas[1:].append(pd.DatetimeIndex([pd.NaT])), index=unicas)
    if len(unicas):
        ultima = unicas[-1] + pd.offsets.BDay(1)
        sig.iloc[-1] = ultima
    h, m = (int(x) for x in hora.split(":"))
    local = pd.DatetimeIndex(sig.reindex(f).to_numpy()) + pd.Timedelta(hours=h, minutes=m)
    return pd.Series(local.tz_localize(huso, ambiguous="raise", nonexistent="raise").tz_convert("UTC"),
                     index=range(len(f)))


def disponible_mismo_dia(fechas: pd.DatetimeIndex, hora: str, huso: str) -> pd.Series:
    """Instante UTC de publicación a ``hora`` (``huso``) del propio día del dato."""
    h, m = (int(x) for x in hora.split(":"))
    local = pd.DatetimeIndex(pd.to_datetime(fechas)).normalize() + pd.Timedelta(hours=h, minutes=m)
    return pd.Series(local.tz_localize(huso).tz_convert("UTC"), index=range(len(local)))


def _hora_publicador_ust() -> tuple[str, str]:
    try:
        pub = visibilidad.cargar_publicaciones(catalogo=False)["publicadores"][PUBLICADOR_UST]
        hora = pub["hora"]
        return (hora.strftime("%H:%M") if hasattr(hora, "strftime") else str(hora)), pub["huso"]
    except Exception:  # noqa: BLE001 - sin config: la hora declarada del Tesoro
        return "18:00", "America/New_York"


def _con_dato(ancho: pd.DataFrame) -> pd.DataFrame:
    """Solo los días con algún valor: un día sin ningún plazo no es un día publicado."""
    return ancho.dropna(how="all") if len(ancho) else ancho


def _larga(ancho: pd.DataFrame, prefijo: str, fuente: str, disponible: pd.Series) -> pd.DataFrame:
    """Ancho (fecha × plazo) → tabla larga del contrato. Por plazo: desde su primer a su último
    valor; los huecos intermedios, nulos con ``NOTA_HUECO``."""
    if ancho.empty:
        return vacio_series()
    disp = pd.Series(disponible.to_numpy(), index=ancho.index)
    partes = []
    for col in ancho.columns:
        s = ancho[col]
        validos = s.notna().to_numpy()
        if not validos.any():
            continue
        i0, i1 = np.argmax(validos), len(s) - 1 - np.argmax(validos[::-1])
        tramo = s.iloc[i0:i1 + 1]
        d = disp.iloc[i0:i1 + 1]
        partes.append(pd.DataFrame({
            "serie": f"{prefijo}{col}", "fecha_referencia": tramo.index, "valor": tramo.to_numpy(),
            "disponible_utc": d.to_numpy(),
            "nota": np.where(tramo.isna().to_numpy(), NOTA_HUECO, None)}))
    if not partes:
        return vacio_series()
    df = pd.concat(partes, ignore_index=True)
    df["disponible_utc"] = pd.to_datetime(df["disponible_utc"], utc=True)
    df["fecha_publicacion"] = primera_decision_vec(df["disponible_utc"])
    df["vintage"] = VINTAGE
    df["unidad"] = UNIDAD
    df["divisa"] = None
    df["fuente"] = fuente
    df["metodo_publicacion"] = METODO
    return validar_series(df[list(COLUMNAS_SERIES)])


def tabla_ust(ancho: pd.DataFrame) -> pd.DataFrame:
    ancho = _con_dato(ancho)
    hora, huso = _hora_publicador_ust()
    return _larga(ancho, "ust:PAR.", "ust_curva", disponible_mismo_dia(ancho.index, hora, huso))


def tabla_bce(ancho: pd.DataFrame) -> pd.DataFrame:
    ancho = _con_dato(ancho)
    p = PUBLICADORES["bce_curvas"]
    disp = disponible_habil_siguiente(ancho.index, p["hora"], p["huso"])
    return _larga(ancho, "ecb:YC.", "bce_curvas", disp)


def tabla_mof(ancho: pd.DataFrame) -> pd.DataFrame:
    ancho = _con_dato(ancho)
    p = PUBLICADORES["mof_jgb"]
    return _larga(ancho, "mof:JGB.", "mof_jgb", disponible_habil_siguiente(ancho.index, p["hora"], p["huso"]))


def tabla_boc(ancho: pd.DataFrame) -> pd.DataFrame:
    ancho = _con_dato(ancho)
    p = PUBLICADORES["boc_curva"]
    return _larga(ancho, "boc:BOND.", "boc_curva", disponible_habil_siguiente(ancho.index, p["hora"], p["huso"]))


def tabla_boe(ancho: pd.DataFrame) -> pd.DataFrame:
    ancho = _con_dato(ancho)
    p = PUBLICADORES["boe_curvas"]
    return _larga(ancho, "boe:NOM.", "boe_curvas", disponible_habil_siguiente(ancho.index, p["hora"], p["huso"]))


# --- Desde los crudos ---------------------------------------------------------------------------
def ancho_desde_crudos(fuente: str, raiz: Path | None = None) -> pd.DataFrame:
    """Ancho (fecha × plazo) de ``fuente`` con el último crudo de cada fichero."""
    if fuente == "ust_curva":
        partes = [leer_ust(p) for p in _ultimos(fuente, "ust_*.csv", raiz)]
    elif fuente == "bce_curvas":
        partes = [leer_bce(p) for p in _ultimos(fuente, "yc_svensson.csv", raiz)]
    elif fuente == "mof_jgb":
        # El histórico primero; el mes en curso pisa sus días (keep='last').
        partes = [leer_mof(p) for n in ("jgbcme_all.csv", "jgbcme.csv") for p in _ultimos(fuente, n, raiz)]
    elif fuente == "boc_curva":
        partes = [leer_boc(p) for p in _ultimos(fuente, "bond_yields_benchmark.csv", raiz)]
    elif fuente == "boe_curvas":
        partes = [leer_boe_zip(p) for n in ("glcnominalddata.zip", "latest-yield-curve-data.zip")
                  for p in _ultimos(fuente, n, raiz)]
    else:
        raise ValueError(f"fuente {fuente!r} no es una de {list(FUENTES)}")
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes)
    return out[~out.index.duplicated(keep="last")].sort_index()


_TABLAS = {"ust_curva": tabla_ust, "bce_curvas": tabla_bce, "mof_jgb": tabla_mof, "boc_curva": tabla_boc,
           "boe_curvas": tabla_boe}


def tabla_curvas(fuentes: Iterable[str] = FUENTES, *, raiz: Path | None = None) -> pd.DataFrame:
    """Tabla larga validada de las curvas con los crudos locales."""
    partes = [_TABLAS[f](ancho_desde_crudos(f, raiz)) for f in fuentes]
    partes = [p for p in partes if len(p)]
    return validar_series(pd.concat(partes, ignore_index=True)) if partes else vacio_series()


def escribir(tabla: pd.DataFrame, ruta: Path | None = None) -> Path:
    from ingesta.mercados_series import almacen

    ruta = Path(ruta) if ruta is not None else corpus_dir(TEMA) / "series" / FICHERO_TABLA
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return almacen.escribir_series(tabla, ruta)


# --- Mediciones y fichas ------------------------------------------------------------------------
def resumen(tabla: pd.DataFrame) -> pd.DataFrame:
    """Metadatos por serie (sin valores): inicio, fin, n.º de observaciones y huecos, retraso
    (días naturales entre la fecha de referencia y la de publicación)."""
    if tabla.empty:
        return pd.DataFrame(columns=["serie", "fuente", "inicio", "fin", "n_obs", "n_huecos", "retraso_dias_mediana",
                                     "retraso_dias_max", "hora_utc_mediana"])
    t = tabla.assign(ret=(tabla["fecha_publicacion"] - tabla["fecha_referencia"]).dt.days,
                     hueco=tabla["valor"].isna(), hora=tabla["disponible_utc"].dt.strftime("%H:%M"))
    g = t.groupby("serie")
    out = pd.DataFrame({
        "fuente": g["fuente"].first(), "inicio": g["fecha_referencia"].min().dt.date.astype(str),
        "fin": g["fecha_referencia"].max().dt.date.astype(str), "n_obs": g["valor"].count(),
        "n_huecos": g["hueco"].sum().astype(int), "retraso_dias_mediana": g["ret"].median(),
        "retraso_dias_max": g["ret"].max(), "hora_utc_mediana": g["hora"].agg(lambda s: s.mode().iat[0])})
    return out.reset_index()


_NOMBRES = {
    "ust_curva": "US Treasury · Daily Treasury Par Yield Curve Rates",
    "bce_curvas": "BCE · curvas de rendimiento de la zona euro (Svensson; AAA y todos los emisores)",
    "mof_jgb": "Ministerio de Finanzas de Japón · JGB Interest Rate (rendimientos de referencia)",
    "boc_curva": "Banco de Canadá · Valet, rendimientos de bonos de referencia",
    "boe_curvas": "Banco de Inglaterra · curvas nominales del gilt (spot, Anderson-Sleath)",
}
_URLS = {
    "ust_curva": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
                 "?type=daily_treasury_yield_curve",
    "bce_curvas": "https://data.ecb.europa.eu/data/datasets/YC",
    "mof_jgb": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm",
    "boc_curva": "https://www.bankofcanada.ca/valet/docs",
    "boe_curvas": "https://www.bankofengland.co.uk/statistics/yield-curves",
}
_LICENCIAS = {
    "ust_curva": "dominio público (Gobierno de EE. UU.)",
    "bce_curvas": "reutilización libre citando la fuente (ECB Data Portal, condiciones de uso del BCE)",
    "mof_jgb": "condiciones de uso del sitio del MoF (reutilización compatible con CC BY 4.0, citando la fuente)",
    "boc_curva": "términos del Banco de Canadá (https://www.bankofcanada.ca/terms/): uso no comercial citando",
    "boe_curvas": "© Bank of England; uso permitido citando la fuente (términos del sitio)",
}
_RETRASO = {
    "ust_curva": "dato de D publicado el mismo D por la tarde (publicador treasury_curva, 18:00 NY) → decisión D+1",
    "bce_curvas": "dato de D publicado a las 12:00 Europe/Berlin del hábil TARGET siguiente → decisión D+1",
    "mof_jgb": "dato de D publicado el hábil japonés siguiente (08:30 JST, Last-Modified) → decisión D+1 en NY",
    "boc_curva": ("dato de D publicado el hábil canadiense siguiente; se fija a las 18:00 Toronto (conservador) → "
                  "decisión D+2"),
    "boe_curvas": "dato de D publicado el hábil británico siguiente antes de las 12:00 Londres → decisión D+1",
}


def _metas(fuente: str, raiz: Path | None) -> list[dict[str, Any]]:
    carpeta = ruta_crudo(fuente, raiz)
    ultimas: dict[str, dict[str, Any]] = {}
    for p in sorted(carpeta.glob("*.meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        ultimas[m.get("clave", p.name)] = m
    return list(ultimas.values())


def sondas_bloqueo() -> dict[str, Any]:
    """Una petición (un reintento) a Bundesbank y RBA para documentar su acceso."""
    out = {}
    for k, url in URL_SONDAS.items():
        if not http.robots_permite(url):
            out[k] = {"estado": None, "error": "robots.txt no lo permite"}
            continue
        try:
            r = http.get(url, reintentos=2, timeout=60)
            out[k] = {"estado": r.status_code, "bytes": len(r.content),
                      "tipo": r.headers.get("Content-Type"), "last_modified": r.headers.get("Last-Modified")}
        except Exception as e:  # noqa: BLE001
            out[k] = {"estado": getattr(getattr(e, "response", None), "status_code", None),
                      "error": f"{type(e).__name__}: {http.sin_clave(str(e))[:200]}"}
    return out


def fichas_curvas(tabla: pd.DataFrame, *, raiz: Path | None = None, extra: dict[str, dict[str, Any]] | None = None):
    """``FichaFuente`` de las cinco fuentes con lo medido (cobertura y retraso de la tabla)."""
    from ingesta.comun.esquema import FichaFuente

    extra = extra or {}
    res = resumen(tabla)
    fichas = []
    for f in FUENTES:
        r = res[res["fuente"] == f]
        metas = _metas(f, raiz)
        errores = [m for m in metas if m.get("error")]
        estado = "fallo" if metas and len(errores) == len(metas) else (
            "parcial" if errores or (metas and r.empty) else ("ok" if metas else "no_probada"))
        mediciones: dict[str, Any] = {
            "descargas": [{k: m.get(k) for k in ("clave", "estado", "bytes", "descargado_utc", "error")}
                          | {"last_modified": (m.get("cabeceras") or {}).get("Last-Modified")} for m in metas],
            "series": {row.serie: {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                                   for k, v in row._asdict().items() if k not in ("serie", "fuente", "Index")}
                       for row in r.itertuples(index=False)}}
        if len(r):
            mediciones["retraso_dias_mediana"] = float(r["retraso_dias_mediana"].median())
            mediciones["retraso_dias_max"] = int(r["retraso_dias_max"].max())
        e = extra.get(f, {})
        mediciones |= e.get("mediciones", {})
        volumen = sum(int(m.get("bytes") or 0) for m in metas)
        fichas.append(FichaFuente(
            fuente=f, nombre=_NOMBRES[f], tema=TEMA, agentes=["macro", "risk", "portfolio"], rag="serie",
            url=_URLS[f], acceso="api" if f in ("bce_curvas", "boc_curva") else "descarga",
            autenticacion="ninguna", coste="0 €", licencia_y_terminos=_LICENCIAS[f],
            cobertura_inicio=r["inicio"].min() if len(r) else None, cobertura_fin=r["fin"].max() if len(r) else None,
            frecuencia="diaria (hábiles del publicador)",
            retraso_publicacion=_RETRASO[f] + (f"; medido en la tabla: mediana {mediciones['retraso_dias_mediana']:.0f} "
                                               f"días naturales, máximo {mediciones['retraso_dias_max']}"
                                               if len(r) else ""),
            formato={"ust_curva": "csv (uno por año)", "bce_curvas": "csv SDMX", "mof_jgb": "csv",
                     "boc_curva": "csv (Valet)", "boe_curvas": "zip con xlsx"}[f],
            volumen_estimado=f"{volumen / 1e6:.2f} MB de crudos descargados ({len(metas)} fichero(s))",
            limites=e.get("limites", "1 petición/s por host; robots.txt permite la descarga"),
            estado_sonda=estado, mediciones=mediciones,
            notas=[f"Cargador: ingesta.mercados_series.fuentes.curvas (crudos en <corpus>/mercados_series/crudo/{f}/; "
                   f"tabla en <corpus>/mercados_series/series/{FICHERO_TABLA})",
                   "Vintage 'unico' y metodo 'calendario': las curvas no se revisan tras publicarse",
                   *e.get("notas", [])]))
    return fichas


NOTAS_FICHA: dict[str, list[str]] = {
    "ust_curva": [
        "Plazos que cambian con los años, leídos por cabecera: 1990 empieza en 3 Mo; 1 Mo desde 2001-07-31, "
        "2 Mo desde 2018-10-16, 4 Mo desde 2022-10-19, 1.5 Month (id 6W) desde 2025-02-18; el 30 años no se "
        "publicó entre 2002-02-19 y 2006-02-08 (nulos con nota dentro de su tramo)",
        "Ruptura de método: desde 2021-12-06 la curva par se estima con monotone convex (antes, quasi-cúbica "
        "hermítica); se declara, no se corrige",
        "Hora: publicador treasury_curva de publicaciones.yaml (18:00 NY, sin verificar con fuente oficial)",
        "Descarga lenta (unos 20 s por CSV anual el 2026-10-13): 37 ficheros en dos pasadas reanudables "
        "(descargar('ust_curva', desde_anio=N) reutiliza los crudos del día)",
    ],
    "bce_curvas": [
        "Ids ecb:YC.AAA.<plazo> (G_N_A) y ecb:YC.ALL.<plazo> (G_N_C): Svensson, tipos al contado (SR_), "
        "composición cambiante de la zona euro (U2); plazos 3M a 30Y",
        "Hora: Last-Modified de la API 12:00 Europe/Berlin con el dato del hábil anterior (medido el 2026-10-13)",
        "Bundesbank y RBA: una petición cada una para documentar el acceso (ver mediciones.sondas); no se cargan "
        "(Alemania queda cubierta por la curva del BCE; Australia fuera del universo de países)",
    ],
    "mof_jgb": [
        "jgbcme_all.csv (histórico, rehecho a primeros de mes) + jgbcme.csv (mes en curso): el mensual pisa al "
        "histórico en sus días",
        "Formato: título y '(Unit : %)' en la primera fila, cabecera Date,1Y..40Y en la segunda; '-' es faltante "
        "(plazos que aún no existían: 15Y a 40Y en los años setenta y ochenta); al pie, una fila vacía y una nota "
        "sobre la caché del navegador que se descartan",
        "Hora: Last-Modified de jgbcme.csv 2026-10-12 23:30 UTC (08:30 JST del 13) con el dato del 9 (el 12 fue "
        "festivo en Japón): el dato sale el hábil japonés siguiente",
    ],
    "boc_curva": [
        "Grupo Valet bond_yields_benchmark: 2, 3, 5, 7 y 10 años, LONG (largo de referencia) y RRB (bono real); "
        "los datos van tras el marcador \"OBSERVATIONS\"",
        "Hora no verificada: el 2026-10-13 a las 07:44 Toronto el último dato era el del 8 (viernes 9 aún sin "
        "publicar tras el festivo del 12): se fija a las 18:00 Toronto del hábil siguiente (conservador)",
        "Solo curva: los tipos oficiales del Banco de Canadá los carga otro bloque",
    ],
    "boe_curvas": [
        "latest-yield-curve-data.zip: mes en curso (GLC Nominal/Real/Inflation y OIS); glcnominalddata.zip: "
        "histórico nominal, con la CLI --historico-boe (descarga grande)",
        "Hoja '4. spot curve' (en los xlsx antiguos '4. nominal spot curve'; plazos 0,5 a 40 años); se cargan 6M, 1Y, "
        "2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 25Y, 30Y, 40Y",
        "El tramo de plazos depende de los gilts vivos: desde 1979 hasta 20Y (con huecos), 25Y desde 1992, 30Y y 40Y solo "
        "en el fichero 2016 en adelante; los festivos que el BoE lista en blanco se descartan (no son días publicados)",
        "Hora: Last-Modified del zip 2026-10-13 10:04 UTC (11:04 Londres) con el dato del 12: el dato sale el "
        "hábil siguiente antes de las 12:00 Londres",
    ],
}


def main(argv: list[str] | None = None) -> None:
    from ingesta.comun.fichas import guardar_ficha

    ap = argparse.ArgumentParser(description="Curvas oficiales (bloque 6 del Tema 1)")
    ap.add_argument("--fuentes", nargs="*", default=list(FUENTES))
    ap.add_argument("--descargar", action="store_true")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--desde-anio", type=int, default=UST_DESDE)
    ap.add_argument("--historico-boe", action="store_true")
    ap.add_argument("--fichas", action="store_true")
    ap.add_argument("--escribir", action="store_true")
    a = ap.parse_args(argv)
    if a.descargar:
        for f in a.fuentes:
            metas = descargar(f, desde_anio=a.desde_anio, historico_boe=a.historico_boe, forzar=a.forzar)
            print(f, [(m["clave"], m.get("estado"), m.get("bytes"), m.get("error")) for m in metas][-3:])
    tabla = tabla_curvas(a.fuentes)
    print(f"filas: {len(tabla)}; series: {tabla['serie'].nunique()}")
    if a.escribir:
        print("escrita:", escribir(tabla))
    if a.fichas:
        extra = {f: {"notas": NOTAS_FICHA[f]} for f in FUENTES}
        extra["bce_curvas"]["mediciones"] = {"sondas": sondas_bloqueo()}
        for ficha in fichas_curvas(tabla, extra=extra):
            if ficha.fuente in a.fuentes:
                print("ficha:", guardar_ficha(ficha))
        from ingesta.comun.rutas import resultados_dir

        ruta = resultados_dir(TEMA) / "tablas" / "b6_curvas_resumen.csv"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        resumen(tabla).to_csv(ruta, index=False)   # metadatos por serie, sin valores
        print("resumen:", ruta)


if __name__ == "__main__":
    main()
