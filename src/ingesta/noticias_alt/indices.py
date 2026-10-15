"""Índices alternativos en serie (A6): GPR, EPU, CBOE SKEW / VIX / put/call.

Todo va ENTERO a la base de series (tabla larga con ``fecha_publicacion``); ninguno es texto. La
fecha de publicación sale de una regla de calendario documentada por cada fuente (``metodo_publicacion``):

- GPR diario (Caldara-Iacoviello): «updated every Monday» (festivo federal -> siguiente hábil). Regla
  conservadora: el lunes hábil SIGUIENTE al día (D+1..D+7). Hay vintages reales en GitHub desde
  2022-02-24 (``gpr_vintages``): con ellos se MIDE el retraso en los días testigo.
- GPR mensual: «updated at the beginning of each month» -> primer día hábil del mes siguiente.
- EPU diario EE. UU. (Baker-Bloom-Davis): CSV actualizado cada día -> D+1.
- EPU global mensual: publicación irregular (medida con Last-Modified) -> último día del mes m+2.
- CBOE SKEW, VIX y put/call: el CSV se actualiza ≈5 h después del cierre (Last-Modified ≈01:00 UTC del
  día siguiente) y el JSON diario de put/call 5-6 h después: ``fecha_publicacion`` = siguiente día
  hábil, ``disponible_utc`` = cierre (16:15 ET) + 6 h. Un D+0 para el VIX exige una fuente en tiempo real.

GPR y EPU se recalculan enteros y se REVISAN (la sonda mide ±16 % en días testigo frente al primer
vintage): la descarga de hoy se guarda con ``metodo_publicacion = base.REVISABLE`` y ``vintage`` = hoy, y
la regla de calendario va en la nota. ``base.a_fecha_publicacion(..., excluir_revisables=True)`` las
descarta. Para el GPR diario desde 2022-02-24 hay vintages reales en GitHub: ``serie_desde_vintages``
monta la serie point-in-time (primer vintage que contiene cada día); bajarlos todos son ≈240 xls de
≈3 MB (≈0,7 GB), así que en la sonda solo se hace con ``cfg.extra['gpr_vintages_pit']``.

El put/call histórico de CBOE (CSV) termina el 2019-10-04; después solo hay un JSON por día (el que
pinta la página diaria de estadísticas), que aquí se muestrea en los días testigo.
"""
from __future__ import annotations

import email.utils
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from ingesta.comun.http import get, robots_permite
from ingesta.noticias_alt import base

MODULO = "indices"

GPR_WEB = "https://www.matteoiacoviello.com/gpr.htm"
GPR_DIARIO = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GPR_MENSUAL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_VINTAGES = "https://api.github.com/repos/iacoviel/iacoviel.github.io/contents/gpr_archive_files"
EPU_WEB = "https://www.policyuncertainty.com"
EPU_DIARIO = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
EPU_GLOBAL = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx"
CBOE_SKEW = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CBOE_PC = {k: f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{k}pc.csv"
           for k in ("total", "equity", "index")}
CBOE_PC_DIARIO = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{fecha}_daily_options"
CBOE_PC_PAGINA = "https://www.cboe.com/us/options/market_statistics/daily/"

GPR_DIARIO_COLS = ("GPRD", "GPRD_ACT", "GPRD_THREAT", "N10D")
GPR_MENSUAL_COLS = ("GPR", "GPRT", "GPRA", "GPRH")
GPR_PAISES = ("USA", "CHN", "RUS", "UKR", "ISR", "DEU", "GBR", "ESP")
CIERRE_ET = "16:15"            # cierre de opciones de índice de CBOE (hora de Nueva York)
FICHERO_TRAS_CIERRE_H = 6      # CSV/JSON de CBOE: Last-Modified medido ≈5 h tras el cierre (+ margen)
VENTANA_VINTAGE_DIAS = {"diario": 21, "mensual": 75}   # vintage válido para medir el retraso
LAG_REGENERADO_H = 24 * 30  # Last-Modified más tardío que esto = fichero regenerado, no publicación

_CAL = USFederalHolidayCalendar()
_HABIL = CustomBusinessDay(calendar=_CAL)

# Descripción de cada serie guardada: fuente, frecuencia ('D' natural, 'B' hábil, 'M' mensual) y regla.
_LAG_CBOE = "D+1 hábil (fichero ~5 h tras el cierre)"
SERIES: dict[str, dict[str, str]] = {
    "gpr_diario": dict(fuente="gpr", frecuencia="D", metodo=base.REVISABLE,
                       lag="lunes hábil siguiente al día (D+1..D+7; festivo federal -> siguiente hábil)"),
    "gpr_diario_pit": dict(fuente="gpr", frecuencia="D", metodo="primer_vintage",
                           lag="fecha del primer vintage (GitHub) que contiene el día"),
    "gpr_mensual": dict(fuente="gpr", frecuencia="M", metodo=base.REVISABLE,
                        lag="primer día hábil del mes siguiente"),
    "epu_diario_usa": dict(fuente="epu", frecuencia="D", metodo=base.REVISABLE, lag="D+1"),
    "epu_global_mensual": dict(fuente="epu", frecuencia="M", metodo=base.REVISABLE,
                               lag="último día del mes m+2 (publicación irregular)"),
    "cboe_skew": dict(fuente="cboe_skew", frecuencia="B", metodo="fichero_tras_cierre", lag=_LAG_CBOE),
    "cboe_vix": dict(fuente="cboe_vix", frecuencia="B", metodo="fichero_tras_cierre", lag=_LAG_CBOE),
    "cboe_putcall_total": dict(fuente="cboe_putcall", frecuencia="B", metodo="fichero_tras_cierre", lag=_LAG_CBOE),
    "cboe_putcall_equity": dict(fuente="cboe_putcall", frecuencia="B", metodo="fichero_tras_cierre", lag=_LAG_CBOE),
    "cboe_putcall_index": dict(fuente="cboe_putcall", frecuencia="B", metodo="fichero_tras_cierre", lag=_LAG_CBOE),
}


# --------------------------------------------------------------------------- descarga con caché

def _meta_ruta(ruta: Path) -> Path:
    return ruta.with_name(ruta.name + ".meta.json")


def last_modified(meta: dict) -> datetime | None:
    lm = meta.get("last_modified")
    try:
        return email.utils.parsedate_to_datetime(lm) if lm else None
    except (TypeError, ValueError):
        return None


def descargar(url: str, nombre: str, res: base.Resultado, cfg: base.ConfigSonda, *, fuente: str, item: str,
              dia: str = "", timeout: int = 180, inmutable: bool = False, reintentos: int = 2) -> tuple[bytes, dict]:
    """GET con caché en ``corpus('crudo', 'indices')``; devuelve (contenido, meta).

    Reutiliza el crudo si ``cfg.usar_cache`` y (``inmutable`` o tiene menos de ``cfg.extra['cache_dias']``
    días, 1 por defecto). Registra la petición (o el acierto de caché) y relanza el error.
    """
    ruta = base.corpus("crudo", MODULO) / nombre
    meta_p = _meta_ruta(ruta)
    if cfg.usar_cache and ruta.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        edad = datetime.now(timezone.utc) - datetime.fromisoformat(meta["descargado_utc"])
        if inmutable or edad <= timedelta(days=float(cfg.extra.get("cache_dias", 1))):
            contenido = ruta.read_bytes()
            res.registrar(fuente, f"{item} (caché)", url, True, dia=dia, bytes=len(contenido))
            return contenido, meta
    t0 = base.cronometro()
    try:
        r = get(url, timeout=timeout, reintentos=reintentos)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        res.registrar(fuente, item, url, False, dia=dia, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
        raise
    contenido = r.content
    meta = dict(url=url, last_modified=r.headers.get("Last-Modified"), bytes=len(contenido),
                descargado_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    base.escribir_atomico(ruta, contenido)
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    res.registrar(fuente, item, url, True, dia=dia, bytes=len(contenido), segundos=base.segundos(t0))
    return contenido, meta


# --------------------------------------------------------------------------- parseo

def leer_gpr(contenido: bytes) -> pd.DataFrame:
    """Excel de GPR (diario o mensual) -> DataFrame numérico indexado por fecha.

    Diario: columna ``date`` (o ``DAY`` AAAAMMDD); mensual: ``month``. Se quitan las columnas de
    metadatos (``var_name``, ``var_label``, ``event``) y las filas sin fecha.
    """
    df = pd.read_excel(io.BytesIO(contenido))
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
    elif "month" in df.columns:
        idx = pd.to_datetime(df["month"], errors="coerce")
    elif "DAY" in df.columns:
        idx = pd.to_datetime(df["DAY"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        idx = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.drop(columns=[c for c in ("date", "month", "DAY", "var_name", "var_label", "event") if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = idx
    df = df[df.index.notna()].sort_index()
    df.index.name = "fecha"
    return df


def leer_epu_diario(texto: str) -> pd.Series:
    """CSV ``day,month,year,daily_policy_index`` -> serie diaria."""
    df = pd.read_csv(io.StringIO(texto))
    idx = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")
    s = pd.Series(pd.to_numeric(df["daily_policy_index"], errors="coerce").values, index=idx, name="epu_usa")
    return s[s.index.notna()].dropna().sort_index()


def leer_epu_global(contenido: bytes) -> pd.DataFrame:
    """xlsx ``Year, Month, GEPU_current, GEPU_ppp`` (+ filas de cita al final) -> mensual por fecha."""
    df = pd.read_excel(io.BytesIO(contenido))
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Year", "Month"])
    idx = pd.to_datetime(dict(year=df["Year"].astype(int), month=df["Month"].astype(int), day=1))
    out = df.drop(columns=["Year", "Month"]).apply(pd.to_numeric, errors="coerce")
    out.index = idx
    out.index.name = "fecha"
    return out.sort_index()


def leer_csv_cboe(texto: str) -> pd.DataFrame:
    """CSV de CBOE: aviso legal opcional antes de la cabecera ``DATE``; números con espacios."""
    lineas = texto.splitlines()
    i = next(k for k, l in enumerate(lineas) if l.strip().upper().startswith(("DATE", "TRADE_DATE")))
    df = pd.read_csv(io.StringIO("\n".join(lineas[i:])), skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    idx = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip(), format="mixed", errors="coerce")
    df = df.iloc[:, 1:].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce"))
    df.index = idx
    df = df[df.index.notna()].sort_index()
    df.index.name = "fecha"
    return df


def leer_cboe_diario(j: dict) -> dict[str, float]:
    """JSON diario de estadísticas de CBOE -> ratios put/call (``total``, ``equity``, ``index``, ``vix``...)."""
    claves = {"TOTAL PUT/CALL RATIO": "total", "EQUITY PUT/CALL RATIO": "equity", "INDEX PUT/CALL RATIO": "index",
              "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "vix",
              "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp", "SPX + SPXW PUT/CALL RATIO": "spx"}
    out: dict[str, float] = {}
    for r in j.get("ratios", []):
        k = claves.get(str(r.get("name", "")).strip().upper())
        if k:
            try:
                out[k] = float(r.get("value"))
            except (TypeError, ValueError):
                pass
    return out


# --------------------------------------------------------------------------- reglas de publicación

def _rodar_habil(fechas: pd.Series) -> pd.Series:
    """Lleva cada fecha al primer día hábil (EE. UU., festivos federales) >= fecha."""
    return pd.Series([_HABIL.rollforward(f) for f in fechas], index=fechas.index)


def pub_gpr_diario(fechas: pd.Series) -> pd.Series:
    """Lunes SIGUIENTE (estricto) al día; si es festivo federal, el siguiente hábil."""
    f = pd.to_datetime(fechas).dt.normalize()
    dias = (7 - f.dt.weekday) % 7
    lunes = f + pd.to_timedelta(dias.where(dias > 0, 7), unit="D")
    return _rodar_habil(lunes)


def pub_primer_habil_mes_siguiente(fechas: pd.Series) -> pd.Series:
    f = pd.to_datetime(fechas).dt.normalize()
    return _rodar_habil(f.dt.to_period("M").dt.to_timestamp() + pd.offsets.MonthBegin(1))


def pub_fin_mes(fechas: pd.Series, meses: int) -> pd.Series:
    """Último día del mes ``m + meses`` (``meses=0``: fin del propio mes)."""
    f = pd.to_datetime(fechas).dt.to_period("M").dt.to_timestamp()
    return f + pd.offsets.MonthEnd(meses + 1)


def disponible_cierre(fechas: pd.Series, horas_tras_cierre: float = 0) -> pd.Series:
    """Hora UTC del cierre (16:15 Nueva York) de cada día (+ ``horas_tras_cierre``), ISO."""
    f = pd.to_datetime(fechas).dt.normalize() + pd.Timedelta(CIERRE_ET + ":00")
    f = f.dt.tz_localize("America/New_York").dt.tz_convert("UTC") + pd.Timedelta(hours=horas_tras_cierre)
    return f.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def pub_habil_siguiente(fechas: pd.Series) -> pd.Series:
    """Siguiente día hábil (EE. UU.) estrictamente posterior: fichero publicado tras el cierre."""
    return _rodar_habil(pd.to_datetime(fechas).dt.normalize() + pd.Timedelta(days=1))


# --------------------------------------------------------------------------- tabla larga

def a_larga(df: pd.DataFrame, columnas: dict[str, str], *, unidad: dict[str, str] | str, fuente: str,
            pub: pd.Series, metodo: str, vintage: str, nota: str = "",
            disponible: pd.Series | None = None) -> pd.DataFrame:
    """DataFrame ancho (índice fecha) -> tabla larga; ``columnas`` = {columna_origen: nombre_serie}."""
    trozos = []
    base_df = pd.DataFrame({"fecha_referencia": df.index, "fecha_publicacion": pub.values})
    if disponible is not None:
        base_df["disponible_utc"] = disponible.values
    for col, nombre in columnas.items():
        if col not in df.columns:
            continue
        t = base_df.copy()
        t["valor"] = df[col].values
        t["serie"] = nombre
        t["unidad"] = unidad if isinstance(unidad, str) else unidad.get(col, "indice")
        trozos.append(t.dropna(subset=["valor"]))
    out = pd.concat(trozos, ignore_index=True)
    out["fuente"], out["vintage"], out["metodo_publicacion"], out["nota"] = fuente, vintage, metodo, nota
    return out


def huecos(fechas: pd.Series, frecuencia: str) -> int:
    """Saltos mayores que el paso esperado (D: >1 día; B: >4 días naturales; M: >1 mes)."""
    f = pd.Series(pd.to_datetime(fechas).sort_values().unique())
    if len(f) < 2:
        return 0
    if frecuencia == "M":
        p = f.dt.to_period("M").astype("int64").diff().dropna()
        return int((p > 1).sum())
    d = f.diff().dt.days.dropna()
    return int((d > (1 if frecuencia == "D" else 4)).sum())


def resumen(nombre: str, larga: pd.DataFrame) -> list[dict]:
    """Filas de ``indices_resumen`` (una por serie dentro del fichero ``nombre``)."""
    info = SERIES[nombre]
    filas = []
    for serie, g in larga.groupby("serie"):
        filas.append(dict(fichero=nombre, serie=serie, fuente=info["fuente"],
                          inicio=g["fecha_referencia"].min().date().isoformat(),
                          fin=g["fecha_referencia"].max().date().isoformat(), n=int(len(g)),
                          frecuencia=info["frecuencia"], huecos=huecos(g["fecha_referencia"], info["frecuencia"]),
                          lag_publicacion=info["lag"], metodo=info["metodo"],
                          lag_mediano_regla_dias=round(float((g["fecha_publicacion"] - g["fecha_referencia"])
                                                             .dt.days.median()), 1),
                          valor_min=round(float(g["valor"].min()), 3), valor_max=round(float(g["valor"].max()), 3),
                          valor_ultimo=round(float(g.sort_values("fecha_referencia")["valor"].iloc[-1]), 3)))
    return filas


# --------------------------------------------------------------------------- vintages de GPR

def listar_vintages_gpr(j: list[dict]) -> pd.DataFrame:
    """Listado de la API de GitHub -> (fichero, tipo, fecha_vintage, bytes, url) solo .xls."""
    filas = []
    for x in j:
        m = re.fullmatch(r"data_gpr_(daily_recent|export)_(\d{6,8})\.xls", x.get("name", ""))
        if not m:
            continue
        tipo = "diario" if m.group(1) == "daily_recent" else "mensual"
        s = m.group(2)
        f = pd.to_datetime(s, format="%Y%m%d" if len(s) == 8 else "%Y%m")
        filas.append(dict(fichero=x["name"], tipo=tipo, fecha_vintage=f, bytes=int(x.get("size", 0)),
                          url=x.get("download_url")))
    df = pd.DataFrame(filas, columns=["fichero", "tipo", "fecha_vintage", "bytes", "url"])
    if df.empty:
        return df
    df = df.sort_values(["tipo", "fecha_vintage"]).reset_index(drop=True)
    df["dias_desde_anterior"] = df.groupby("tipo")["fecha_vintage"].diff().dt.days
    df["dia_semana"] = df["fecha_vintage"].dt.day_name()
    return df


def serie_desde_vintages(vintages: list[tuple[pd.Timestamp, str, pd.DataFrame]], columnas: dict[str, str], *,
                         unidad: dict[str, str] | str, fuente: str = "gpr") -> pd.DataFrame:
    """Serie point-in-time desde vintages ``[(fecha_vintage, fichero, df)]``: cada fecha_referencia toma el
    valor y la ``fecha_publicacion`` del PRIMER vintage que la contiene (lo que se veía ese día)."""
    filas, vistos = [], set()
    for fv, fichero, df in sorted(vintages, key=lambda x: x[0]):
        for col, nombre in columnas.items():
            if col not in df.columns:
                continue
            for ref, v in df[col].dropna().items():
                if (nombre, ref) in vistos or ref > fv:
                    continue
                vistos.add((nombre, ref))
                filas.append(dict(serie=nombre, fecha_referencia=ref, fecha_publicacion=fv, valor=float(v),
                                  unidad=unidad if isinstance(unidad, str) else unidad.get(col, "indice"),
                                  fuente=fuente, vintage=fichero, metodo_publicacion="primer_vintage",
                                  nota="valor del primer vintage publicado que contiene el día (GitHub)"))
    return pd.DataFrame(filas)


def _gpr_pit(vint: pd.DataFrame, res: base.Resultado, cfg: base.ConfigSonda) -> pd.DataFrame:
    """Baja TODOS los vintages diarios (inmutables, con caché) y monta ``gpr_diario_pit``."""
    lista = []
    for _, v in vint[vint["tipo"] == "diario"].iterrows():
        try:
            cont, _ = descargar(v["url"], v["fichero"], res, cfg, fuente="gpr", item=f"vintage {v['fichero']}",
                                inmutable=True)
            lista.append((v["fecha_vintage"], v["fichero"], leer_gpr(cont)))
        except Exception:  # noqa: BLE001  (ya registrado)
            continue
    unidad = {"N10D": "articulos"} | {c: "indice (1985-2019=100)" for c in GPR_DIARIO_COLS[:3]}
    return serie_desde_vintages(lista, {c: c.lower() for c in GPR_DIARIO_COLS}, unidad=unidad)


def _vintages_testigo(vint: pd.DataFrame, actual_d: pd.DataFrame, actual_m: pd.DataFrame, res: base.Resultado,
                      cfg: base.ConfigSonda) -> list[dict]:
    """Para cada día testigo con vintages: primer vintage que contiene el día -> retraso y revisión."""
    filas: list[dict] = []
    dias = [d for d in cfg.dias if d.clave != "hoy"]
    max_desc = 1 if cfg.modo_rapido else 3
    for d in dias:
        dia = pd.Timestamp(d.fecha)
        for tipo, actual, col, objetivo in (("diario", actual_d, "GPRD", dia),
                                            ("mensual", actual_m, "GPR", dia.to_period("M").to_timestamp())):
            # Solo vintages publicados poco después del hecho (si no, el «retraso» no mide nada).
            limite = objetivo + pd.Timedelta(days=VENTANA_VINTAGE_DIAS[tipo])
            cand = vint[(vint["tipo"] == tipo) & (vint["fecha_vintage"] >= objetivo)
                        & (vint["fecha_vintage"] <= limite)].head(max_desc)
            fila = dict(dia=d.clave, fecha=d.fecha.isoformat(), tipo=tipo, resolucion="día" if tipo == "diario" else "mes",
                        vintage=None, fecha_vintage=None, lag_dias=None, valor_vintage=None, valor_hoy=None,
                        revision_pct=None, estado="sin cobertura de vintages" if cand.empty else "no encontrado")
            for _, v in cand.iterrows():
                try:
                    cont, _ = descargar(v["url"], v["fichero"], res, cfg, fuente="gpr", item=f"vintage {v['fichero']}",
                                        dia=d.fecha.isoformat(), inmutable=True)
                    dv = leer_gpr(cont)
                except Exception:  # noqa: BLE001  (ya registrado)
                    continue
                if objetivo in dv.index and col in dv.columns and pd.notna(dv.at[objetivo, col]):
                    vv = float(dv.at[objetivo, col])
                    vh = float(actual.at[objetivo, col]) if objetivo in actual.index else None
                    fila.update(vintage=v["fichero"], fecha_vintage=v["fecha_vintage"].date().isoformat(),
                                lag_dias=int((v["fecha_vintage"] - (dia if tipo == "diario"
                                                                     else objetivo + pd.offsets.MonthBegin(1))).days),
                                valor_vintage=round(vv, 3), valor_hoy=None if vh is None else round(vh, 3),
                                revision_pct=None if not vh else round(100 * (vh - vv) / vv, 2), estado="ok")
                    break
            filas.append(fila)
    return filas


# --------------------------------------------------------------------------- tablas de apoyo

def tabla_testigo(larga: pd.DataFrame, dias: list) -> list[dict]:
    """Valor de cada serie en cada día testigo con TimeGate: último dato PUBLICADO a esa fecha y su percentil
    sobre lo publicado hasta entonces. El dato del propio día (aún sin publicar) va aparte como ``*_ex_post``.

    ``revisable``: la serie es el vintage de hoy (GPR, EPU): el valor publicado entonces pudo ser otro.
    """
    filas = []
    for serie, g in larga.groupby("serie"):
        g = g.sort_values("fecha_referencia")
        revisable = bool((g["metodo_publicacion"] == base.REVISABLE).any())
        for d in dias:
            f = pd.Timestamp(d.fecha)
            hasta = g[g["fecha_referencia"] <= f]
            pub = g[g["fecha_publicacion"] <= f].sort_values("fecha_referencia")
            if hasta.empty:
                continue
            ex = hasta.iloc[-1]
            v = float(pub.iloc[-1]["valor"]) if not pub.empty else None
            filas.append(dict(serie=serie, dia=d.clave, fecha=d.fecha.isoformat(),
                              fecha_dato=None if pub.empty else pub.iloc[-1]["fecha_referencia"].date().isoformat(),
                              valor=None if v is None else round(v, 3),
                              percentil_causal=None if v is None else round(float((pub["valor"] <= v).mean() * 100), 1),
                              dias_retraso_dato=None if pub.empty else int((f - pub.iloc[-1]["fecha_referencia"]).days),
                              fecha_dato_ex_post=ex["fecha_referencia"].date().isoformat(),
                              valor_ex_post=round(float(ex["valor"]), 3), revisable=revisable))
    return filas


def tabla_mensual(larga: pd.DataFrame) -> pd.DataFrame:
    """Media, máximo y n por serie y mes (para pintar todas las series en el mismo eje)."""
    larga = larga[~larga["serie"].str.endswith(("_calls", "_puts"))]   # volúmenes: no hacen falta para pintar
    g = larga.assign(mes=larga["fecha_referencia"].dt.to_period("M").dt.to_timestamp())
    out = g.groupby(["serie", "mes"])["valor"].agg(media="mean", maximo="max", n="count").reset_index()
    out["media"], out["maximo"] = out["media"].round(3), out["maximo"].round(3)
    return out


# --------------------------------------------------------------------------- fuentes

def _tg(res: base.Resultado, fuente: str, item: str, hecho: Any, publicado: Any, campo: str) -> None:
    """Ejemplo de TimeGate. Si el Last-Modified es muy posterior, se marca como regenerado (no publicación)."""
    if hecho is None or publicado is None:
        return
    lag = base.lag_horas(hecho, publicado)
    if campo.startswith("Last-Modified") and lag is not None and lag > LAG_REGENERADO_H:
        campo += "; fichero regenerado o congelado después: no mide publicación"
    res.timegate.append(dict(fuente=fuente, item=item, fecha_hecho=pd.Timestamp(hecho).isoformat(),
                             fecha_publicacion=pd.Timestamp(publicado).isoformat(), campo_fecha=campo,
                             lag_h=lag))


def _tg_regla(res: base.Resultado, fuente: str, larga: pd.DataFrame, nombre: str, dias: list) -> None:
    """Ejemplos de la regla de calendario en los días testigo."""
    for d in dias:
        f = pd.Timestamp(d.fecha)
        g = larga[larga["fecha_referencia"] <= f].sort_values("fecha_referencia")
        if g.empty or (f - g["fecha_referencia"].iloc[-1]).days > 92:   # día fuera de la cobertura
            continue
        r = g.iloc[-1]
        _tg(res, fuente, f"{nombre} {d.clave}", r["fecha_referencia"], r["fecha_publicacion"],
            f"regla: {SERIES[nombre]['lag']}")


def _sondear_gpr(cfg: base.ConfigSonda, res: base.Resultado, hoy: str, larga_todas: list, resumen_f: list) -> dict:
    med: dict[str, Any] = {}
    cd, md = descargar(GPR_DIARIO, "data_gpr_daily_recent.xls", res, cfg, fuente="gpr", item="diario")
    cm, mm = descargar(GPR_MENSUAL, "data_gpr_export.xls", res, cfg, fuente="gpr", item="mensual")
    dd, dm = leer_gpr(cd), leer_gpr(cm)
    nota = ("vintage de hoy, REVISADO (no es lo que se veía en su día); vintages reales en GitHub desde 2022 "
            "(gpr_vintages, gpr_diario_pit). Regla de fecha: ")
    ld = a_larga(dd, {c: c.lower() for c in GPR_DIARIO_COLS},
                 unidad={"N10D": "articulos"} | {c: "indice (1985-2019=100)" for c in GPR_DIARIO_COLS[:3]},
                 fuente="gpr", pub=pub_gpr_diario(pd.Series(dd.index)), metodo=base.REVISABLE, vintage=hoy,
                 nota=nota + SERIES["gpr_diario"]["lag"])
    cols_m = {c: c.lower() for c in GPR_MENSUAL_COLS} | {f"GPRC_{p}": f"gprc_{p.lower()}" for p in GPR_PAISES}
    lm = a_larga(dm, cols_m, unidad="indice (1985-2019=100)", fuente="gpr",
                 pub=pub_primer_habil_mes_siguiente(pd.Series(dm.index)), metodo=base.REVISABLE, vintage=hoy,
                 nota=nota + SERIES["gpr_mensual"]["lag"])
    for nombre, larga in (("gpr_diario", ld), ("gpr_mensual", lm)):
        res.series.append(base.guardar_serie(larga, nombre, rapido=cfg.modo_rapido).stem)
        larga_todas.append(larga)
        resumen_f.extend(resumen(nombre, larga))
        _tg_regla(res, "gpr", larga[larga["serie"] == ("gprd" if nombre == "gpr_diario" else "gpr")], nombre, cfg.dias)
    # Retraso medido 1: Last-Modified del fichero vigente frente al último dato.
    _tg(res, "gpr", "diario: último dato", dd["GPRD"].dropna().index.max(), last_modified(md), "Last-Modified (HTTP)")
    _tg(res, "gpr", "mensual: último mes", dm["GPR"].dropna().index.max() + pd.offsets.MonthBegin(1),
        last_modified(mm), "Last-Modified (HTTP); hecho = fin del mes; el fichero se reescribe con cada "
        "actualización semanal del diario (la mensual sale el día 1)")
    paises = sorted(c[5:] for c in dm.columns if c.startswith("GPRC_"))
    med.update(diario=dict(filas=len(dd), inicio=dd.index.min().date().isoformat(),
                           fin=dd["GPRD"].dropna().index.max().date().isoformat(), last_modified=md.get("last_modified")),
               mensual=dict(filas=len(dm), inicio=dm.index.min().date().isoformat(),
                            fin=dm["GPR"].dropna().index.max().date().isoformat(), columnas=dm.shape[1],
                            n_paises=len(paises), last_modified=mm.get("last_modified")),
               bytes=md["bytes"] + mm["bytes"])
    # Retraso medido 2: vintages (GitHub) en los días testigo.
    try:
        t0 = base.cronometro()
        r = get(GPR_VINTAGES, timeout=60, reintentos=2, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        res.registrar("gpr", "listado vintages", GPR_VINTAGES, True, bytes=len(r.content), segundos=base.segundos(t0))
        vint = listar_vintages_gpr(r.json())
    except Exception as e:  # noqa: BLE001
        res.registrar("gpr", "listado vintages", GPR_VINTAGES, False, error=f"{type(e).__name__}: {e}")
        vint = pd.DataFrame()
    if not vint.empty:
        res.tablas["gpr_vintages"] = vint.drop(columns=["url"]).assign(
            fecha_vintage=lambda x: x["fecha_vintage"].dt.date.astype(str))
        vt = pd.DataFrame(_vintages_testigo(vint, dd, dm, res, cfg))
        res.tablas["gpr_vintages_testigo"] = vt
        for _, f in vt[vt["estado"] == "ok"].iterrows():
            hecho = pd.Timestamp(f["fecha"]) if f["tipo"] == "diario" else \
                pd.Timestamp(f["fecha"]).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
            _tg(res, "gpr", f"{f['tipo']} {f['dia']}", hecho, f["fecha_vintage"], f"nombre del vintage ({f['vintage']})")
        dv = vint[vint["tipo"] == "diario"]
        mv = vint[vint["tipo"] == "mensual"]
        ult = dv[dv["fecha_vintage"] >= dv["fecha_vintage"].max() - pd.Timedelta(days=365)]
        med["vintages"] = dict(
            n_diarios=int(len(dv)), n_mensuales=int(len(mv)),
            primer_diario=dv["fecha_vintage"].min().date().isoformat() if len(dv) else None,
            primer_mensual=mv["fecha_vintage"].min().date().isoformat() if len(mv) else None,
            dias_entre_vintages_diarios_ult_ano=float(ult["dias_desde_anterior"].median()) if len(ult) else None,
            pct_lunes_ult_ano=round(float((ult["dia_semana"] == "Monday").mean() * 100), 1) if len(ult) else None,
            lag_testigo_dias=vt.loc[vt["estado"] == "ok", ["dia", "tipo", "lag_dias", "revision_pct"]]
            .to_dict("records"))
        if cfg.extra.get("gpr_vintages_pit") and not cfg.modo_rapido:
            pit = _gpr_pit(vint, res, cfg)
            if not pit.empty:
                res.series.append(base.guardar_serie(pit, "gpr_diario_pit").stem)
                larga_todas.append(pit)
                resumen_f.extend(resumen("gpr_diario_pit", pit))
                med["vintages"]["pit_filas"] = int(len(pit))
    return med


def _sondear_epu(cfg: base.ConfigSonda, res: base.Resultado, hoy: str, larga_todas: list, resumen_f: list) -> dict:
    cd, md = descargar(EPU_DIARIO, "All_Daily_Policy_Data.csv", res, cfg, fuente="epu", item="diario EE. UU.")
    cg, mg = descargar(EPU_GLOBAL, "Global_Policy_Uncertainty_Data.xlsx", res, cfg, fuente="epu", item="global mensual")
    s = leer_epu_diario(cd.decode("utf-8", errors="replace"))
    g = leer_epu_global(cg)
    nota = "sin vintages: descarga de hoy; el índice se recalcula (revisiones posibles). Regla de fecha: "
    ld = a_larga(s.to_frame(), {"epu_usa": "epu_usa_diario"}, unidad="indice", fuente="epu",
                 pub=pd.Series(s.index) + pd.Timedelta(days=1), metodo=base.REVISABLE, vintage=hoy,
                 nota=nota + SERIES["epu_diario_usa"]["lag"])
    lg = a_larga(g, {"GEPU_current": "gepu_current", "GEPU_ppp": "gepu_ppp"}, unidad="indice (1997-2015=100)",
                 fuente="epu", pub=pub_fin_mes(pd.Series(g.index), 2), metodo=base.REVISABLE, vintage=hoy,
                 nota=nota + SERIES["epu_global_mensual"]["lag"])
    for nombre, larga, principal in (("epu_diario_usa", ld, "epu_usa_diario"), ("epu_global_mensual", lg, "gepu_current")):
        res.series.append(base.guardar_serie(larga, nombre, rapido=cfg.modo_rapido).stem)
        larga_todas.append(larga)
        resumen_f.extend(resumen(nombre, larga))
        _tg_regla(res, "epu", larga[larga["serie"] == principal], nombre, cfg.dias)
    _tg(res, "epu", "diario: último dato", s.index.max(), last_modified(md), "Last-Modified (HTTP)")
    fin_g = g["GEPU_current"].dropna().index.max()
    _tg(res, "epu", "global: último mes", fin_g + pd.offsets.MonthBegin(1), last_modified(mg),
        "Last-Modified (HTTP); hecho = fin del mes")
    hoy_ts = pd.Timestamp(hoy)
    return dict(diario_usa=dict(filas=len(s), inicio=s.index.min().date().isoformat(),
                                fin=s.index.max().date().isoformat(), last_modified=md.get("last_modified")),
                global_mensual=dict(filas=len(g), inicio=g.index.min().date().isoformat(),
                                    fin=fin_g.date().isoformat(), columnas=list(g.columns),
                                    last_modified=mg.get("last_modified"),
                                    dias_desde_fin_ultimo_mes=int((hoy_ts - (fin_g + pd.offsets.MonthEnd(0))).days)),
                bytes=md["bytes"] + mg["bytes"])


def _serie_cboe(cfg, res, hoy, url, fichero, fuente, item, columnas, unidad, nombre, larga_todas, resumen_f):
    cont, meta = descargar(url, fichero, res, cfg, fuente=fuente, item=item)
    df = leer_csv_cboe(cont.decode("utf-8", errors="replace"))
    ref = pd.Series(df.index)
    larga = a_larga(df, columnas, unidad=unidad, fuente=fuente, pub=pub_habil_siguiente(ref),
                    metodo="fichero_tras_cierre", vintage=hoy,
                    disponible=disponible_cierre(ref, FICHERO_TRAS_CIERRE_H),
                    nota=f"valor de cierre; CSV de CBOE actualizado ≈5 h tras el cierre -> {_LAG_CBOE}")
    res.series.append(base.guardar_serie(larga, nombre, rapido=cfg.modo_rapido).stem)
    larga_todas.append(larga)
    resumen_f.extend(resumen(nombre, larga))
    ultimo = df.dropna(how="all").index.max()
    cierre = pd.Timestamp(disponible_cierre(pd.Series([ultimo])).iloc[0])
    _tg(res, fuente, f"{nombre}: último dato (cierre)", cierre, last_modified(meta), "Last-Modified (HTTP)")
    _tg_regla(res, fuente, larga[larga["serie"] == next(iter(columnas.values()))], nombre, cfg.dias)
    return df, meta


def _ultimo_habil(d: date) -> date:
    f = pd.Timestamp(d) - pd.Timedelta(days=1)
    return _HABIL.rollback(f).date()


def _pedir_json_diario(res: base.Resultado, f: date, item: str) -> tuple[dict | None, str | None]:
    """JSON diario de CBOE de la fecha ``f`` -> (json, Last-Modified); (None, motivo) si no hay fichero.

    El CDN (S3) responde 403 cuando no existe el objeto: es «sin fichero», no un bloqueo (las demás
    fechas responden 200 con las mismas cabeceras).
    """
    url = CBOE_PC_DIARIO.format(fecha=f.isoformat())
    t0 = base.cronometro()
    try:
        r = get(url, timeout=30, reintentos=1)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        status = getattr(getattr(e, "response", None), "status_code", None)
        sin_fichero = status in (403, 404) or " 403" in str(e) or " 404" in str(e)
        res.registrar("cboe_putcall", item + (" (sin fichero)" if sin_fichero else ""), url, sin_fichero,
                      dia=f.isoformat(), segundos=base.segundos(t0),
                      error="" if sin_fichero else f"{type(e).__name__}: {e}")
        return None, "sin fichero (403/404)" if sin_fichero else f"fallo: {type(e).__name__}"
    res.registrar("cboe_putcall", item, url, True, dia=f.isoformat(), bytes=len(r.content), segundos=base.segundos(t0))
    base.escribir_atomico(base.corpus("crudo", MODULO) / f"cboe_daily_options_{f.isoformat()}.json", r.content)
    return j, r.headers.get("Last-Modified")


def _sondear_putcall_diario(cfg: base.ConfigSonda, res: base.Resultado, csv_total: pd.DataFrame) -> list[dict]:
    """Put/call en los días testigo: del CSV si lo cubre; si no, del JSON diario (y búsqueda de su inicio)."""
    fin_csv = csv_total.index.max().date() if not csv_total.empty else date(2019, 10, 4)
    dias: list[tuple[str, date]] = []
    for d in cfg.dias:
        dias.append((d.clave, _ultimo_habil(d.fecha) if d.clave == "hoy" else d.fecha))
    filas = []
    n_json = 0
    for clave, f in dias:
        fila = dict(dia=clave, fecha=f.isoformat(), origen=None, total=None, equity=None, index=None, vix=None,
                    last_modified=None, lag_h=None)
        if f <= fin_csv:
            ts = pd.Timestamp(f)
            if ts in csv_total.index:
                fila.update(origen="csv", total=float(csv_total.at[ts, "P/C Ratio"]))
            filas.append(fila)
            continue
        if cfg.tope and n_json >= cfg.tope:
            continue
        n_json += 1
        j, lm = _pedir_json_diario(res, f, f"json diario {clave}")
        if j is None:
            fila["origen"] = lm
        else:
            rat = leer_cboe_diario(j)
            lm_dt = email.utils.parsedate_to_datetime(lm) if lm else None
            cierre = pd.Timestamp(disponible_cierre(pd.Series([pd.Timestamp(f)])).iloc[0])
            fila.update(origen="json_diario", last_modified=lm, lag_h=base.lag_horas(cierre, lm_dt),
                        **{k: rat.get(k) for k in ("total", "equity", "index", "vix")})
            _tg(res, "cboe_putcall", f"json diario {clave}", cierre, lm_dt, "Last-Modified (HTTP) del JSON diario")
        filas.append(fila)
    # Inicio del JSON diario: búsqueda binaria entre el fin del CSV y el primer día testigo con JSON.
    ok = [pd.Timestamp(x["fecha"]) for x in filas if x["origen"] == "json_diario"]
    if ok and not cfg.modo_rapido:
        habiles = pd.bdate_range(pd.Timestamp(fin_csv) + pd.Timedelta(days=1), min(ok), freq=_HABIL)
        lo, hi = 0, len(habiles) - 1        # habiles[hi] tiene fichero
        while lo < hi:
            mid = (lo + hi) // 2
            j, _ = _pedir_json_diario(res, habiles[mid].date(), "búsqueda del inicio del json")
            lo, hi = (lo, mid) if j is not None else (mid + 1, hi)
        filas.append(dict(dia="inicio_json", fecha=habiles[hi].date().isoformat(), origen="json_diario (primer día)",
                          total=None, equity=None, index=None, vix=None, last_modified=None, lag_h=None))
    return filas


# --------------------------------------------------------------------------- sondear

def _rango(res_f: list[dict], fuente: str) -> tuple[str | None, str | None]:
    fs = [r for r in res_f if r["fuente"] == fuente]
    return (min(r["inicio"] for r in fs), max(r["fin"] for r in fs)) if fs else (None, None)


def _lag_med(res: base.Resultado, fuente: str) -> float | None:
    """Mediana del retraso MEDIDO de la serie más frecuente (sin reglas, mensuales ni ficheros regenerados)."""
    return base.resumen_lag(t["lag_h"] for t in res.timegate
                            if t["fuente"] == fuente and not t["campo_fecha"].startswith("regla")
                            and "regenerado" not in t["campo_fecha"]
                            and not any(k in t["item"] for k in ("mensual", "global")))["mediana_h"]


def sondear(cfg: base.ConfigSonda | None = None) -> base.Resultado:
    cfg = cfg or base.ConfigSonda()
    res = base.Resultado(MODULO)
    hoy = date.today().isoformat()
    larga_todas: list[pd.DataFrame] = []
    resumen_f: list[dict] = []
    med: dict[str, dict] = {}
    errores: dict[str, str] = {}
    t_fuente: dict[str, float] = {}

    for fuente, fn in (("gpr", _sondear_gpr), ("epu", _sondear_epu)):
        t0 = base.cronometro()
        cfg.log(f"[indices] {fuente}…")
        try:
            med[fuente] = fn(cfg, res, hoy, larga_todas, resumen_f)
        except Exception as e:  # noqa: BLE001
            errores[fuente] = f"{type(e).__name__}: {str(e)[:200]}"
        t_fuente[fuente] = base.segundos(t0)

    cboe = [("cboe_skew", CBOE_SKEW, "SKEW_History.csv", "SKEW", {"SKEW": "skew"}, "indice", "cboe_skew"),
            ("cboe_vix", CBOE_VIX, "VIX_History.csv", "VIX", {"CLOSE": "vix_cierre", "HIGH": "vix_maximo"}, "puntos",
             "cboe_vix")]
    for fuente, url, fichero, item, cols, unidad, nombre in cboe:
        t0 = base.cronometro()
        cfg.log(f"[indices] {fuente}…")
        try:
            df, meta = _serie_cboe(cfg, res, hoy, url, fichero, fuente, item, cols, unidad, nombre, larga_todas,
                                   resumen_f)
            med[fuente] = dict(filas=len(df), inicio=df.index.min().date().isoformat(),
                               fin=df.index.max().date().isoformat(), last_modified=meta.get("last_modified"),
                               bytes=meta["bytes"])
        except Exception as e:  # noqa: BLE001
            errores[fuente] = f"{type(e).__name__}: {str(e)[:200]}"
        t_fuente[fuente] = base.segundos(t0)

    # Put/call: CSV históricos (hasta 2019) + JSON diario en los días testigo.
    t0 = base.cronometro()
    cfg.log("[indices] cboe_putcall…")
    med["cboe_putcall"] = {}
    csv_total = pd.DataFrame()
    for k, url in CBOE_PC.items():
        try:
            df, meta = _serie_cboe(cfg, res, hoy, url, f"{k}pc.csv", "cboe_putcall", f"{k}pc.csv",
                                   {"P/C Ratio": f"pc_{k}_ratio", df_col(k, "call"): f"pc_{k}_calls",
                                    df_col(k, "put"): f"pc_{k}_puts"}, "ratio", f"cboe_putcall_{k}",
                                   larga_todas, resumen_f)
            med["cboe_putcall"][k] = dict(filas=len(df), inicio=df.index.min().date().isoformat(),
                                          fin=df.index.max().date().isoformat(), columnas=list(df.columns),
                                          last_modified=meta.get("last_modified"))
            if k == "total":
                csv_total = df
        except Exception as e:  # noqa: BLE001
            med["cboe_putcall"][k] = f"{type(e).__name__}: {str(e)[:150]}"
    try:
        filas_pc = _sondear_putcall_diario(cfg, res, csv_total)
        res.tablas["cboe_putcall_diario"] = pd.DataFrame(filas_pc)
        med["cboe_putcall"]["json_diario"] = [{k: f.get(k) for k in ("dia", "fecha", "origen", "total", "lag_h")}
                                             for f in filas_pc]
    except Exception as e:  # noqa: BLE001
        errores["cboe_putcall"] = f"{type(e).__name__}: {str(e)[:200]}"
    med["cboe_putcall"]["robots_pagina_diaria"] = _robots(CBOE_PC_PAGINA)
    med["cboe_putcall"]["robots_cdn"] = _robots(CBOE_PC_DIARIO.format(fecha="2025-04-07"))
    t_fuente["cboe_putcall"] = base.segundos(t0)

    # Tablas comunes del módulo.
    if resumen_f:
        res.tablas["indices_resumen"] = pd.DataFrame(resumen_f)
    if larga_todas:
        todas = pd.concat([base.serie_larga(x) for x in larga_todas], ignore_index=True)
        res.tablas["indices_testigo"] = pd.DataFrame(tabla_testigo(todas, cfg.dias))
        res.tablas["indices_mensual"] = tabla_mensual(todas)

    _cobertura_y_fichas(res, med, errores, resumen_f, t_fuente)
    return res


def df_col(k: str, lado: str) -> str:
    """Nombre de columna de volumen en los CSV de put/call (``totalpc`` usa CALLS/PUTS; el resto CALL/PUT)."""
    c = lado.upper()
    return c + "S" if k == "total" else c


def _robots(url: str) -> bool | str:
    try:
        return robots_permite(url)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}"


def _cobertura_y_fichas(res: base.Resultado, med: dict, errores: dict, resumen_f: list[dict],
                        t_fuente: dict[str, float]) -> None:
    cfgs = {
        "gpr": dict(nombre="Geopolitical Risk Index (Caldara e Iacoviello, 2022)", url=GPR_WEB, acceso="descarga",
                    agentes=["macro", "risk"], formato="xls (+ vintages xls en GitHub)", idiomas="en",
                    licencia="libre para investigación citando Caldara e Iacoviello (2022, AER)",
                    frecuencia="diaria (desde 1985) y mensual (desde 1900; 44 países)",
                    limites="servidor personal (la sonda de 2026-10-06 tardó ≈700 s; hoy ≈2 s); API de GitHub 60 pet./h sin clave"),
        "epu": dict(nombre="Economic Policy Uncertainty (Baker, Bloom y Davis, 2016)", url=EPU_WEB, acceso="descarga",
                    agentes=["macro", "risk"], formato="csv / xlsx", idiomas="en",
                    licencia="libre citando a Baker, Bloom y Davis (2016, QJE) y Davis (2016) para el global",
                    frecuencia="diaria (EE. UU.) y mensual (global)", limites="sin clave ni límites conocidos"),
        "cboe_skew": dict(nombre="CBOE SKEW Index", url="https://www.cboe.com/tradable_products/vix/skew/",
                          acceso="descarga", agentes=["risk"], formato="csv", idiomas="",
                          licencia="datos de CBOE para conveniencia del visitante, sin garantía; términos de CBOE",
                          frecuencia="diaria (hábiles)", limites="sin clave"),
        "cboe_vix": dict(nombre="CBOE VIX (histórico diario)", url="https://www.cboe.com/tradable_products/vix/",
                         acceso="descarga", agentes=["risk"], formato="csv", idiomas="",
                         licencia="datos de CBOE para conveniencia del visitante, sin garantía; términos de CBOE",
                         frecuencia="diaria (hábiles)", limites="sin clave"),
        "cboe_putcall": dict(nombre="CBOE ratios put/call (total, equity, index)",
                             url="https://www.cboe.com/us/options/market_statistics/historical_data/",
                             acceso="descarga", agentes=["risk"], formato="csv (hasta 2019) + json diario", idiomas="",
                             licencia="datos de CBOE para conveniencia del visitante, sin garantía; términos de CBOE",
                             frecuencia="diaria (hábiles)",
                             limites="CSV históricos congelados; después, un JSON por día (≈1.750 peticiones para "
                                     "rehacer 2019-2026); robots.txt de cdn.cboe.com responde 403 (S3)"),
    }
    for fuente, c in cfgs.items():
        ini, fin = _rango(resumen_f, fuente)
        est = base.estado(res.registro, fuente)
        m = med.get(fuente, {})
        notas: list[str] = []
        if fuente in errores:
            notas.append(f"error: {errores[fuente]}")
            est = "fallo" if est == "no_probada" else ("parcial" if est == "ok" else est)
        series_f = [r for r in resumen_f if r["fuente"] == fuente]
        nota_cob = ""
        if fuente == "gpr":
            v = m.get("vintages", {})
            nota_cob = (f"vintages diarios desde {v.get('primer_diario')} (cada {v.get('dias_entre_vintages_diarios_ult_ano')} "
                        f"días el último año); retraso medido en testigo: {v.get('lag_testigo_dias')}")
            notas += ["La web documenta: mensual 'updated at the beginning of each month'; diario 'updated every "
                      "Monday' (festivo federal -> siguiente hábil). Los datos recientes son preliminares y se revisan.",
                      nota_cob,
                      "Revisiones: el GPRD diario del primer vintage difiere del de hoy un "
                      f"{[x.get('revision_pct') for x in v.get('lag_testigo_dias', []) if x.get('tipo') == 'diario']} %"
                      " en los días testigo; el dato del propio lunes ya sale ese lunes (preliminar). La regla del "
                      "lunes SIGUIENTE es conservadora y la serie guardada es el vintage de hoy (revisado)."]
        elif fuente == "epu":
            gm = m.get("global_mensual", {})
            nota_cob = (f"global: último mes {gm.get('fin')} (Last-Modified {gm.get('last_modified')}; "
                        f"{gm.get('dias_desde_fin_ultimo_mes')} días desde su fin a hoy)")
            notas += ["Diario EE. UU. muy ruidoso: suavizar (7-30 días) antes de usar.", nota_cob,
                      "Regla del global (fin de m+2) es una aproximación: la publicación es irregular."]
        elif fuente == "cboe_putcall":
            fins = {k: v.get("fin") for k, v in m.items() if isinstance(v, dict) and "fin" in v}
            js = m.get("json_diario", [])
            ok_js = [x for x in js if x.get("origen") == "json_diario"]
            ini_js = next((x["fecha"] for x in js if x.get("dia") == "inicio_json"), None)
            nota_cob = (f"CSV terminan en {sorted(set(fins.values()))}; JSON diario desde {ini_js} "
                        f"(días testigo con JSON: {[x['dia'] for x in ok_js]}); sin solape CSV/JSON para validar el empalme")
            notas += [nota_cob, "Después de 2019-10-04 no hay CSV histórico público: la página diaria de "
                      "estadísticas (robots.txt de www.cboe.com la permite) carga un JSON por fecha; reconstruir "
                      "la serie exige una petición por día hábil (no se hace en la sonda)."]
            if est == "ok":
                est = "parcial"   # cobertura cortada en 2019 para la serie descargable entera
        elif fuente in ("cboe_skew", "cboe_vix"):
            nota_cob = f"{len(series_f)} series; huecos (> 4 días): {[r['huecos'] for r in series_f]}"
            notas.append("Referencia de mercado para la validez de los alternativos (VIX)." if fuente == "cboe_vix" else
                         "Riesgo de cola; la misma serie se descarga también en el Tema 1 (mercados).")
            notas.append(f"Huecos > 4 días naturales (cierres excepcionales, p. ej. 11-S): {nota_cob}")
        lag = _lag_med(res, fuente)
        res.cobertura.append(dict(
            fuente=fuente, familia="indices_ews", tipo="serie", inicio=ini, fin=fin, hora=False, idiomas=c["idiomas"],
            lag_mediana_h=lag, destino="serie", backtest=est in ("ok", "parcial"), estado=est, nota=nota_cob[:300]))
        n_obs = sum(r["n"] for r in series_f)
        bytes_f = sum(r["bytes"] for r in res.registro if r["fuente"] == fuente)
        res.fichas.append(base.ficha(
            fuente=fuente, nombre=c["nombre"], agentes=c["agentes"], rag="serie", url=c["url"], acceso=c["acceso"],
            licencia_y_terminos=c["licencia"], cobertura_inicio=ini, cobertura_fin=fin, frecuencia=c["frecuencia"],
            retraso_publicacion="; ".join(sorted({f"{r['fichero']}: {r['lag_publicacion']}" for r in series_f}))
            + (f" | medido (mediana): {lag} h" if lag is not None else ""),
            formato=c["formato"], volumen_estimado=f"{bytes_f / 1e6:.2f} MB en esta sonda; {n_obs} observaciones",
            limites=c["limites"], estado_sonda=est,
            mediciones=dict(m, series=[{k: r[k] for k in ("serie", "inicio", "fin", "n", "huecos")} for r in series_f],
                            segundos=t_fuente.get(fuente)),
            notas=notas))
