"""OCDE: indicadores de corto plazo con ediciones (CLI, IPC y paro) del Tema 1.

Fuente principal: ``sdmx.oecd.org`` (SDMX REST, CSV). El conjunto ``DF_STES_REVISIONS``
(``OECD.SDD.STES``, versión 4.0) guarda cada edición mensual de los indicadores de corto plazo
(dimensión ``EDITION`` = ``AAAAMM``): son vintages reales. Clave: ``REF_AREA.FREQ.MEASURE.
UNIT_MEASURE.ACTIVITY.EDITION`` (p. ej. ``USA.M.LI_AA...``). Medidas usadas (``MEDIDAS``): ``LI_AA``
(CLI con amplitud ajustada), ``CP`` (IPC) y ``UNEMP`` (paro). Áreas (``AREAS``): EE. UU., zona euro
(``EA20``, composición cambiante de la OCDE), G7 y China.

Vintage y fecha (conservadoras): ``vintage`` = ``AAAA-MM`` de la edición; el dato se da por público
el ÚLTIMO día del mes de la edición a las 23:59 de París (publicador ``oecd_edicion`` del
fragmento): la OCDE publica a mitad de mes, así que nunca se adelanta. Solo valores nuevos o
revisados entre ediciones consecutivas (``b4_comun.compactar_vintages``).

Plan B (``descargar_dbnomics``): si ``sdmx.oecd.org`` responde 403 o un reto anti-bot no se
esquiva: se usa DBnomics (proveedor ``OECD``, conjunto ``DSD_STES@DF_CLI``), que solo da la ÚLTIMA
edición: se guarda como vintage ``AAAA-MM`` del mes de descarga (fin de mes, conservador) con la nota
de revisión. Sonda del 2026-10-13: ``sdmx.oecd.org`` respondió 200 (sin bloqueo); DBnomics también.

CLI: ``python -m ingesta.mercados_series.fuentes.oecd [--areas USA ...] [--medidas LI_AA ...]
[--desde 2000-01] [--descargar] [--ficha]`` (reanudable: un CSV por área y medida en el corpus;
``--forzar`` vuelve a pedirlo).
"""
from __future__ import annotations

import argparse
import io
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.http import get
from ingesta.mercados_series.fuentes.b4_comun import (
    TEMA,
    compactar_vintages,
    guardar_crudo,
    ruta_crudo,
    tabla_contrato,
)

FUENTE = "oecd"
PUBLICADOR = "oecd_edicion"
URL_SDMX = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,4.0/"
            "{area}.M.{medida}...")
URL_DBNOMICS = "https://api.db.nomics.world/v22/series/OECD/DSD_STES@DF_CLI"
AREAS = ("USA", "EA20", "DEU", "FRA", "ITA", "GBR", "JPN", "CAN", "CHN")
MEDIDAS = {"LI_AA": "índice (100 = tendencia de largo plazo)", "CP": "índice (año base de la edición)",
           "UNEMP": "% de la población activa"}
NOTA_EDICION = "OCDE, edición {edicion}: fecha conservadora = fin del mes de la edición"
NOTA_DBNOMICS = ("DBnomics (última edición de la OCDE): vintage = mes de descarga; valores del último vintage: "
                 "posible fuga de revisiones")

__all__ = ["FUENTE", "AREAS", "MEDIDAS", "url_sdmx", "leer_csv_revisiones", "tabla_oecd", "descargar",
           "leer_dbnomics", "tabla_dbnomics", "descargar_dbnomics", "ficha_oecd"]


def url_sdmx(area: str, medida: str) -> str:
    return URL_SDMX.format(area=area, medida=medida)


def _ruta(area: str, medida: str) -> Path:
    return ruta_crudo(FUENTE) / f"revisions_{area}_{medida}.csv"


def leer_csv_revisiones(fuente: Path | str | bytes) -> pd.DataFrame:
    """CSV SDMX de ``DF_STES_REVISIONS`` → ``area, medida, unidad_sdmx, actividad, edicion, periodo, valor``.

    Solo frecuencia mensual (``TIME_PERIOD`` = ``AAAA-MM``); valores no numéricos fuera.
    """
    if isinstance(fuente, bytes):
        fuente = io.BytesIO(fuente)
    d = pd.read_csv(fuente, dtype=str)
    d = d[d["FREQ"] == "M"]
    out = pd.DataFrame({
        "area": d["REF_AREA"], "medida": d["MEASURE"], "unidad_sdmx": d["UNIT_MEASURE"], "actividad": d["ACTIVITY"],
        "edicion": pd.to_datetime(d["EDITION"].str.strip(), format="%Y%m"),
        "periodo": pd.to_datetime(d["TIME_PERIOD"].str.strip(), format="%Y-%m"),
        "valor": pd.to_numeric(d["OBS_VALUE"], errors="coerce")})
    return out.dropna(subset=["valor"]).reset_index(drop=True)


def _filtrar_medida(d: pd.DataFrame, medida: str) -> pd.DataFrame:
    """Una sola variante por medida: CLI en índice (no tasa), IPC total, paro total."""
    d = d[d["medida"] == medida]
    if medida == "LI_AA":
        d = d[d["unidad_sdmx"] == "IX"]
    if d["actividad"].nunique() > 1:
        preferida = "_T" if (d["actividad"] == "_T").any() else ("_Z" if (d["actividad"] == "_Z").any()
                                                                  else d["actividad"].iloc[0])
        d = d[d["actividad"] == preferida]
    if d["unidad_sdmx"].nunique() > 1:
        unidad = "IX" if (d["unidad_sdmx"] == "IX").any() else d["unidad_sdmx"].mode().iloc[0]
        d = d[d["unidad_sdmx"] == unidad]
    return d


def tabla_oecd(area: str, medida: str, crudo: pd.DataFrame, *,
               publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Filas del contrato de ``oecd:<medida>.<area>`` (una por valor nuevo o revisado en cada edición)."""
    d = _filtrar_medida(crudo[crudo["area"] == area], medida)
    largo = pd.DataFrame({"periodo": d["periodo"], "orden": d["edicion"], "valor": d["valor"]})
    largo = compactar_vintages(largo)
    fin_mes = largo["orden"] + pd.offsets.MonthEnd(0)
    largo = largo[largo["periodo"] <= fin_mes]
    fin_mes = largo["orden"] + pd.offsets.MonthEnd(0)
    edicion = largo["orden"].dt.strftime("%Y-%m").reset_index(drop=True)
    return tabla_contrato(
        f"oecd:{medida}.{area}", largo["periodo"], fin_mes, largo["valor"], vintage=edicion,
        unidad=MEDIDAS[medida], divisa=None, fuente=FUENTE, metodo="vintage_exacto", publicador=PUBLICADOR,
        nota=edicion.map(lambda e: NOTA_EDICION.format(edicion=e)), publicaciones=publicaciones)


def descargar(areas: Iterable[str] = AREAS, medidas: Iterable[str] = tuple(MEDIDAS), *, desde: str | None = None,
              forzar: bool = False, limite_segundos: float = 420.0) -> dict[str, dict[str, Any]]:
    """Descarga ``DF_STES_REVISIONS`` por área y medida (reanudable: un CSV por par en el corpus).

    Un 403 o un reto anti-bot se anota (``estado`` = ``bloqueado``) y se sigue: no se esquiva.
    ``limite_segundos`` corta la tanda entre peticiones (comandos cortos).
    """
    inicio = time.monotonic()
    informe: dict[str, dict[str, Any]] = {}
    for area in areas:
        for medida in medidas:
            clave = f"{area}.{medida}"
            ruta = _ruta(area, medida)
            if ruta.exists() and not forzar:
                informe[clave] = {"estado": "en_cache", "bytes": ruta.stat().st_size}
                continue
            if time.monotonic() - inicio > limite_segundos:
                informe[clave] = {"estado": "pendiente_tiempo"}
                continue
            params = {"format": "csvfile"}
            if desde:
                params["startPeriod"] = desde
            t = time.monotonic()
            try:
                r = get(url_sdmx(area, medida), params=params, timeout=120, reintentos=2)
            except Exception as e:  # noqa: BLE001 — se documenta y se sigue
                txt = str(e)
                estado = "bloqueado" if ("403" in txt or "cloudflare" in txt.lower()) else (
                    "sin_datos" if "404" in txt else "fallo")
                informe[clave] = {"estado": estado, "error": txt[:200]}
                continue
            guardar_crudo(FUENTE, ruta.name, r.content)
            informe[clave] = {"estado": "ok", "bytes": len(r.content), "segundos": round(time.monotonic() - t, 2)}
    return informe


def cargar_crudo(area: str, medida: str) -> pd.DataFrame | None:
    ruta = _ruta(area, medida)
    return leer_csv_revisiones(ruta) if ruta.exists() and ruta.stat().st_size > 0 else None


# --- Plan B: DBnomics (solo la última edición) ----------------------------------------------------
def leer_dbnomics(doc: dict[str, Any]) -> pd.DataFrame:
    """JSON de ``/v22/series/OECD/...`` → ``serie, area, periodo, valor`` (``'NA'`` fuera)."""
    filas = []
    for s in (doc.get("series") or {}).get("docs", []):
        area = (s.get("dimensions") or {}).get("REF_AREA")
        for p, v in zip(s.get("period", []), s.get("value", [])):
            if v is None or v == "NA":
                continue
            filas.append({"serie": s.get("series_code"), "area": area, "periodo": pd.Timestamp(p), "valor": float(v)})
    return pd.DataFrame(filas, columns=["serie", "area", "periodo", "valor"])


def tabla_dbnomics(area: str, d: pd.DataFrame, descarga: Any, *, publicaciones: dict[str, Any] | None = None,
                   serie_dbnomics: str | None = None) -> pd.DataFrame:
    """Filas de ``oecd:LI_AA.<area>`` con la edición de DBnomics como vintage del mes de ``descarga``."""
    g = d[d["area"] == area]
    if serie_dbnomics:
        g = g[g["serie"] == serie_dbnomics]
    fin = pd.Timestamp(descarga).normalize() + pd.offsets.MonthEnd(0)
    g = g[g["periodo"] <= fin]
    return tabla_contrato(f"oecd:LI_AA.{area}", g["periodo"], [fin] * len(g), g["valor"],
                          vintage=fin.strftime("%Y-%m"), unidad=MEDIDAS["LI_AA"], divisa=None, fuente=FUENTE,
                          metodo="vintage_exacto", publicador=PUBLICADOR, nota=NOTA_DBNOMICS,
                          publicaciones=publicaciones)


def descargar_dbnomics(areas: Iterable[str] = AREAS) -> dict[str, Any]:
    """CLI de la OCDE vía DBnomics (plan B). Guarda el JSON crudo por área."""
    informe: dict[str, Any] = {}
    for area in areas:
        dims = json.dumps({"REF_AREA": [area], "MEASURE": ["LI"], "ADJUSTMENT": ["AA"], "FREQ": ["M"]})
        try:
            r = get(URL_DBNOMICS, params={"dimensions": dims, "observations": 1, "limit": 50}, timeout=90)
        except Exception as e:  # noqa: BLE001
            informe[area] = {"estado": "fallo", "error": str(e)[:200]}
            continue
        guardar_crudo(FUENTE, f"dbnomics_cli_{area}.json", r.content)
        informe[area] = {"estado": "ok", "bytes": len(r.content)}
    return informe


# --- Ficha ----------------------------------------------------------------------------------------
def ficha_oecd(areas: Iterable[str] = AREAS, medidas: Iterable[str] = tuple(MEDIDAS),
               informe: dict[str, dict] | None = None, *, guardar: bool = True):
    """Ficha ``oecd``: ediciones, cobertura y retraso (edición → último periodo) medidos, sin valores."""
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    med: dict[str, Any] = {}
    for area in areas:
        for medida in medidas:
            crudo = cargar_crudo(area, medida)
            if crudo is None or crudo.empty:
                med[f"{area}.{medida}"] = {"estado": (informe or {}).get(f"{area}.{medida}", {}).get("estado", "sin_datos")}
                continue
            d = _filtrar_medida(crudo, medida)
            if d.empty:
                med[f"{area}.{medida}"] = {"estado": "sin_datos"}
                continue
            ult = d.groupby("edicion")["periodo"].max()
            meses = (ult.index.year - ult.dt.year) * 12 + (ult.index.month - ult.dt.month)
            tabla = tabla_oecd(area, medida, crudo)
            med[f"{area}.{medida}"] = {
                "primera_edicion": d["edicion"].min().strftime("%Y-%m"),
                "ultima_edicion": d["edicion"].max().strftime("%Y-%m"), "n_ediciones": int(d["edicion"].nunique()),
                "periodo_inicio": d["periodo"].min().strftime("%Y-%m"), "periodo_fin": d["periodo"].max().strftime("%Y-%m"),
                "meses_edicion_a_ultimo_periodo_mediana": float(meses.median()),
                "filas_crudo": int(len(d)), "filas_contrato": int(len(tabla))}
    ok = {k: v for k, v in med.items() if "n_ediciones" in v}
    bytes_crudo = sum(p.stat().st_size for p in ruta_crudo(FUENTE).glob("revisions_*.csv"))
    ficha = FichaFuente(
        fuente=FUENTE, nombre="OCDE: ediciones de indicadores de corto plazo (DF_STES_REVISIONS: CLI, IPC, paro)",
        tema=TEMA, agentes=["macro"], rag="serie", url="https://sdmx.oecd.org/public/rest/",
        acceso="api", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="OCDE: CC BY 4.0 (términos de la OCDE); citar la fuente",
        cobertura_inicio=min((v["periodo_inicio"] for v in ok.values()), default=None),
        cobertura_fin=max((v["periodo_fin"] for v in ok.values()), default=None),
        frecuencia="mensual (una edición al mes)",
        retraso_publicacion=("medido: meses de la edición al último periodo publicado (mediana): "
                             + "; ".join(f"{k} {v['meses_edicion_a_ultimo_periodo_mediana']:g}" for k, v in ok.items())
                             + ". Fecha conservadora: fin del mes de la edición, 23:59 París"),
        formato="csv (SDMX)",
        volumen_estimado=f"{bytes_crudo / 1e6:.1f} MB de CSV crudo para {len(ok)} pares área-medida",
        limites=("sin clave; límite de peticiones por IP no documentado (aquí 1/s); sonda 2026-10-13: 200 sin "
                 "Cloudflare. Plan B DBnomics (solo última edición) si devuelve 403"),
        estado_sonda="ok" if ok and len(ok) == len(med) else ("parcial" if ok else "fallo"),
        mediciones={"pares": med, "descarga": informe or {}, "bytes_crudo": bytes_crudo},
        notas=["vintage = edición AAAA-MM; fecha_publicacion = primera decisión tras el fin del mes de la edición",
               "Solo valores nuevos o revisados entre ediciones consecutivas",
               "EA20: composición cambiante según la OCDE (zona euro a 21 en 2026: comprobar el código del área)",
               "IPC de Japón: se toma de aquí (el Banco de Japón no publica el IPC; lo hace el MIC)",
               "Crudo en <corpus>/mercados_series/crudo/oecd/"])
    ruta = guardar_ficha(ficha) if guardar else None
    return ficha, ruta


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--areas", nargs="*", default=list(AREAS))
    p.add_argument("--medidas", nargs="*", default=list(MEDIDAS))
    p.add_argument("--desde", default=None, help="startPeriod AAAA-MM")
    p.add_argument("--descargar", action="store_true")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--limite-segundos", type=float, default=420.0)
    p.add_argument("--dbnomics", action="store_true", help="plan B: CLI de DBnomics (solo última edición)")
    p.add_argument("--ficha", action="store_true")
    a = p.parse_args(argv)
    informe = None
    if a.descargar:
        informe = descargar(a.areas, a.medidas, desde=a.desde, forzar=a.forzar, limite_segundos=a.limite_segundos)
        for k, v in informe.items():
            print(k, v.get("estado"), v.get("bytes"))
    if a.dbnomics:
        print(descargar_dbnomics(a.areas))
    if a.ficha:
        _, ruta = ficha_oecd(a.areas, a.medidas, informe)
        print("ficha:", ruta)


if __name__ == "__main__":
    main()
