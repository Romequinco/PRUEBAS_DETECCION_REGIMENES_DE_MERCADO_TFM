"""FMI: World Economic Outlook por EDICIÓN (vintages ``AAAA-MM``), bloque 4 del Tema 1.

Orígenes:

- DBnomics (``api.db.nomics.world/v22``, proveedor ``IMF``, dataset ``WEO:<AAAA-MM>``): ediciones
  2008-04 … 2025-04 (falta la de septiembre de 2011: DBnomics no la tiene). Una petición por edición
  con los países y conceptos filtrados (``dimensions``). Medido 2026-10-13: el robots.txt de
  api.db.nomics.world es ``Disallow: /`` y ``robots_permite`` lo respeta, así que estas ediciones
  quedan como fallo documentado en la ficha (el código queda listo si DBnomics lo permite).
- API SDMX 3.0 del FMI (``api.imf.org/external/sdmx/3.0``, flujo ``IMF.RES/WEO``): solo sirve las
  versiones vigentes del flujo (medido 2026-10-13: 9.0.0 y 6.0.0). La edición se deduce del atributo
  ``COUNTRY_UPDATE_DATE`` (mes <= 6 → ``AAAA-04``; si no, ``AAAA-10``). Las ediciones intermedias que
  ninguna API sirve quedan pendientes (se declara en la ficha).

Conceptos: PIB real (``NGDP_RPCH``), inflación media (``PCPIPCH``), paro (``LUR``), saldo fiscal
(``GGXCNL_NGDP``, % del PIB) y deuda bruta (``GGXWDG_NGDP``, % del PIB) de EE. UU., G7 y China
(la zona euro está en el WEO de agregados: pendiente). Ids ``imf:WEO.<ISO3>.<concepto>``;
``fecha_referencia`` = 1 de enero del año.

Publicación: ``vintage`` = edición; ``fecha_publicacion`` = día de publicación de la edición
(``FECHAS_EDICION``, documentadas por las notas de prensa del FMI; ``verificado: false``) a las
09:00 de Washington (``disponible_utc``). Medición: ``indexed_at`` de DBnomics (cuándo la indexó)
debe ser posterior; si no, la ficha lo señala. Cada edición conserva todos sus años: las
proyecciones (años posteriores al de publicación) se descartan porque el contrato no admite filas
publicadas antes de su periodo; el año en curso (estimación) sí se conserva con nota.

CLI: ``python -m ingesta.mercados_series.fuentes.imf [--ediciones 2019-10 2020-04] [--sin-sdmx]``.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.mercados_series.fuentes import _organismos as org

FUENTE = "imf_weo"
URL_DBNOMICS = "https://api.db.nomics.world/v22/series/IMF/WEO:{edicion}"
URL_SDMX = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.RES/WEO/{version}/{clave}"
VERSIONES_SDMX = ("9.0.0", "6.0.0")
PAISES = ("USA", "DEU", "FRA", "ITA", "GBR", "JPN", "CAN", "CHN")
CONCEPTOS = {
    "NGDP_RPCH": ("PIB real, variación anual", "% anual"),
    "PCPIPCH": ("Inflación (IPC medio), variación anual", "% anual"),
    "LUR": ("Tasa de paro", "% de la población activa"),
    "GGXCNL_NGDP": ("Saldo de las AA. PP. (capacidad/necesidad de financiación)", "% del PIB"),
    "GGXWDG_NGDP": ("Deuda bruta de las AA. PP.", "% del PIB"),
}
HORA_PUBLICACION = "09:00"
HUSO_PUBLICACION = "America/New_York"
NOTA_ESTIMACION = "estimación del año en curso en la edición (no dato cerrado)"
# Día de publicación de cada edición (notas de prensa del FMI). verificado: false.
FECHAS_EDICION = {
    "2008-04": "2008-04-09", "2008-10": "2008-10-08", "2009-04": "2009-04-22", "2009-10": "2009-10-01",
    "2010-04": "2010-04-21", "2010-10": "2010-10-06", "2011-04": "2011-04-11", "2011-09": "2011-09-20",
    "2012-04": "2012-04-17", "2012-10": "2012-10-09", "2013-04": "2013-04-16", "2013-10": "2013-10-08",
    "2014-04": "2014-04-08", "2014-10": "2014-10-07", "2015-04": "2015-04-14", "2015-10": "2015-10-06",
    "2016-04": "2016-04-12", "2016-10": "2016-10-04", "2017-04": "2017-04-18", "2017-10": "2017-10-10",
    "2018-04": "2018-04-17", "2018-10": "2018-10-09", "2019-04": "2019-04-09", "2019-10": "2019-10-15",
    "2020-04": "2020-04-14", "2020-10": "2020-10-13", "2021-04": "2021-04-06", "2021-10": "2021-10-12",
    "2022-04": "2022-04-19", "2022-10": "2022-10-11", "2023-04": "2023-04-11", "2023-10": "2023-10-10",
    "2024-04": "2024-04-16", "2024-10": "2024-10-22", "2025-04": "2025-04-22", "2025-10": "2025-10-14",
    "2026-04": "2026-04-14", "2026-10": "2026-10-13",
}
EDICIONES_DBNOMICS = tuple(e for e in FECHAS_EDICION if e <= "2025-04" and e != "2011-09")


def id_serie(pais: str, concepto: str) -> str:
    return f"imf:WEO.{pais}.{concepto}"


def disponible(edicion: str) -> pd.Timestamp:
    """Instante UTC de publicación de la edición (09:00 de Washington del día documentado)."""
    if edicion not in FECHAS_EDICION:
        raise ValueError(f"edición {edicion!r} sin fecha de publicación documentada")
    local = pd.Timestamp(f"{FECHAS_EDICION[edicion]} {HORA_PUBLICACION}").tz_localize(HUSO_PUBLICACION)
    return local.tz_convert("UTC")


def edicion_de_fecha(texto: str) -> str:
    """``9/30/2026`` → ``2026-10``; ``3/20/2026`` → ``2026-04`` (atributo COUNTRY_UPDATE_DATE)."""
    t = pd.Timestamp(texto)
    return f"{t.year}-{'04' if t.month <= 6 else '10'}"


# --- Lectores (sin red) --------------------------------------------------------------------------
def leer_dbnomics(contenido: bytes | str | dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """JSON de ``/series/IMF/WEO:<ed>`` → ``pais, concepto, anio, valor`` y metadatos (indexed_at)."""
    j = contenido if isinstance(contenido, dict) else json.loads(contenido)
    filas, indexado = [], []
    for s in j["series"]["docs"]:
        dim = s.get("dimensions", {})
        pais, concepto = dim.get("weo-country"), dim.get("weo-subject")
        if s.get("indexed_at"):
            indexado.append(s["indexed_at"])
        for p, v in zip(s["period"], s["value"]):
            filas.append((pais, concepto, int(str(p)[:4]), v))
    df = pd.DataFrame(filas, columns=["pais", "concepto", "anio", "valor"])
    df["valor"] = pd.to_numeric(df["valor"].replace({"NA": None}), errors="coerce")
    meta = {"n_series": len(j["series"]["docs"]), "num_found": j["series"].get("num_found"),
            "indexed_at_min": min(indexado) if indexado else None}
    return df, meta


def leer_sdmx(contenido: bytes | str | dict) -> tuple[pd.DataFrame, str | None]:
    """SDMX-JSON 3.0 del FMI → ``pais, concepto, anio, valor`` y la edición (COUNTRY_UPDATE_DATE)."""
    j = contenido if isinstance(contenido, dict) else json.loads(contenido)
    st = j["data"]["structures"][0]
    dims = st["dimensions"]["series"]
    pos = {d["id"]: i for i, d in enumerate(dims)}
    anios = [int(v["value"][:4]) for v in st["dimensions"]["observation"][0]["values"]]
    attrs = [a["id"] for a in st["attributes"].get("series", [])]
    i_fecha = attrs.index("COUNTRY_UPDATE_DATE") if "COUNTRY_UPDATE_DATE" in attrs else None
    filas, fechas = [], []
    for clave, s in j["data"]["dataSets"][0]["series"].items():
        idx = [int(k) for k in clave.split(":")]
        pais = dims[pos["COUNTRY"]]["values"][idx[pos["COUNTRY"]]]["id"]
        concepto = dims[pos["INDICATOR"]]["values"][idx[pos["INDICATOR"]]]["id"]
        if i_fecha is not None and i_fecha < len(s.get("attributes", [])) and s["attributes"][i_fecha]:
            fechas.append(pd.Timestamp(s["attributes"][i_fecha]))
        for k, v in s.get("observations", {}).items():
            filas.append((pais, concepto, anios[int(k)], v[0] if v else None))
    df = pd.DataFrame(filas, columns=["pais", "concepto", "anio", "valor"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    edicion = edicion_de_fecha(str(max(fechas).date())) if fechas else None
    return df, edicion


def tabla_edicion(datos: pd.DataFrame, edicion: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filas del contrato de una edición: años <= año de publicación; el año en curso con nota."""
    disp = disponible(edicion)
    dia_pub = pd.Timestamp(FECHAS_EDICION[edicion])
    d = datos[datos["concepto"].isin(CONCEPTOS) & datos["pais"].isin(PAISES) & datos["valor"].notna()]
    d = d.drop_duplicates(["pais", "concepto", "anio"], keep="first")
    proyecciones = d["anio"] > dia_pub.year
    trozos = []
    for (pais, concepto), g in d[~proyecciones].groupby(["pais", "concepto"], sort=True):
        fechas = pd.to_datetime(g["anio"].astype(str) + "-01-01")
        nota = [NOTA_ESTIMACION if a == dia_pub.year else None for a in g["anio"]]
        trozos.append(org.filas(id_serie(pais, concepto), fechas.values, [dia_pub] * len(g), g["valor"].values,
                                vintage=edicion, unidad=CONCEPTOS[concepto][1], divisa=None, fuente=FUENTE,
                                metodo="vintage_exacto", nota=nota, disponible_utc=[disp] * len(g)))
    return org.unir(trozos), {"filas": int((~proyecciones).sum()), "proyecciones_descartadas": int(proyecciones.sum())}


# --- Descarga ------------------------------------------------------------------------------------
def descargar_dbnomics(edicion: str, *, forzar: bool = False, raiz: Path | None = None):
    params = {"dimensions": json.dumps({"weo-country": list(PAISES), "weo-subject": list(CONCEPTOS)}),
              "observations": 1, "limit": 1000}
    return org.descargar(FUENTE, f"dbnomics_WEO_{edicion}.json", URL_DBNOMICS.format(edicion=edicion),
                         params=params, forzar=forzar, raiz=raiz)


def descargar_sdmx(version: str, *, forzar: bool = False, raiz: Path | None = None):
    clave = f"{'+'.join(PAISES)}.{'+'.join(CONCEPTOS)}.A"
    return org.descargar(FUENTE, f"sdmx_WEO_{version}.json", URL_SDMX.format(version=version, clave=clave),
                         params={"dimensionAtObservation": "TIME_PERIOD", "attributes": "all"},
                         headers={"Accept": "application/json"}, forzar=forzar, raiz=raiz)


def tabla_weo(ediciones: Iterable[str] = EDICIONES_DBNOMICS, *, sdmx: bool = True, forzar: bool = False,
              raiz: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tabla del contrato con todas las ediciones (DBnomics + SDMX vigente) e informe por edición."""
    trozos, informe = [], {}
    for ed in ediciones:
        contenido, meta = descargar_dbnomics(ed, forzar=forzar, raiz=raiz)
        info = {"origen": "dbnomics", "error": meta.get("error"), "bytes": meta.get("bytes")}
        if contenido is not None:
            datos, m = leer_dbnomics(contenido)
            t, cuenta = tabla_edicion(datos, ed)
            info |= m | cuenta
            if m.get("indexed_at_min"):
                info["indexado_tras_publicacion"] = bool(pd.Timestamp(m["indexed_at_min"]) >= disponible(ed))
            trozos.append(t)
        informe[ed] = info
    if sdmx:
        vistas = {e for e in informe}
        for v in VERSIONES_SDMX:
            contenido, meta = descargar_sdmx(v, forzar=forzar, raiz=raiz)
            if contenido is None:
                informe[f"sdmx_{v}"] = {"origen": "sdmx", "error": meta.get("error")}
                continue
            datos, ed = leer_sdmx(contenido)
            if ed is None or ed in vistas or ed not in FECHAS_EDICION:
                informe[f"sdmx_{v}"] = {"origen": "sdmx", "edicion": ed, "descartada": "repetida o sin fecha"}
                continue
            t, cuenta = tabla_edicion(datos, ed)
            informe[ed] = {"origen": f"sdmx {v}", "bytes": meta.get("bytes"), **cuenta}
            vistas.add(ed)
            trozos.append(t)
    return org.unir(trozos), informe


def frecuencias(tabla: pd.DataFrame) -> dict[str, str]:
    return {sid: "anual" for sid in tabla["serie"].unique()}


def fragmento(ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Entradas del catálogo de las series del WEO (vintage por edición)."""
    out = {}
    for sid in sorted(ids):
        _, pais, concepto = sid.split(":", 1)[1].split(".")
        nombre, unidad = CONCEPTOS[concepto]
        out[sid] = {"alias": None, "nombre": f"WEO del FMI: {nombre} - {pais}", "fuente": FUENTE,
                    "id_origen": f"WEO/{pais}.{concepto}", "frecuencia": "anual", "unidad": unidad, "divisa": None,
                    "metodo": "vintage_exacto", "publicador": "imf_weo", "revisable": True, "rol_fase1": None,
                    "licencia": "FMI: uso libre con cita (términos de imf.org); DBnomics como espejo",
                    "estado": "activa", "agentes": ["macro"],
                    "notas": ["vintage = edición del WEO (AAAA-MM); proyecciones posteriores al año de la edición "
                              "descartadas (contrato)", "falta la edición de 2011-09 (DBnomics no la tiene)"],
                    "verificado": False}
    return out


def publicador_fragmento() -> dict[str, Any]:
    return {"imf_weo": {"hora": HORA_PUBLICACION, "huso": HUSO_PUBLICACION, "dia": "mismo",
                        "fuente": "https://www.imf.org/en/Publications/WEO", "verificado": False,
                        "notas": ["la fecha de cada edición está en fuentes.imf.FECHAS_EDICION (notas de prensa)"]}}


def ficha_weo(tabla: pd.DataFrame, informe: dict[str, Any], meta: pd.DataFrame) -> Path:
    med = org.resumen_medido(meta)
    for k in ("retraso_observado_dias_min", "retraso_observado_dias_mediana", "lag_aplicado_dias_mediana"):
        med.pop(k, None)   # sin sentido con vintages por edición (el año en curso aún no ha terminado)
    robots = sorted(e for e, i in informe.items() if "robots" in str(i.get("error")))
    ediciones = sorted(e for e in informe if e in FECHAS_EDICION and not informe[e].get("error")
                       and informe[e].get("filas"))
    faltan = sorted(set(FECHAS_EDICION) - set(ediciones))
    errores = {e: i["error"] for e, i in informe.items() if i.get("error")}
    sospechas = [e for e, i in informe.items() if i.get("indexado_tras_publicacion") is False]
    ficha = FichaFuente(
        fuente=FUENTE, nombre="FMI World Economic Outlook por edición (DBnomics + API SDMX del FMI)",
        tema=org.TEMA, agentes=["macro"], rag="serie", url="https://api.db.nomics.world/v22/series/IMF",
        acceso="api", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="FMI: reutilización con cita (copyright IMF); DBnomics redistribuye sin cambios",
        cobertura_inicio=med.get("cobertura_inicio"), cobertura_fin=med.get("cobertura_fin"),
        frecuencia="anual; dos ediciones al año (abril y octubre)",
        retraso_publicacion=("vintage exacto por edición: día de publicación documentado a las 09:00 de Washington; "
                             "año en curso como estimación; proyecciones descartadas"),
        formato="json (DBnomics) y SDMX-JSON 3.0 (api.imf.org)",
        volumen_estimado=f"{len(ediciones)} ediciones, {med.get('observaciones', 0)} filas, {med.get('series', 0)} series",
        limites="sin clave; DBnomics 1 petición/s; la API del FMI solo sirve las versiones vigentes del flujo",
        estado_sonda="fallo" if tabla.empty else ("parcial" if faltan or errores else "ok"),
        mediciones={**med, "ediciones": ediciones, "ediciones_pendientes": faltan, "errores": errores,
                    "ediciones_indexadas_antes_de_la_fecha_documentada": sospechas,
                    "proyecciones_descartadas": int(sum(i.get("proyecciones_descartadas", 0) for i in informe.values())),
                    "por_edicion": informe},
        notas=[*([f"DBnomics: robots.txt de api.db.nomics.world es 'Disallow: /' (medido {org.hoy_ny()}): "
                  f"{len(robots)} ediciones no descargadas; no se esquiva (pedir permiso a DBnomics o ficheros "
                  "del archivo del FMI)"] if robots else []),
               "fechas de publicación de FECHAS_EDICION documentadas, no medidas contra la API (verificado: false)",
               "zona euro pendiente (WEO de agregados, WEOAGG en DBnomics)",
               "ediciones fuera de DBnomics y de las versiones vigentes de la API SDMX: ver ediciones_pendientes"])
    return guardar_ficha(ficha)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FMI WEO por edición")
    ap.add_argument("--ediciones", nargs="*", default=list(EDICIONES_DBNOMICS))
    ap.add_argument("--sin-sdmx", action="store_true", help="no consulta la API SDMX del FMI")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args(argv)
    tabla, informe = tabla_weo(a.ediciones, sdmx=not a.sin_sdmx, forzar=a.forzar)
    meta = org.medir(tabla, frecuencias(tabla))
    print("tabla:", org.guardar_tabla(tabla, FUENTE), len(tabla), "filas")
    print("metadatos:", org.guardar_metadatos(meta, FUENTE))
    print("ficha:", ficha_weo(tabla, informe, meta))


if __name__ == "__main__":
    main()
