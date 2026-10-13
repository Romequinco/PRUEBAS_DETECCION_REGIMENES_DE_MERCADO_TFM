"""Extractor de Eurostat del Tema 1 (bloque 4): HICP por país, paro, sentimiento económico y PIB.

API SDMX 2.1 de difusión (``ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data``) con
``format=SDMX-CSV`` (``DATAFLOW, LAST UPDATE, <dimensiones>, TIME_PERIOD, OBS_VALUE, OBS_FLAG``).
Red con ``ingesta.comun.http.get`` (1 petición/s); caché cruda en
``<corpus>/mercados_series/crudo/eurostat/`` (``bce.ClienteCsv``). Una petición por conjunto con la
clave sin país (todas las zonas) y filtro local a ``GEOS``: zona euro (``EA`` composición cambiante
cuando existe, si no ``EA21``/``EA20``), UE-27 y los países del G7 que publica cada conjunto
(DE, FR, IT; UK, US, JP, CA en paro).

Conjuntos (``CONJUNTOS``):

- ``prc_hicp_minr`` (NUEVO, ECOICOP v2 desde la publicación de febrero de 2026): ``RCH_A`` del total.
- ``prc_hicp_manr`` (antiguo, ECOICOP v1; última actualización 2026-02-06, último dato 2025-12):
  ``retirada`` con ``sucesor``; ``medir_ruptura_hicp`` compara ambos en los meses comunes.
- ``une_rt_m``: paro desestacionalizado, 15-74, % de la población activa.
- ``ei_bssi_m_r2``: indicador de sentimiento económico (ESI, desestacionalizado).
- ``namq_10_gdp``: PIB real, tasa interanual (``CLV_PCH_SM``, desestacionalizado y de calendario).

Publicación: Eurostat publica a las 11:00 Europe/Luxembourg (publicador ``eurostat``), pero no da
vintages: todo es revisable y se carga con ``lag_fijo`` (regla de ``lags.yaml``: HICP y paro
``1M1D`` desde el fin de mes, ESI ``1D`` (sale a final del propio mes), PIB ``1M1D`` desde el fin
de trimestre) y la nota obligatoria de revisables (último valor publicado). Las banderas
(``OBS_FLAG``: p provisional, e estimado, b ruptura, d definición distinta) van a la nota.

CLI: ``python -m ingesta.mercados_series.fuentes.eurostat [--sin-red] [--fichas] [--escribir]``.
"""
from __future__ import annotations

import argparse
import io
from typing import Any

import pandas as pd

from ingesta.comun.rutas import resultados_dir
from ingesta.mercados_series import catalogo
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, UTC, vacio_series, validar_series
from ingesta.mercados_series.fuentes.bce import ClienteCsv, entrada_catalogo, fecha_periodo

API_EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
TEMA = "mercados_series"
FUENTE = "eurostat"
LICENCIA_EUROSTAT = "Eurostat: reutilización libre con cita (Decisión 2011/833/UE; © Unión Europea)"
GEOS = ("EA", "EA21", "EA20", "EU27_2020", "DE", "FR", "IT", "UK", "US", "JP", "CA")
BANDERAS = {"p": "provisional", "e": "estimado", "b": "ruptura de serie", "d": "definición distinta",
            "u": "baja fiabilidad", "c": "confidencial", "s": "estimación de Eurostat", "f": "previsión"}
NOTA_RUPTURA_HICP = ("ruptura de 2026: HICP pasa a ECOICOP v2 (prc_hicp_minr); prc_hicp_manr (ECOICOP v1) deja de "
                     "actualizarse en febrero de 2026")

# Agregados que no existían antes de esa fecha (primera publicación de Eurostat, aproximación conservadora):
# sus valores retropolados no son visibles antes de su creación.
CREACION_GEO = {"EA20": "2023-03-01", "EA21": "2026-03-01"}
NOTA_POSTERIOR = ("serie o agregado creado después del periodo (retropolación): visible solo desde su creación; "
                  "para el pasado usar la versión vigente entonces (EA cambiante, prc_hicp_manr)")

# clave -> especificación (clave SDMX sin la zona: la zona es la última dimensión).
CONJUNTOS: dict[str, dict[str, Any]] = {
    "prc_hicp_minr": {"clave": "M.RCH_A.TOTAL.", "nombre": "HICP general (ECOICOP v2), tasa anual",
                      "frecuencia": "mensual", "unidad": "% interanual", "lag_clave": "CPIAUCSL", "estado": "activa",
                      "no_antes_de": "2026-01-28", "notas": [NOTA_RUPTURA_HICP]},
    "prc_hicp_manr": {"clave": "M.RCH_A.CP00.", "nombre": "HICP general (ECOICOP v1, serie antigua), tasa anual",
                      "frecuencia": "mensual", "unidad": "% interanual", "lag_clave": "CPIAUCSL", "estado": "retirada",
                      "sucesor": "prc_hicp_minr", "notas": [NOTA_RUPTURA_HICP]},
    "une_rt_m": {"clave": "M.SA.TOTAL.PC_ACT.T.", "nombre": "Tasa de paro 15-74, desestacionalizada",
                 "frecuencia": "mensual", "unidad": "% población activa", "lag_clave": "EUROSTAT_PARO", "estado": "activa",
                 "lag_clave_geo": {"UK": "EUROSTAT_PARO_UK"}},
    "ei_bssi_m_r2": {"clave": "M.BS-ESI-I.SA.", "nombre": "Indicador de sentimiento económico (ESI), desestacionalizado",
                     "frecuencia": "mensual", "unidad": "índice (media de largo plazo = 100)", "lag_clave": "UMCSENT",
                     "estado": "activa"},
    "namq_10_gdp": {"clave": "Q.CLV_PCH_SM.SCA.B1GQ.", "nombre": "PIB real, tasa interanual (desestacionalizado)",
                    "frecuencia": "trimestral", "unidad": "% interanual", "lag_clave": "EUROSTAT_PIB", "estado": "activa"},
}

__all__ = ["API_EUROSTAT", "CONJUNTOS", "GEOS", "ClienteEurostat", "parsear_sdmx_csv", "id_eurostat",
           "tabla_conjunto", "tabla_eurostat", "no_antes_de", "CREACION_GEO", "medir_ruptura_hicp",
           "entradas_eurostat", "ficha_eurostat", "main"]


class ClienteEurostat(ClienteCsv):
    """Cliente SDMX 2.1 de Eurostat (``format=SDMX-CSV``) con caché cruda."""

    nombre = "eurostat"
    api = API_EUROSTAT

    def datos(self, conjunto: str, clave: str, **params: Any) -> pd.DataFrame:
        return parsear_sdmx_csv(self.texto(f"{conjunto}/{clave}", {"format": "SDMX-CSV", **params}))


def parsear_sdmx_csv(texto: str) -> pd.DataFrame:
    """SDMX-CSV de Eurostat → ``geo, TIME_PERIOD, OBS_VALUE, OBS_FLAG, LAST_UPDATE`` (+ dimensiones)."""
    if not texto.strip():
        return pd.DataFrame(columns=["geo", "TIME_PERIOD", "OBS_VALUE", "OBS_FLAG", "LAST_UPDATE"])
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False, na_values=[""])
    df = df.rename(columns={"LAST UPDATE": "LAST_UPDATE"})
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    if "OBS_FLAG" not in df:
        df["OBS_FLAG"] = None
    if "LAST_UPDATE" in df:
        df["LAST_UPDATE"] = pd.to_datetime(df["LAST_UPDATE"], format="%d/%m/%y %H:%M:%S", errors="coerce")
    return df


def id_eurostat(conjunto: str, clave: str, geo: str) -> str:
    """``eurostat:<conjunto>.<clave completa>`` (p. ej. ``eurostat:une_rt_m.M.SA.TOTAL.PC_ACT.T.DE``)."""
    return f"eurostat:{conjunto}.{clave}{geo}"


def entradas_eurostat(geos_por_conjunto: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Especificación de cada serie (conjunto × zona) con el esquema del catálogo."""
    out = {}
    for conj, geos in geos_por_conjunto.items():
        c = CONJUNTOS[conj]
        for geo in geos:
            sid = id_eurostat(conj, c["clave"], geo)
            e = {"clave": f"{conj}/{c['clave']}{geo}", "nombre": f"{c['nombre']} ({geo})",
                 "frecuencia": c["frecuencia"], "unidad": c["unidad"], "divisa": None, "metodo": "lag_fijo",
                 "lag_clave": c.get("lag_clave_geo", {}).get(geo, c["lag_clave"]), "revisable": True, "fuente": FUENTE, "estado": c["estado"],
                 "notas": [catalogo.NOTA_REVISABLE_EDICION, *c.get("notas", [])]}
            if c.get("sucesor"):
                e["sucesor"] = id_eurostat(c["sucesor"], CONJUNTOS[c["sucesor"]]["clave"], geo)
            out[sid] = e
    return out


def no_antes_de(clave: str) -> pd.Timestamp | None:
    """Primera fecha de decisión en que existía la serie ``conjunto/clave+geo`` (``None`` si siempre existió).

    Es la mayor de la creación del agregado (``CREACION_GEO``) y la del conjunto (``no_antes_de``).
    """
    conj, resto = clave.split("/", 1)
    geo = resto.rsplit(".", 1)[1]
    fechas = [f for f in (CREACION_GEO.get(geo), CONJUNTOS.get(conj, {}).get("no_antes_de")) if f]
    return max(pd.Timestamp(f) for f in fechas) if fechas else None


def tabla_conjunto(sid: str, obs: pd.DataFrame, espec: dict[str, Any]) -> pd.DataFrame:
    """Observaciones de UNA zona → filas ``lag_fijo`` (vintage ``unico``, nota de revisable y banderas).

    Valores vacíos con bandera ``c``/``:`` → nulo con nota (hueco declarado); vacíos sin bandera, fuera.
    """
    if obs.empty:
        return vacio_series()
    df = obs.copy()
    bandera = df["OBS_FLAG"].fillna("").astype(str).str.strip()
    df = df[df["OBS_VALUE"].notna() | (bandera != "")].reset_index(drop=True)
    bandera = df["OBS_FLAG"].fillna("").astype(str).str.strip()
    if df.empty:
        return vacio_series()
    ref = fecha_periodo(df["TIME_PERIOD"])
    lag = catalogo.cargar_lags()[espec["lag_clave"]]
    pub = catalogo.fecha_publicacion_lag(ref, lag["lag"], lag["fin_periodo"], habiles=lag.get("habiles"))
    pub = pd.Series(pub.to_numpy(), index=df.index)
    creacion = no_antes_de(espec["clave"])
    posterior = pd.Series(False, index=df.index)
    if creacion is not None:
        posterior = pub < creacion
        pub = pub.where(~posterior, creacion)
    nr = catalogo.NOTA_REVISABLE_EDICION
    extra = bandera.map(lambda b: "; ".join(f"bandera Eurostat {c}: {BANDERAS.get(c, 'otra')}" for c in b if c.isalpha()))
    nota = pd.Series(nr, index=df.index, dtype=object)
    nota = nota.where(extra == "", nr + "; " + extra)
    nota = nota.where(df["OBS_VALUE"].notna(), nota + "; hueco de la fuente (sin valor publicado)")
    nota = nota.where(~posterior, nota + "; " + NOTA_POSTERIOR)
    out = pd.DataFrame({
        "serie": sid, "fecha_referencia": ref.to_numpy(), "fecha_publicacion": pub.to_numpy(),
        "disponible_utc": pd.Series(pd.NaT, index=df.index, dtype=UTC).to_numpy(), "vintage": "unico",
        "valor": df["OBS_VALUE"].to_numpy(), "unidad": espec["unidad"], "divisa": None, "fuente": FUENTE,
        "metodo_publicacion": "lag_fijo", "nota": nota.to_numpy(),
    })
    out["disponible_utc"] = pd.to_datetime(out["disponible_utc"], utc=True)
    return out[list(COLUMNAS_SERIES)]


def tabla_eurostat(cliente: ClienteEurostat | None = None, *, sin_red: bool = False,
                   conjuntos: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Series de Eurostat validadas + mediciones (una petición por conjunto)."""
    cli = cliente or ClienteEurostat(sin_red=sin_red)
    partes, med, geos = [], {}, {}
    crudos = {}
    for conj in conjuntos or list(CONJUNTOS):
        c = CONJUNTOS[conj]
        df = cli.datos(conj, c["clave"])
        df = df[df["geo"].isin(GEOS)]
        crudos[conj] = df
        geos[conj] = sorted(df["geo"].unique())
        lu = df["LAST_UPDATE"].dropna() if "LAST_UPDATE" in df else pd.Series(dtype="datetime64[ns]")
        med[conj] = {"geos": geos[conj], "ultima_actualizacion": lu.max().isoformat() if len(lu) else None}
    espec = entradas_eurostat(geos)
    for sid, e in espec.items():
        conj, geo = e["clave"].split("/", 1)[0], e["clave"].rsplit(".", 1)[1]
        obs = crudos[conj][crudos[conj]["geo"] == geo]
        t = tabla_conjunto(sid, obs, e)
        partes.append(t)
        if len(t):
            med[sid] = {"n_obs": int(len(t)), "inicio": t["fecha_referencia"].min().date().isoformat(),
                        "fin": t["fecha_referencia"].max().date().isoformat(),
                        "n_nulos": int(t["valor"].isna().sum())}
    df = validar_series(pd.concat([p for p in partes if len(p)], ignore_index=True))
    if {"prc_hicp_minr", "prc_hicp_manr"} <= set(crudos):
        med["ruptura_hicp"] = medir_ruptura_hicp(crudos["prc_hicp_minr"], crudos["prc_hicp_manr"])
    return df, {"conjuntos": med, "peticiones": cli.peticiones, "bytes": cli.bytes, "espec": espec}


def medir_ruptura_hicp(nuevo: pd.DataFrame, antiguo: pd.DataFrame) -> dict[str, Any]:
    """Diferencia (puntos) entre ECOICOP v2 y v1 en los meses comunes, por zona."""
    a = nuevo[["geo", "TIME_PERIOD", "OBS_VALUE"]].merge(antiguo[["geo", "TIME_PERIOD", "OBS_VALUE"]],
                                                         on=["geo", "TIME_PERIOD"], suffixes=("_v2", "_v1"))
    a = a.dropna(subset=["OBS_VALUE_v2", "OBS_VALUE_v1"])
    out = {}
    for geo, g in a.groupby("geo"):
        d = (g["OBS_VALUE_v2"] - g["OBS_VALUE_v1"]).abs()
        out[geo] = {"meses_comunes": int(len(g)), "inicio": g["TIME_PERIOD"].min(), "fin": g["TIME_PERIOD"].max(),
                    "dif_abs_media": round(float(d.mean()), 4), "dif_abs_max": round(float(d.max()), 4),
                    "meses_distintos": int((d > 1e-9).sum())}
    return out


def ficha_eurostat(df: pd.DataFrame, med: dict[str, Any]):
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    conj = {k: v for k, v in med["conjuntos"].items() if k in CONJUNTOS or k == "ruptura_hicp"}
    ficha = FichaFuente(
        fuente=FUENTE, nombre="Eurostat: HICP (prc_hicp_minr / prc_hicp_manr), paro (une_rt_m), ESI, PIB (namq_10_gdp)",
        tema=TEMA, agentes=["macro"], rag="serie", url="https://ec.europa.eu/eurostat/web/main/data/web-services",
        acceso="api", autenticacion="ninguna", coste="0 €", licencia_y_terminos=LICENCIA_EUROSTAT,
        cobertura_inicio=df["fecha_referencia"].min().date().isoformat(),
        cobertura_fin=df["fecha_referencia"].max().date().isoformat(),
        frecuencia="mensual (HICP, paro, ESI), trimestral (PIB)",
        retraso_publicacion=("publicación a las 11:00 Europe/Luxembourg (calendario de Euro indicators); sin vintages: "
                             "lag_fijo conservador (HICP y paro 1M1D desde fin de mes, ESI 1D, PIB 1M1D desde fin de "
                             "trimestre) y nota de revisables; LAST UPDATE medido por conjunto en mediciones"),
        formato="csv (SDMX-CSV 2.1)",
        volumen_estimado=f"{med.get('bytes', 0) / 1e6:.1f} MB de CSV por descarga completa ({len(CONJUNTOS)} peticiones)",
        limites="sin clave; 1 petición/s por host; claves inválidas → HTTP 400; robots.txt permite la API",
        estado_sonda="ok",
        mediciones={"n_series": int(df["serie"].nunique()), "n_filas": int(len(df)), "conjuntos": conj},
        notas=[NOTA_RUPTURA_HICP,
               "La zona euro de composición cambiante ('EA') no está en todos los conjuntos: se guarda la que haya "
               "(EA, EA21, EA20) y la UE-27",
               "Revisables sin vintages: valor del último fichero descargado (posible fuga de revisiones)",
               "Las banderas OBS_FLAG (p, e, b, d...) se conservan en la nota de cada fila"],
    )
    return guardar_ficha(ficha)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extractor de Eurostat (Tema 1, bloque 4)")
    ap.add_argument("--sin-red", action="store_true")
    ap.add_argument("--fichas", action="store_true")
    ap.add_argument("--escribir", action="store_true")
    a = ap.parse_args(argv)
    cli = ClienteEurostat(sin_red=a.sin_red)
    df, med = tabla_eurostat(cli)
    print(f"Eurostat: {df['serie'].nunique()} series, {len(df)} filas, {cli.peticiones} peticiones")
    if a.fichas:
        ficha_eurostat(df, med)
        carpeta = resultados_dir(TEMA) / "tablas"
        carpeta.mkdir(parents=True, exist_ok=True)
        filas = [{"id": k, "clave_sdmx": e["clave"], "nombre": e["nombre"], "frecuencia": e["frecuencia"],
                  "unidad": e["unidad"], "lag_clave": e["lag_clave"], "estado": e["estado"],
                  **{c: med["conjuntos"].get(k, {}).get(c) for c in ("inicio", "fin", "n_obs")}}
                 for k, e in sorted(med["espec"].items())]
        pd.DataFrame(filas).to_csv(carpeta / "eurostat_series.csv", index=False)
    if a.escribir:
        from ingesta.mercados_series import almacen
        almacen.escribir_series(df)


if __name__ == "__main__":
    main()
