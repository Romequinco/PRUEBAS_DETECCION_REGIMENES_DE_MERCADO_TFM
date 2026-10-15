"""BIS (Banco de Pagos Internacionales): API SDMX de stats.bis.org en CSV (bloque 4 del Tema 1).

Dataflows (países del alcance: EE. UU., zona euro ``XM``, G7 y China):

- ``WS_CBPOL``       tipo oficial del banco central, mensual a fin de periodo (``M.<país>``).
- ``WS_CREDIT_GAP``  brecha crédito/PIB del sector privado no financiero (``Q.<país>.P.A.C``:
                     crédito de todos los sectores, brecha = ratio - tendencia HP unilateral, pp del PIB).
- ``WS_DSR``         ratio de servicio de la deuda del sector privado no financiero (``Q.<país>.P``).
- ``WS_EER``         tipo de cambio efectivo real, cesta amplia (``M.R.B.<país>``, 2020 = 100).

URL: ``https://stats.bis.org/api/v1/data/<flujo>/<clave>/all?format=csv&detail=dataonly``; una
petición por flujo con los países unidos por ``+``. Ids canónicos ``bis:<flujo>.<clave>``
(p. ej. ``bis:WS_CBPOL.M.US``); ``fecha_referencia`` = inicio del periodo.

Publicación: el BIS no publica vintages ni hora. ``metodo`` = ``lag_fijo`` con retraso conservador
desde el fin del periodo (``LAGS``), ``disponible_utc`` nulo y ``vintage`` = ``unico``. Brecha de
crédito, DSR y EER se revisan (crédito, PIB, ponderaciones y la tendencia HP con cada dato nuevo):
llevan la nota de revisable (``catalogo.NOTA_REVISABLE_EDICION``); el archivo de crudos por día de
descarga empieza con este bloque. El retraso real se mide como cota superior: días entre el fin del
último periodo publicado y la descarga (``_organismos.medir``).

CLI: ``python -m ingesta.mercados_series.fuentes.bis [--forzar] [--sin-red]``.
"""
from __future__ import annotations

import argparse
import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.mercados_series import catalogo as cat_mod
from ingesta.mercados_series.fuentes import _organismos as org

FUENTE = "bis"
URL_BASE = "https://stats.bis.org/api/v1/data/{flujo}/{clave}/all"
PAISES = ("US", "XM", "DE", "FR", "IT", "GB", "JP", "CA", "CN")
NOMBRES_PAIS = {"US": "EE. UU.", "XM": "zona euro", "DE": "Alemania", "FR": "Francia", "IT": "Italia",
                "GB": "Reino Unido", "JP": "Japón", "CA": "Canadá", "CN": "China"}
NOTA_REVISABLE = cat_mod.NOTA_REVISABLE_EDICION

# flujo -> plantilla de clave, dimensiones (orden de la clave), frecuencia, unidad, lag (desde el fin
# del periodo, formato de lags.yaml), fin_periodo, revisable, nombre.
FLUJOS: dict[str, dict[str, Any]] = {
    "WS_CBPOL": {"clave": "M.{p}", "dims": ("FREQ", "REF_AREA"), "frecuencia": "mensual",
                 "unidad": "% anual (fin de mes)", "lag": "7D", "fin_periodo": "mes", "revisable": False,
                 "nombre": "Tipo oficial del banco central (BIS, fin de mes)",
                 "justificacion": "el tipo se conoce el día de la decisión; el BIS lo recopila con retraso "
                                  "de días: 7 días tras el fin de mes (conservador)"},
    "WS_CREDIT_GAP": {"clave": "Q.{p}.P.A.C", "dims": ("FREQ", "BORROWERS_CTY", "TC_BORROWERS", "TC_LENDERS",
                                                       "CG_DTYPE"),
                      "frecuencia": "trimestral", "unidad": "puntos porcentuales del PIB", "lag": "6M",
                      "fin_periodo": "trimestre", "revisable": True,
                      "nombre": "Brecha crédito/PIB, sector privado no financiero (BIS)",
                      "justificacion": "publicación trimestral del BIS (crédito total) con 3-6 meses de "
                                       "retraso: 6 fines de mes tras el trimestre (conservador)"},
    "WS_DSR": {"clave": "Q.{p}.P", "dims": ("FREQ", "BORROWERS_CTY", "DSR_BORROWERS"), "frecuencia": "trimestral",
               "unidad": "% de la renta disponible", "lag": "6M", "fin_periodo": "trimestre", "revisable": True,
               "nombre": "Ratio de servicio de la deuda, sector privado no financiero (BIS)",
               "justificacion": "se publica con el crédito total: 6 fines de mes tras el trimestre"},
    "WS_EER": {"clave": "M.R.B.{p}", "dims": ("FREQ", "EER_TYPE", "EER_BASKET", "REF_AREA"),
               "frecuencia": "mensual", "unidad": "índice (2020 = 100)", "lag": "1M", "fin_periodo": "mes",
               "revisable": True, "nombre": "Tipo de cambio efectivo real, cesta amplia (BIS)",
               "justificacion": "publicación mensual hacia mediados del mes siguiente: fin del mes siguiente"},
}


def url_flujo(flujo: str, paises: Iterable[str] = PAISES) -> str:
    clave = FLUJOS[flujo]["clave"].format(p="+".join(paises))
    return URL_BASE.format(flujo=flujo, clave=clave)


def id_serie(flujo: str, clave: str) -> str:
    return f"bis:{flujo}.{clave}"


def pais_de(flujo: str, clave: str) -> str:
    partes = clave.split(".")
    dims = FLUJOS[flujo]["dims"]
    for nombre in ("REF_AREA", "BORROWERS_CTY"):
        if nombre in dims:
            return partes[dims.index(nombre)]
    raise ValueError(f"{flujo}: sin dimensión de país")


def periodo_a_fecha(periodos: pd.Series) -> pd.Series:
    """``2024-01`` → 2024-01-01; ``2023-Q2`` → 2023-04-01; ``2024`` → 2024-01-01; ``2024-01-15`` → ese día."""
    def _uno(p: str) -> pd.Timestamp:
        p = str(p).strip()
        if "-Q" in p:
            return pd.Period(p.replace("-", ""), freq="Q").start_time
        return pd.Timestamp(p)
    return pd.to_datetime(periodos.map(_uno)).dt.normalize()


def leer_csv(contenido: bytes | str, flujo: str) -> pd.DataFrame:
    """CSV del BIS → ``clave, fecha, valor`` (clave SDMX sin el flujo)."""
    texto = contenido.decode("utf-8-sig") if isinstance(contenido, bytes) else contenido
    df = pd.read_csv(io.StringIO(texto), dtype=str)
    dims = list(FLUJOS[flujo]["dims"])
    faltan = [c for c in (*dims, "TIME_PERIOD", "OBS_VALUE") if c not in df.columns]
    if faltan:
        raise ValueError(f"CSV del BIS ({flujo}) sin columnas {faltan}")
    out = pd.DataFrame({"clave": df[dims].fillna("").agg(".".join, axis=1),
                        "fecha": periodo_a_fecha(df["TIME_PERIOD"]),
                        "valor": pd.to_numeric(df["OBS_VALUE"].replace({"NaN": None, "": None}), errors="coerce")})
    return out.sort_values(["clave", "fecha"]).reset_index(drop=True)


def tabla_flujo(datos: pd.DataFrame, flujo: str) -> pd.DataFrame:
    """Filas del contrato de un flujo (lag fijo, vintage ``unico``, huecos interiores con nota)."""
    cfg = FLUJOS[flujo]
    trozos = []
    for clave, g in datos.groupby("clave", sort=True):
        s = g.set_index("fecha")["valor"]
        s = s[~s.index.duplicated(keep="last")]
        s, nota = org.recortar_huecos(s)
        if s.empty:
            continue
        pub = cat_mod.fecha_publicacion_lag(pd.Series(s.index), cfg["lag"], cfg["fin_periodo"])
        if cfg["revisable"]:
            nota = nota.map(lambda n: f"{n}; {NOTA_REVISABLE}" if n else NOTA_REVISABLE)
        trozos.append(org.filas(id_serie(flujo, clave), s.index, pub.values, s.values, vintage="unico",
                                unidad=cfg["unidad"], divisa=None, fuente=FUENTE, metodo="lag_fijo",
                                nota=nota.values))
    return org.unir(trozos)


def descargar_bis(flujos: Iterable[str] = tuple(FLUJOS), *, forzar: bool = False,
                  raiz: Path | None = None) -> dict[str, tuple[bytes | None, dict[str, Any]]]:
    """Un crudo CSV por flujo (``<hoy>_<flujo>.csv``)."""
    return {f: org.descargar(FUENTE, f"{f}.csv", url_flujo(f), params={"format": "csv", "detail": "dataonly"},
                             forzar=forzar, raiz=raiz) for f in flujos}


def _ultimo_crudo(flujo: str, raiz: Path | None) -> bytes | None:
    rutas = sorted(org.ruta_crudo(FUENTE, raiz).glob(f"*_{flujo}.csv"))
    return rutas[-1].read_bytes() if rutas else None


def tabla_bis(flujos: Iterable[str] = tuple(FLUJOS), *, sin_red: bool = False, forzar: bool = False,
              raiz: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tabla del contrato de todos los flujos y el informe de descarga (metas por flujo)."""
    flujos = list(flujos)
    metas = {} if sin_red else descargar_bis(flujos, forzar=forzar, raiz=raiz)
    trozos, informe = [], {}
    for f in flujos:
        contenido = metas[f][0] if f in metas else None
        contenido = contenido or _ultimo_crudo(f, raiz)
        informe[f] = {k: v for k, v in (metas.get(f, (None, {}))[1]).items() if k != "params"}
        if contenido is None:
            continue
        trozos.append(tabla_flujo(leer_csv(contenido, f), f))
    return org.unir(trozos), informe


def frecuencias(tabla: pd.DataFrame) -> dict[str, str]:
    return {sid: FLUJOS[sid.split(":", 1)[1].split(".", 1)[0]]["frecuencia"] for sid in tabla["serie"].unique()}


def fragmento(ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Entradas del catálogo (esquema de ``catalogo_series.yaml``) de las series descargadas."""
    out = {}
    for sid in sorted(ids):
        flujo, clave = sid.split(":", 1)[1].split(".", 1)
        cfg = FLUJOS[flujo]
        pais = pais_de(flujo, clave)
        notas = [f"retraso conservador {cfg['lag']} desde el fin del periodo: {cfg['justificacion']}"]
        if cfg["revisable"]:
            notas.insert(0, NOTA_REVISABLE)
        out[sid] = {"alias": None, "nombre": f"{cfg['nombre']} - {NOMBRES_PAIS.get(pais, pais)}", "fuente": FUENTE,
                    "id_origen": f"{flujo}/{clave}", "frecuencia": cfg["frecuencia"], "unidad": cfg["unidad"],
                    "divisa": None, "metodo": "lag_fijo", "lag_clave": f"BIS_{flujo}", "revisable": cfg["revisable"],
                    "rol_fase1": None, "licencia": "BIS: uso libre con cita (términos de stats.bis.org)",
                    "estado": "activa", "agentes": ["macro", "risk"], "notas": notas, "verificado": False}
    return out


def lags_fragmento() -> dict[str, dict[str, Any]]:
    """Entradas para ``config/lags.yaml`` (sección ``adicionales``) que usan las series del BIS."""
    return {f"BIS_{f}": {"lag": c["lag"], "fin_periodo": c["fin_periodo"], "verificado": False,
                         "justificacion": c["justificacion"]} for f, c in FLUJOS.items()}


def ficha_bis(tabla: pd.DataFrame, informe: dict[str, Any], meta: pd.DataFrame) -> Path:
    med = org.resumen_medido(meta)
    errores = {f: m.get("error") for f, m in informe.items() if m.get("error")}
    por_flujo = {}
    for f in FLUJOS:
        sub = meta[meta["serie"].str.startswith(f"bis:{f}.")] if not meta.empty else meta
        if len(sub):
            por_flujo[f] = {"series": int(len(sub)), "inicio": str(sub["inicio"].min()), "fin": str(sub["fin"].max()),
                            "retraso_observado_dias_min": int(sub["retraso_observado_dias"].min()),
                            "lag_aplicado": FLUJOS[f]["lag"]}
    estado = "fallo" if tabla.empty else ("parcial" if errores else "ok")
    ficha = FichaFuente(
        fuente=FUENTE, nombre="BIS Data Portal (stats.bis.org): tipos oficiales, brecha de crédito, DSR, EER",
        tema=org.TEMA, agentes=["macro", "risk"], rag="serie", url="https://stats.bis.org/api/v1/",
        acceso="api", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="BIS: reutilización libre citando la fuente (términos de uso de bis.org)",
        cobertura_inicio=med.get("cobertura_inicio"), cobertura_fin=med.get("cobertura_fin"),
        frecuencia="mensual (CBPOL, EER) y trimestral (brecha de crédito, DSR)",
        retraso_publicacion=("sin vintages ni hora: lag fijo conservador desde el fin del periodo "
                             + ", ".join(f"{f} {c['lag']}" for f, c in FLUJOS.items())
                             + f"; retraso observado mínimo medido {med.get('retraso_observado_dias_min')} días"),
        formato="csv (SDMX)", volumen_estimado=f"{med.get('observaciones', 0)} observaciones en {med.get('series', 0)} series",
        limites="sin clave; 1 petición/s por cortesía; una petición por flujo", estado_sonda=estado,
        mediciones={**med, "por_flujo": por_flujo, "errores": errores,
                    "bytes": {f: m.get("bytes") for f, m in informe.items()},
                    "segundos": {f: m.get("segundos") for f, m in informe.items()}},
        notas=["sin vintages: brecha de crédito, DSR y EER se cargan con el último valor y la nota de revisable",
               "zona euro = XM; China sin DSR en el BIS si la descarga no la trae (ver por_flujo)",
               "ids bis:<flujo>.<clave SDMX>; fecha_referencia = inicio del periodo"])
    return guardar_ficha(ficha)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="BIS: tipos oficiales, brecha de crédito, DSR y EER")
    ap.add_argument("--forzar", action="store_true", help="vuelve a descargar aunque haya crudo de hoy")
    ap.add_argument("--sin-red", action="store_true", help="usa el último crudo archivado")
    a = ap.parse_args(argv)
    tabla, informe = tabla_bis(sin_red=a.sin_red, forzar=a.forzar)
    meta = org.medir(tabla, frecuencias(tabla))
    print("tabla:", org.guardar_tabla(tabla, FUENTE), len(tabla), "filas")
    print("metadatos:", org.guardar_metadatos(meta, FUENTE))
    print("ficha:", ficha_bis(tabla, informe, meta))


if __name__ == "__main__":
    main()
