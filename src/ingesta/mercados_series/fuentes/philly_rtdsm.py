"""Real-Time Data Set for Macroeconomists (RTDSM) de la Fed de Filadelfia: historia de EE. UU.
ANTERIOR a ALFRED.

Cada variable es una matriz de vintages en Excel (``DATE`` + una columna por vintage, p. ej.
``ROUTPUT65Q4`` o ``EMPLOY64M12``). Solo se cargan los vintages anteriores al primer vintage de
ALFRED de la serie equivalente de FRED (``MAPEO``); el resto lo da ALFRED con fechas exactas.

Siglo del vintage (``anios_vintage``): el nombre solo trae dos cifras. La regla ingenua «yy < 65 →
20yy» es un error: ``EMPLOY64M12`` es diciembre de **1964**. El siglo se deduce del ORDEN de las
columnas (cronológico en la matriz): la primera columna es del siglo XX si ``yy >= 40`` (ningún
vintage del RTDSM es anterior a 1940 ni posterior a 2039 en su primera columna) y cada vez que las
dos cifras bajan respecto de la columna anterior se suma un siglo.

Fecha del vintage (conservadora): último día del mes del vintage; en los trimestrales (``Q``), del
mes central del trimestre (el RTDSM fecha el vintage a mediados de ese mes). Hora: 23:59 de Nueva
York (publicador ``philly_rtdsm`` del fragmento) → ``fecha_publicacion`` = primera decisión
posterior. ``vintage`` = ``AAAA-MM``. Solo valores nuevos o revisados (``b4_comun.compactar_vintages``).

Mapeo id Philly → id FRED (``MAPEO``): ROUTPUT → GDPC1 (PNB real hasta 1991, PIB real después; base
del vintage), PCPI → CPIAUCSL, EMPLOY → PAYEMS, RUC → UNRATE. Primer vintage de ALFRED de cada una:
``regla_previa_alfred.primer_vintage`` del catálogo (GDPC1 1991-12-04, CPIAUCSL 1972-07-21, PAYEMS
1955-05-06, UNRATE 1960-03-15): solo ROUTPUT aporta vintages previos (1965T4-1991T3); los vintages
de PCPI (desde 1998), EMPLOY (desde 1964) y RUC (desde 1965) son todos posteriores al inicio de ALFRED.

CLI: ``python -m ingesta.mercados_series.fuentes.philly_rtdsm [--variables ROUTPUT ...] [--descargar] [--ficha]``.
"""
from __future__ import annotations

import argparse
import io
import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.http import get, robots_permite
from ingesta.mercados_series.fuentes.b4_comun import (
    TEMA,
    compactar_vintages,
    guardar_crudo,
    ruta_crudo,
    tabla_contrato,
)

FUENTE = "philly_rtdsm"
PUBLICADOR = "philly_rtdsm"
URL_PAGINA = "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/{var}"
URL_FICHERO = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/"
               "data-files/xlsx/{fichero}")
# variable Philly -> (fichero, id FRED equivalente, primer vintage de ALFRED de respaldo, frecuencia, unidad, divisa)
MAPEO: dict[str, dict[str, Any]] = {
    "ROUTPUT": {"fichero": "ROUTPUTQvQd.xlsx", "fred": "fred:GDPC1", "primer_alfred": "1991-12-04",
                "frecuencia": "trimestral", "divisa": "USD",
                "unidad": "miles de millones de dólares constantes (base del vintage), tasa anual desestacionalizada"},
    "PCPI": {"fichero": "pcpiMvMd.xlsx", "fred": "fred:CPIAUCSL", "primer_alfred": "1972-07-21",
             "frecuencia": "mensual", "divisa": None, "unidad": "índice 1982-84=100 (desestacionalizado)"},
    "EMPLOY": {"fichero": "employMvMd.xlsx", "fred": "fred:PAYEMS", "primer_alfred": "1955-05-06",
               "frecuencia": "mensual", "divisa": None, "unidad": "miles de personas (desestacionalizado)"},
    "RUC": {"fichero": "rucQvMd.xlsx", "fred": "fred:UNRATE", "primer_alfred": "1960-03-15",
            "frecuencia": "mensual", "divisa": None, "unidad": "% (desestacionalizado)"},
}
NOTA_PHILLY = ("RTDSM (Fed de Filadelfia): vintage anterior a ALFRED de {fred}; fecha conservadora = fin del mes "
               "del vintage")
_COLUMNA = re.compile(r"^([A-Za-z_]+?)(\d{2})([QM])(\d{1,2})$")

__all__ = ["FUENTE", "MAPEO", "partes_columna", "anios_vintage", "fechas_vintage", "leer_matriz",
           "primer_vintage_alfred", "tabla_philly", "descargar", "ficha_philly"]


def partes_columna(nombre: str) -> tuple[str, int, str, int]:
    """``'EMPLOY64M12'`` → ``('EMPLOY', 64, 'M', 12)``. ``ValueError`` si no es una columna de vintage."""
    m = _COLUMNA.fullmatch(str(nombre).strip())
    if not m:
        raise ValueError(f"columna de vintage no reconocida: {nombre!r}")
    var, yy, tipo, k = m.group(1).upper(), int(m.group(2)), m.group(3), int(m.group(4))
    if (tipo == "Q" and not 1 <= k <= 4) or (tipo == "M" and not 1 <= k <= 12):
        raise ValueError(f"periodo de vintage fuera de rango: {nombre!r}")
    return var, yy, tipo, k


def anios_vintage(columnas: Sequence[str]) -> list[int]:
    """Año de cuatro cifras de cada columna de vintage, en el orden dado (cronológico).

    Primera columna: 19yy si ``yy >= 40`` y 20yy si no; después, cada vez que ``yy`` baja respecto
    de la columna anterior (paso de 99 a 00) se suma un siglo. ``EMPLOY64M12`` → 1964.
    """
    anios: list[int] = []
    siglo = None
    previo = None
    for c in columnas:
        _, yy, _, _ = partes_columna(c)
        if siglo is None:
            siglo = 1900 if yy >= 40 else 2000
        elif yy < previo:
            siglo += 100
        previo = yy
        anios.append(siglo + yy)
    return anios


def fechas_vintage(columnas: Sequence[str]) -> pd.DatetimeIndex:
    """Fecha conservadora de cada vintage: último día del mes del vintage (mes central si es trimestral)."""
    anios = anios_vintage(columnas)
    meses = []
    for c in columnas:
        _, _, tipo, k = partes_columna(c)
        meses.append(3 * k - 1 if tipo == "Q" else k)
    inicio = pd.to_datetime([f"{a:04d}-{m:02d}-01" for a, m in zip(anios, meses)])
    return inicio + pd.offsets.MonthEnd(0)


def _fecha_periodo(texto: Any) -> pd.Timestamp:
    """``'1947:Q1'`` → 1947-01-01; ``'1947:01'`` → 1947-01-01."""
    s = str(texto).strip()
    m = re.fullmatch(r"(\d{4}):Q([1-4])", s)
    if m:
        return pd.Timestamp(int(m.group(1)), 3 * int(m.group(2)) - 2, 1)
    m = re.fullmatch(r"(\d{4}):(\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    raise ValueError(f"periodo RTDSM no reconocido: {texto!r}")


def leer_matriz(fuente: Path | bytes | pd.DataFrame) -> pd.DataFrame:
    """Matriz de vintages → tabla larga ``periodo, columna, orden, fecha_vintage, valor`` (sin nulos).

    ``#N/A`` y celdas vacías son «no publicado en ese vintage» (no hueco): se descartan.
    """
    if isinstance(fuente, pd.DataFrame):
        m = fuente.copy()
    else:
        m = pd.read_excel(io.BytesIO(fuente) if isinstance(fuente, bytes) else fuente)
    m = m.rename(columns={m.columns[0]: "DATE"})
    cols = [c for c in m.columns if c != "DATE"]
    fechas = fechas_vintage(cols)
    m["periodo"] = m["DATE"].map(_fecha_periodo)
    largo = m.melt(id_vars="periodo", value_vars=cols, var_name="columna", value_name="valor")
    largo["valor"] = pd.to_numeric(largo["valor"], errors="coerce")
    orden = {c: i for i, c in enumerate(cols)}
    largo["orden"] = largo["columna"].map(orden)
    largo["fecha_vintage"] = largo["columna"].map(dict(zip(cols, fechas)))
    return largo.dropna(subset=["valor"]).reset_index(drop=True)


def primer_vintage_alfred(variable: str, catalogo_series: dict[str, dict] | None = None) -> pd.Timestamp:
    """Primer vintage de ALFRED de la serie FRED equivalente (catálogo; si no, el de ``MAPEO``)."""
    info = MAPEO[variable]
    if catalogo_series is None:
        try:
            from ingesta.mercados_series import catalogo
            catalogo_series = catalogo.cargar_catalogo()
        except Exception:  # noqa: BLE001 — sin catálogo legible se usa el valor de respaldo
            catalogo_series = {}
    previa = (catalogo_series.get(info["fred"]) or {}).get("regla_previa_alfred") or {}
    return pd.Timestamp(previa.get("primer_vintage") or info["primer_alfred"])


def tabla_philly(variable: str, matriz: Path | bytes | pd.DataFrame, *, primer_alfred: Any = None,
                 publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Filas del contrato de ``philly:<variable>``: vintages anteriores al primero de ALFRED."""
    info = MAPEO[variable]
    corte = pd.Timestamp(primer_alfred) if primer_alfred is not None else primer_vintage_alfred(variable)
    largo = leer_matriz(matriz)
    largo = largo[largo["fecha_vintage"] < corte]
    largo = compactar_vintages(largo)
    # Un vintage no puede publicar un periodo que aún no ha empezado (protección ante matrices raras).
    largo = largo[largo["periodo"] <= largo["fecha_vintage"]]
    return tabla_contrato(
        f"philly:{variable}", largo["periodo"], largo["fecha_vintage"], largo["valor"],
        vintage=largo["fecha_vintage"].dt.strftime("%Y-%m").reset_index(drop=True), unidad=info["unidad"],
        divisa=info["divisa"], fuente=FUENTE, metodo="vintage_exacto", publicador=PUBLICADOR,
        nota=NOTA_PHILLY.format(fred=info["fred"]), publicaciones=publicaciones)


def descargar(variables: Iterable[str] = tuple(MAPEO)) -> dict[str, dict[str, Any]]:
    """Descarga las matrices (copia cruda en el corpus). Comprueba robots.txt antes."""
    informe: dict[str, dict[str, Any]] = {}
    for v in variables:
        url = URL_FICHERO.format(fichero=MAPEO[v]["fichero"])
        if not robots_permite(URL_PAGINA.format(var=v.lower())):
            informe[v] = {"estado": "vetado_robots", "url": url}
            continue
        t = time.monotonic()
        try:
            r = get(url, timeout=90)
        except Exception as e:  # noqa: BLE001 — se documenta y se sigue
            informe[v] = {"estado": "fallo", "error": str(e)[:200], "url": url}
            continue
        ruta = guardar_crudo(FUENTE, MAPEO[v]["fichero"], r.content)
        informe[v] = {"estado": "ok", "bytes": len(r.content), "segundos": round(time.monotonic() - t, 2),
                      "ruta": str(ruta), "url": url}
    return informe


def _ruta_local(variable: str) -> Path:
    return ruta_crudo(FUENTE) / MAPEO[variable]["fichero"]


def ficha_philly(variables: Iterable[str] = tuple(MAPEO), informe: dict[str, dict] | None = None, *,
                 guardar: bool = True):
    """Ficha ``philly_rtdsm`` con cobertura de vintages, filas previas a ALFRED y retraso medidos."""
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    med: dict[str, Any] = {}
    for v in variables:
        ruta = _ruta_local(v)
        if not ruta.exists():
            med[v] = {"estado": "sin_descargar"}
            continue
        largo = leer_matriz(ruta)
        cols = largo.drop_duplicates("columna").sort_values("orden")
        corte = primer_vintage_alfred(v)
        previos = cols[cols["fecha_vintage"] < corte]
        tabla = tabla_philly(v, ruta, primer_alfred=corte)
        ult = largo.groupby("fecha_vintage")["periodo"].max()
        retraso = (ult.index - ult.values).days
        med[v] = {"fred": MAPEO[v]["fred"], "primer_vintage_alfred": corte.date().isoformat(),
                  "primer_vintage": cols["columna"].iloc[0], "primer_vintage_fecha": cols["fecha_vintage"].iloc[0].date().isoformat(),
                  "ultimo_vintage": cols["columna"].iloc[-1], "n_vintages": int(len(cols)),
                  "n_vintages_previos_alfred": int(len(previos)), "filas_contrato": int(len(tabla)),
                  "periodo_inicio": largo["periodo"].min().date().isoformat(),
                  "periodo_fin": largo["periodo"].max().date().isoformat(),
                  "dias_vintage_a_ultimo_periodo_mediana": float(pd.Series(retraso).median()),
                  "bytes": ruta.stat().st_size}
    ok = [m for m in med.values() if "n_vintages" in m]
    ficha = FichaFuente(
        fuente=FUENTE, nombre="Real-Time Data Set for Macroeconomists (Federal Reserve Bank of Philadelphia)",
        tema=TEMA, agentes=["macro"], rag="serie",
        url="https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists",
        acceso="descarga", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="Uso libre citando la fuente (Federal Reserve Bank of Philadelphia)",
        cobertura_inicio=min((m["periodo_inicio"] for m in ok), default=None),
        cobertura_fin=max((m["periodo_fin"] for m in ok), default=None),
        frecuencia="vintages trimestrales (ROUTPUT, RUC) y mensuales (PCPI, EMPLOY)",
        retraso_publicacion=("medido: días del fin del mes del vintage al inicio del último periodo publicado "
                             "(mediana): " + "; ".join(f"{v} {m['dias_vintage_a_ultimo_periodo_mediana']:g}"
                                                       for v, m in med.items() if "n_vintages" in m)
                             + ". Fecha del vintage conservadora: fin del mes (23:59 Nueva York)"),
        formato="xlsx", volumen_estimado=f"{sum(m.get('bytes', 0) for m in ok) / 1e6:.1f} MB ({len(ok)} matrices)",
        limites="sin límites declarados; 1 petición/s; robots.txt permite las páginas de datos",
        estado_sonda="ok" if len(ok) == len(med) and ok else ("parcial" if ok else "fallo"),
        mediciones={"variables": med, "descarga": informe or {}},
        notas=["Solo vintages ANTERIORES al primer vintage de ALFRED de la serie equivalente (mapeo ROUTPUT→GDPC1, "
               "PCPI→CPIAUCSL, EMPLOY→PAYEMS, RUC→UNRATE)",
               "Corregido el error de siglo: el año sale del orden de columnas (EMPLOY64M12 = 1964, no 2064)",
               "ROUTPUT: PNB real hasta 1991 y en dólares constantes de la base de cada vintage (unidad_por_vintage)",
               "Matrices crudas en <corpus>/mercados_series/crudo/philly_rtdsm/"])
    ruta = guardar_ficha(ficha) if guardar else None
    return ficha, ruta


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variables", nargs="*", default=list(MAPEO))
    p.add_argument("--descargar", action="store_true")
    p.add_argument("--ficha", action="store_true")
    a = p.parse_args(argv)
    informe = descargar(a.variables) if a.descargar else None
    if informe:
        for v, i in informe.items():
            print(v, i.get("estado"), i.get("bytes"))
    if a.ficha:
        _, ruta = ficha_philly(a.variables, informe)
        print("ficha:", ruta)


if __name__ == "__main__":
    main()
