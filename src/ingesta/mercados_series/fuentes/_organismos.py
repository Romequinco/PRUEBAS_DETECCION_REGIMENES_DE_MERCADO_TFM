"""Utilidades comunes de las fuentes de organismos internacionales (bloque 4: BIS, FMI, Banco Mundial).

- Crudos en ``<corpus>/mercados_series/crudo/<fuente>/<AAAAMMDD>_<nombre>`` (día de descarga en
  Nueva York) con su ``.meta.json`` (URL, estado, bytes, sha256, instante de descarga). La misma
  descarga el mismo día se sirve del archivo.
- Red solo con ``ingesta.comun.http.get`` tras ``robots_permite``. Un 403, un reto o una respuesta
  HTML no se esquivan: quedan como ``error`` en el ``.meta.json``.
- ``filas``: construye filas del contrato de series (``contrato.validar_series``).
- ``medir``: cobertura y retraso observado por serie (para fichas y CSV de metadatos).
- Tablas del bloque en ``<corpus>/mercados_series/bloque4/<fuente>.parquet`` (fuera de la base
  común: la integra ``base`` cuando se añada el cargador).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun import http
from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, HUSO_DECISION, validar_series

TEMA = "mercados_series"
CARPETA_TABLAS = "bloque4"
CSV_METADATOS = "b4_bis_fmi_bm_series.csv"
NOTA_HUECO = "hueco permanente: la fuente no publica dato de este periodo"


def ruta_crudo(fuente: str, raiz: Path | None = None) -> Path:
    """``<corpus>/mercados_series/crudo/<fuente>/`` (fuera del repositorio)."""
    base = Path(raiz) if raiz is not None else corpus_dir(TEMA) / "crudo"
    p = base / fuente
    p.mkdir(parents=True, exist_ok=True)
    return p


def hoy_ny() -> str:
    return pd.Timestamp.now(tz=HUSO_DECISION).strftime("%Y%m%d")


def descargar(fuente: str, nombre: str, url: str, *, params: dict | None = None, headers: dict | None = None,
              forzar: bool = False, raiz: Path | None = None, timeout: int = 90,
              por_segundo: float = 1.0) -> tuple[bytes | None, dict[str, Any]]:
    """Descarga ``url`` (con ``params``) como crudo ``<hoy>_<nombre>``: ``(contenido, meta)``.

    No lanza: un fallo (robots, 4xx/5xx, HTML) queda en ``meta['error']`` y el contenido es None.
    """
    carpeta = ruta_crudo(fuente, raiz)
    ruta = carpeta / f"{hoy_ny()}_{nombre}"
    ruta_meta = ruta.with_name(ruta.name + ".meta.json")
    if not forzar and ruta.exists() and ruta_meta.exists():
        return ruta.read_bytes(), json.loads(ruta_meta.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {"fuente": fuente, "nombre": nombre, "url": http.sin_clave(url), "params": params or {},
                            "descargado_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    if not http.robots_permite(url):
        meta["error"] = "robots.txt no permite la descarga con nuestro User-Agent"
    else:
        t0 = pd.Timestamp.now()
        try:
            r = http.get(url, params=params, headers=headers, timeout=timeout, reintentos=2, por_segundo=por_segundo)
            meta["estado"] = r.status_code
            meta["segundos"] = round((pd.Timestamp.now() - t0).total_seconds(), 2)
            tipo = r.headers.get("Content-Type", "")
            meta["content_type"] = tipo
            if "html" in tipo.lower():
                meta["error"] = "respuesta HTML en lugar de datos (posible reto o bloqueo; no se esquiva)"
            else:
                meta |= {"bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()}
                ruta.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001 - se documenta en el .meta.json
            resp = getattr(e, "response", None)
            meta["estado"] = getattr(resp, "status_code", None)
            meta["error"] = f"{type(e).__name__}: {http.sin_clave(str(e))[:300]} (no se esquiva)"
    ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return (ruta.read_bytes() if "error" not in meta else None), meta


def filas(serie: str, fechas_ref: Any, fechas_pub: Any, valores: Any, *, vintage: Any, unidad: str,
          divisa: str | None, fuente: str, metodo: str, nota: Any = None, disponible_utc: Any = None) -> pd.DataFrame:
    """Filas del contrato de series (sin validar: ``validar_series`` al final de cada tabla)."""
    fr = pd.to_datetime(pd.Series(fechas_ref)).reset_index(drop=True)
    n = len(fr)

    def _col(v: Any, dtype: Any = object) -> pd.Series:
        if isinstance(v, (pd.Series, np.ndarray, list)):
            return pd.Series(v, dtype=dtype).reset_index(drop=True)
        return pd.Series([v] * n, dtype=dtype)

    disp = (pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]") if disponible_utc is None
            else pd.to_datetime(_col(disponible_utc), utc=True))
    return pd.DataFrame({
        "serie": serie, "fecha_referencia": fr, "fecha_publicacion": pd.to_datetime(_col(fechas_pub)),
        "disponible_utc": disp, "vintage": _col(vintage), "valor": _col(valores, "float64"),
        "unidad": unidad, "divisa": divisa, "fuente": fuente, "metodo_publicacion": metodo, "nota": _col(nota),
    })


def unir(trozos: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatena y valida contra el contrato (tabla vacía con las columnas si no hay trozos)."""
    if not trozos:
        vacio = pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNAS_SERIES.items()})
        return validar_series(vacio)
    return validar_series(pd.concat(trozos, ignore_index=True))


def recortar_huecos(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Quita nulos antes del primer y tras el último valor; los interiores llevan ``NOTA_HUECO``."""
    s = s.sort_index()
    validos = s.dropna()
    if validos.empty:
        return s.iloc[:0], pd.Series(dtype=object)
    s = s.loc[validos.index[0]:validos.index[-1]]
    nota = pd.Series(np.where(s.isna(), NOTA_HUECO, None), index=s.index, dtype=object)
    return s, nota


def medir(tabla: pd.DataFrame, frecuencia: dict[str, str] | None = None,
          hoy: pd.Timestamp | None = None) -> pd.DataFrame:
    """Metadatos medidos por serie: inicio, fin, observaciones, vintages y retraso observado.

    ``retraso_observado_dias``: días entre el fin del último periodo con dato y hoy (cota superior
    del retraso real de publicación en fuentes sin vintages); ``lag_mediano_dias``: mediana de
    ``fecha_publicacion - fin del periodo`` en la tabla (lo que aplica el TimeGate).
    """
    hoy = pd.Timestamp(hoy or pd.Timestamp.now(tz=HUSO_DECISION).date())
    if tabla.empty:
        return pd.DataFrame(columns=["serie", "fuente", "frecuencia", "inicio", "fin", "n_obs", "n_vintages",
                                     "retraso_observado_dias", "lag_mediano_dias", "unidad"])
    out = []
    reglas = {"diaria": "D", "mensual": "M", "trimestral": "Q", "anual": "Y"}
    for sid, g in tabla[tabla["valor"].notna()].groupby("serie", sort=True):
        fr = (frecuencia or {}).get(sid, "mensual")
        fin_periodo = g["fecha_referencia"].dt.to_period(reglas.get(fr, "M")).dt.end_time.dt.normalize()
        ultimo = fin_periodo.max()
        out.append({"serie": sid, "fuente": g["fuente"].iloc[0], "frecuencia": fr,
                    "inicio": g["fecha_referencia"].min().date().isoformat(),
                    "fin": g["fecha_referencia"].max().date().isoformat(), "n_obs": int(len(g)),
                    "n_vintages": int(g["vintage"].nunique()),
                    "retraso_observado_dias": int((hoy - ultimo).days),
                    "lag_mediano_dias": float((g["fecha_publicacion"] - fin_periodo).dt.days.median()),
                    "unidad": g["unidad"].iloc[0]})
    return pd.DataFrame(out)


def guardar_tabla(tabla: pd.DataFrame, fuente: str, raiz: Path | None = None) -> Path:
    """``<corpus>/mercados_series/bloque4/<fuente>.parquet`` (validada; fuera del repositorio)."""
    base = Path(raiz) if raiz is not None else corpus_dir(TEMA) / CARPETA_TABLAS
    base.mkdir(parents=True, exist_ok=True)
    ruta = base / f"{fuente}.parquet"
    validar_series(tabla).to_parquet(ruta, index=False)
    return ruta


def guardar_metadatos(meta: pd.DataFrame, fuente: str, destino: Path | None = None) -> Path:
    """Actualiza las filas de ``fuente`` en el CSV de metadatos del bloque (sin valores)."""
    ruta = destino or resultados_dir(TEMA) / "tablas" / CSV_METADATOS
    ruta.parent.mkdir(parents=True, exist_ok=True)
    trozos = [meta]
    if ruta.exists():
        previo = pd.read_csv(ruta)
        previo = previo[previo["fuente"] != fuente]
        if len(previo):
            trozos.insert(0, previo)
    pd.concat(trozos, ignore_index=True).sort_values(["fuente", "serie"]).to_csv(ruta, index=False)
    return ruta


def resumen_medido(meta: pd.DataFrame) -> dict[str, Any]:
    """Cobertura global y retraso observado (mínimo y mediano) para la ficha."""
    if meta.empty:
        return {"cobertura_inicio": None, "cobertura_fin": None, "series": 0}
    return {"cobertura_inicio": str(meta["inicio"].min()), "cobertura_fin": str(meta["fin"].max()),
            "series": int(len(meta)), "observaciones": int(meta["n_obs"].sum()),
            "retraso_observado_dias_min": int(meta["retraso_observado_dias"].min()),
            "retraso_observado_dias_mediana": float(meta["retraso_observado_dias"].median()),
            "lag_aplicado_dias_mediana": float(meta["lag_mediano_dias"].median())}
