"""Extractor del BCE (Data Portal, ``data-api.ecb.europa.eu``) del Tema 1 (bloque 4).

API SDMX REST con ``format=csvdata`` (CSV plano: ``KEY, ..., TIME_PERIOD, OBS_VALUE``) y
``detail=dataonly``. Red solo con ``ingesta.comun.http.get`` (User-Agent identificable, 1 petición/s
por host); robots.txt permite la API. Caché cruda: un CSV por petición en
``<corpus>/mercados_series/crudo/bce/<flujo>/<AAAAMMDD>_<hash>.csv`` (fecha de descarga en Nueva
York); la misma petición el mismo día se sirve de la caché.

Series (``SERIES_BCE`` + todos los tipos de cambio de referencia):

- ``ecb:EXR.D.<X>.EUR`` (``EXR.D.<X>.EUR.SP00.A``): unidades de X por 1 EUR, TODAS las divisas que
  sirve el flujo (vigentes y retiradas; descubiertas con ``lastNObservations=1``). ``calendario``
  con el publicador ``ecb_fx`` (16:00 Europe/Berlin del mismo día: visible en la decisión de ese
  día). Las usa ``divisas.SERIE_FX``.
- ``ecb:ESTR`` (``EST.B.EU000A2X2A25.WT``): publicador ``ecb_estr`` (08:00 Berlin del hábil TARGET
  siguiente). ``ecb:EONIA`` (``EON.D.EONIA_TO.RATE``, 1999-2021, retirada; ya no está en ``FM``):
  mismo publicador (conservador: antes de 2019 salía el mismo día a las 19:00 CET).
- ``ecb:DFR`` y ``ecb:MRO`` (``FM.B.U2.EUR.4F.KR.<DFR|MRR_FR>.LEV``): una fila por fecha de cambio
  (fecha de efecto). Publicador ``ecb_tipos_oficiales`` (00:00 Berlin del día de efecto; la decisión
  se anuncia días antes: conservador).
- Revisables con historial de versiones (``includeHistory=true``): HICP de la zona euro general y
  subyacente del flujo NUEVO ``HICP`` (proveedor ``4D0``, ECOICOP v2, ``U2`` = composición
  cambiante), paro (``LFSI``, ``U2``) y PIB real (``MNA``, ``I10`` = zona euro 21 de composición
  fija: el flujo no tiene ``U2``). ``vintage_exacto``: una fila por versión (``ACTION=Replace``) con
  ``disponible_utc`` = ``VALID_FROM`` (instante real de carga en el Data Portal) y vintage = su día en
  Berlín. Medido (2026-10-13): en estos flujos ``includeHistory`` devuelve UNA versión por
  observación (la vigente, con su ``VALID_FROM``), no la historia de revisiones. Por eso cada
  observación cuya regla de publicación (``previa``: clave de lags.yaml con sus tramos, ``catalogo.fecha_publicacion_clave``)
  cae antes de su primera versión guardada se fecha con la regla (``lag_fijo``) y la nota
  ``NOTA_PREVIA`` (valor revisado: posible fuga de revisiones); el resto, con su ``VALID_FROM``.

``medir_include_history``: qué devuelve ``includeHistory`` sin ``startPeriod`` (versiones por
observación, acciones, fechas de la primera y última versión) → ficha ``bce``.

Datos solo en local. Al repositorio: código, ficha y metadatos (``tablas/bce_series.csv``).
CLI: ``python -m ingesta.mercados_series.fuentes.bce [--sin-red] [--fichas] [--escribir]``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ingesta.comun import http
from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.mercados_series import catalogo, visibilidad
from ingesta.mercados_series.contrato import (
    COLUMNAS_SERIES,
    HUSO_DECISION,
    UTC,
    primera_decision_vec,
    vacio_series,
    validar_series,
)

API_BCE = "https://data-api.ecb.europa.eu/service/data"
TEMA = "mercados_series"
FUENTE = "bce"
RUTA_FRAGMENTO = Path(__file__).resolve().parents[1] / "config" / "fragmentos" / "b4_bce_eurostat.yaml"
LICENCIA_BCE = ("BCE: reutilización libre con cita de la fuente (ecb.europa.eu, «Reuse of ECB statistics»); "
                "EONIA: datos de EMMI/EBF, solo local")
NOTA_PREVIA = ("anterior al historial de versiones del BCE: valor de la primera versión guardada, fechado con la "
               "regla de publicación (posible fuga de revisiones)")
NOTA_POSTERIOR_BCE = ("serie creada después del periodo (retropolación): visible solo desde su primera versión en el "
                      "BCE; para el pasado usar la versión vigente entonces (ICP/HICP ECOICOP v1, EA cambiante)")
NOTA_FX_RETIRADA = "divisa retirada del fijado de referencia del BCE (sin datos posteriores)"
# Divisas de referencia del BCE (vigentes): el resto que devuelve el flujo son retiradas (BGN, HRK, GRD...).
PERIODO_INICIO = "1999-01-01"

# id -> especificación. 'clave' = flujo/clave SDMX; 'previa' = lag_clave de lags.yaml (vintage_exacto).
SERIES_BCE: dict[str, dict[str, Any]] = {
    "ecb:ESTR": {"clave": "EST/B.EU000A2X2A25.WT", "nombre": "€STR (euro short-term rate), media recortada",
                 "frecuencia": "diaria", "unidad": "% anual", "metodo": "calendario", "publicador": "ecb_estr",
                 "revisable": False, "estado": "activa"},
    "ecb:EONIA": {"clave": "EON/D.EONIA_TO.RATE", "nombre": "EONIA (euro overnight index average)",
                  "frecuencia": "diaria", "unidad": "% anual", "metodo": "calendario", "publicador": "ecb_estr",
                  "revisable": False, "estado": "retirada", "sucesor": "ecb:ESTR",
                  "licencia": "EMMI/EBF vía BCE: solo local, sin redistribuir"},
    "ecb:DFR": {"clave": "FM/B.U2.EUR.4F.KR.DFR.LEV", "nombre": "Tipo de la facilidad de depósito del BCE (fecha de efecto)",
                "frecuencia": "diaria", "unidad": "% anual", "metodo": "calendario",
                "publicador": "ecb_tipos_oficiales", "revisable": False, "estado": "activa"},
    "ecb:MRO": {"clave": "FM/B.U2.EUR.4F.KR.MRR_FR.LEV",
                "nombre": "Tipo de las operaciones principales de financiación del BCE (fecha de efecto)",
                "frecuencia": "diaria", "unidad": "% anual", "metodo": "calendario",
                "publicador": "ecb_tipos_oficiales", "revisable": False, "estado": "activa"},
    "ecb:HICP.M.U2.N.000000.4D0.ANR": {
        "clave": "HICP/M.U2.N.000000.4D0.ANR", "nombre": "HICP zona euro (composición cambiante), general, tasa anual",
        "frecuencia": "mensual", "unidad": "% interanual", "metodo": "vintage_exacto", "publicador": "eurostat",
        "revisable": True, "estado": "activa", "previa": "CPIAUCSL", "solo_desde_primer_vintage": True},
    "ecb:HICP.M.U2.N.XEF000.4D0.ANR": {
        "clave": "HICP/M.U2.N.XEF000.4D0.ANR",
        "nombre": "HICP zona euro (composición cambiante) sin energía, alimentos, alcohol ni tabaco, tasa anual",
        "frecuencia": "mensual", "unidad": "% interanual", "metodo": "vintage_exacto", "publicador": "eurostat",
        "revisable": True, "estado": "activa", "previa": "CPILFESL", "solo_desde_primer_vintage": True},
    "ecb:LFSI.M.U2.S.UNEHRT.TOTAL0.15_74.T": {
        "clave": "LFSI/M.U2.S.UNEHRT.TOTAL0.15_74.T",
        "nombre": "Tasa de paro de la zona euro (composición cambiante), 15-74 años, desestacionalizada",
        "frecuencia": "mensual", "unidad": "% población activa", "metodo": "vintage_exacto", "publicador": "eurostat",
        "revisable": True, "estado": "activa", "previa": "EUROSTAT_PARO"},
    "ecb:MNA.Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY": {
        "clave": "MNA/Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY",
        "nombre": "PIB real de la zona euro 21 (composición fija), tasa interanual",
        "frecuencia": "trimestral", "unidad": "% interanual", "metodo": "vintage_exacto", "publicador": "eurostat",
        "revisable": True, "estado": "activa", "previa": "EUROSTAT_PIB", "solo_desde_primer_vintage": True,
        "notas": ["MNA no tiene la zona euro de composición cambiante (U2): se usa I10 (EA21 fija desde 2026)"]},
}
CLAVE_FX = "EXR/D..EUR.SP00.A"

Transporte = Callable[[str, dict[str, Any]], str]

__all__ = [
    "API_BCE", "SERIES_BCE", "NOTA_PREVIA", "ClienteBce", "parsear_csv", "id_fx", "series_fx",
    "tabla_calendario", "tabla_versiones", "tabla_bce", "medir_include_history", "publicaciones_b4",
    "entradas_bce", "ficha_bce", "main",
]


def _hoy_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=HUSO_DECISION).tz_localize(None).normalize()


# --- Cliente con caché cruda ---------------------------------------------------------------------
class ClienteCsv:
    """Cliente SDMX-CSV con caché cruda por petición (``<corpus>/mercados_series/crudo/<nombre>/``).

    ``transporte``: ``(url, params) -> texto`` (por defecto ``http.get``; los tests pasan uno
    sintético). ``peticiones`` cuenta las que salen a la red; ``bytes``, lo leído (red o caché).
    """

    nombre = "csv"
    api = ""

    def __init__(self, *, transporte: Transporte | None = None, carpeta: Path | None = None,
                 cache: bool = True, sin_red: bool = False, hoy: Any = None):
        self._transporte = transporte
        self.carpeta = Path(carpeta) if carpeta is not None else None
        self.cache = cache
        self.sin_red = sin_red
        self.hoy = pd.Timestamp(hoy).normalize() if hoy is not None else _hoy_ny()
        self.peticiones = 0
        self.bytes = 0

    def _carpeta(self) -> Path:
        if self.carpeta is None:
            self.carpeta = corpus_dir(TEMA) / "crudo" / self.nombre
        return self.carpeta

    @staticmethod
    def huella(ruta: str, params: dict[str, Any]) -> str:
        texto = ruta + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]

    def _dir(self, ruta: str) -> Path:
        return self._carpeta() / ruta.split("/", 1)[0]

    def ruta_cache(self, ruta: str, params: dict[str, Any]) -> Path:
        return self._dir(ruta) / f"{self.hoy:%Y%m%d}_{self.huella(ruta, params)}.csv"

    def ultima_cache(self, ruta: str, params: dict[str, Any]) -> Path | None:
        """Última descarga archivada de la petición (cualquier día) o None."""
        h = self.huella(ruta, params)
        candidatos = sorted(self._dir(ruta).glob(f"*_{h}.csv")) if self._dir(ruta).exists() else []
        return candidatos[-1] if candidatos else None

    def texto(self, ruta: str, params: dict[str, Any] | None = None) -> str:
        """CSV de ``ruta`` (caché del día, red o, con ``sin_red``, la última archivada)."""
        params = dict(params or {})
        destino = self.ruta_cache(ruta, params)
        if self.cache and destino.exists():
            t = destino.read_text(encoding="utf-8")
            self.bytes += len(t.encode("utf-8"))
            return t
        if self.sin_red:
            previa = self.ultima_cache(ruta, params)
            if previa is None:
                raise FileNotFoundError(f"{self.nombre}: sin descarga archivada de {ruta} {params}")
            t = previa.read_text(encoding="utf-8")
            self.bytes += len(t.encode("utf-8"))
            return t
        url = f"{self.api}/{ruta}"
        if self._transporte is not None:
            t = self._transporte(url, params)
        else:
            t = http.get(url, params=params, por_segundo=1.0, reintentos=3, timeout=180).text
        self.peticiones += 1
        self.bytes += len(t.encode("utf-8"))
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(".csv.tmp")
        tmp.write_text(t, encoding="utf-8")
        os.replace(tmp, destino)
        return t


class ClienteBce(ClienteCsv):
    """Cliente del Data Portal del BCE (``format=csvdata``)."""

    nombre = "bce"
    api = API_BCE

    def datos(self, clave: str, **params: Any) -> pd.DataFrame:
        p = {"format": "csvdata", **params}
        return parsear_csv(self.texto(clave, p))


# --- Parseo ---------------------------------------------------------------------------------------
def parsear_csv(texto: str) -> pd.DataFrame:
    """CSV del BCE → ``KEY, TIME_PERIOD, OBS_VALUE`` (+ ``ACTION, VALID_FROM, VALID_TO, TITLE`` si vienen).

    ``OBS_VALUE`` numérico (vacío → NaN); ``VALID_FROM``/``VALID_TO`` a UTC. Texto vacío → tabla vacía.
    """
    cols = ["KEY", "TIME_PERIOD", "OBS_VALUE"]
    if not texto.strip():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False, na_values=[""])
    extra = [c for c in ("ACTION", "VALID_FROM", "VALID_TO", "TITLE") if c in df.columns]
    df = df[[*cols, *extra]].copy()
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    for c in ("VALID_FROM", "VALID_TO"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df


def fecha_periodo(periodos: pd.Series) -> pd.Series:
    """``TIME_PERIOD`` SDMX (``AAAA-MM-DD``, ``AAAA-MM``, ``AAAA-Qn``, ``AAAA``) → primer día del periodo."""
    s = periodos.astype(str)
    trimestre = s.str.fullmatch(r"\d{4}-Q[1-4]")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if trimestre.any():
        out[trimestre] = pd.PeriodIndex(s[trimestre].str.replace("-", ""), freq="Q").to_timestamp()
    if (~trimestre).any():
        out[~trimestre] = pd.to_datetime(s[~trimestre], format="mixed")
    return out.astype("datetime64[ns]")


def id_fx(clave_sdmx: str) -> str:
    """``EXR.D.USD.EUR.SP00.A`` → ``ecb:EXR.D.USD.EUR`` (id de ``divisas.SERIE_FX``)."""
    partes = clave_sdmx.split(".")
    return f"ecb:EXR.D.{partes[2]}.EUR"


# --- Publicación ----------------------------------------------------------------------------------
def _fragmento() -> dict[str, Any]:
    if not RUTA_FRAGMENTO.exists():
        return {}
    return yaml.safe_load(RUTA_FRAGMENTO.read_text(encoding="utf-8")) or {}


def publicaciones_b4(publicaciones: dict[str, Any] | None = None) -> dict[str, Any]:
    """Configuración de publicaciones con los publicadores del fragmento del bloque 4 añadidos.

    Hasta que el integrador funda el fragmento en ``publicaciones.yaml``, sus publicadores
    (``ecb_tipos_oficiales``) solo existen aquí; uno ya definido en la base no se pisa.
    """
    base = publicaciones if publicaciones is not None else visibilidad.cargar_publicaciones()
    nuevos = {k: v for k, v in (_fragmento().get("publicadores") or {}).items() if k not in base["publicadores"]}
    if not nuevos:
        return base
    out = dict(base)
    pubs = dict(base["publicadores"])
    for k, v in nuevos.items():
        v = dict(v)
        v.setdefault("margen_min", 0)
        if v.get("dia") != "mismo":
            v.setdefault("habiles", "target")
        pubs[k] = v
    out["publicadores"] = pubs
    return out


def _disponible_vec(publicador: str, fechas: pd.Series, publicaciones: dict[str, Any]) -> pd.Series:
    res = {pd.Timestamp(d): visibilidad.disponible_publicador(publicador, pd.Timestamp(d), publicaciones)
           for d in pd.unique(fechas.dropna())}
    return pd.Series([res.get(pd.Timestamp(d), pd.NaT) for d in fechas], index=fechas.index, dtype=UTC)


def _filas(sid: str, ref: pd.Series, pub: pd.Series, disp: pd.Series, vintage: Any, valor: pd.Series,
           espec: dict[str, Any], metodo: Any, nota: Any) -> pd.DataFrame:
    out = pd.DataFrame({
        "serie": sid, "fecha_referencia": ref.to_numpy(), "fecha_publicacion": pub.to_numpy(),
        "disponible_utc": disp.to_numpy(), "vintage": vintage, "valor": valor.to_numpy(),
        "unidad": espec["unidad"], "divisa": espec.get("divisa"), "fuente": espec.get("fuente", FUENTE),
        "metodo_publicacion": metodo, "nota": nota,
    })
    out["disponible_utc"] = pd.to_datetime(out["disponible_utc"], utc=True)
    return out[list(COLUMNAS_SERIES)]


def tabla_calendario(sid: str, obs: pd.DataFrame, espec: dict[str, Any], *,
                     publicaciones: dict[str, Any] | None = None) -> pd.DataFrame:
    """Observaciones vigentes (``KEY, TIME_PERIOD, OBS_VALUE``) → filas ``calendario`` (vintage ``unico``).

    Valores vacíos: se descartan (en el BCE, un día sin dato es un día sin fijado). Fecha de
    publicación = la del publicador para la fecha de referencia.
    """
    if obs.empty:
        return vacio_series()
    df = obs.dropna(subset=["OBS_VALUE"]).reset_index(drop=True)
    if df.empty:
        return vacio_series()
    pubs = publicaciones_b4(publicaciones)
    ref = fecha_periodo(df["TIME_PERIOD"])
    disp = _disponible_vec(espec["publicador"], ref, pubs)
    return _filas(sid, ref, primera_decision_vec(disp), disp, "unico", df["OBS_VALUE"], espec, "calendario", None)


def tabla_versiones(sid: str, hist: pd.DataFrame, espec: dict[str, Any]) -> pd.DataFrame:
    """Historial de versiones (``includeHistory=true``) → filas ``vintage_exacto``.

    Una fila por versión ``Replace`` con valor (la última de cada día en Berlín si hay varias);
    ``disponible_utc`` = ``VALID_FROM``. ``Delete`` y versiones sin valor se descartan. Las
    observaciones cuya regla (``espec['previa']``, lags.yaml) cae antes del primer vintage de la
    serie fechan su PRIMERA versión con la regla (``lag_fijo`` + ``NOTA_PREVIA``).
    """
    if hist.empty or "VALID_FROM" not in hist:
        return vacio_series()
    df = hist[(hist.get("ACTION", "Replace") == "Replace") & hist["OBS_VALUE"].notna()].copy()
    if df.empty:
        return vacio_series()
    df["ref"] = fecha_periodo(df["TIME_PERIOD"])
    df["vintage"] = df["VALID_FROM"].dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m-%d")
    df = (df.sort_values(["ref", "VALID_FROM"]).drop_duplicates(["ref", "vintage"], keep="last")
          .reset_index(drop=True))
    disp = df["VALID_FROM"].astype(UTC)
    pub = primera_decision_vec(disp)
    metodo = pd.Series("vintage_exacto", index=df.index, dtype=object)
    nota = pd.Series(None, index=df.index, dtype=object)
    if espec.get("previa"):
        regla = catalogo.fecha_publicacion_clave(df["ref"], espec["previa"])
        # Por observación: la regla cae antes que su primera versión guardada (carga en bloque o revisión
        # posterior): la primera versión se fecha con la regla (nunca después de lo que ya se veía).
        primera = ~df.duplicated("ref", keep="first")
        previa = primera & (regla < pub)
        if espec.get("solo_desde_primer_vintage"):
            # Serie creada después (HICP ECOICOP v2, agregado I10 de composición fija): su retropolación
            # no existía antes de su primera versión; nunca visible antes de ella. Solo es retropolación lo
            # que la regla daba por publicado antes de crear la serie; una observación posterior cuya
            # primera versión guardada ya es una revisión lleva NOTA_PREVIA (posible fuga de revisiones).
            retro = previa & (regla < pub.min())
            regla = regla.where(regla >= pub.min(), pub.min())
            nota[retro] = NOTA_POSTERIOR_BCE
        pub = pub.where(~previa, regla)
        disp = disp.where(~previa, pd.NaT)
        metodo[previa] = "lag_fijo"
        nota[previa] = nota[previa].fillna(NOTA_PREVIA)
    return _filas(sid, df["ref"], pd.Series(pub), pd.Series(disp, dtype=UTC), df["vintage"].to_numpy(),
                  df["OBS_VALUE"], espec, metodo.to_numpy(), nota.to_numpy())


# --- Descarga y tabla -------------------------------------------------------------------------------
def series_fx(cliente: ClienteBce) -> pd.DataFrame:
    """Divisas del flujo EXR diario de referencia (``SP00.A``): clave, divisa, título y último dato."""
    df = cliente.datos(CLAVE_FX, lastNObservations=1)
    df["divisa"] = df["KEY"].str.split(".").str[2]
    df["id"] = df["KEY"].map(id_fx)
    df["ultimo"] = df["TIME_PERIOD"]
    return df[["id", "KEY", "divisa", "TITLE", "ultimo"]].sort_values("id").reset_index(drop=True)


def entradas_bce(fx: pd.DataFrame | None = None) -> dict[str, dict[str, Any]]:
    """Especificación de todas las series del BCE (``SERIES_BCE`` + una por divisa de ``fx``)."""
    out = {k: dict(v, fuente=FUENTE, divisa=v.get("divisa")) for k, v in SERIES_BCE.items()}
    if fx is not None:
        hoy = _hoy_ny()
        for r in fx.itertuples(index=False):
            retirada = pd.Timestamp(r.ultimo) < hoy - pd.Timedelta(days=60)
            out[r.id] = {"clave": f"EXR/{r.KEY.split('.', 1)[1]}", "nombre": f"{r.TITLE} ({r.divisa} por 1 EUR)",
                         "frecuencia": "diaria", "unidad": f"{r.divisa} por 1 EUR", "divisa": r.divisa,
                         "metodo": "calendario", "publicador": "ecb_fx", "revisable": False, "fuente": FUENTE,
                         "estado": "retirada" if retirada else "activa",
                         "notas": [NOTA_FX_RETIRADA] if retirada else []}
    return out


def tabla_bce(cliente: ClienteBce | None = None, *, sin_red: bool = False,
              publicaciones: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Todas las series del BCE validadas con el contrato + mediciones por serie.

    Pocas peticiones: una para descubrir divisas, una con TODO el EXR diario, una por serie de
    calendario y una ``includeHistory`` por revisable.
    """
    cli = cliente or ClienteBce(sin_red=sin_red)
    fx = series_fx(cli)
    espec = entradas_bce(fx)
    partes: list[pd.DataFrame] = []
    med: dict[str, Any] = {}
    todo_fx = cli.datos(CLAVE_FX, detail="dataonly", startPeriod=PERIODO_INICIO)
    for sid, e in espec.items():
        if sid.startswith("ecb:EXR."):
            clave_sdmx = "EXR." + e["clave"].split("/", 1)[1]
            obs = todo_fx[todo_fx["KEY"] == clave_sdmx]
            t = tabla_calendario(sid, obs, e, publicaciones=publicaciones)
        elif e["metodo"] == "calendario":
            obs = cli.datos(e["clave"], detail="dataonly")
            t = tabla_calendario(sid, obs, e, publicaciones=publicaciones)
        else:
            hist = cli.datos(e["clave"], detail="dataonly", includeHistory="true")
            t = tabla_versiones(sid, hist, e)
            med[sid] = {"include_history": _resumen_historial(hist)}
        partes.append(t)
        med.setdefault(sid, {}).update(_resumen_tabla(t))
    df = validar_series(pd.concat([p for p in partes if len(p)], ignore_index=True))
    return df, {"series": med, "peticiones": cli.peticiones, "bytes": cli.bytes}


def _resumen_tabla(t: pd.DataFrame) -> dict[str, Any]:
    if t.empty:
        return {"n_filas": 0}
    ultima = t.sort_values("fecha_publicacion").drop_duplicates("fecha_referencia", keep="first")
    med = {"n_filas": int(len(t)), "n_obs": int(t["fecha_referencia"].nunique()),
           "inicio": t["fecha_referencia"].min().date().isoformat(),
           "fin": t["fecha_referencia"].max().date().isoformat()}
    retraso = (ultima["fecha_publicacion"] - ultima["fecha_referencia"]).dt.days
    if (ultima["metodo_publicacion"] == "vintage_exacto").any():
        r = retraso[ultima["metodo_publicacion"] == "vintage_exacto"]
        med["retraso_primera_version_dias"] = {"mediana": float(r.median()), "min": int(r.min()), "max": int(r.max())}
        med["n_previa_regla"] = int((t["metodo_publicacion"] == "lag_fijo").sum())
    else:
        med["retraso_dias_mediana"] = float(retraso.median())
    return med


def _resumen_historial(hist: pd.DataFrame) -> dict[str, Any]:
    if hist.empty or "VALID_FROM" not in hist:
        return {"n_filas": 0}
    acciones = hist["ACTION"].value_counts().to_dict() if "ACTION" in hist else {}
    rep = hist[hist.get("ACTION", "Replace") == "Replace"]
    versiones = rep.groupby("TIME_PERIOD").size()
    vf = rep["VALID_FROM"].dropna()
    return {"n_filas": int(len(hist)), "acciones": {k: int(v) for k, v in acciones.items()},
            "versiones_por_obs_max": int(versiones.max()) if len(versiones) else 0,
            "obs_con_mas_de_una_version": int((versiones > 1).sum()),
            "primera_version": vf.min().isoformat() if len(vf) else None,
            "ultima_version": vf.max().isoformat() if len(vf) else None,
            "dias_con_version": int(vf.dt.tz_convert("Europe/Berlin").dt.date.nunique())}


def medir_include_history(cliente: ClienteBce, clave: str = "ICP/M.U2.N.000000.4.ANR") -> dict[str, Any]:
    """¿Devuelve ``includeHistory`` historia real de versiones SIN ``startPeriod``? (flujo ICP antiguo).

    Compara el número de filas con el de observaciones vigentes y resume acciones y fechas.
    """
    hist = cliente.datos(clave, detail="dataonly", includeHistory="true")
    vigente = cliente.datos(clave, detail="dataonly")
    res = _resumen_historial(hist)
    res.update(clave=clave, n_obs_vigentes=int(vigente["OBS_VALUE"].notna().sum()))
    res["devuelve_historia"] = bool(res.get("obs_con_mas_de_una_version", 0) > 0)
    return res


# --- Metadatos, fragmento y ficha -------------------------------------------------------------------
def tabla_metadatos(espec: dict[str, dict[str, Any]], med: dict[str, Any]) -> pd.DataFrame:
    filas = []
    for sid, e in sorted(espec.items()):
        m = med.get(sid, {})
        filas.append({"id": sid, "clave_sdmx": e["clave"], "nombre": e["nombre"], "frecuencia": e["frecuencia"],
                      "unidad": e["unidad"], "divisa": e.get("divisa"), "metodo": e["metodo"],
                      "publicador": e["publicador"], "revisable": e["revisable"], "estado": e["estado"],
                      "inicio": m.get("inicio"), "fin": m.get("fin"), "n_obs": m.get("n_obs"),
                      "n_filas": m.get("n_filas")})
    return pd.DataFrame(filas)


def entrada_catalogo(sid: str, e: dict[str, Any], licencia: str, primer_vintage: str | None = None) -> dict[str, Any]:
    """Entrada con el esquema de ``catalogo_series.yaml`` (para el fragmento del bloque 4)."""
    out: dict[str, Any] = {
        "alias": None, "nombre": e["nombre"], "fuente": e.get("fuente", FUENTE), "id_origen": e["clave"],
        "frecuencia": e["frecuencia"], "unidad": e["unidad"], "divisa": e.get("divisa"), "metodo": e["metodo"],
        "revisable": e["revisable"], "rol_fase1": None, "licencia": e.get("licencia", licencia),
        "estado": e["estado"], "verificado": False,
    }
    if e.get("publicador"):
        out["publicador"] = e["publicador"]
    if e.get("lag_clave"):
        out["lag_clave"] = e["lag_clave"]
    if e.get("sucesor"):
        out["sucesor"] = e["sucesor"]
    if e.get("previa") and primer_vintage:
        out["regla_previa_alfred"] = {"lag_clave": e["previa"], "primer_vintage": primer_vintage}
    if e.get("notas"):
        out["notas"] = list(e["notas"])
    return out


def ficha_bce(df: pd.DataFrame, med: dict[str, Any], include_history: dict[str, Any]):
    from ingesta.comun.esquema import FichaFuente
    from ingesta.comun.fichas import guardar_ficha

    fx = df[df["serie"].str.startswith("ecb:EXR.")]
    ficha = FichaFuente(
        fuente=FUENTE, nombre="Banco Central Europeo: Data Portal (EXR, EST, EON, FM, HICP 4D0, LFSI, MNA)",
        tema=TEMA, agentes=["macro", "risk", "portfolio"], rag="serie", url="https://data-api.ecb.europa.eu/",
        acceso="api", autenticacion="ninguna", coste="0 €", licencia_y_terminos=LICENCIA_BCE,
        cobertura_inicio=df["fecha_referencia"].min().date().isoformat(),
        cobertura_fin=df["fecha_referencia"].max().date().isoformat(),
        frecuencia="diaria (FX, €STR, EONIA, tipos oficiales), mensual (HICP, paro), trimestral (PIB)",
        retraso_publicacion=(
            "FX: mismo día 16:00 Berlin (visible en la decisión del día); €STR: 08:00 Berlin del hábil TARGET "
            "siguiente; HICP/paro/PIB: VALID_FROM medido por versión (ver mediciones.series.*.retraso_primera_version_dias)"),
        formato="csv (SDMX csvdata)",
        volumen_estimado=f"{med.get('bytes', 0) / 1e6:.1f} MB de CSV por descarga completa (~10 peticiones)",
        limites="sin límite publicado; 1 petición/s por host; robots.txt permite /service/data",
        estado_sonda="ok",
        mediciones={"n_series": int(df["serie"].nunique()), "n_filas": int(len(df)),
                    "n_divisas_fx": int(fx["serie"].nunique()), "include_history_sin_startPeriod": include_history,
                    **med},
        notas=[
            "EONIA ya no está en FM (404): se sirve en el flujo EON (EON.D.EONIA_TO.RATE), 1999-2021",
            "HICP: flujo NUEVO HICP con proveedor 4D0 (ECOICOP v2, U2 = composición cambiante); el flujo ICP "
            "antiguo queda como referencia de la ruptura de 2026",
            "PIB: MNA no publica U2; se usa I10 (zona euro 21, composición fija desde el 1-1-2026)",
            "includeHistory=true (sin startPeriod ni updatedAfter) responde con ACTION/VALID_FROM/VALID_TO, pero en "
            "HICP 4D0, LFSI y MNA da UNA versión por observación (la vigente): no hay historia de revisiones; en el "
            "flujo ICP antiguo, ver include_history_sin_startPeriod",
            "Revisables: observación con regla de publicación anterior a su VALID_FROM → fechada con la regla "
            "(lag_fijo, valor revisado, nota); si no, VALID_FROM (instante real de carga)",
            "Tipos oficiales: una fila por fecha de efecto; visibles ese día (la decisión se anuncia antes)",
            "TWD y otras divisas no están en el fijado de referencia del BCE",
        ],
    )
    return guardar_ficha(ficha)


def escribir_fragmento_bce(espec: dict[str, dict[str, Any]], df: pd.DataFrame) -> dict[str, Any]:
    """Entradas de catálogo del BCE (para fundir en el fragmento del bloque 4)."""
    out = {}
    for sid, e in sorted(espec.items()):
        pv = None
        sub = df[(df["serie"] == sid) & (df["metodo_publicacion"] == "vintage_exacto")]
        if len(sub):
            pv = str(sub["vintage"].min())
        out[sid] = entrada_catalogo(sid, e, LICENCIA_BCE, pv)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extractor del BCE (Tema 1, bloque 4)")
    ap.add_argument("--sin-red", action="store_true", help="usa solo las descargas archivadas")
    ap.add_argument("--fichas", action="store_true", help="escribe la ficha y la tabla de metadatos")
    ap.add_argument("--escribir", action="store_true", help="escribe las series en la base del tema")
    a = ap.parse_args(argv)
    cli = ClienteBce(sin_red=a.sin_red)
    df, med = tabla_bce(cli)
    print(f"BCE: {df['serie'].nunique()} series, {len(df)} filas, {cli.peticiones} peticiones")
    if a.fichas:
        ih = medir_include_history(cli)
        ficha_bce(df, med, ih)
        espec = entradas_bce(series_fx(cli))
        carpeta = resultados_dir(TEMA) / "tablas"
        carpeta.mkdir(parents=True, exist_ok=True)
        tabla_metadatos(espec, med["series"]).to_csv(carpeta / "bce_series.csv", index=False)
    if a.escribir:
        from ingesta.mercados_series import almacen
        almacen.escribir_series(df)


if __name__ == "__main__":
    main()
