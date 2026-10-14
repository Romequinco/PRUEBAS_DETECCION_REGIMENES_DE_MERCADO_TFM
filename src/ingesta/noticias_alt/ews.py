"""EWS · «Apocalypse Early Warning System» de jets privados (Kyle McDonald): botón de cataclismo.

El panel ``ews.kylemcdonald.net`` carga ``dashboard.json`` de un bucket público de Cloudflare R2
(código MIT en github.com/kylemcdonald/ews; feed sin documentar). Cada ~30 min cuenta los jets de negocio
en el aire (ADS-B Exchange) y los compara con una línea base semanal ajustada por festivos de EE. UU.:
``zScore`` y ``emergencyLevel`` 1-5 (alarma con z >= 4,7; elevado con z >= 2,35).

``trends.archive`` trae el histórico de UN AÑO MÓVIL (``t0`` avanza con cada instantánea): ``c`` conteo,
``p`` esperado, ``s`` desviación, ``z``; los instantes van codificados por tramos en ``tr``
(``[[paso_ms, repeticiones], ...]``), y el muestreo NO es regular. ``p``/``s``/``z`` del archivo se
recalculan en cada instantánea (revisión ex post, ver ``ews_revision``); ``c`` no cambia. Por eso: solo
vivo (``backtest=False``), archivando cada instantánea en el corpus, y un z causal propio para medir.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingesta.comun.http import get, robots_permite
from ingesta.noticias_alt import base

MODULO = "ews"
WEB = "https://ews.kylemcdonald.net"
R2 = "https://pub-49bb6a6f314c47be9b481c25e5f6ca9e.r2.dev/"
PANELES = {"jets": "dashboard.json", "militar": "military-dashboard.json", "sin_icao": "untracked-dashboard.json"}
RSS = "https://ews.kylemcdonald.net/rss.xml"
ALARMA, ELEVADO = 4.7, 2.35
UMBRALES = (ELEVADO, 3.0, 4.0, ALARMA)
TZ_PANEL = "America/Los_Angeles"   # zona de la línea base del panel (``concurrentPredictionTimeZone``)
LICENCIA = "código MIT; datos derivados de ADS-B Exchange (términos de ADSBx: uso no comercial); feed no documentado"


# --------------------------------------------------------------------------- red

def panel(nombre: str = "jets") -> dict[str, Any]:
    """Instantánea JSON de un panel (``jets``, ``militar``, ``sin_icao``)."""
    r = get(R2 + PANELES[nombre], timeout=60, reintentos=2)
    r.raise_for_status()
    return r.json()


def _ruta_instantanea(d: dict[str, Any]) -> Path:
    gen = d.get("snapshotGeneratedAt") or d.get("current", {}).get("asOf") or datetime.now(timezone.utc).isoformat()
    ts = pd.Timestamp(gen).tz_convert("UTC") if pd.Timestamp(gen).tzinfo else pd.Timestamp(gen)
    return base.corpus("crudo", MODULO) / f"dashboard_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"


def instantaneas() -> list[Path]:
    """Instantáneas archivadas en el corpus, de la más antigua a la más reciente."""
    return sorted(base.corpus("crudo", MODULO).glob("dashboard_*.json"))


# --------------------------------------------------------------------------- parseo

def estado_actual(d: dict[str, Any]) -> dict[str, Any]:
    c = d.get("current", {})
    return {k: c.get(k) for k in ("asOf", "concurrentCount", "baselineMean", "baselineStdDev", "zScore", "alertLevel",
                                  "emergencyLevel", "alarmSigmaThreshold", "elevatedSigmaThreshold")}


def tiempos_archivo(a: dict[str, Any]) -> pd.DatetimeIndex:
    """Instantes del archivo: ``t0`` + pasos de ``tr`` (tramos ``[paso_ms, n]``); sin ``tr``, 30 min."""
    n = len(a.get("c", []))
    t0 = pd.Timestamp(a["t0"])
    t0 = t0.tz_convert("UTC") if t0.tzinfo else t0.tz_localize("UTC")
    if "tr" in a:
        pasos = np.concatenate([[0]] + [np.full(int(k), int(ms), dtype="int64") for ms, k in a["tr"]])
    else:
        pasos = np.r_[0, np.full(max(n - 1, 0), 1_800_000, dtype="int64")]
    if len(pasos) != n:
        raise ValueError(f"archivo incoherente: {len(pasos)} instantes para {n} conteos")
    return t0 + pd.to_timedelta(np.cumsum(pasos), unit="ms")


def serie_archivo(d: dict[str, Any]) -> pd.DataFrame:
    """``trends.archive`` -> DataFrame (ts, conteo, esperado, sd, z_publicado), un año móvil."""
    a = d["trends"]["archive"]
    return pd.DataFrame(dict(ts=tiempos_archivo(a), conteo=pd.to_numeric(pd.Series(a["c"]), errors="coerce"),
                             esperado=a.get("p"), sd=a.get("s"), z_publicado=a.get("z")))


PISO_REL = 0.12      # suelo de la desviación (× media) del z causal que se guarda: sin él, sd≈0 da z de −60


def z_causal(serie: pd.DataFrame | pd.Series, ventana: int = 8, minimo: int = 4, piso_rel: float = 0.0,
             tz: str = TZ_PANEL) -> pd.Series:
    """z con línea base SOLO del pasado: mismo hueco semanal (día × media hora, hora de ``tz``).

    Para cada punto, media y desviación de los ``ventana`` valores anteriores del mismo hueco (al menos
    ``minimo``; si no, NaN). ``piso_rel`` acota la desviación por debajo a ``piso_rel × media`` (el panel
    usa un suelo parecido). Nunca usa el punto actual ni posteriores.
    """
    if isinstance(serie, pd.DataFrame):
        ts, x = pd.to_datetime(serie["ts"], utc=True), serie["conteo"].astype(float)
    else:
        ts, x = pd.to_datetime(serie.index, utc=True).to_series(index=serie.index), serie.astype(float)
    local = pd.DatetimeIndex(ts).tz_convert(tz)
    hueco = pd.Series(local.weekday * 48 + local.hour * 2 + local.minute // 30, index=x.index)
    previo = x.groupby(hueco).shift(1)
    g = previo.groupby(hueco)
    media = g.transform(lambda s: s.rolling(ventana, min_periods=minimo).mean())
    sd = g.transform(lambda s: s.rolling(ventana, min_periods=minimo).std(ddof=0))
    if piso_rel > 0:
        sd = np.maximum(sd, piso_rel * media)
    sd = sd.where(sd > 0)
    return ((x - media) / sd).rename("z_causal")


def nivel(z: float | None, alarma: float = ALARMA, elevado: float = ELEVADO) -> int:
    """z -> nivel 1-5 aproximado (0 si no hay z). El mapeo exacto del panel no está documentado."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return 0
    if z >= alarma:
        return 5
    if z >= elevado:
        return 4 if z >= (alarma + elevado) / 2 else 3
    return 2 if z >= 1 else 1


def leer_rss(texto: str) -> list[dict[str, Any]]:
    """RSS 2.0 de alertas de nivel 5 -> [{titulo, enlace, fecha}]."""
    raiz = ET.fromstring(texto)
    out = []
    for it in raiz.iter("item"):
        out.append(dict(titulo=(it.findtext("title") or "").strip(), enlace=(it.findtext("link") or "").strip(),
                        fecha=(it.findtext("pubDate") or "").strip()))
    return out


def revision(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    """Compara dos instantáneas del archivo en los instantes comunes: qué se revisa ex post."""
    m = a.set_index("ts").join(b.set_index("ts"), how="inner", lsuffix="_a", rsuffix="_b")
    out: dict[str, Any] = dict(n_comunes=int(len(m)))
    for k in ("conteo", "esperado", "sd", "z_publicado"):
        dif = (m[f"{k}_a"] - m[f"{k}_b"]).abs()
        out[f"pct_{k}_cambia"] = round(float((dif > 1e-9).mean() * 100), 2) if len(m) else None
        out[f"max_dif_{k}"] = round(float(dif.max()), 4) if len(m) else None
    if len(m):
        cambia = (m["esperado_a"] - m["esperado_b"]).abs() > 1e-9
        out["primer_cambio"] = m.index[cambia].min().isoformat() if cambia.any() else None
        out["ultimo_cambio"] = m.index[cambia].max().isoformat() if cambia.any() else None
    return out


# --------------------------------------------------------------------------- tablas

def tabla_z(arch: pd.DataFrame, zc: pd.Series, zc_piso: pd.Series) -> pd.DataFrame:
    """Tabla semihoraria: conteo, z publicado (revisado), z causal y nivel 1-5 aproximado del publicado."""
    t = pd.DataFrame(dict(ts=arch["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"), valor=arch["conteo"],
                          z_publicado=arch["z_publicado"], z_causal=zc.round(2), z_causal_piso=zc_piso.round(2)))
    t["nivel_publicado"] = [nivel(z) for z in t["z_publicado"]]
    return t


def tabla_alarmas(t: pd.DataFrame) -> pd.DataFrame:
    filas = []
    dia = t["ts"].str[:10]
    for u in UMBRALES:
        fila: dict[str, Any] = dict(umbral=u)
        for col, nom in (("z_publicado", "publicado"), ("z_causal", "causal"), ("z_causal_piso", "causal_piso")):
            m = t[col] >= u
            fila[f"n_alarmas_{nom}"] = int(m.sum())
            fila[f"dias_{nom}"] = int(dia[m].nunique())
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_diaria(t: pd.DataFrame) -> pd.DataFrame:
    g = t.assign(dia=t["ts"].str[:10]).groupby("dia")
    return pd.DataFrame(dict(n_muestras=g["valor"].count(), conteo_medio=g["valor"].mean().round(1),
                             conteo_max=g["valor"].max(), z_publicado_max=g["z_publicado"].max(),
                             z_causal_max=g["z_causal"].max().round(3))).reset_index()


def tabla_muestreo(arch: pd.DataFrame) -> pd.DataFrame:
    paso = arch["ts"].diff().dt.total_seconds().div(60).round().dropna()
    t = paso.value_counts().rename_axis("intervalo_min").reset_index(name="n").sort_values("intervalo_min")
    t["pct"] = (100 * t["n"] / t["n"].sum()).round(2)
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------- series

def _larga(arch: pd.DataFrame, col: str, serie: str, unidad: str, *, pub: pd.Series, metodo: str, vintage: str,
           nota: str) -> pd.DataFrame:
    t = arch.dropna(subset=[col])
    return pd.DataFrame(dict(serie=serie, fecha_referencia=t["ts"].dt.tz_convert(None).dt.normalize(),
                             fecha_publicacion=pub.loc[t.index], valor=t[col].astype(float), unidad=unidad,
                             fuente="ews", vintage=vintage,
                             disponible_utc=t["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             metodo_publicacion=metodo, nota=nota))


def series_ews(arch: pd.DataFrame, snapshot_dia: str) -> dict[str, pd.DataFrame]:
    dia_ts = arch["ts"].dt.tz_convert(None).dt.normalize()
    act = _larga(arch, "conteo", "ews_conteo_30min", "aviones en el aire", pub=dia_ts, metodo="tiempo_real",
                 vintage=snapshot_dia, nota="conteo semihorario (instante en disponible_utc); no se revisa")
    diario = arch.assign(dia=dia_ts).groupby("dia")["conteo"].agg(["mean", "count"]).reset_index()
    act_d = pd.DataFrame(dict(serie="ews_conteo_diario_medio", fecha_referencia=diario["dia"],
                              fecha_publicacion=diario["dia"] + pd.Timedelta(days=1), valor=diario["mean"].round(2),
                              unidad="aviones en el aire (media del día)", fuente="ews", vintage=snapshot_dia,
                              metodo_publicacion="lag_fijo", nota="media diaria (UTC); se conoce al acabar el día"))
    zp = _larga(arch, "z_publicado", "ews_z_publicado", "z", pub=pd.Series(pd.Timestamp(snapshot_dia), index=arch.index),
                metodo="vintage_unico", vintage=snapshot_dia,
                nota="z del archivo REVISADO en cada instantánea: solo se conoce en la fecha del vintage")
    zc = _larga(arch, "z_causal", "ews_z_causal", "z", pub=dia_ts, metodo="calculado_causal", vintage=snapshot_dia,
                nota=f"z propio con línea base solo del pasado (mismo hueco semanal) y suelo de desviación "
                     f"{PISO_REL} × media")
    return {"ews_actividad": pd.concat([act, act_d], ignore_index=True), "ews_z_publicado": zp, "ews_z_causal": zc}


# --------------------------------------------------------------------------- sondear

def _descargar_panel(res: base.Resultado, cfg: base.ConfigSonda) -> tuple[dict, dict]:
    """Instantánea nueva de ``dashboard.json`` (se archiva con su hora); si falla y hay caché, la última."""
    url = R2 + PANELES["jets"]
    t0 = base.cronometro()
    try:
        r = get(url, timeout=60, reintentos=2)
        r.raise_for_status()
        d = r.json()
        base.escribir_atomico(_ruta_instantanea(d), r.content)
        res.registrar("ews", "dashboard.json", url, True, dia=datetime.now(timezone.utc).date().isoformat(),
                      bytes=len(r.content), segundos=base.segundos(t0))
        return d, dict(last_modified=r.headers.get("Last-Modified"), bytes=len(r.content), cache=False)
    except Exception as e:  # noqa: BLE001
        res.registrar("ews", "dashboard.json", url, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
        previas = instantaneas()
        if cfg.usar_cache and previas:
            return json.loads(previas[-1].read_text(encoding="utf-8")), dict(bytes=previas[-1].stat().st_size, cache=True)
        raise


def sondear(cfg: base.ConfigSonda | None = None) -> base.Resultado:
    cfg = cfg or base.ConfigSonda()
    res = base.Resultado(MODULO)
    t_ini = base.cronometro()
    med: dict[str, Any] = {}
    notas: list[str] = []
    ini = fin = None
    lag_h = None
    cfg.log("[ews] panel…")
    try:
        d, meta = _descargar_panel(res, cfg)
        ahora = datetime.now(timezone.utc)
        act = estado_actual(d)
        as_of = pd.Timestamp(act["asOf"])
        gen = pd.Timestamp(d["snapshotGeneratedAt"]) if d.get("snapshotGeneratedAt") else None
        snapshot_dia = (gen or as_of).date().isoformat()
        arch = serie_archivo(d)
        zc_simple = z_causal(arch, ventana=8, minimo=4)          # solo para comparar (tablas)
        arch["z_causal"] = z_causal(arch, ventana=8, minimo=4, piso_rel=PISO_REL)   # el que se guarda
        ini, fin = arch["ts"].min().date().isoformat(), arch["ts"].max().date().isoformat()

        tz = tabla_z(arch, zc_simple, arch["z_causal"])
        res.tablas["ews_z"] = tz
        res.tablas["ews_alarmas"] = tabla_alarmas(tz)
        res.tablas["ews_diario"] = tabla_diaria(tz)
        res.tablas["ews_muestreo"] = tabla_muestreo(arch)

        for nombre, larga in series_ews(arch, snapshot_dia).items():
            res.series.append(base.guardar_serie(larga, nombre, rapido=cfg.modo_rapido).stem)

        # TimeGate: muestra -> instantánea generada -> lectura.
        lag_gen = base.lag_horas(as_of, gen)
        res.timegate.append(dict(fuente="ews", item="última muestra", fecha_hecho=as_of.isoformat(),
                                 fecha_publicacion=gen.isoformat() if gen is not None else None,
                                 campo_fecha="snapshotGeneratedAt", lag_h=lag_gen))
        lag_h = lag_gen
        edad_min = round((ahora - as_of.to_pydatetime()).total_seconds() / 60, 1)
        res.timegate.append(dict(fuente="ews", item="z del archivo (primer punto)", fecha_hecho=arch["ts"].iloc[0].isoformat(),
                                 fecha_publicacion=(gen or as_of).isoformat(),
                                 campo_fecha="snapshotGeneratedAt (z revisado en cada instantánea)",
                                 lag_h=base.lag_horas(arch["ts"].iloc[0], gen or as_of)))

        # Revisión: comparar con la instantánea anterior archivada.
        previas = [p for p in instantaneas() if p.name != _ruta_instantanea(d).name]
        rev_filas = []
        for p in previas[-3:]:
            try:
                ant = json.loads(p.read_text(encoding="utf-8"))
                r_ = revision(serie_archivo(ant), arch[["ts", "conteo", "esperado", "sd", "z_publicado"]])
                rev_filas.append(dict(instantanea_a=p.name, instantanea_b=_ruta_instantanea(d).name,
                                      t0_a=ant["trends"]["archive"]["t0"], t0_b=d["trends"]["archive"]["t0"], **r_))
            except Exception as e:  # noqa: BLE001
                notas.append(f"no se pudo comparar con {p.name}: {type(e).__name__}")
        if rev_filas:
            res.tablas["ews_revision"] = pd.DataFrame(rev_filas)

        muestreo = res.tablas["ews_muestreo"]
        al = res.tablas["ews_alarmas"].set_index("umbral")
        med.update(actual=act, cohorte=d.get("cohort", {}).get("trackedCount"),
                   cadencia_nominal_min=d.get("liveStatus", {}).get("cadenceMinutes"),
                   proveedor=d.get("liveStatus", {}).get("providerLabel"),
                   modelo=d.get("page", {}).get("predictionModel"),
                   puntos_archivo=int(len(arch)), inicio_archivo=arch["ts"].iloc[0].isoformat(),
                   fin_archivo=arch["ts"].iloc[-1].isoformat(),
                   pct_intervalos_30min=float(muestreo.loc[muestreo["intervalo_min"] == 30, "pct"].sum()),
                   intervalo_max_h=round(float(muestreo["intervalo_min"].max()) / 60, 1),
                   z_max_publicado=float(arch["z_publicado"].max()),
                   z_max_causal=round(float(zc_simple.max()), 2),
                   z_min_causal=round(float(zc_simple.min()), 2),
                   z_max_causal_piso=round(float(arch["z_causal"].max()), 2),
                   z_min_causal_piso=round(float(arch["z_causal"].min()), 2),
                   alarmas_z47=dict(publicado=int(al.at[ALARMA, "n_alarmas_publicado"]),
                                    causal=int(al.at[ALARMA, "n_alarmas_causal"]),
                                    causal_piso=int(al.at[ALARMA, "n_alarmas_causal_piso"])),
                   corr_z_publicado_causal=round(float(tz[["z_publicado", "z_causal"]].corr().iloc[0, 1]), 3),
                   minutos_desde_ultima_muestra=edad_min, horas_muestra_a_instantanea=lag_gen,
                   bytes=meta.get("bytes"), desde_cache=meta.get("cache"), revision=rev_filas)
        if rev_filas:
            r0 = rev_filas[-1]
            notas.append(f"Revisión ex post confirmada: entre {r0['instantanea_a']} y {r0['instantanea_b']} cambia "
                         f"'esperado' en el {r0['pct_esperado_cambia']} % de los instantes comunes (máx. "
                         f"{r0['max_dif_esperado']} aviones) y z en el {r0['pct_z_publicado_cambia']} % (máx. "
                         f"{r0['max_dif_z_publicado']}); el conteo no cambia ({r0['pct_conteo_cambia']} %). "
                         "El z del archivo NO es el que se vio en vivo: usa información posterior.")
        notas.append(f"El archivo es un año móvil (t0 {med['inicio_archivo'][:10]}): lo anterior se pierde si no se "
                     "archivan instantáneas (se guardan en el corpus con su hora).")
        notas.append(f"Muestreo irregular: {med['pct_intervalos_30min']} % de pasos de 30 min; hueco máximo "
                     f"{med['intervalo_max_h']} h.")
        notas.append(f"Alarmas z>=4,7: publicado {med['alarmas_z47']['publicado']} medias horas, causal simple "
                     f"{med['alarmas_z47']['causal']}, causal con suelo de desviación {med['alarmas_z47']['causal_piso']}. "
                     f"La serie ews_z_causal es la del suelo ({PISO_REL} × media): la simple da z absurdos "
                     f"(mín. {med['z_min_causal']}) al dividir por desviaciones casi nulas.")
    except Exception as e:  # noqa: BLE001
        notas.append(f"error en el panel: {type(e).__name__}: {str(e)[:200]}")

    # Otros paneles y RSS (no en modo rápido).
    if not cfg.modo_rapido:
        otros = {}
        for k in ("militar", "sin_icao"):
            url = R2 + PANELES[k]
            t0 = base.cronometro()
            try:
                r = get(url, timeout=60, reintentos=2)
                r.raise_for_status()
                p = r.json()
                res.registrar("ews", PANELES[k], url, True, bytes=len(r.content), segundos=base.segundos(t0))
                otros[k] = dict(asOf=p.get("current", {}).get("asOf"),
                                cohorte=p.get("cohort", {}).get("trackedCount"),
                                puntos=len(p.get("trends", {}).get("archive", {}).get("c", [])))
            except Exception as e:  # noqa: BLE001
                res.registrar("ews", PANELES[k], url, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
                otros[k] = f"{type(e).__name__}"
        med["otros_paneles"] = otros
    t0 = base.cronometro()
    try:
        permitido = robots_permite(RSS)
        med["robots_rss"] = permitido
        if permitido:
            r = get(RSS, timeout=30, reintentos=2)
            r.raise_for_status()
            base.escribir_atomico(base.corpus("crudo", MODULO) / "rss.xml", r.content)
            items = leer_rss(r.text)
            res.registrar("ews", "rss.xml", RSS, True, bytes=len(r.content), segundos=base.segundos(t0))
            med["rss_items"] = len(items)
            m = re.search(r"<lastBuildDate>([^<]+)</lastBuildDate>", r.text)
            med["rss_last_build"] = m.group(1) if m else None
            if not items:
                notas.append("El RSS de alertas de nivel 5 está vacío (lastBuildDate 1970): nunca ha saltado una alerta.")
    except Exception as e:  # noqa: BLE001
        res.registrar("ews", "rss.xml", RSS, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
    med["segundos"] = base.segundos(t_ini)

    est = base.estado(res.registro, "ews")
    if med.get("desde_cache"):
        est = "parcial"
    res.cobertura.append(dict(
        fuente="ews", familia="indices_ews", tipo="instantanea", inicio=ini, fin=fin, hora=True, idiomas="",
        lag_mediana_h=lag_h, destino="vivo", backtest=False, estado=est,
        nota=(f"año móvil semihorario irregular; última muestra hace {med.get('minutos_desde_ultima_muestra')} min; "
              f"z del archivo revisado ex post")[:300]))
    res.fichas.append(base.ficha(
        fuente="ews", nombre="EWS · Apocalypse Early Warning System (jets privados, ADS-B)", agentes=["cio", "risk"],
        rag="serie", url=WEB, acceso="api", licencia_y_terminos=LICENCIA, cobertura_inicio=ini, cobertura_fin=fin,
        frecuencia=f"{med.get('cadencia_nominal_min')} min nominal (irregular: {med.get('pct_intervalos_30min')} % de "
                   "pasos de 30 min)",
        retraso_publicacion=(f"{med.get('horas_muestra_a_instantanea')} h de la muestra a la instantánea; "
                             f"{med.get('minutos_desde_ultima_muestra')} min de antigüedad en la sonda"),
        formato="json (bucket R2) + rss", volumen_estimado=f"{(med.get('bytes') or 0) / 1e6:.2f} MB por instantánea "
                                                          "(≈20 MB/día archivando cada 30 min)",
        limites="bucket R2 público sin límites documentados; puede cambiar de URL sin aviso (no es una API)",
        estado_sonda=est, mediciones=med,
        notas=notas + ["Uso: solo vivo/demo con el emergencyLevel PUBLICADO (botón de cataclismo del CIO); "
                       "backtest=False.", "liveAircraft trae matrículas y posiciones: solo en el corpus, nunca a results."]))
    return res
