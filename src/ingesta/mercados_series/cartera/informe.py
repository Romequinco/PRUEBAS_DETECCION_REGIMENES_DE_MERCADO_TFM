"""Mediciones, ficha y tablas del bloque 8 (``python -m ingesta.mercados_series.cartera.informe``).

Pasos (cada uno por debajo de unos 5 minutos; se pueden lanzar por separado y en paralelo):

- ``--parte brutos``: baselines sin costes → ``D:/.../mercados_series/cartera/baselines_brutos.parquet``.
- ``--parte costes``: 60/40, momentum y 1/N con DEGIRO, IBKR y DEGIRO sin multiplicador →
  ``cartera/baselines_costes.parquet``; multiplicador de SPY → ``cartera/multiplicador_spy.parquet``.
- ``--parte ficha``: lee lo anterior y escribe ``fichas/cartera_baselines_costes.json`` y
  ``tablas/costes_modelo.csv`` (parámetros, sin precios).

Los parquet con curvas van al corpus (no al repositorio). Los precios se leen con
``almacen.leer_precios`` (validados por contrato; ARCX y NAV_US ya están en ``config/mercados.yaml``).
Si un ticker no supera el contrato, se lee su parquet tal cual y queda en ``avisos_carga``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.mercados_series import almacen
from ingesta.mercados_series.cartera import baselines as b
from ingesta.mercados_series.cartera import costes as c
from ingesta.mercados_series.cartera import libre_riesgo as lr
from ingesta.mercados_series.contrato import ErrorContrato

TEMA = "mercados_series"
FUENTE = "cartera_baselines_costes"
INICIO, FIN = "1993-01-01", "2026-10-12"
CAPITAL_EUR = 10_000.0


def carpeta() -> Path:
    d = corpus_dir(TEMA) / "cartera"
    d.mkdir(parents=True, exist_ok=True)
    return d


def nucleo() -> pd.DataFrame:
    return pd.read_csv(resultados_dir(TEMA) / "tablas" / "yahoo_nucleo.csv")


def cargar_precios(tickers, proveedor: str = "yahoo") -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Precios del contrato por ticker: validados si se puede; si no, el parquet tal cual + aviso."""
    out, avisos = {}, []
    for tk in tickers:
        try:
            t = almacen.leer_precios(tk, proveedor)
        except ErrorContrato as e:
            ruta = almacen.ruta_precios(proveedor, tk)
            t = pd.read_parquet(ruta) if ruta.exists() else None
            avisos.append(f"{tk}: leído sin validar ({str(e).splitlines()[0][:120]})")
        if t is not None and len(t):
            out[tk] = t.sort_values("sesion").reset_index(drop=True)
    return out, avisos


def estrategias(etf: list[str]) -> list[b.Estrategia]:
    return [b.buy_hold("SPY"), b.sesenta_cuarenta(), b.uno_n(etf, nombre=f"1/N {len(etf)} ETF"), b.buy_hold("RSP"),
            b.momentum_sectorial(), b.sesenta_cuarenta_europeo()]


def _eur_usd(precios: dict[str, pd.DataFrame]) -> pd.Series | float:
    t = precios.get("EURUSD=X")
    if t is None:
        return 1.10
    return pd.Series(t["close"].to_numpy(float), index=pd.DatetimeIndex(t["fecha_publicacion"])).sort_index()


def _curvas(resultados: list[b.Resultado], etiqueta: str = "") -> pd.DataFrame:
    filas = []
    for r in resultados:
        filas.append(pd.DataFrame({"estrategia": r.nombre, "variante": etiqueta, "fecha": r.curva.index,
                                   "valor_usd": r.curva.to_numpy()}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def parte_brutos() -> dict:
    nuc = nucleo()
    etf = sorted(nuc.loc[nuc["clase"] == "ETF", "ticker"])
    precios, avisos = cargar_precios(sorted(set(etf) | {"SPY", "AGG", "VBMFX", "RSP", "CSPX.L", "EUNA.DE"}))
    med, res = {}, []
    for est in estrategias(etf):
        t0 = time.time()
        r = b.simular(est, precios, INICIO, FIN)
        res.append(r)
        med[r.nombre] = {**b.metricas(r), "segundos": round(time.time() - t0, 1), "avisos": r.avisos[:5],
                         "notas": est.notas}
    _curvas(res, "bruto").to_parquet(carpeta() / "baselines_brutos.parquet", index=False)
    return {"metricas": med, "avisos_carga": avisos}


def parte_costes() -> dict:
    nuc = nucleo()
    etf = sorted(nuc.loc[nuc["clase"] == "ETF", "ticker"])
    precios, _ = cargar_precios(sorted(set(etf) | {"SPY", "AGG", "VBMFX", "EURUSD=X"}))
    info = {r.ticker: {"mercado": r.mercado, "divisa": r.divisa, "clase": r.clase} for r in nuc.itertuples()}
    mult = c.multiplicador_crisis(precios["SPY"])
    mult.to_parquet(carpeta() / "multiplicador_spy.parquet")
    m_pub = mult.set_index("fecha_publicacion")["m"]
    eur = _eur_usd(precios)
    variantes = {
        "degiro": c.ModeloCostes(broker=c.DEGIRO, info=info, multiplicador=m_pub, eur_usd=eur),
        "ibkr": c.ModeloCostes(broker=c.IBKR, info=info, multiplicador=m_pub, eur_usd=eur),
        "degiro_sin_multiplicador": c.ModeloCostes(broker=c.DEGIRO, info=info, multiplicador=None, eur_usd=eur),
    }
    capital = CAPITAL_EUR * c.valor_en(eur, pd.Timestamp(INICIO), 1.10)
    med, curvas = {}, []
    for nombre, modelo in variantes.items():
        ests = [b.sesenta_cuarenta(), b.momentum_sectorial()]
        if nombre == "degiro":
            ests.append(b.uno_n(etf, nombre=f"1/N {len(etf)} ETF"))
        for est in ests:
            t0 = time.time()
            r = b.simular(est, precios, INICIO, FIN, capital=capital, costes=modelo)
            curvas.append(_curvas([r], nombre))
            med[f"{r.nombre} [{nombre}]"] = {**b.metricas(r), "segundos": round(time.time() - t0, 1)}
    pd.concat(curvas, ignore_index=True).to_parquet(carpeta() / "baselines_costes.parquet", index=False)
    crisis = {p: {"m_media": round(float(mult.loc[a:z, "m"].mean()), 2),
                  "m_mediana": round(float(mult.loc[a:z, "m"].median()), 2),
                  "chl_mediana_pb": round(float(mult.loc[a:z, "chl_21d"].median()) * 1e4, 1)}
              for p, (a, z) in {"calma_2017": ("2017-01-01", "2017-12-31"),
                                "crisis_2008": ("2008-09-15", "2008-12-31"),
                                "covid_2020": ("2020-02-24", "2020-04-30")}.items()}
    calma = mult["mediana_calma"].dropna()
    return {"costes": med, "multiplicador_spy": {
        "periodos": crisis, "sesiones": int(len(mult)), "fraccion_chl_cero": round(float((mult["chl_21d"] == 0).mean()), 3),
        "mediana_calma_final_pb": round(float(calma.iloc[-1]) * 1e4, 2) if len(calma) else None,
        "capital_inicial_usd": round(capital, 2)}}


def medir_libre_riesgo() -> dict:
    base = almacen.leer_series([lr.SERIE_DTB3, lr.SERIE_FRENCH_RF])
    dias = pd.bdate_range("1993-01-04", FIN)
    dtb3 = lr.tasa_dtb3(dias, base=base)
    retraso = (dtb3["fecha_publicacion"] - dtb3["fecha_referencia"]).dt.days
    fr = lr.rf_french(base)
    fr_anual = (1 + fr).rolling(21).apply(np.prod, raw=True) ** (252 / 21) - 1
    comun = dtb3["rendimiento"].reindex(fr_anual.index).dropna()
    dif = (comun - fr_anual.reindex(comun.index)).dropna() * 1e4
    eur = lr.tasa_eur(dias[-5:])
    return {"dtb3": {"inicio": str(base.loc[base["serie"] == lr.SERIE_DTB3, "fecha_referencia"].min().date()),
                     "fin": str(base.loc[base["serie"] == lr.SERIE_DTB3, "fecha_referencia"].max().date()),
                     "retraso_dias_mediana": float(retraso.median()), "retraso_dias_p95": float(retraso.quantile(0.95))},
            "french_rf": {"inicio": str(fr.index.min().date()), "fin": str(fr.index.max().date())},
            "dtb3_menos_french_pb": {"mediana": round(float(dif.median()), 1), "p05": round(float(dif.quantile(0.05)), 1),
                                     "p95": round(float(dif.quantile(0.95)), 1), "n": int(len(dif))},
            "eur": {"disponible": bool(eur["tasa"].notna().any()), "avisos": eur.attrs.get("avisos", [])}}


def ficha(mediciones: dict) -> FichaFuente:
    met = mediciones.get("brutos", {}).get("metricas", {})
    inicios = [m["inicio"] for m in met.values() if m.get("inicio")]
    fines = [m["fin"] for m in met.values() if m.get("fin")]
    eu = [k for k, m in met.items() if "UCITS" in k and not m]
    return FichaFuente(
        fuente=FUENTE, nombre="Bloque 8: libre de riesgo, baselines y modelo de costes (derivado)", tema=TEMA,
        agentes=["portfolio", "risk", "cio"], rag="serie", url="(derivado de precios/yahoo y de la base de series)",
        acceso="libreria", autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="derivado de Yahoo (uso personal), FRED (dominio público) y Ken French (cita)",
        cobertura_inicio=min(inicios) if inicios else None, cobertura_fin=max(fines) if fines else None,
        frecuencia="curvas diarias (día hábil); rebalanceo mensual o anual",
        retraso_publicacion=("punto en el tiempo: cada valor usa solo precios con fecha_publicacion <= el día "
                             "(16:30 Nueva York); DTB3 en T+1 (medido en mediciones.libre_riesgo); French RF solo ex post"),
        formato="parquet (curvas, en el corpus) y csv de parámetros",
        volumen_estimado="< 5 MB de curvas", limites="sin red; coste de cálculo medido en segundos por estrategia",
        estado_sonda="parcial" if eu or not mediciones.get("libre_riesgo", {}).get("eur", {}).get("disponible") else "ok",
        mediciones=mediciones,
        notas=[
            "Baselines con get_precios en cada control (último hábil de mes) y base recortada a la ventana; test de truncado",
            "60/40: VBMFX hasta el primer control con AGG visible (2003-09-30); VBMFX publica en D+1 (NAV)",
            "1/N sobre los ETF del núcleo de Yahoo vivos en la fecha (lista estática del núcleo: sesgo de supervivencia "
            "declarado; WRDS pendiente)",
            "60/40 europeo (anexo) con CSPX.L/EUNA.DE: sin precios UCITS descargados, no invierte" if eu else
            "60/40 europeo con UCITS",
            "EONIA/€STR del bloque 4 aún no en la base: sin libre de riesgo en EUR; Euribor 3M no catalogado",
            "Comisiones DEGIRO/IBKR de la tarifa pública sin verificar (verificado=False en costes_modelo.csv)",
            _nota_carga(mediciones.get("brutos", {}).get("avisos_carga", [])),
        ])


def _nota_carga(avisos: list[str]) -> str:
    if not avisos:
        return "Precios validados por contrato con almacen.leer_precios (ningún ticker leído sin validar)"
    return f"{len(avisos)} tickers leídos sin validar (no superan el contrato); ver mediciones.brutos.avisos_carga"


def parte_ficha() -> dict:
    import json
    d = carpeta()
    med = {}
    for parte in ("brutos", "costes"):
        f = d / f"mediciones_{parte}.json"
        if f.exists():
            med[parte] = json.loads(f.read_text(encoding="utf-8"))
    med["libre_riesgo"] = medir_libre_riesgo()
    print(guardar_ficha(ficha(med)))
    ruta = resultados_dir(TEMA) / "tablas" / "costes_modelo.csv"
    c.tabla_parametros().to_csv(ruta, index=False, encoding="utf-8")
    print(ruta)
    return med


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parte", choices=["brutos", "costes", "ficha"], required=True)
    a = ap.parse_args(argv)
    if a.parte == "ficha":
        parte_ficha()
        return 0
    med = parte_brutos() if a.parte == "brutos" else parte_costes()
    (carpeta() / f"mediciones_{a.parte}.json").write_text(json.dumps(med, ensure_ascii=False, indent=1, default=str),
                                                          encoding="utf-8")
    print(json.dumps(med, ensure_ascii=True, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
