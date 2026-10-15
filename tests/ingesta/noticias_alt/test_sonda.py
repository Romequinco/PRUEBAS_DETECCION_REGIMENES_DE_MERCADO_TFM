"""Sonda del Tema 4 sin red: módulos falsos que escriben documentos y series en un corpus temporal."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ingesta.comun.esquema import Documento, doc_id
from ingesta.noticias_alt import base, sonda
from ingesta.noticias_alt.muestra import DiaTestigo

DIAS = [DiaTestigo("covid", date(2020, 3, 16), "prueba"), DiaTestigo("hoy", date(2026, 10, 14), "vivo")]


def _doc(fuente: str, titulo: str, url: str, medio: str, fecha: str, dia: str | None) -> dict:
    d = Documento(doc_id=doc_id(url), fuente=fuente, tipo_documento="titular", titulo=titulo,
                  fecha_publicacion=date.fromisoformat(fecha), url=url, agente="comun", rag="cajon",
                  institucion=medio, extra={"dia_testigo": dia} if dia else {})
    return base.documento_dict(d)


def _sondear_texto(cfg: base.ConfigSonda) -> base.Resultado:
    res = base.Resultado("texto_falso")
    res.registrar("gdelt_doc", "consulta", "https://x/1", True, dia="2020-03-16", bytes=2_000_000, segundos=2.0)
    res.registrar("gdelt_doc", "consulta (caché)", "https://x/2", True, dia="2020-03-16", bytes=1000)
    res.registrar("rss_medios", "feed", "https://y/1", False, error="HTTPError: 404")
    docs = [
        _doc("gdelt_doc", "Stocks plunge as virus fears grip markets worldwide", "https://a.com/1", "a.com",
             "2020-03-16", "covid"),
        _doc("gdelt_doc", "Stocks plunge as virus fears grip markets worldwide today", "https://b.com/9", "b.com",
             "2020-03-16", "covid"),
        _doc("gdelt_doc", "Stocks plunge as virus fears grip markets worldwide", "https://a.com/1?utm_source=x",
             "a.com", "2020-03-16", "covid"),
        _doc("gdelt_doc", "Fed cuts rates to zero in emergency move", "https://c.com/2", "c.com", "2020-03-16",
             "covid"),
    ]
    ruta, n = base.guardar_documentos(docs, "gdelt_doc_titulares")
    res.documentos[ruta.stem] = n
    vivos = [_doc("rss_medios", "Bank shares slide after earnings", "https://r.com/1", "R", "2026-10-14", None)]
    ruta, n = base.guardar_documentos(vivos, "rss_titulares")
    res.documentos[ruta.stem] = n
    res.cobertura.append(dict(fuente="gdelt_doc", familia="eventos", tipo="titular", inicio="2017-01-01",
                              fin="2026-10-14", hora=True, idiomas="en", lag_mediana_h=0.5, destino="cajon",
                              backtest=True, estado="ok", nota=""))
    res.timegate.append(dict(fuente="gdelt_doc", item="a.com", fecha_hecho="2020-03-16T10:00:00",
                             fecha_publicacion="2020-03-16", campo_fecha="seendate", lag_h=0.0))
    res.tablas["gdelt_doc_muestra"] = pd.DataFrame({"titulo": ["x"], "n": [1]})
    res.fichas.append(base.ficha(fuente="gdelt_doc", nombre="GDELT DOC", agentes=["macro"], rag="cajon",
                                 url="https://gdelt", licencia_y_terminos="GDELT: uso libre; citar", frecuencia="15 min",
                                 retraso_publicacion="≈15 min", formato="json",
                                 volumen_estimado="≈ 2 GB/año", limites="429", estado_sonda="ok",
                                 cobertura_inicio="2017-01-01", cobertura_fin="2019-01-01"))
    res.fichas.append(base.ficha(fuente="rss_medios", nombre="RSS", agentes=["macro"], rag="cajon", url="https://r",
                                 licencia_y_terminos="copyright del medio (uso académico)", frecuencia="vivo",
                                 retraso_publicacion="minutos", formato="rss", volumen_estimado="sin cifra",
                                 limites="", estado_sonda="fallo"))
    return res


def _sondear_serie(cfg: base.ConfigSonda) -> base.Resultado:
    res = base.Resultado("serie_falsa")
    res.registrar("indice_x", "csv", "https://z/1", True, bytes=500_000, segundos=1.0)
    s = pd.DataFrame({"serie": "indice_x", "fecha_referencia": ["2020-01-01", "2020-01-02"],
                      "fecha_publicacion": ["2020-01-02", "2020-01-03"], "valor": [1.0, 2.0], "unidad": "indice",
                      "fuente": "indice_x"})
    res.series.append(base.guardar_serie(s, "indice_x").stem)
    res.fichas.append(base.ficha(fuente="indice_x", nombre="Índice X", agentes=["risk"], rag="serie", url="https://z",
                                 licencia_y_terminos="libre", frecuencia="diaria", retraso_publicacion="D+1",
                                 formato="csv", volumen_estimado="0,5 MB en esta sonda; total 3 MB",
                                 limites="", estado_sonda="ok", cobertura_inicio="2020-01-01",
                                 cobertura_fin="2020-01-02"))
    return res


def _sondear_roto(cfg: base.ConfigSonda) -> base.Resultado:
    raise RuntimeError("fuente caída")


@pytest.fixture
def modulos_falsos(monkeypatch):
    mods = [SimpleNamespace(__name__="ingesta.noticias_alt.texto_falso", sondear=_sondear_texto),
            SimpleNamespace(__name__="ingesta.noticias_alt.roto", sondear=_sondear_roto),
            SimpleNamespace(__name__="ingesta.noticias_alt.serie_falsa", sondear=_sondear_serie)]
    monkeypatch.setattr(sonda, "MODULOS", mods)
    return mods


def test_sonda_escribe_csv_y_fichas(sin_red, modulos_falsos):
    cfg = base.ConfigSonda(dias=DIAS, verbose=False)
    res = sonda.ejecutar(cfg)
    rutas = sonda.guardar(res)
    carpeta = sin_red / "results" / "noticias_alt"
    assert all(p.exists() for p in rutas)
    for nombre in ("registro_descargas", "cobertura", "timegate", "volumen", "fuentes_resumen", "limpieza_embudo",
                   "limpieza_idiomas", "limpieza_sindicacion", "gdelt_doc_muestra", "sonda_modulos"):
        assert (carpeta / f"{nombre}.csv").exists(), nombre
    fichas = sorted(p.stem for p in (carpeta / "fichas").glob("*.json"))
    assert fichas == ["gdelt_doc", "indice_x", "rss_medios"]
    assert json.loads((carpeta / "fichas" / "indice_x.json").read_text(encoding="utf-8"))["estado_sonda"] == "ok"

    # El módulo roto queda registrado y no para la sonda.
    mods = pd.read_csv(carpeta / "sonda_modulos.csv")
    assert mods.set_index("modulo")["ok"].to_dict() == {"texto_falso": True, "roto": False, "serie_falsa": True}
    reg = pd.read_csv(carpeta / "registro_descargas.csv")
    assert len(reg) == 4 and set(reg["modulo"]) == {"texto_falso", "serie_falsa"}


def test_embudo_idiomas_y_sindicacion(sin_red, modulos_falsos):
    res = sonda.ejecutar(base.ConfigSonda(dias=DIAS, verbose=False))
    emb = res.tablas["limpieza_embudo"]
    covid = emb.loc[emb["dia"] == "covid"].set_index("etapa")["n"].to_dict()
    assert covid == {"bruto": 4, "tras_url": 3, "tras_jaccard": 2, "tras_simhash": 2}
    assert set(emb["dia"]) == {"covid", "hoy"}          # el RSS sin día testigo cae en «hoy»
    sind = res.tablas["limpieza_sindicacion"]
    assert len(sind) == 1 and sind.iloc[0]["n_medios"] == 2 and sind.iloc[0]["medio_original"] == "a.com"
    assert sind["titular"].str.len().max() <= sonda.MAX_TITULAR
    idi = res.tablas["limpieza_idiomas"]
    assert idi.set_index("fuente")["n"].to_dict() == {"gdelt_doc": 4, "rss_medios": 1}


def test_volumen_y_resumen(sin_red, modulos_falsos):
    res = sonda.ejecutar(base.ConfigSonda(dias=DIAS, verbose=False))
    vol = res.volumen.set_index("fuente")
    assert list(res.volumen.columns[:6]) == ["fuente", "n_items_muestra", "mb_muestra", "gb_corpus_estimado",
                                             "horas_descarga_estimadas", "n_peticiones"]
    assert vol.loc["gdelt_doc", "n_items_muestra"] == 4
    assert vol.loc["indice_x", "n_items_muestra"] == 2
    assert vol.loc["indice_x", "gb_corpus_estimado"] == pytest.approx(0.003)       # última cantidad absoluta
    assert vol.loc["gdelt_doc", "gb_corpus_estimado"] == pytest.approx(2 * 730 / 365.25, rel=1e-3)  # tasa × periodo
    assert vol.loc["gdelt_doc", "mb_s_medidos"] == pytest.approx(1.0)              # 2 MB en 2 s (la caché no cuenta)
    assert pd.isna(vol.loc["rss_medios", "gb_corpus_estimado"])
    resumen = res.fuentes_resumen.set_index("fuente")
    assert resumen.loc["gdelt_doc", "familia"] == "eventos" and resumen.loc["gdelt_doc", "licencia"] == "GDELT: uso libre"
    assert resumen.loc["indice_x", "destino"] == "serie"
    ser = res.series.set_index("serie")
    assert ser.loc["indice_x", "n_filas"] == 2 and ser.loc["indice_x", "lag_mediana_d"] == 1


def test_gb_de_texto():
    assert sonda.gb_de_texto("≈ 1,5 TB zip; 2 MB/15 min")[0] == pytest.approx(1500)
    assert sonda.gb_de_texto("3 GB/mes", "2020-01-01", "2020-01-31")[0] == pytest.approx(3 * 30 / 30.44)
    assert sonda.gb_de_texto("no medido") == (None, "sin cifra de volumen en la ficha")


def test_titular_corto():
    assert base.titular_corto("  Una   línea\ncon saltos ") == "Una línea con saltos"
    t = base.titular_corto("x" * 200)
    assert len(t) == base.MAX_TITULAR and t.endswith("…")


# --------------------------------------------------------------------------- results reales (si existen)

RESULTS = Path(__file__).resolve().parents[3] / "results" / "ingesta" / "noticias_alt"


@pytest.mark.skipif(not (RESULTS / "cobertura.csv").exists(), reason="sin results de la sonda")
def test_results_titulares_cortos():
    """Copyright: ninguna columna titular/título de results supera ``MAX_TITULAR`` caracteres."""
    largos = []
    for ruta in sorted(RESULTS.glob("*.csv")):
        d = pd.read_csv(ruta, low_memory=False)
        for c in d.columns:
            if c.lower().startswith(("titular", "titulo", "título")):
                n = d[c].dropna().astype(str).str.len().max()
                if n and n > base.MAX_TITULAR:
                    largos.append(f"{ruta.name}:{c}={n}")
    assert not largos, largos


@pytest.mark.skipif(not (RESULTS / "cobertura.csv").exists(), reason="sin results de la sonda")
def test_results_destino_coincide_con_ficha():
    """Fuentes medidas (ok/parcial) y no solo en vivo: el destino de cobertura es el ``rag`` de su ficha."""
    cob = pd.read_csv(RESULTS / "cobertura.csv")
    fichas = {}
    for ruta in (RESULTS / "fichas").glob("*.json"):
        f = json.loads(ruta.read_text(encoding="utf-8"))
        fichas[f["fuente"]] = f.get("rag")
    medidas = cob[cob["estado"].astype(str).str.split().str[0].isin(["ok", "parcial"]) & (cob["destino"] != "vivo")]
    distintas = [f"{r.fuente}: {r.destino} != {fichas.get(r.fuente)}" for r in medidas.itertuples()
                 if r.fuente in fichas and fichas[r.fuente] != r.destino]
    assert not distintas, distintas
