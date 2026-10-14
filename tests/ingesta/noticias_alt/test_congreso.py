"""Tests del módulo ``congreso`` (PTR de la Cámara, Senado, Stock Watcher, agregadores, ETFs). Sin red."""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from ingesta.noticias_alt import base, congreso
from ingesta.noticias_alt.muestra import DiaTestigo

FX = __import__("pathlib").Path(__file__).parent / "fixtures" / "congreso"


def _zip_indice() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2024FD.xml", b"\xef\xbb\xbf" + (FX / "fd_indice.xml").read_bytes())
        z.writestr("2024FD.txt", "Prefix\tLast\tFirst\n")
    return buf.getvalue()


# --------------------------------------------------------------------------- índice

def test_parse_indice_filingdate_y_electronicos():
    fs = congreso.indice_zip(_zip_indice())
    ptr = [f for f in fs if f["tipo"] == "P"]
    assert len(fs) == 4 and len(ptr) == 3
    assert ptr[0]["fecha_publicacion"] == date(2024, 1, 11)       # FilingDate m/d/AAAA sin ceros
    assert ptr[0]["representante"] == "Ana Ejemplo" and ptr[0]["distrito"] == "ZZ01"
    assert [f["electronico"] for f in ptr] == [True, False, True]


@pytest.mark.parametrize("doc_id, esperado", [("20026140", True), ("8220682", False), ("9112233", False),
                                              ("10060479", False), ("", False), (None, False)])
def test_es_electronico(doc_id, esperado):
    assert congreso.es_electronico(doc_id) is esperado


def test_urls():
    assert congreso.url_indice(2020).endswith("financial-pdfs/2020FD.zip")
    assert congreso.url_ptr(2024, "20026140").endswith("ptr-pdfs/2024/20026140.pdf")


# --------------------------------------------------------------------------- PTR

def test_parse_ptr_formato_nuevo():
    ops = congreso.parse_ptr_texto((FX / "ptr_nuevo.txt").read_text(encoding="utf-8"))
    assert len(ops) == 5
    o = ops[0]
    assert (o["titular"], o["ticker"], o["clase"], o["tipo"]) == ("SP", "EIU", "ST", "compra")
    assert o["activo"].startswith("Empresa Inventada Uno")
    assert o["fecha_operacion"] == date(2023, 12, 18) and o["fecha_notificacion"] == date(2023, 12, 19)
    assert (o["importe_min"], o["importe_max"]) == (1001, 15000)
    # Preferente con '$' en el ticker y activo en varias líneas; venta parcial.
    assert ops[1]["ticker"] == "CFD$A" and ops[1]["tipo"] == "venta parcial"
    assert "Depositary Shares" in ops[1]["activo"]
    # Bono sin ticker: clase GS y sin buscar en la continuación.
    assert ops[2]["ticker"] is None and ops[2]["clase"] == "GS" and ops[2]["importe_min"] == 50001
    # Activo partido entre páginas: el ticker sale de la continuación; importe «Over».
    assert ops[3]["titular"] == "JT" and ops[3]["ticker"] == "HIMG"
    assert (ops[3]["importe_min"], ops[3]["importe_max"]) == (1000000, None)
    assert ops[4]["tipo"] == "canje" and ops[4]["ticker"] == "TEF"


def test_parse_ptr_formato_antiguo_pegado():
    """PDF 2014-2022: sin saltos de línea, mayúsculas cambiadas y fechas sin ceros."""
    ops = congreso.parse_ptr_texto((FX / "ptr_antiguo.txt").read_text(encoding="utf-8"))
    assert [o["ticker"] for o in ops] == ["AFT", "BIM.B", "GFC"]
    assert [o["tipo"] for o in ops] == ["venta parcial", "compra", "compra"]
    assert ops[1]["titular"] == "JT" and ops[1]["clase"] == "ST"
    assert ops[2]["fecha_operacion"] == date(2018, 6, 4) and ops[2]["fecha_notificacion"] == date(2018, 6, 5)
    assert "Gamma Falsa Corp" in ops[2]["activo"] and ":" not in ops[2]["activo"]


def test_parse_ptr_vacio_y_glifos_nulos():
    assert congreso.parse_ptr_texto("") == []
    assert congreso.parse_ptr_texto("\n" * 20) == []  # escaneado: pypdf no saca texto
    t = "Name: Hon. X\nN" + congreso.NULO * 4 + "ombre\nAcme (ACM) [ST]\nP 01/02/202401/03/2024$1,001 - $15,000\n"
    (o,) = congreso.parse_ptr_texto(t)
    assert o["ticker"] == "ACM"


def test_importe():
    assert congreso._importe("$1,001 - $15,000") == (1001, 15000)
    assert congreso._importe("Over $50,000,000") == (50000000, None)
    assert congreso._importe("") == (None, None)


# --------------------------------------------------------------------------- punta a punta

@pytest.fixture
def red_congreso(red_falsa, monkeypatch):
    texto = (FX / "ptr_nuevo.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(congreso, "texto_pdf", lambda contenido: (texto, 2))
    monkeypatch.setattr(congreso, "robots_permite", lambda url: False)
    nanc = pd.DataFrame(dict(fecha=pd.to_datetime(["2023-02-07", "2023-02-08", "2023-02-09"]), cierre=[25.0, 25.5, 26.0]))
    monkeypatch.setattr(congreso, "precios_etf", lambda tickers=("NANC", "KRUZ"): {
        "NANC": nanc, "KRUZ": pd.DataFrame(columns=["fecha", "cierre"])})
    return red_falsa(congreso, {"financial-pdfs/": _zip_indice(), "ptr-pdfs/": b"%PDF-1.4 falso",
                                "stock-watcher-data": (403, b"Forbidden")})


def test_sondear_punta_a_punta_sin_red(sin_red, red_congreso):
    cfg = base.ConfigSonda(modo_rapido=True, verbose=False, dias=[DiaTestigo("covid", date(2020, 3, 16), "")])
    res = congreso.sondear(cfg)

    cob = pd.DataFrame(res.cobertura)
    assert list(cob.columns) == base.COLUMNAS_COBERTURA
    assert cob["fuente"].is_unique and len(cob) == 5
    est = dict(zip(cob["fuente"], cob["estado"]))
    assert est["congreso_house_ptr"] == "ok" and est["congreso_stock_watcher"] == "fallo"
    assert est["congreso_senate_efd"] == "no_probada" and est["congreso_agregadores_pago"] == "no_probada"
    assert est["congreso_etfs_nanc_kruz"] == "parcial"  # KRUZ sin datos
    house = cob.set_index("fuente").loc["congreso_house_ptr"]
    assert house["inicio"] == "2024-01-11" and house["fin"] == "2024-06-20" and house["backtest"]

    anio = res.tablas["congreso_ptr_anio"]
    assert list(anio["anio"]) == list(range(congreso.ANIO_INICIO, date.today().year + 1))
    assert (anio["n_ptr"] == 3).all() and (anio["n_electronicos"] == 2).all() and (anio["pct_electronicos"] == 66.7).all()

    ops = res.tablas["congreso_operaciones"]
    assert len(ops) == 10 and set(ops["doc_id"]) == {"20099901", "20099902"}
    # TimeGate: fecha_publicacion = FilingDate + k hábiles (sin Last-Modified medido: k = 1), nunca la operación.
    assert dict(zip(ops["doc_id"], ops["fecha_presentacion"])) == {"20099901": date(2024, 1, 11),
                                                                   "20099902": date(2024, 6, 20)}
    pub = dict(zip(ops["doc_id"], ops["fecha_publicacion"]))
    assert pub == {"20099901": date(2024, 1, 12), "20099902": date(2024, 6, 21)}
    assert (ops["lag_dias"] == ops.apply(lambda r: (r["fecha_publicacion"] - r["fecha_operacion"]).days, axis=1)).all()
    assert (ops["fecha_publicacion"] >= ops["fecha_operacion"]).all()
    assert res.timegate and all(t["campo_fecha"].startswith("FilingDate + 1") for t in res.timegate)
    assert all(t["lag_h"] >= 0 for t in res.timegate)

    assert set(res.series) == {"congreso_ptr_diarios_rapido", "congreso_etf_nanc_kruz_rapido"}
    serie = base.leer_serie("congreso_ptr_diarios_rapido")
    assert serie["valor"].sum() == 3 and list(serie.columns) == base.COLUMNAS_SERIE
    assert (serie["fecha_publicacion"] == serie["fecha_referencia"] + pd.offsets.BDay(1)).all()
    assert res.fichas[0].mediciones["filing_a_web_habiles"]["k"] == 1
    assert {f.fuente for f in res.fichas} == set(cob["fuente"])
    assert all(f.tema == "noticias_alt" for f in res.fichas)
    # Caché: los crudos quedan en el corpus (fuera del repo).
    assert (base.corpus("crudo", "congreso") / "2024FD.zip").exists()
    assert not any(r["error"] for r in res.registro if r["fuente"] == "congreso_house_ptr")


def test_k_publicacion_desde_last_modified():
    # jueves 11-ene; subido el lunes 15 a las 10:00 ET -> 2 hábiles; a las 17:00 ET -> cuenta el martes (3)
    assert congreso.habiles_hasta_web(date(2024, 1, 11), "Mon, 15 Jan 2024 15:00:00 GMT") == 2
    assert congreso.habiles_hasta_web(date(2024, 1, 11), "Mon, 15 Jan 2024 22:00:00 GMT") == 3
    assert congreso.habiles_hasta_web(date(2024, 1, 11), "Thu, 11 Jan 2024 15:00:00 GMT") == 0
    assert congreso.habiles_hasta_web(date(2024, 1, 11), "Wed, 01 Jan 2025 10:00:00 GMT") is None   # regenerado
    assert congreso.habiles_hasta_web(date(2024, 1, 11), None) is None
    k, med = congreso.k_publicacion([0, 1, 1, 2, 2, 2, 3, 3, 4, 9, None])
    assert k == 5 and med["n"] == 10 and med["origen"] == "p90 medido"
    assert congreso.k_publicacion([0, 0])[0] == 1 and congreso.k_publicacion([])[0] == 1     # mínimo D+1
    assert congreso.publicar(date(2024, 1, 12), 1) == date(2024, 1, 15)                       # viernes -> lunes


def test_sondear_fuente_que_revienta_no_detiene(sin_red, red_congreso, monkeypatch):
    def _explota(cfg, res):
        raise RuntimeError("caída simulada")

    monkeypatch.setattr(congreso, "_sondear_house", _explota)
    res = congreso.sondear(base.ConfigSonda(modo_rapido=True, verbose=False))
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob["congreso_house_ptr"]["estado"] == "fallo" and "caída simulada" in cob["congreso_house_ptr"]["nota"]
    assert len(cob) == 5 and len(res.fichas) == 5


def test_reutiliza_cache(sin_red, red_congreso, red_falsa):
    cfg = base.ConfigSonda(modo_rapido=True, verbose=False)
    congreso.sondear(cfg)
    # Segunda pasada sin red para índices pasados y PTR: todo sale de la caché (el año en curso caduca a las 24 h).
    red_falsa(congreso, {"stock-watcher-data": (403, b"")})
    res = congreso.sondear(cfg)
    house = [r for r in res.registro if r["fuente"] == "congreso_house_ptr"]
    assert house and all(r["ok"] and r["item"].endswith("(caché)") for r in house)


@pytest.mark.live
def test_live_indice_y_ptr():
    ptr = congreso.listar_ptr(2024)
    assert len(ptr) > 300
    e = next(f for f in ptr if f["electronico"])
    ops, texto = congreso.leer_ptr(2024, e["doc_id"])
    assert texto.strip() and ops and all(o["fecha_operacion"] <= e["fecha_publicacion"] for o in ops)
