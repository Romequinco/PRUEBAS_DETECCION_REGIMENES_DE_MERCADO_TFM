"""Tests del Tema 3 (empresas). Sin red: las respuestas de la SEC salen de ``fixtures/``.

Los fixtures imitan el formato real de cada endpoint (submissions, company_tickers,
companyfacts, index.json, .hdr.sgml) con datos recortados o inventados.
"""
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ingesta.empresas import sec_edgar, secciones, sentimiento, sonda, universo, xbrl
from ingesta.empresas.universo import Empresa

FIX = Path(__file__).parent / "fixtures"


class _Resp:
    def __init__(self, contenido: bytes):
        self.content = contenido
        self.text = contenido.decode("utf-8")

    def json(self):
        return json.loads(self.content)


# Hora oficial (Nueva York) de cada filing del fixture, como la daría su cabecera .hdr.sgml.
# Los dos filings de los 90 no traen ACCEPTANCE-DATETIME.
HORA_OFICIAL = {
    "0000320193-24-000123": "20241101060136",   # 10-K antes de abrir
    "0000320193-24-000120": "20241031163039",   # resultados tras el cierre
    "0000320193-24-000081": "20240802060300",
    "0000320193-24-000070": "20240703164500",
    "0000320193-24-000010": "20240201163100",
}

RUTAS = {
    "/submissions/CIK0000320193.json": "submissions_CIK0000320193.json",
    "/submissions/CIK0000320193-submissions-001.json": "CIK0000320193-submissions-001.json",
    "/files/company_tickers.json": "company_tickers.json",
    "/api/xbrl/companyfacts/CIK0000320193.json": "companyfacts_CIK0000320193.json",
    "-index.htm": "index_8k.htm",
    "q4fy24earnings.htm": "ex991.htm",
    "aapl-20240928.htm": "10k_mini.htm",
    "aapl-20240629.htm": "10q_mini.htm",
    "0000320193-96-000023.txt": "10k_1996.txt",
    "0000320193-94-000016.txt": "10k_1996.txt",
}


def _get_falso(url, **_):
    if url.endswith(".hdr.sgml"):
        acc = url.rsplit("/", 1)[-1].removesuffix(".hdr.sgml")
        linea = f"<ACCEPTANCE-DATETIME>{HORA_OFICIAL[acc]}\n" if acc in HORA_OFICIAL else ""
        return _Resp(f"<SEC-HEADER>\n{linea}<ACCESSION-NUMBER>{acc}\n</SEC-HEADER>\n".encode())
    for sufijo, fichero in RUTAS.items():
        if url.endswith(sufijo):
            return _Resp((FIX / fichero).read_bytes())
    raise AssertionError(f"URL no prevista en los fixtures: {url}")


@pytest.fixture
def sin_red(tmp_path, monkeypatch):
    """Corpus y resultados en tmp + ``get`` falso en todos los módulos que lo usan."""
    monkeypatch.setenv("INGESTA_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setattr("ingesta.comun.rutas.ENV_FILE", tmp_path / "no_existe.env")
    monkeypatch.setattr("ingesta.comun.rutas.RESULTS_INGESTA", tmp_path / "results")
    for mod in ("ingesta.empresas.sec_edgar", "ingesta.empresas.universo"):
        monkeypatch.setattr(f"{mod}.get", _get_falso)
    return tmp_path


def _filings(sin_red):
    return sec_edgar.tabla_filings(320193)


# --------------------------------------------------------------------------- TimeGate

def test_parsear_aceptacion_de_la_api():
    assert sec_edgar.parsear_aceptacion("2024-11-01T14:01:36.000Z") == datetime(2024, 11, 1, 14, 1, 36)
    assert sec_edgar.parsear_aceptacion("") is None and sec_edgar.parsear_aceptacion(None) is None


@pytest.mark.parametrize("filing, aceptacion, esperado", [
    (date(2024, 11, 1), datetime(2024, 11, 1, 6, 1), date(2024, 11, 1)),      # antes de abrir: mismo día
    (date(2024, 10, 31), datetime(2024, 10, 31, 16, 30), date(2024, 11, 1)),  # tras el cierre: día siguiente
    (date(2024, 2, 2), datetime(2024, 2, 2, 16, 5), date(2024, 2, 5)),        # viernes tras cierre: lunes
    (date(2024, 7, 3), datetime(2024, 7, 3, 16, 45), date(2024, 7, 5)),       # víspera del 4 de julio
    (date(2024, 3, 28), datetime(2024, 3, 28, 17, 0), date(2024, 4, 1)),      # jueves antes de Viernes Santo
    (date(2024, 5, 1), datetime(2024, 4, 30, 18, 0), date(2024, 5, 1)),       # tras 17:30: la SEC ya movió la fecha
    (date(1996, 12, 19), None, date(1996, 12, 20)),                           # sin hora oficial: prudente
    (date(2021, 12, 30), datetime(2021, 12, 30, 16, 10), date(2021, 12, 31)), # 1-ene-2022 en sábado: el 31 se abre
    (date(2001, 9, 10), datetime(2001, 9, 10, 16, 30), date(2001, 9, 17)),    # cierre del 11-S
    (date(1997, 1, 17), datetime(1997, 1, 17, 16, 30), date(1997, 1, 20)),    # MLK no cerraba antes de 1998
    (date(2012, 10, 26), datetime(2012, 10, 26, 16, 30), date(2012, 10, 31)), # huracán Sandy
])
def test_fecha_publicacion(filing, aceptacion, esperado):
    assert sec_edgar.fecha_publicacion(filing, aceptacion) == esperado


def test_aceptacion_desde_hdr():
    assert sec_edgar.aceptacion_desde_hdr((FIX / "hdr.sgml").read_text()) == datetime(2024, 11, 1, 6, 1, 36)
    assert sec_edgar.aceptacion_desde_hdr("<SEC-HEADER>\n</SEC-HEADER>") is None


# --------------------------------------------------------------------------- filings

def test_tabla_filings_une_paginas_y_aplica_timegate(sin_red):
    df = _filings(sin_red)
    assert len(df) == 7                                    # 5 recientes + 2 antiguos
    assert df["filing_date"].is_monotonic_increasing
    k = df.loc[df["accession"] == "0000320193-24-000123"].iloc[0]
    assert k["familia"] == "10-K" and k["retraso_dias"] == 34
    assert k["url_documento"].endswith("/edgar/data/320193/000032019324000123/aapl-20240928.htm")
    # Sin verificar: fecha prudente (día hábil siguiente) para todos; la hora de la API no se usa.
    assert not df["aceptacion_verificada"].any() and k["fecha_publicacion"] == date(2024, 11, 4)

    v = sec_edgar.verificar_aceptacion(df).set_index("accession")
    assert v.loc["0000320193-24-000123", "fecha_publicacion"] == date(2024, 11, 1)   # 06:01 -> mismo día
    assert v.loc["0000320193-24-000120", "fecha_publicacion"] == date(2024, 11, 1)   # 16:30 -> día siguiente
    assert v.loc["0000320193-24-000070", "fecha_publicacion"] == date(2024, 7, 5)    # víspera del 4 de julio
    viejo = v.loc["0000320193-94-000016"]
    assert not viejo["aceptacion_verificada"] and viejo["fecha_publicacion"] == date(1994, 12, 14)
    assert viejo["url_documento"].endswith("/0000320193-94-000016.txt")
    # La segunda vez sale de la caché, sin red.
    import ingesta.empresas.sec_edgar as mod
    mod.get, original = (lambda *a, **k: (_ for _ in ()).throw(AssertionError("sin red"))), mod.get
    try:
        assert sec_edgar.verificar_aceptacion(df)["aceptacion_verificada"].sum() == 5
    finally:
        mod.get = original


def test_filtrar_deja_solo_8k_de_resultados(sin_red):
    sel = sec_edgar.filtrar(_filings(sin_red))
    assert set(sel["form"]) == {"10-K", "10-Q", "8-K"}
    assert "0000320193-24-000070" not in set(sel["accession"])   # 8-K de Item 5.02, no resultados
    assert sec_edgar.es_resultados("2.02,9.01") and sec_edgar.es_resultados("12") and not sec_edgar.es_resultados("5.02")


def test_elegir_exhibit_99():
    nombres = json.loads((FIX / "index_8k.json").read_text())["directory"]["item"]
    assert sec_edgar.elegir_exhibit_99([n["name"] for n in nombres]) == "aapl-20241031xex991.htm"
    assert sec_edgar.elegir_exhibit_99(["ex99-2.htm", "ex99-1.htm"]) == "ex99-1.htm"
    for nombre in ["exhibit991.htm", "a8-kexhibit991.htm", "q12024exhibit991.htm", "exh99-1.htm", "d123dex991.htm"]:
        assert sec_edgar.elegir_exhibit_99(["d8k.htm", nombre]) == nombre
    assert sec_edgar.elegir_exhibit_99(["exhibit992.htm", "exhibit991.htm"]) == "exhibit991.htm"
    assert sec_edgar.elegir_exhibit_99(["d10k.htm", "R1.htm"]) is None


def test_anexo_99_por_tipo_aunque_el_nombre_no_lo_diga():
    docs = sec_edgar.documentos_del_indice((FIX / "index_8k.htm").read_text())
    assert ("q4fy24earnings.htm", "EX-99.1") in docs and ("aapl-20241031.htm", "8-K") in docs
    assert sec_edgar.elegir_anexo_99(docs) == "q4fy24earnings.htm"          # por tipo, no por nombre
    assert sec_edgar.elegir_anexo_99([("d8k.htm", "8-K"), ("r.htm", "EX-99.2")]) == "r.htm"
    assert sec_edgar.elegir_anexo_99([("d8k.htm", ""), ("x-ex991.htm", "")]) == "x-ex991.htm"   # sin tipos: nombre
    assert sec_edgar.elegir_anexo_99([("d8k.htm", "8-K"), ("img.jpg", "GRAPHIC")]) is None


def test_cache_corrupta_se_vuelve_a_descargar(sin_red):
    df = sec_edgar.tabla_filings(320193)                        # crea la caché
    ruta = sin_red / "corpus" / "empresas" / "sec" / "submissions" / "CIK0000320193.json.gz"
    ruta.write_bytes(ruta.read_bytes()[:20])                    # simula una descarga interrumpida
    assert len(sec_edgar.tabla_filings(320193)) == len(df)      # se detecta y se vuelve a descargar


def test_universo():
    tabla = universo.tickers_a_tabla(json.loads((FIX / "company_tickers.json").read_text()))
    assert universo.resolver_cik("aapl", tabla) == 320193
    with pytest.raises(KeyError):
        universo.resolver_cik("LEH", tabla)
    assert universo.cik10(320193) == "0000320193"
    assert len(universo.MUESTRA_SONDA) == 10 and sum(not e.activa for e in universo.MUESTRA_SONDA) == 2


# --------------------------------------------------------------------------- secciones

def test_secciones_10k_toma_el_cuerpo_y_no_el_indice():
    texto = secciones.html_a_texto((FIX / "10k_mini.htm").read_bytes())
    assert "hidden xbrl metadata" not in texto                 # ix:header fuera
    sec = secciones.extraer_secciones(texto, "10-K")
    assert {"1", "1A", "7", "7A", "8"} <= set(sec)
    assert "1B" not in sec                                     # "None." es demasiado corto
    assert sec["1A"].startswith("Item 1A. Risk Factors") and "risk factors in this section" in sec["1A"]
    assert "MD&A" in sec["7"] and "market risk" not in sec["7"]


def test_secciones_10q_por_partes():
    sec = secciones.extraer_secciones(secciones.html_a_texto((FIX / "10q_mini.htm").read_bytes()), "10-Q")
    assert {"I-1", "I-2", "I-3", "I-4", "II-1", "II-1A", "II-6"} == set(sec)
    assert "quarterly MD&A" in sec["I-2"] and "updated risk factors" in sec["II-1A"]


def test_referencia_cruzada_partida_no_es_encabezado():
    # En el HTML de los 2000 el código fuente parte las líneas: "...in\nItem 8 of this Form 10-K".
    cuerpo = "The discussion continues with details about revenue and costs. " * 20
    html = ("<html><body><p><b>Item 7. Management's Discussion</b></p>"
            f"<p>{cuerpo} Refer to Note 12 included in\nItem 8 of this Form 10-K for more. {cuerpo}</p>"
            f"<p><b>Item 7A. Market Risk</b></p><p>{cuerpo}</p>"
            f"<p><b>Item 8. Financial Statements</b></p><p>{cuerpo}</p></body></html>")
    sec = secciones.extraer_secciones(secciones.html_a_texto(html), "10-K")
    assert "Refer to Note 12 included in Item 8 of this Form" in sec["7"]
    assert sec["7"].count("revenue and costs") == 40 and "7A" in sec and sec["8"].startswith("Item 8. Financial")
    # En texto plano tampoco cuenta una línea que empieza por "Item 8 of this report".
    plano = f"Item 7. MD&A\n{cuerpo}\nItem 8 of this report has the details.\n{cuerpo}\nItem 8. Financial Statements\n{cuerpo}"
    assert secciones.extraer_secciones(plano, "10-K")["7"].count("revenue and costs") == 40


def test_cabeceras_de_pagina_no_trocean_la_seccion():
    pagina = "Operating results improved across segments during the year. " * 15
    texto = "\n".join(["Item 7. Management's Discussion and Analysis", pagina,
                       "PART II", "Item 7", pagina, "PART II", "Item 7", pagina,
                       "Item 8. Financial Statements", pagina])
    sec = secciones.extraer_secciones(texto, "10-K")
    assert sec["7"].count("Operating results") == 45


def test_variantes_de_encabezado():
    relleno = "Text of the section with enough words to pass the minimum length. " * 8
    texto = (f"Items 1 and 2. Business and Properties\n{relleno}\nITEM 1(A). RISK FACTORS\n{relleno}\n"
             f"Item 3 — Legal Proceedings\n{relleno}")
    assert set(secciones.extraer_secciones(texto, "10-K")) == {"1", "1A", "3"}


def test_sgml_con_cabecera_larga():
    cabecera = "<SEC-HEADER>\n" + "FILER: COMPANY DATA x\n" * 400 + "</SEC-HEADER>\n"
    envio = cabecera + "<DOCUMENT>\n<TYPE>10-K\n<TEXT>\nPrincipal\n</TEXT>\n</DOCUMENT>\n<DOCUMENT>\n<TEXT>\nAnexo 13\n</TEXT>\n</DOCUMENT>"
    assert len(cabecera) > 5000
    assert secciones.html_a_texto(envio) == "Principal"


def test_texto_sgml_con_tablas_conserva_las_lineas():
    # Filings de texto de 2001 (p. ej. 10-Q de Enron): texto plano con marcas <TABLE>/<S>/<C>.
    relleno = "Revenues increased due to higher volumes in wholesale services during the quarter. " * 6
    envio = ("<DOCUMENT>\n<TYPE>10-Q\n<TEXT>\nPART I. FINANCIAL INFORMATION\n\nITEM 1. FINANCIAL STATEMENTS\n"
             f"<TABLE>\n<CAPTION>\n<S>                 <C>\nRevenues            $ 50,129\n</TABLE>\n{relleno}\n"
             f"ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS\n{relleno}\n{relleno}\n"
             f"PART II. OTHER INFORMATION\n\nITEM 1. LEGAL PROCEEDINGS\n{relleno}\n</TEXT>\n</DOCUMENT>")
    texto = secciones.html_a_texto(envio)
    assert "<TABLE>" not in texto and "Revenues $ 50,129" in texto
    sec = secciones.extraer_secciones(texto, "10-Q")
    assert {"I-1", "I-2", "II-1"} <= set(sec) and sec["I-2"].count("wholesale services") == 12


def test_filing_antiguo_en_texto_y_por_referencia():
    texto = secciones.html_a_texto((FIX / "10k_1996.txt").read_bytes())
    assert "should not be read from here" not in texto         # solo el primer <DOCUMENT>
    sec = secciones.extraer_secciones(texto, "10-K")
    assert "1" in sec and secciones.es_por_referencia(sec["7"]) and not secciones.es_por_referencia(sec["1"])


def test_documentos_cumplen_el_contrato(sin_red):
    fila = sec_edgar.verificar_aceptacion(_filings(sin_red)).set_index("accession").loc["0000320193-24-000123"].copy()
    fila["accession"] = "0000320193-24-000123"
    sec = secciones.extraer_secciones(secciones.html_a_texto((FIX / "10k_mini.htm").read_bytes()), "10-K")
    docs = secciones.documentos_de_filing(fila, sec, "AAPL")
    por_seccion = {d["seccion"]: d for d in docs}
    d = por_seccion["Item 1A"]
    assert d["fecha_publicacion"] == "2024-11-01" and d["fecha_referencia"] == "2024-09-28"
    assert d["rag"] == "base" and d["agente"] == "equity" and d["tickers"] == ["AAPL"]
    assert d["texto"].startswith("Item 1A") and d["extra"]["accession"] == "0000320193-24-000123"
    assert len({x["doc_id"] for x in docs}) == len(docs)
    assert d["extra"]["aceptacion"] == "2024-11-01T06:01:36"


def test_filing_sin_hora_no_escribe_nat(sin_red):
    df = _filings(sin_red)
    viejo = df.iloc[0]
    assert viejo["aceptacion"] is None
    sec = secciones.extraer_secciones(secciones.html_a_texto((FIX / "10k_1996.txt").read_bytes()), "10-K")
    (d,) = [x for x in secciones.documentos_de_filing(viejo, sec, "AAPL") if x["seccion"] == "Item 1"]
    assert d["extra"]["aceptacion"] is None


# --------------------------------------------------------------------------- XBRL

def test_companyfacts_point_in_time():
    cf = json.loads((FIX / "companyfacts_CIK0000320193.json").read_text())
    t = xbrl.companyfacts_a_tabla(cf, "AAPL")
    assert list(t.columns) == xbrl.COLUMNAS and len(t) == 6
    # Sin la tabla de filings no se sabe la hora: por prudencia, día hábil siguiente a "filed".
    assert set(t["fecha_publicacion"]) == {date(2023, 11, 6), date(2024, 11, 4)}
    activos = t.loc[t["concepto"] == "Assets"]
    antes = xbrl.vista_a_fecha(activos, date(2024, 1, 1))
    despues = xbrl.vista_a_fecha(activos, date(2024, 12, 1))
    assert antes["valor"].tolist() == [352583000000]       # el 1-ene-2024 solo se conocía la cifra original
    assert despues["valor"].tolist() == [352600000000]     # tras el 10-K de 2024, la corregida
    rex = xbrl.reexpresiones(t)
    assert rex["serie"].tolist() == ["AAPL.us-gaap:Assets"]
    assert xbrl.vista_a_fecha(t, date(2023, 11, 2)).empty  # nada publicado aún


def test_companyfacts_usa_timegate_de_filings(sin_red):
    cf = json.loads((FIX / "companyfacts_CIK0000320193.json").read_text())
    fil = pd.DataFrame({"accession": ["0000320193-24-000123"], "fecha_publicacion": [date(2024, 11, 1)]})
    t = xbrl.companyfacts_a_tabla(cf, "AAPL", fil)
    assert set(t.loc[t["accession"] == "0000320193-24-000123", "fecha_publicacion"]) == {date(2024, 11, 1)}


# --------------------------------------------------------------------------- sentimiento

def test_lm_y_risk_delta():
    lm = sentimiento.cargar_lm(FIX / "lm_mini.csv")
    assert "LOSSES" in lm["Negative"] and "OLDWORD" not in lm["Negative"]   # año negativo = retirada
    p = sentimiento.puntuar("Losses could rise. Uncertain demand may improve.", lm)
    assert p["n_palabras"] == 7 and p["lm_negative"] == pytest.approx(1 / 7)
    assert p["lm_tono"] == pytest.approx(0.0)
    assert sentimiento.risk_delta("a b c", None)["risk_delta"] is None
    assert sentimiento.risk_delta("risk risk war", "risk risk war")["risk_delta"] == pytest.approx(0.0)
    assert sentimiento.risk_delta("pandemic supply war", "competition pricing")["risk_delta"] == pytest.approx(1.0)


def test_cargar_lm_sin_fichero_da_error_claro(tmp_path, monkeypatch):
    monkeypatch.setenv("INGESTA_CORPUS_DIR", str(tmp_path))
    monkeypatch.setattr("ingesta.comun.rutas.ENV_FILE", tmp_path / "no_existe.env")
    with pytest.raises(FileNotFoundError, match="Loughran-McDonald"):
        sentimiento.cargar_lm()


def test_risk_delta_encadena_por_empresa_y_fecha():
    def doc(tk, fecha, texto, id_):
        return {"doc_id": id_, "tickers": [tk], "tipo_documento": "10-K", "fecha_publicacion": fecha,
                "seccion": "Item 1A", "version": None, "texto": texto}
    docs = [doc("A", "2021-02-01", "war pandemic supply", "a2"), doc("A", "2020-02-01", "competition pricing", "a1"),
            doc("B", "2021-02-01", "rates credit", "b2"), doc("B", "2020-02-01", "rates credit", "b1")]
    out = sonda.paso4_sentimiento(docs, lm=None).set_index("doc_id")
    assert pd.isna(out.loc["a1", "risk_delta"]) and pd.isna(out.loc["b1", "risk_delta"])   # primer año de cada una
    assert out.loc["a2", "risk_delta"] == pytest.approx(1.0)    # A reescribió sus riesgos
    assert out.loc["b2", "risk_delta"] == pytest.approx(0.0)    # B los copió


# --------------------------------------------------------------------------- sonda de punta a punta

def test_sonda_completa_sin_red(sin_red):
    cfg = sonda.ConfigSonda(empresas=(Empresa("AAPL", 320193, "Apple Inc.", "Tecnología"),), verbose=False)
    filings = sonda.paso1_filings(cfg)
    cob = sonda.medir_cobertura(filings)
    assert set(cob["familia"]) == {"10-K", "10-Q", "8-K resultados", "8-K otros"}
    assert cob.loc[cob["familia"] == "10-K", "primero"].iloc[0] == date(1994, 12, 13)

    filings = sonda.verificar_timegate(filings, cfg)
    # Solo se verifican los que se van a descargar (el 8-K de Item 5.02 no) y los de los 90 no traen hora.
    assert filings["aceptacion_verificada"].sum() == 4
    tg = sonda.contrastar_timegate(filings)
    assert len(tg) == 4 and (tg["desfase_h"] == 8).all() and not tg["coinciden"].any()
    cob = sonda.medir_cobertura(filings)
    assert cob.loc[cob["familia"] == "8-K resultados", "pct_tras_cierre"].iloc[0] == 1.0

    registro, metas = sonda.paso2_documentos(filings, cfg)
    assert registro["ok"].sum() >= 4 and registro["error"].isna().sum() == registro["ok"].sum()
    assert "7" in registro.loc[registro["form"] == "10-K", "por_referencia"].str.cat(sep=",")
    assert {"base", "cajon"} <= set(metas["rag"])
    docs = sonda.leer_documentos(cfg, [320193])
    assert len(docs) == len(metas) and all("texto" in d for d in docs)
    assert any(d["fuente"] == "sec_8k_resultados" for d in docs)

    fund = sonda.paso3_fundamentales(cfg, filings)
    assert len(fund) == 6 and (sin_red / "corpus" / "empresas" / "fundamentales" / "320193.parquet").exists()

    lm = sentimiento.cargar_lm(FIX / "lm_mini.csv")
    sent = sonda.paso4_sentimiento(docs, lm)
    assert "lm_negative" in sent and (sent["lm_negative"] > 0).any()
    # Solo hay un Item 1A (el de 1996 no existe: es obligatorio desde 2005): sin año anterior, risk_delta vacío.
    item1a = sent.loc[sent["seccion"] == "Item 1A"]
    assert len(item1a) == 1 and item1a["risk_delta"].isna().all()

    vol = sonda.extrapolar_volumen(registro, n_empresas=1)
    assert (vol["filings_universo"] >= vol["filings_muestra"]).all()
    fichas = sonda.fichas_tema3(cob, registro, vol, fund, lm_disponible=True, n_empresas=1)
    assert len(fichas) == 4 and (sin_red / "results" / "empresas" / "fichas" / "sec_10k_10q.json").exists()
    sonda.guardar_tablas(cobertura=cob, registro_descargas=registro)
    assert (sin_red / "results" / "empresas" / "cobertura.csv").exists()


def test_10k_sin_items_se_guarda_entero_en_el_cajon(sin_red, monkeypatch):
    sin_items = ("<html><body>" + "<p>Our businesses performed well across aerospace and power segments.</p>" * 80
                 + "<p>Form 10-K Cross Reference Index</p></body></html>").encode()
    monkeypatch.setattr(sonda.sec_edgar, "descargar_documento",
                        lambda f, url=None, usar_cache=True: (sin_items, FIX / "ex991.htm"))
    fila = _filings(sin_red).set_index("accession").loc["0000320193-24-000123"].copy()
    fila["accession"], fila["ticker"] = "0000320193-24-000123", "GE"
    docs, reg = sonda._procesar_filing(fila, sonda.ConfigSonda(verbose=False))
    assert len(docs) == 1 and docs[0]["seccion"] == "Documento completo" and docs[0]["rag"] == "cajon"
    assert not reg["ok"] and "guardado entero" in reg["error"]


def test_prueba_rapida_no_pisa_la_completa(sin_red):
    empresa = (Empresa("AAPL", 320193, "Apple Inc.", "Tecnología"),)
    completa = sonda.ConfigSonda(empresas=empresa, verbose=False)
    rapida = sonda.ConfigSonda(empresas=empresa, verbose=False, max_por_familia=1)
    filings = sonda.paso1_filings(completa)
    sonda.paso2_documentos(filings, completa)
    sonda.paso2_documentos(filings, rapida)
    assert len(sonda.leer_documentos(completa)) > len(sonda.leer_documentos(rapida)) > 0


def test_sonda_sin_ninguna_empresa_da_error_claro(sin_red):
    cfg = sonda.ConfigSonda(empresas=(Empresa("X", 999999, "Sin datos", "-"),), verbose=False)
    with pytest.raises(RuntimeError, match="ninguna empresa"):
        sonda.paso1_filings(cfg)


# --------------------------------------------------------------------------- con red (INGESTA_LIVE=1)

@pytest.mark.live
def test_live_submissions_apple():
    df = sec_edgar.tabla_filings(320193, usar_cache=False)
    k = df.loc[df["familia"] == "10-K"]
    assert len(k) >= 25 and k["filing_date"].min() < date(1997, 1, 1)
    v = sec_edgar.verificar_aceptacion(k.tail(3), usar_cache=False)
    assert v["aceptacion_verificada"].all()
    # Hora oficial dentro del horario de EDGAR (6:00-22:00 de Nueva York) y en el día del filing o el anterior.
    for _, f in v.iterrows():
        assert 6 <= f["aceptacion"].hour < 22
        assert 0 <= (f["filing_date"] - f["aceptacion"].date()).days <= 1
    # El último 8-K de resultados de Apple tiene su nota de prensa como anexo EX-99.1 en la página de índice.
    ocho_k = df.loc[(df["familia"] == "8-K") & df["items"].map(sec_edgar.es_resultados)].iloc[-1]
    url, ficheros = sec_edgar.exhibit_resultados(ocho_k, usar_cache=False)
    assert url is not None, ficheros
    assert any("(EX-99.1)" in x for x in ficheros), ficheros
