"""Operaciones bursátiles del Congreso de EE. UU. («Pelosi tracker», STOCK Act 2012).

- **Cámara** (``disclosures-clerk.house.gov``): índice anual ``<AAAA>FD.zip`` (XML + TXT con todas
  las declaraciones: tipo, miembro, distrito, ``FilingDate``, ``DocID``) y un PDF por PTR
  (``ptr-pdfs/<AAAA>/<DocID>.pdf``). DocID ``2xxxxxxx`` = PTR electrónico (texto extraíble con
  pypdf); ``8xxxxxx``/``9xxxxxx`` = escaneado (imagen: necesitaría OCR, que no se hace).
- **Senado** (eFD): puerta de aceptación de condiciones; no se automatiza (ficha ``no_probada``).
- **Agregadores**: House/Senate Stock Watcher (S3, se comprueba), Quiver / Capitol Trades (pago o
  prohíben scraping: ``no_probada``).
- **Proxy**: ETFs NANC y KRUZ (réplica de carteras del Congreso, desde 2023-02) vía yfinance.

TimeGate: ``fecha_referencia`` = ``FilingDate`` del PTR (día de presentación), NUNCA la fecha de la
operación (hasta 45 días antes). El Clerk sube el PDF días después del FilingDate, así que
``fecha_publicacion`` = FilingDate + k días hábiles, con k = p90 del retraso medido entre el FilingDate y
el ``Last-Modified`` del PDF en la muestra (mínimo ``K_MIN`` = 1: el conteo del día D solo se cierra al
acabar D). Los índices anuales se siguen ampliando con presentaciones tardías (cada una con su propio
FilingDate). Importes en rangos.
"""
from __future__ import annotations

import io
import math
import random
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.http import get, robots_permite
from ingesta.noticias_alt import base
from ingesta.noticias_alt.limpieza import a_fecha

MODULO = "congreso"
HOUSE = "https://disclosures-clerk.house.gov/public_disc/"
F_HOUSE = "congreso_house_ptr"
F_SENADO = "congreso_senate_efd"
F_WATCHER = "congreso_stock_watcher"
F_AGREG = "congreso_agregadores_pago"
F_ETF = "congreso_etfs_nanc_kruz"
LICENCIA_GOV = ("registro público del Gobierno de EE. UU. (sin copyright); uso comercial prohibido por la "
                "Ethics in Government Act (5 U.S.C. §13107) salvo prensa")
ANIO_INICIO = 2014            # primer índice con PTR (STOCK Act 2012-04; e-filing de PTR desde 2014)
ANIOS_MUESTRA = range(2020, 2026)
K_MIN = 1                     # días hábiles mínimos FilingDate -> publicación (el día D se cierra al acabar D)
LAG_WEB_MAX_D = 60            # Last-Modified más tardío = PDF regenerado (no mide la subida original)
OBJETIVO_TICKERS = 60         # operaciones con ticker para la rentabilidad anormal ilustrativa
WATCHER = {
    "house": "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
    "senate": "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
}
SENADO = "https://efdsearch.senate.gov/search/home/"
OPERACION = {"P": "compra", "S": "venta", "S (partial)": "venta parcial", "E": "canje"}


# --------------------------------------------------------------------------- URLs

def url_indice(anio: int) -> str:
    return f"{HOUSE}financial-pdfs/{anio}FD.zip"


def url_ptr(anio: int, doc_id: str) -> str:
    return f"{HOUSE}ptr-pdfs/{anio}/{doc_id}.pdf"


# --------------------------------------------------------------------------- descarga con caché

def _bajar(res: base.Resultado, cfg: base.ConfigSonda, fuente: str, item: str, url: str, ruta: Path | None,
           *, dia: str = "", max_edad_h: float | None = None, **kw: Any) -> bytes | None:
    """Bytes de ``url`` (o de la caché ``ruta``, escrita de forma atómica); registra la petición. None si falla."""
    return base.get_cacheado(res, cfg, fuente, item, url, ruta, getter=get, dia=dia, max_edad_h=max_edad_h, **kw)


# --------------------------------------------------------------------------- publicación (TimeGate)

def habiles_hasta_web(presentado: date, last_modified: str | None) -> int | None:
    """Días hábiles entre el FilingDate y la subida del PDF (Last-Modified en hora de NY; desde las 16:00 cuenta
    el hábil siguiente). None si no hay cabecera o es una regeneración (fuera de 0..``LAG_WEB_MAX_D`` días)."""
    lm = a_fecha(last_modified) if last_modified else None
    if lm is None or presentado is None:
        return None
    ny = pd.Timestamp(lm).tz_convert("America/New_York")
    d = ny.date() + timedelta(days=1) if ny.hour >= 16 else ny.date()
    if not 0 <= (d - presentado).days <= LAG_WEB_MAX_D:
        return None
    return int(pd.bdate_range(presentado, d).size - 1) if d > presentado else 0


def k_publicacion(lags_habiles: list[int | None]) -> tuple[int, dict[str, Any]]:
    """k = p90 (redondeado hacia arriba) del retraso FilingDate -> web, con mínimo ``K_MIN``."""
    v = pd.Series([x for x in lags_habiles if x is not None], dtype=float)
    if v.empty:
        return K_MIN, dict(n=0, mediana=None, p90=None, k=K_MIN, origen="sin Last-Modified: mínimo D+1")
    p90 = float(v.quantile(.9))
    k = max(K_MIN, math.ceil(p90))
    return k, dict(n=int(v.size), mediana=float(v.median()), p90=round(p90, 2), k=k, origen="p90 medido")


def publicar(presentado: date | None, k: int) -> date | None:
    return (pd.Timestamp(presentado) + pd.offsets.BDay(k)).date() if presentado else None


# --------------------------------------------------------------------------- índice anual

def _fecha_mdy(txt: str | None) -> date | None:
    try:
        return datetime.strptime((txt or "").strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def es_electronico(doc_id: str | None) -> bool:
    """PTR electrónico (formulario web con texto): DocID de 8 cifras que empieza por 2."""
    d = (doc_id or "").strip()
    return len(d) == 8 and d.startswith("2")


def parse_indice(xml: bytes) -> list[dict[str, Any]]:
    """XML del índice anual -> declaraciones con ``fecha_publicacion`` (= ``FilingDate``)."""
    import lxml.etree as ET

    raiz = ET.fromstring(xml.lstrip(b"\xef\xbb\xbf"), parser=ET.XMLParser(recover=True, resolve_entities=False))
    filas = []
    for m in raiz.findall("Member"):
        f = {c.tag: (c.text or "").strip() for c in m}
        nombre = " ".join(x for x in (f.get("First", ""), f.get("Last", "")) if x)
        filas.append(dict(representante=nombre, distrito=f.get("StateDst") or None, tipo=f.get("FilingType"),
                          fecha_publicacion=_fecha_mdy(f.get("FilingDate")), doc_id=f.get("DocID") or "",
                          anio=int(f.get("Year") or 0), electronico=es_electronico(f.get("DocID"))))
    return filas


def indice_zip(contenido: bytes) -> list[dict[str, Any]]:
    """``<AAAA>FD.zip`` -> declaraciones (usa el XML del zip)."""
    z = zipfile.ZipFile(io.BytesIO(contenido))
    return parse_indice(z.read(next(n for n in z.namelist() if n.lower().endswith(".xml"))))


def listar_ptr(anio: int) -> list[dict[str, Any]]:
    """PTR (tipo ``P``) del índice de ``anio`` (sin caché ni registro: uso de las tools)."""
    return [f for f in indice_zip(get(url_indice(anio), timeout=120).content) if f["tipo"] == "P"]


# --------------------------------------------------------------------------- PTR (PDF)

_IMPORTE = r"(?:\$[\d,]+\s*-\s*\$[\d,]+|Over\s+\$[\d,]+|\$[\d,]+\s*\+?)"
# Ancla de cada operación: «<tipo> <fecha op> <fecha notificación> <importe>». Los PDF de 2014-2022 salen de
# pypdf sin saltos de línea y con letras en minúscula cambiada («s (partial)», «(CoP) [sT]», «P02/14/2022»):
# el tipo es la letra pegada a la primera fecha.
_OP = re.compile(r"(?P<op>[Ss] \(partial\)|[PSEpse])\s*(?P<f_op>\d{1,2}/\d{1,2}/\d{4})\s*"
                 r"(?P<f_not>\d{1,2}/\d{1,2}/\d{4})\s*(?P<importe>" + _IMPORTE + ")")
# Ticker: paréntesis al FINAL del activo, con o sin código de clase [ST]/[OT]...
_TICKER_FIN = re.compile(r"\((?P<ticker>[A-Za-z][A-Za-z0-9.\-$]{0,9})\)\s*(?:\[(?P<clase>[A-Za-z]{2})\])?\s*$")
_TICKER = re.compile(r"\((?P<ticker>[A-Za-z][A-Za-z0-9.\-$]{0,9})\)\s*(?:\[(?P<clase>[A-Za-z]{2})\])?")
_CLASE_FIN = re.compile(r"\[(?P<clase>[A-Za-z]{2})\]\s*$")
_CABECERA = re.compile(r"^(?:ID Owner Asset|ID$|Owner\s*Asset|Type$|Date$|Date Notification|Amount Cap|Gains >|\$200\?|"
                       r"Filing ID #|F$)")
_MARCA = re.compile(r"(?i):|\$200\?|fedcb?|filing id #\d+|notification\s*date\s*amount")
_ESTADO = re.compile(r"(?i)(?:f\s*)?iling\s*status:\s*(?:new|amended)")
NULO = chr(0)  # los PTR electrónicos traen glifos nulos en los títulos


def texto_pdf(contenido: bytes) -> tuple[str, int]:
    """Texto (sin glifos nulos) y nº de páginas de un PDF."""
    import pypdf

    lector = pypdf.PdfReader(io.BytesIO(contenido))
    texto = "\n".join(p.extract_text() or "" for p in lector.pages)
    return texto.replace(NULO, ""), len(lector.pages)


def _activo(segmento: str, nuevo: bool = True) -> str:
    """Nombre del activo: cola del segmento previo a la operación, tras el último marcador de ruido
    («Filing Status: New», «Subholding Of: ...», cabecera de la tabla, casillas «fedc»)."""
    seg = _ESTADO.sub("\n:", segmento)
    cola: list[str] = []
    for l in reversed([x.strip() for x in seg.splitlines()]):
        if not l:
            continue
        marcas = list(_MARCA.finditer(l))
        if marcas or _CABECERA.match(l) or (nuevo and len(l) > 70):
            resto = l[marcas[-1].end():].strip() if marcas else ""
            # Formato antiguo (todo en una línea): el activo va pegado tras la etiqueta. En el nuevo
            # lo que sigue a «Etiqueta:» es su valor, no el activo; una línea larga es un comentario.
            if resto and resto not in ("New", "Amended") and not nuevo:
                cola.insert(0, resto)
            break
        cola.insert(0, l)
    # Cola de un comentario de varias líneas («...without my input or / direction.»): empieza en minúscula.
    while nuevo and cola and cola[0][:1].islower():
        cola.pop(0)
    return " ".join(cola).strip()


def es_formato_nuevo(texto: str) -> bool:
    """PTR con una línea por campo (≈2023-hoy) frente al antiguo pegado con casillas «g fedc» (2014-2022)."""
    return "fedc" not in texto


def _importe(txt: str) -> tuple[int | None, int | None]:
    cifras = [int(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", txt)]
    if not cifras:
        return None, None
    if txt.lower().startswith("over") or len(cifras) == 1:
        return cifras[0], None
    return cifras[0], cifras[1]


def _fecha_ptr(txt: str) -> date:
    return datetime.strptime(txt, "%m/%d/%Y").date()


def parse_ptr_texto(texto: str) -> list[dict[str, Any]]:
    """Operaciones de un PTR electrónico (texto de pypdf) -> lista de dicts.

    Ancla: ``<tipo> <fecha op> <fecha notificación> <importe>``. El activo es el texto anterior
    hasta el último marcador de ruido; el ticker, el paréntesis al final del activo. Si el activo
    se parte entre páginas, el ticker se busca en la continuación (tras la operación, hasta la
    primera línea con ``:``).
    """
    t = texto.replace(NULO, "")
    nuevo = es_formato_nuevo(t)
    ops: list[dict[str, Any]] = []
    previo = 0
    for m in _OP.finditer(t):
        activo = _activo(t[previo:m.start()], nuevo)
        previo = m.end()
        mt = _TICKER_FIN.search(activo)
        if mt is None and not _CLASE_FIN.search(activo):  # continuación tras salto de página
            sig = _OP.search(t, m.end())
            lineas = []
            for l in (x.strip() for x in t[m.end(): sig.start() if sig else len(t)].splitlines()):
                if ":" in l:
                    break
                if l and not _CABECERA.match(l):
                    lineas.append(l)
            extra = " ".join(lineas)
            mt = _TICKER.search(extra)
            if mt:
                activo = f"{activo} {extra}".strip()
        cm = _CLASE_FIN.search(activo)
        clase = (mt.group("clase") if mt and mt.group("clase") else (cm.group("clase") if cm else None))
        titular = "self"
        mm = re.match(r"^(SP|JT|DC)(?=\s|[A-Z][A-Za-z])\s*(.*)$", activo)
        if mm:
            titular, activo = mm.group(1), mm.group(2)
        imin, imax = _importe(re.sub(r"\s+", " ", m.group("importe")).strip())
        op = m.group("op")
        ops.append(dict(titular=titular, activo=activo[:200], ticker=mt.group("ticker").upper() if mt else None,
                        clase=clase.upper() if clase else None,
                        tipo=OPERACION["S (partial)" if "partial" in op else op.upper()],
                        fecha_operacion=_fecha_ptr(m.group("f_op")), fecha_notificacion=_fecha_ptr(m.group("f_not")),
                        importe_min=imin, importe_max=imax))
    return ops


def leer_ptr(anio: int, doc_id: str) -> tuple[list[dict[str, Any]], str]:
    """Descarga y parsea un PTR (sin caché ni registro: uso de las tools)."""
    texto, _ = texto_pdf(get(url_ptr(anio, doc_id), timeout=60).content)
    return parse_ptr_texto(texto), texto


# --------------------------------------------------------------------------- sondas

def _sondear_house(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    crudo = base.corpus("crudo", MODULO)
    hoy = date.today()
    anios = list(range(ANIO_INICIO, hoy.year + 1))
    filas_anio, todos = [], []
    for a in anios:
        datos = _bajar(res, cfg, F_HOUSE, f"indice {a}", url_indice(a), crudo / f"{a}FD.zip", dia=str(a),
                       max_edad_h=24 if a == hoy.year else None, timeout=120)
        if datos is None:
            continue
        try:
            fs = indice_zip(datos)
        except Exception as e:  # noqa: BLE001
            res.registrar(F_HOUSE, f"parseo indice {a}", url_indice(a), False, dia=str(a), error=repr(e))
            continue
        ptr = list({f["doc_id"]: f for f in fs if f["tipo"] == "P"}.values())
        fechas = [f["fecha_publicacion"] for f in ptr if f["fecha_publicacion"]]
        n_e = sum(f["electronico"] for f in ptr)
        filas_anio.append(dict(anio=a, n_declaraciones=len(fs), n_ptr=len(ptr), n_electronicos=n_e,
                               n_escaneados=len(ptr) - n_e,
                               pct_electronicos=round(100 * n_e / len(ptr), 1) if ptr else None,
                               primera_presentacion=min(fechas).isoformat() if fechas else None,
                               ultima_presentacion=max(fechas).isoformat() if fechas else None,
                               bytes_indice=len(datos)))
        vistos = {f["doc_id"] for f in todos}
        todos += [f for f in ptr if f["doc_id"] not in vistos]
    tabla_anio = pd.DataFrame(filas_anio, columns=["anio", "n_declaraciones", "n_ptr", "n_electronicos", "n_escaneados",
                                                   "pct_electronicos", "primera_presentacion", "ultima_presentacion",
                                                   "bytes_indice"])
    res.tablas["congreso_ptr_anio"] = tabla_anio

    fechas_ptr = pd.Series([f["fecha_publicacion"] for f in todos if f["fecha_publicacion"]], dtype="object")
    if not fechas_ptr.empty:
        mes = pd.to_datetime(fechas_ptr).dt.to_period("M").value_counts().sort_index()
        res.tablas["congreso_ptr_mes"] = pd.DataFrame(dict(mes=mes.index.astype(str), n_ptr=mes.values))

    # Muestra de PTR electrónicos repartida por años; se amplía hasta OBJETIVO_TICKERS operaciones con ticker.
    n_obj = cfg.max_items or (6 if cfg.modo_rapido else 40)
    n_max = n_obj if cfg.modo_rapido else 2 * n_obj
    colas = {a: [f for f in todos if f["anio"] == a and f["electronico"]] for a in ANIOS_MUESTRA}
    for a, cola in colas.items():
        random.Random(a).shuffle(cola)
    colas = {a: c for a, c in colas.items() if c}
    muestra: list[dict] = []
    ops_tabla: list[dict] = []
    while colas and len(muestra) < n_max:
        con_ticker = sum(o["ticker"] is not None for o in ops_tabla)
        if len(muestra) >= n_obj and con_ticker >= OBJETIVO_TICKERS:
            break
        for a in list(colas):
            if not colas[a]:
                colas.pop(a)
                continue
            if len(muestra) >= n_max:
                break
            f = colas[a].pop()
            url = url_ptr(f["anio"], f["doc_id"])
            ruta = crudo / f"ptr_{f['anio']}_{f['doc_id']}.pdf"
            datos = _bajar(res, cfg, F_HOUSE, f"ptr {f['doc_id']}", url, ruta,
                           dia=f["fecha_publicacion"].isoformat() if f["fecha_publicacion"] else "", timeout=60,
                           guardar_meta=True)
            if datos is None:
                continue
            try:
                texto, paginas = texto_pdf(datos)
                ops = parse_ptr_texto(texto)
            except Exception as e:  # noqa: BLE001
                res.registrar(F_HOUSE, f"parseo ptr {f['doc_id']}", url, False, error=repr(e))
                continue
            lm = base.leer_meta(ruta).get("last_modified")
            muestra.append(dict(doc_id=f["doc_id"], anio=f["anio"], representante=f["representante"],
                                distrito=f["distrito"], fecha_presentacion=f["fecha_publicacion"], last_modified=lm,
                                habiles_hasta_web=habiles_hasta_web(f["fecha_publicacion"], lm), bytes=len(datos),
                                paginas=paginas, chars_texto=len(texto.strip()), n_operaciones=len(ops),
                                con_texto=len(texto.strip()) > 50))
            for o in ops:
                ops_tabla.append(dict(doc_id=f["doc_id"], anio=f["anio"], representante=f["representante"],
                                      distrito=f["distrito"], titular=o["titular"], activo=o["activo"],
                                      ticker=o["ticker"], clase=o["clase"], tipo=o["tipo"],
                                      fecha_operacion=o["fecha_operacion"], fecha_notificacion=o["fecha_notificacion"],
                                      fecha_presentacion=f["fecha_publicacion"],
                                      importe_min=o["importe_min"], importe_max=o["importe_max"]))
    # Control: unos pocos escaneados para confirmar que no traen texto (no se hace OCR).
    if not cfg.modo_rapido:
        for a in (2020, 2024):
            esc = [f for f in todos if f["anio"] == a and not f["electronico"]]
            if not esc:
                continue
            f = random.Random(a).choice(esc)
            url = url_ptr(a, f["doc_id"])
            datos = _bajar(res, cfg, F_HOUSE, f"ptr escaneado {f['doc_id']}", url,
                           crudo / f"ptr_{a}_{f['doc_id']}.pdf", timeout=90)
            if datos is None:
                continue
            try:
                texto, paginas = texto_pdf(datos)
            except Exception as e:  # noqa: BLE001
                res.registrar(F_HOUSE, f"parseo ptr {f['doc_id']}", url, False, error=repr(e))
                continue
            muestra.append(dict(doc_id=f["doc_id"], anio=a, representante=f["representante"], distrito=f["distrito"],
                                fecha_presentacion=f["fecha_publicacion"], last_modified=None, habiles_hasta_web=None,
                                bytes=len(datos), paginas=paginas, chars_texto=len(texto.strip()), n_operaciones=0,
                                con_texto=len(texto.strip()) > 50))

    # k medido: FilingDate -> Last-Modified del PDF (p90, mínimo D+1).
    k, med_k = k_publicacion([m["habiles_hasta_web"] for m in muestra])
    metodo = f"FilingDate + {k} días hábiles (k = {med_k['origen']}, n={med_k['n']})"
    for m in muestra:
        m["fecha_publicacion"] = publicar(m["fecha_presentacion"], k)
    for o in ops_tabla:
        o["fecha_publicacion"] = publicar(o["fecha_presentacion"], k)
        o["lag_dias"] = (o["fecha_publicacion"] - o["fecha_operacion"]).days if o["fecha_publicacion"] else None

    # Serie: PTR presentados por día (toda la historia de los índices), conocidos en FilingDate + k.
    if not fechas_ptr.empty:
        por_dia = pd.to_datetime(fechas_ptr).value_counts().sort_index()
        serie = pd.DataFrame(dict(serie="congreso_ptr_presentados", fecha_referencia=por_dia.index,
                                  fecha_publicacion=por_dia.index + pd.offsets.BDay(k), valor=por_dia.values,
                                  unidad="PTR/día", fuente=F_HOUSE, metodo_publicacion=metodo,
                                  nota="índice anual del Clerk; las presentaciones tardías entran con su FilingDate"))
        res.series.append(base.guardar_serie(serie, "congreso_ptr_diarios", rapido=cfg.modo_rapido).stem)

    cols_ops = ["doc_id", "anio", "representante", "distrito", "titular", "activo", "ticker", "clase", "tipo",
                "fecha_operacion", "fecha_notificacion", "fecha_presentacion", "fecha_publicacion", "lag_dias",
                "importe_min", "importe_max"]
    t_ops = pd.DataFrame(ops_tabla, columns=cols_ops)
    t_mu = pd.DataFrame(muestra, columns=["doc_id", "anio", "representante", "distrito", "fecha_presentacion",
                                          "last_modified", "habiles_hasta_web", "fecha_publicacion", "bytes",
                                          "paginas", "chars_texto", "n_operaciones", "con_texto"])
    res.tablas["congreso_operaciones"] = t_ops
    res.tablas["congreso_ptr_muestra"] = t_mu

    # TimeGate: fecha de la operación (hecho) frente a FilingDate + k (publicación).
    for o in ops_tabla[:: max(1, len(ops_tabla) // 12)][:12]:
        res.timegate.append(dict(fuente=F_HOUSE, item=f"{o['doc_id']} {o['ticker'] or o['activo'][:30]}",
                                 fecha_hecho=o["fecha_operacion"].isoformat(),
                                 fecha_publicacion=o["fecha_publicacion"].isoformat() if o["fecha_publicacion"] else None,
                                 campo_fecha=metodo,
                                 lag_h=base.lag_horas(o["fecha_operacion"], o["fecha_publicacion"])))
    for m in [m for m in muestra if m["habiles_hasta_web"] is not None][:4]:
        res.timegate.append(dict(fuente=F_HOUSE, item=f"PDF {m['doc_id']}", fecha_hecho=m["fecha_presentacion"].isoformat(),
                                 fecha_publicacion=a_fecha(m["last_modified"]).isoformat(),
                                 campo_fecha="FilingDate -> Last-Modified del PDF (subida al web del Clerk)",
                                 lag_h=base.lag_horas(m["fecha_presentacion"], a_fecha(m["last_modified"]))))

    # Medidas, cobertura y ficha.
    lags = base.resumen_lag(t_ops["lag_dias"].dropna().astype(float) * 24 if not t_ops.empty else [])
    fechas_todas = [f["fecha_publicacion"] for f in todos if f["fecha_publicacion"]]
    ini = min(fechas_todas).isoformat() if fechas_todas else None
    fin = max(fechas_todas).isoformat() if fechas_todas else None
    electr = t_mu[t_mu["n_operaciones"] > 0] if not t_mu.empty else t_mu
    bytes_pdf = float(t_mu["bytes"].mean()) if not t_mu.empty else 0.0
    n_ptr_total = int(tabla_anio["n_ptr"].sum()) if not tabla_anio.empty else 0
    pct_e = round(100 * tabla_anio["n_electronicos"].sum() / n_ptr_total, 1) if n_ptr_total else None
    n_tick = int(t_ops["ticker"].notna().sum())
    est = base.estado(res.registro, F_HOUSE)
    med = dict(indices=len(tabla_anio), ptr_total=n_ptr_total, pct_electronicos=pct_e,
               ptr_leidos=int(len(electr)), operaciones=int(len(t_ops)), operaciones_con_ticker=n_tick,
               ops_por_ptr=round(len(t_ops) / len(electr), 2) if len(electr) else None,
               compras=int((t_ops["tipo"] == "compra").sum()), ventas=int(t_ops["tipo"].str.startswith("venta").sum()),
               lag_dias_mediana=round(lags["mediana_h"] / 24, 1) if lags["mediana_h"] is not None else None,
               lag_dias_p90=round(lags["p90_h"] / 24, 1) if lags["p90_h"] is not None else None,
               bytes_pdf_medio=round(bytes_pdf), escaneados_sin_texto=int((~t_mu["con_texto"]).sum()) if not t_mu.empty else 0,
               filing_a_web_habiles=med_k)
    nota = (f"{n_ptr_total} PTR {ANIO_INICIO}-{hoy.year} ({pct_e} % electrónicos); {len(electr)} PDF leídos -> "
            f"{len(t_ops)} operaciones ({n_tick} con ticker); lag operación->publicación mediana "
            f"{med['lag_dias_mediana']} d; publicación = FilingDate + {k} hábiles")
    res.cobertura.append(dict(fuente=F_HOUSE, familia="divulgaciones", tipo="evento", inicio=ini, fin=fin, hora=False,
                              idiomas="en", lag_mediana_h=lags["mediana_h"], destino="serie", backtest=True, estado=est,
                              nota=nota))
    res.fichas.append(base.ficha(
        fuente=F_HOUSE, nombre="House Clerk · Periodic Transaction Reports (STOCK Act)", agentes=["equity", "risk"],
        rag="serie", url="https://disclosures-clerk.house.gov/FinancialDisclosure", acceso="descarga",
        licencia_y_terminos=LICENCIA_GOV, cobertura_inicio=ini, cobertura_fin=fin,
        frecuencia=f"por evento (≈{round(n_ptr_total / max(1, len(tabla_anio)))} PTR/año)",
        retraso_publicacion=(f"operación -> publicación (FilingDate + {k} hábiles): mediana {med['lag_dias_mediana']} d, "
                             f"p90 {med['lag_dias_p90']} d (límite legal 45 d hasta el FilingDate); FilingDate -> PDF "
                             f"en la web: mediana {med_k['mediana']} hábiles, p90 {med_k['p90']} (n={med_k['n']})"),
        formato="zip (xml/txt índice) + pdf por PTR (texto si electrónico, imagen si escaneado)",
        volumen_estimado=(f"{n_ptr_total} PTR × {bytes_pdf / 1e3:.0f} KB ≈ {n_ptr_total * bytes_pdf / 1e9:.2f} GB; "
                          f"≈{med['ops_por_ptr']} operaciones/PTR electrónico"),
        limites="sin clave; 1 petición/s; los PTR escaneados (DocID 8/9…) necesitarían OCR (no se hace)",
        estado_sonda=est, mediciones=med, notas=[
            f"fecha_publicacion = FilingDate + {k} días hábiles ({med_k['origen']}); la fecha de operación y el "
            "FilingDate (fecha_presentacion) van como metadatos.",
            "El Clerk sube el PDF a la web después del FilingDate: se mide con el Last-Modified de cada PDF de la "
            f"muestra (descartando > {LAG_WEB_MAX_D} d, regeneraciones).",
            "Tickers de preferentes con '$' (p. ej. COF$I) y activos sin ticker (bonos, fondos) se conservan; "
            "filtrar clase ST sin '$' para la rentabilidad anormal.",
        ]))


def _sondear_senado(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    try:
        permitido: bool | None = robots_permite(SENADO)
    except Exception:  # noqa: BLE001
        permitido = None
    res.cobertura.append(dict(fuente=F_SENADO, familia="divulgaciones", tipo="evento", inicio=None, fin=None, hora=False,
                              idiomas="en", lag_mediana_h=None, destino="ninguno", backtest=False, estado="no_probada",
                              nota="puerta de aceptación de condiciones por sesión; no se automatiza"))
    res.fichas.append(base.ficha(
        fuente=F_SENADO, nombre="Senate eFD (Electronic Financial Disclosures)", agentes=["equity", "risk"], rag="ninguno",
        url=SENADO, acceso="scraping",
        licencia_y_terminos="puerta de aceptación: 'Prohibition Against Use of Financial Disclosure Reports' "
                            "(5 U.S.C. §13107); uso no comercial",
        frecuencia="por evento", retraso_publicacion="≤45 días tras la operación (STOCK Act)",
        formato="html (PTR electrónicos) + pdf/gif escaneados", volumen_estimado="no medido",
        limites="aceptación de condiciones por sesión + protección anti-bot", estado_sonda="no_probada",
        mediciones={"robots_permite": permitido},
        notas=["Exige aceptar condiciones a mano en cada sesión: no se automatiza (regla ética del proyecto).",
               "Alternativa: descarga manual por el usuario de los PTR del periodo, o un agregador con licencia."]))


def _sondear_watcher(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    med: dict[str, Any] = {}
    for k, u in WATCHER.items():
        t0 = base.cronometro()
        try:
            r = get(u, timeout=60, reintentos=1)
            r.raise_for_status()
            med[k] = dict(status=r.status_code, bytes=len(r.content))
            res.registrar(F_WATCHER, k, u, True, bytes=len(r.content), segundos=base.segundos(t0))
        except Exception as e:  # noqa: BLE001
            med[k] = f"{type(e).__name__}: {str(e)[:80]}"
            res.registrar(F_WATCHER, k, u, False, segundos=base.segundos(t0), error=f"{type(e).__name__}: {e}")
    est = base.estado(res.registro, F_WATCHER)
    res.cobertura.append(dict(fuente=F_WATCHER, familia="divulgaciones", tipo="evento", inicio=None, fin=None, hora=False,
                              idiomas="en", lag_mediana_h=None, destino="ninguno", backtest=False, estado=est,
                              nota="; ".join(f"{k}: {v if isinstance(v, str) else v['status']}" for k, v in med.items())))
    res.fichas.append(base.ficha(
        fuente=F_WATCHER, nombre="House/Senate Stock Watcher (datasets abiertos, S3)", agentes=["equity"], rag="ninguno",
        url="https://housestockwatcher.com", acceso="descarga", licencia_y_terminos="proyecto comunitario; sin licencia formal",
        frecuencia="estático", retraso_publicacion="n/a", formato="json", volumen_estimado="no medido",
        limites="buckets S3 públicos", estado_sonda=est, mediciones=med,
        notas=["Si responde 403 el proyecto ya no sirve datos (abandonado): se sustituye por el parseo propio de los PTR."]))


def _ficha_agregadores(res: base.Resultado) -> None:
    res.cobertura.append(dict(fuente=F_AGREG, familia="divulgaciones", tipo="evento", inicio=None, fin=None, hora=False,
                              idiomas="en", lag_mediana_h=None, destino="ninguno", backtest=False, estado="no_probada",
                              nota="de pago (Quiver) o prohíben scraping (Capitol Trades)"))
    res.fichas.append(base.ficha(
        fuente=F_AGREG, nombre="Quiver Quantitative · Capitol Trades · Unusual Whales", agentes=["equity", "risk"],
        rag="ninguno", url="https://www.quiverquant.com/congresstrading/", acceso="api", autenticacion="pago",
        coste="Quiver API desde ≈30 USD/mes; Unusual Whales 50-120 USD/mes; Capitol Trades: web sin API",
        licencia_y_terminos="licencias comerciales; Capitol Trades prohíbe scraping en sus términos",
        frecuencia="diaria", retraso_publicacion="horas tras el PTR (documentado)", formato="json/csv",
        volumen_estimado="no medido", limites="según plan", estado_sonda="no_probada",
        notas=["No se contrata (coste cero). Aportarían Cámara + Senado normalizados con tickers mapeados."]))


def precios_etf(tickers: tuple[str, ...] = ("NANC", "KRUZ")) -> dict[str, pd.DataFrame]:
    """Cierres diarios ajustados vía yfinance (si está instalado): ``{ticker: DataFrame(fecha, cierre)}``."""
    import yfinance as yf

    out = {}
    for tk in tickers:
        h = yf.Ticker(tk).history(period="max", auto_adjust=True)
        out[tk] = pd.DataFrame(dict(fecha=pd.to_datetime(h.index).tz_localize(None).normalize(),
                                    cierre=h["Close"].to_numpy() if len(h) else [])) if len(h) else \
            pd.DataFrame(columns=["fecha", "cierre"])
    return out


def _sondear_etf(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    t0 = base.cronometro()
    med: dict[str, Any] = {}
    try:
        precios = precios_etf()
    except Exception as e:  # noqa: BLE001
        precios = {}
        res.registrar(F_ETF, "yfinance", "yfinance:NANC,KRUZ", False, segundos=base.segundos(t0),
                      error=f"{type(e).__name__}: {e}")
    filas, partes = [], []
    for tk, df in precios.items():
        ok = len(df) > 0
        res.registrar(F_ETF, tk, f"yfinance:{tk}", ok, segundos=base.segundos(t0), error="" if ok else "sin datos")
        ini = df["fecha"].min().date().isoformat() if ok else None
        fin = df["fecha"].max().date().isoformat() if ok else None
        rent = round(100 * (df["cierre"].iloc[-1] / df["cierre"].iloc[0] - 1), 1) if ok else None
        med[tk] = dict(inicio=ini, fin=fin, dias=len(df))
        filas.append(dict(ticker=tk, inicio=ini, fin=fin, dias=len(df), rent_total_pct=rent))
        if ok:
            partes.append(pd.DataFrame(dict(serie=f"{tk.lower()}_cierre", fecha_referencia=df["fecha"],
                                            fecha_publicacion=df["fecha"], valor=df["cierre"], unidad="USD",
                                            fuente=F_ETF, metodo_publicacion="cierre diario (Yahoo vía yfinance)")))
    res.tablas["congreso_etf_resumen"] = pd.DataFrame(filas, columns=["ticker", "inicio", "fin", "dias", "rent_total_pct"])
    if partes:
        res.series.append(base.guardar_serie(pd.concat(partes, ignore_index=True), "congreso_etf_nanc_kruz",
                                             rapido=cfg.modo_rapido).stem)
    med["segundos"] = base.segundos(t0)
    est = base.estado(res.registro, F_ETF)
    nanc = med.get("NANC", {})
    vivos = [tk for tk, v in med.items() if isinstance(v, dict) and v.get("dias")]
    res.cobertura.append(dict(fuente=F_ETF, familia="divulgaciones", tipo="serie", inicio=nanc.get("inicio"),
                              fin=nanc.get("fin"), hora=False, idiomas="en", lag_mediana_h=0.0, destino="serie",
                              backtest=True, estado=est,
                              nota=f"cotizan: {', '.join(vivos) or 'ninguno'}; proxy solo desde 2023-02"))
    res.fichas.append(base.ficha(
        fuente=F_ETF, nombre="ETFs NANC y KRUZ (réplica de carteras del Congreso)", agentes=["equity", "portfolio"],
        rag="serie", url="https://www.subversiveetfs.com", acceso="libreria",
        licencia_y_terminos="precios vía Yahoo Finance (términos de Yahoo, uso personal)",
        cobertura_inicio=nanc.get("inicio"), cobertura_fin=nanc.get("fin"), frecuencia="diaria",
        retraso_publicacion="cierre del mismo día", formato="dataframe (yfinance)", volumen_estimado="KB",
        limites="yfinance no oficial", estado_sonda=est, mediciones=med,
        notas=["Proxy del modo vivo y para validar la señal propia del Congreso en 2023-hoy."]
        + ([] if "KRUZ" in vivos else ["KRUZ sin datos en Yahoo (posible liquidación del ETF): queda solo NANC."])))


def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    """Sonda del Congreso: índices y muestra de PTR de la Cámara + fichas de Senado, agregadores y ETFs."""
    res = base.Resultado(MODULO)
    for fuente, paso in ((F_HOUSE, lambda: _sondear_house(cfg, res)), (F_SENADO, lambda: _sondear_senado(cfg, res)),
                         (F_WATCHER, lambda: _sondear_watcher(cfg, res)), (F_AGREG, lambda: _ficha_agregadores(res)),
                         (F_ETF, lambda: _sondear_etf(cfg, res))):
        t0 = base.cronometro()
        try:
            paso()
        except Exception as e:  # noqa: BLE001
            res.registrar(fuente, "sonda", "", False, error=f"{type(e).__name__}: {e}")
            _fila_fallo(res, fuente, f"{type(e).__name__}: {str(e)[:150]}")
        cfg.log(f"[congreso] {fuente}: {base.segundos(t0)} s")
    return res


def _fila_fallo(res: base.Resultado, fuente: str, motivo: str) -> None:
    """Cobertura y ficha mínimas de una fuente cuya sonda reventó (si aún no las tiene)."""
    if not any(c["fuente"] == fuente for c in res.cobertura):
        res.cobertura.append(dict(fuente=fuente, familia="divulgaciones", tipo="evento", inicio=None, fin=None,
                                  hora=False, idiomas="en", lag_mediana_h=None, destino="ninguno", backtest=False,
                                  estado="fallo", nota=motivo))
    if not any(f.fuente == fuente for f in res.fichas):
        res.fichas.append(base.ficha(fuente=fuente, nombre=fuente, agentes=["equity"], rag="ninguno", url="",
                                     licencia_y_terminos="", frecuencia="", retraso_publicacion="", formato="",
                                     volumen_estimado="", limites="", estado_sonda="fallo", notas=[motivo]))
