"""Form 4 de insiders (SEC EDGAR): compras y ventas de directivos en su propia empresa.

Dos vías gratuitas:

1. **Índices diarios** ``edgar/daily-index/<AAAA>/QTR<n>/form.<AAAAMMDD>.idx`` + el ``.txt`` de cada
   Form 4 (cabecera SGML con ``ACCEPTANCE-DATETIME`` y el XML ``ownershipDocument``, desde 2003-06).
   Es lo que se mide aquí, en los días testigo.
2. **Insider Transactions Data Sets** (DERA): un zip trimestral con todas las transacciones de los
   Forms 3/4/5 ya tabuladas (desde 2006 T1). Solo se documenta su cobertura (no se baja).

User-Agent con contacto (``SEC_EDGAR_USER_AGENT``, lo pone ``ingesta.comun.http``) y ≤ 5 peticiones/s.
TimeGate: ``fecha_publicacion`` sale de la hora de aceptación de EDGAR (``ACCEPTANCE-DATETIME``, ET):
aceptado antes de las 16:00 ET -> ese día; a partir de las 16:00 (tras el cierre) -> siguiente día hábil.
Nunca la fecha de la transacción (hasta 2 días hábiles antes, o meses si se presenta tarde) ni el
``FILED AS OF DATE`` del índice (un Form 4 aceptado a las 17:00 lleva la fecha de ese día y no se pudo
usar al cierre). ``disponible_utc`` guarda el instante de aceptación en UTC.
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ingesta.comun.http import get
from ingesta.noticias_alt import base

MODULO = "insiders"
SEC = "https://www.sec.gov/"
POR_SEGUNDO = 5.0
HILOS = 4                     # peticiones en vuelo (el ritmo lo limita ``get`` por host)
F_FORM4 = "sec_form4_insiders"
F_DERA = "sec_form345_datasets"
URL_DERA = SEC + "data-research/sec-markets-data/insider-transactions-data-sets"
LICENCIA = "dominio público (SEC EDGAR); User-Agent con contacto; ≤10 peticiones/s"
OBJETIVO_P = 60               # Form 4 con compra abierta (P) y ticker, para la rentabilidad anormal
MAX_XML = 400                 # techo de XML por sonda completa
DIAS_EXTRA = 3                # días hábiles tras cada día testigo donde se buscan compras P
CIERRE_ET = 16                # hora de cierre de NY: lo aceptado desde entonces cuenta para el día hábil siguiente
CARGOS = {"isDirector": "director", "isOfficer": "directivo", "isTenPercentOwner": "10%", "isOther": "otro"}


# --------------------------------------------------------------------------- URLs y parseo

def url_indice_diario(d: date) -> str:
    return f"{SEC}Archives/edgar/daily-index/{d.year}/QTR{(d.month - 1) // 3 + 1}/form.{d:%Y%m%d}.idx"


def url_filing(ruta: str) -> str:
    return SEC + "Archives/" + ruta


_LINEA_IDX = re.compile(r"^(\S+)\s{2,}(.+?)\s{2,}(\d+)\s+(\d{8})\s+(edgar/\S+)")


def parse_indice_diario(texto: str, formularios: tuple[str, ...] | None = ("4", "4/A")) -> list[dict[str, Any]]:
    """Líneas de ``form.idx`` (anchos fijos) -> ``[{form, empresa, cik, fecha, ruta}]``.

    ``formularios=None`` devuelve todas las líneas (para contar el total del día).
    """
    filas = []
    for l in texto.splitlines():
        m = _LINEA_IDX.match(l)
        if m and (formularios is None or m.group(1) in formularios):
            filas.append(dict(form=m.group(1), empresa=m.group(2).strip(), cik=m.group(3),
                              fecha=datetime.strptime(m.group(4), "%Y%m%d").date(), ruta=m.group(5)))
    return filas


def parse_cabecera(texto: str) -> dict[str, Any]:
    """Cabecera SGML del ``.txt``: aceptación (con hora, ET), fecha de presentación y nº de acceso."""
    acc = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", texto)
    fil = re.search(r"FILED AS OF DATE:\s*(\d{8})", texto)
    num = re.search(r"ACCESSION NUMBER:\s*(\S+)", texto)
    return dict(aceptacion=datetime.strptime(acc.group(1), "%Y%m%d%H%M%S") if acc else None,
                fecha_presentacion=datetime.strptime(fil.group(1), "%Y%m%d").date() if fil else None,
                accession=num.group(1) if num else None)


def _num(txt: str) -> float | None:
    try:
        return float(txt.replace(",", "")) if txt else None
    except ValueError:
        return None


def parse_form4(texto: str) -> dict[str, Any]:
    """XML ``ownershipDocument`` de un Form 4 -> emisor, ticker, insider, cargo y transacciones.

    Devuelve ``{}`` si no hay XML (Form 4 en papel/HTML anteriores a 2003-06).
    """
    import lxml.etree as ET

    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", texto, re.S)
    if not m:
        return {}
    raiz = ET.fromstring(m.group(0).encode("utf-8"), parser=ET.XMLParser(recover=True, resolve_entities=False))
    v = lambda e, p: (e.findtext(p) or "").strip()  # noqa: E731
    trans = []
    for derivado, etiqueta in ((False, "nonDerivativeTransaction"), (True, "derivativeTransaction")):
        for t in raiz.iter(etiqueta):
            fecha = v(t, "transactionDate/value")[:10]
            trans.append(dict(derivado=derivado, valor=v(t, "securityTitle/value"),
                              fecha=date.fromisoformat(fecha) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha) else None,
                              codigo=v(t, "transactionCoding/transactionCode") or None,
                              acciones=_num(v(t, "transactionAmounts/transactionShares/value")),
                              precio=_num(v(t, "transactionAmounts/transactionPricePerShare/value")),
                              adq_disp=v(t, "transactionAmounts/transactionAcquiredDisposedCode/value") or None))
    duenos = raiz.findall("reportingOwner")
    rel = duenos[0].find("reportingOwnerRelationship") if duenos else None
    cargo = [n for k, n in CARGOS.items() if rel is not None and v(rel, k).lower() in ("1", "true")]
    titulo = v(rel, "officerTitle") if rel is not None else ""
    if titulo:
        cargo = [f"directivo: {titulo}" if c == "directivo" else c for c in cargo]
    return dict(ticker=v(raiz, "issuer/issuerTradingSymbol").upper() or None, emisor=v(raiz, "issuer/issuerName"),
                cik_emisor=v(raiz, "issuer/issuerCik"),
                insider=v(duenos[0], "reportingOwnerId/rptOwnerName") if duenos else "",
                n_insiders=len(duenos), cargo="; ".join(cargo), periodo=v(raiz, "periodOfReport"), transacciones=trans)


def _habil_siguiente(d: date) -> date:
    return (pd.Timestamp(d) + pd.offsets.BDay(1)).date()


def publicacion(aceptacion: datetime | None, fecha_indice: date) -> tuple[date, str | None]:
    """Fecha TimeGate y ``disponible_utc`` de un Form 4 a partir de su aceptación EDGAR (hora ET, naive).

    Antes de las 16:00 ET -> el día de aceptación (o el hábil siguiente si fue en fin de semana); desde las
    16:00 -> el siguiente día hábil. Sin hora de aceptación, prudente: día hábil siguiente al del índice.
    """
    if aceptacion is None:
        return _habil_siguiente(fecha_indice), None
    et = pd.Timestamp(aceptacion).tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
    d = aceptacion.date()
    if aceptacion.hour >= CIERRE_ET or d.weekday() >= 5:
        d = _habil_siguiente(d) if d.weekday() < 5 else pd.offsets.BDay(1).rollforward(pd.Timestamp(d)).date()
    return d, et.tz_convert("UTC").isoformat()


# --------------------------------------------------------------------------- acceso directo (tools)

def listar_form4(d: date) -> list[dict[str, Any]]:
    """Form 4 y 4/A presentados el día ``d`` (sin caché ni registro: uso de las tools)."""
    return parse_indice_diario(get(url_indice_diario(d), por_segundo=POR_SEGUNDO, timeout=120).text)


def leer_form4(ruta: str) -> dict[str, Any]:
    """Form 4 de ``ruta`` (``edgar/data/...txt``) con cabecera y transacciones."""
    texto = get(url_filing(ruta), por_segundo=POR_SEGUNDO, timeout=60).text
    return {**parse_cabecera(texto), **parse_form4(texto)}


# --------------------------------------------------------------------------- descarga con caché

def _bajar(res: base.Resultado, cfg: base.ConfigSonda, fuente: str, item: str, url: str, ruta: Path | None,
           *, dia: str = "", registrar_fallo: bool = True, **kw: Any) -> bytes | None:
    """Bytes de ``url`` (o de la caché ``ruta``, escrita de forma atómica); registra la petición. None si falla."""
    kw.setdefault("por_segundo", POR_SEGUNDO)
    return base.get_cacheado(res, cfg, fuente, item, url, ruta, getter=get, dia=dia,
                             registrar_fallo=registrar_fallo, **kw)


def _dias_habiles_siguientes(d: date, n: int) -> list[date]:
    out, x = [], d
    while len(out) < n:
        x += timedelta(days=1)
        if x.weekday() < 5:
            out.append(x)
    return out


# --------------------------------------------------------------------------- sonda Form 4

class _Muestreo:
    """Estado compartido de la sonda: índices leídos, colas de filings y filas de salida."""

    def __init__(self, cfg: base.ConfigSonda, res: base.Resultado):
        self.cfg, self.res = cfg, res
        self.crudo = base.corpus("crudo", MODULO)
        (self.crudo / "form4").mkdir(exist_ok=True)
        self.colas: dict[date, list[dict]] = {}
        self.clave: dict[date, str] = {}
        self.filas_idx: list[dict] = []
        self.filings: list[dict] = []
        self.ops: list[dict] = []

    def indice(self, d: date, clave: str, *, buscar_atras: int = 0, opcional: bool = False) -> date | None:
        """Lee el índice de ``d`` (o del último día hábil anterior con índice, hasta ``buscar_atras``).

        ``opcional``: día de búsqueda de compras P; si no hay índice (festivo) no es un fallo.
        """
        candidatos = [d] + [d - timedelta(days=k) for k in range(1, buscar_atras + 1)]
        for x in candidatos:
            if x.weekday() >= 5 or x > date.today():
                continue
            ultimo = (x == candidatos[-1] or buscar_atras == 0) and not opcional
            datos = _bajar(self.res, self.cfg, F_FORM4, f"indice {x}", url_indice_diario(x),
                           self.crudo / f"form.{x:%Y%m%d}.idx", dia=x.isoformat(), timeout=120,
                           reintentos=3 if buscar_atras == 0 and not opcional else 1, registrar_fallo=ultimo)
            if datos is None:
                continue
            texto = datos.decode("latin-1")
            todas = parse_indice_diario(texto, None)
            f4 = [f for f in todas if f["form"] == "4"]
            self.filas_idx.append(dict(dia_testigo=clave, dia=x.isoformat(), n_entradas=len(todas), n_form4=len(f4),
                                       n_form4a=sum(f["form"] == "4/A" for f in todas), bytes=len(datos),
                                       nota="" if x == d else f"sin índice el {d}: se usa el {x}"))
            random.Random(x.toordinal()).shuffle(f4)
            self.colas[x], self.clave[x] = f4, clave
            return x
        self.filas_idx.append(dict(dia_testigo=clave, dia=d.isoformat(), n_entradas=0, n_form4=0, n_form4a=0, bytes=0,
                                   nota="sin cobertura (sin índice diario: ¿festivo?)"))
        return None

    def _uno(self, d: date, f: dict, muestra: str) -> None:
        url = url_filing(f["ruta"])
        nombre = f["ruta"].rsplit("/", 1)[-1]
        datos = _bajar(self.res, self.cfg, F_FORM4, f"form4 {nombre}", url, self.crudo / "form4" / nombre,
                       dia=d.isoformat(), timeout=60)
        if datos is None:
            return
        texto = datos.decode("utf-8", errors="replace")
        try:
            cab, doc = parse_cabecera(texto), parse_form4(texto)
        except Exception as e:  # noqa: BLE001
            self.res.registrar(F_FORM4, f"parseo {nombre}", url, False, dia=d.isoformat(), error=repr(e))
            return
        fp, disp = publicacion(cab["aceptacion"], f["fecha"])  # TimeGate: hora de aceptación, no FILED AS OF
        trans = doc.get("transacciones", [])
        self.filings.append(dict(dia_testigo=self.clave[d], dia=d.isoformat(), accession=cab["accession"],
                                 emisor=doc.get("emisor") or f["empresa"], ticker=doc.get("ticker"),
                                 insider=doc.get("insider"), cargo=doc.get("cargo"), xml=bool(doc),
                                 n_transacciones=len(trans),
                                 codigos=",".join(sorted({t["codigo"] for t in trans if t["codigo"]})),
                                 compra_p=any(t["codigo"] == "P" for t in trans if not t["derivado"]),
                                 fecha_presentacion=f["fecha"], fecha_publicacion=fp, disponible_utc=disp,
                                 aceptacion=cab["aceptacion"], bytes=len(datos), muestra=muestra))
        for t in trans:
            acciones, precio = t["acciones"], t["precio"]
            self.ops.append(dict(dia_testigo=self.clave[d], dia=d.isoformat(), accession=cab["accession"],
                                 emisor=doc.get("emisor"), ticker=doc.get("ticker"), insider=doc.get("insider"),
                                 cargo=doc.get("cargo"), codigo=t["codigo"], derivado=t["derivado"],
                                 fecha_transaccion=t["fecha"], fecha_presentacion=f["fecha"], fecha_publicacion=fp,
                                 disponible_utc=disp, aceptacion=cab["aceptacion"],
                                 lag_dias=(fp - t["fecha"]).days if t["fecha"] else None, acciones=acciones,
                                 precio=precio,
                                 valor_usd=round(acciones * precio, 2) if acciones is not None and precio else None,
                                 adq_disp=t["adq_disp"], muestra=muestra))

    def tanda(self, pedidos: list[tuple[date, dict, str]]) -> None:
        if not pedidos:
            return
        with ThreadPoolExecutor(max_workers=HILOS) as ex:
            list(ex.map(lambda p: self._uno(*p), pedidos))

    def compras_p(self) -> int:
        return sum(1 for f in self.filings if f["compra_p"] and f["ticker"])


def _sondear_form4(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    m = _Muestreo(cfg, res)
    n_dia = cfg.max_items or (5 if cfg.modo_rapido else 25)
    dias_ok: list[date] = []
    for dt in cfg.dias:
        x = m.indice(dt.fecha, dt.clave, buscar_atras=7 if dt.clave == "hoy" else 0)
        if x is not None:
            dias_ok.append(x)
    # (1) Muestra base: n_dia Form 4 por día testigo.
    m.tanda([(d, m.colas[d].pop(), "testigo") for d in dias_ok for _ in range(min(n_dia, len(m.colas[d])))])
    cfg.log(f"  [insiders] muestra base: {len(m.filings)} Form 4, {m.compras_p()} con compra P")
    # (2) Búsqueda de compras abiertas (P) en días hábiles posteriores a cada día testigo histórico.
    if not cfg.modo_rapido and m.compras_p() < OBJETIVO_P:
        for dt in cfg.dias:
            if dt.clave == "hoy":
                continue
            for x in _dias_habiles_siguientes(dt.fecha, DIAS_EXTRA):
                m.indice(x, dt.clave, opcional=True)
        pool = [d for d in m.colas if m.clave[d] != "hoy"]
        while m.compras_p() < OBJETIVO_P and len(m.filings) < MAX_XML and any(m.colas[d] for d in pool):
            pedidos = []
            for d in pool:
                for _ in range(min(4, len(m.colas[d]))):
                    if len(m.filings) + len(pedidos) < MAX_XML:
                        pedidos.append((d, m.colas[d].pop(), "busqueda_p"))
            m.tanda(pedidos)
            cfg.log(f"  [insiders] búsqueda P: {len(m.filings)} Form 4, {m.compras_p()} con compra P")

    cols_idx = ["dia_testigo", "dia", "n_entradas", "n_form4", "n_form4a", "bytes", "nota"]
    cols_ops = ["dia_testigo", "dia", "accession", "emisor", "ticker", "insider", "cargo", "codigo", "derivado",
                "fecha_transaccion", "fecha_presentacion", "fecha_publicacion", "disponible_utc", "aceptacion", "lag_dias", "acciones", "precio", "valor_usd",
                "adq_disp", "muestra"]
    cols_fil = ["dia_testigo", "dia", "accession", "emisor", "ticker", "insider", "cargo", "xml", "n_transacciones",
                "codigos", "compra_p", "fecha_presentacion", "fecha_publicacion", "disponible_utc", "aceptacion", "bytes",
                "muestra"]
    t_idx = pd.DataFrame(m.filas_idx, columns=cols_idx).sort_values("dia").reset_index(drop=True)
    t_ops = pd.DataFrame(m.ops, columns=cols_ops).sort_values(["dia", "accession"]).reset_index(drop=True)
    t_fil = pd.DataFrame(m.filings, columns=cols_fil).sort_values(["dia", "accession"]).reset_index(drop=True)
    res.tablas["insiders_indice"] = t_idx
    res.tablas["insiders_operaciones"] = t_ops
    res.tablas["insiders_form4"] = t_fil

    # TimeGate: transacción (hecho) frente a publicación (aceptación EDGAR con corte a las 16:00 ET).
    base_tg = t_ops[t_ops["codigo"].isin(["P", "S"]) & t_ops["fecha_transaccion"].notna()]
    for _, o in base_tg.groupby("dia_testigo", sort=False).head(2).iterrows():
        res.timegate.append(dict(fuente=F_FORM4, item=f"{o['accession']} {o['ticker'] or ''} {o['codigo']}".strip(),
                                 fecha_hecho=o["fecha_transaccion"].isoformat(),
                                 fecha_publicacion=o["fecha_publicacion"].isoformat(),
                                 campo_fecha="ACCEPTANCE-DATETIME (ET): <16:00 -> ese día; si no, día hábil siguiente",
                                 lag_h=base.lag_horas(o["fecha_transaccion"], o["fecha_publicacion"])))

    # Medidas, cobertura y ficha.
    lags = base.resumen_lag((t_ops["lag_dias"].dropna().astype(float) * 24).tolist() if not t_ops.empty else [])
    acept = pd.to_datetime(t_fil["aceptacion"].dropna()) if not t_fil.empty else pd.Series(dtype="datetime64[ns]")
    hora_med = round(float((acept.dt.hour + acept.dt.minute / 60).median()), 2) if len(acept) else None
    tras_cierre = round(float((acept.dt.hour >= CIERRE_ET).mean()), 3) if len(acept) else None
    con_xml = t_fil[t_fil["xml"]] if not t_fil.empty else t_fil
    dias_xml = sorted(con_xml["dia"].unique()) if not con_xml.empty else []
    form4_dia = float(t_idx.loc[t_idx["n_form4"] > 0, "n_form4"].mean()) if (t_idx["n_form4"] > 0).any() else 0.0
    bytes_med = float(t_fil["bytes"].mean()) if not t_fil.empty else 0.0
    anios = date.today().year - 2003 + 0.5
    codigos = t_ops["codigo"].value_counts().to_dict() if not t_ops.empty else {}
    est = base.estado(res.registro, F_FORM4)
    med = dict(indices=int((t_idx["n_form4"] > 0).sum()), form4_por_dia_medio=round(form4_dia),
               form4_leidos=int(len(t_fil)), con_xml=int(len(con_xml)), transacciones=int(len(t_ops)),
               codigos={str(k): int(v) for k, v in codigos.items()},
               form4_con_compra_p_y_ticker=m.compras_p(),
               lag_dias_mediana=round(lags["mediana_h"] / 24, 1) if lags["mediana_h"] is not None else None,
               lag_dias_p90=round(lags["p90_h"] / 24, 1) if lags["p90_h"] is not None else None,
               hora_aceptacion_mediana_et=hora_med, pct_aceptados_tras_cierre=tras_cierre,
               bytes_form4_medio=round(bytes_med))
    ini, fin = (dias_xml[0], dias_xml[-1]) if dias_xml else (None, None)
    nota = (f"{med['form4_por_dia_medio']} Form 4/día de media; {len(t_fil)} leídos ({len(con_xml)} con XML), "
            f"{m.compras_p()} con compra P; lag transacción->presentación mediana {med['lag_dias_mediana']} d; "
            f"aceptación mediana {hora_med} h ET ({tras_cierre} tras el cierre -> D+1 hábil)")
    res.cobertura.append(dict(fuente=F_FORM4, familia="divulgaciones", tipo="evento", inicio=ini, fin=fin, hora=True,
                              idiomas="en", lag_mediana_h=lags["mediana_h"], destino="serie", backtest=True,
                              estado=est, nota=nota))
    res.fichas.append(base.ficha(
        fuente=F_FORM4, nombre="SEC Form 4 · operaciones de insiders (EDGAR)", agentes=["equity", "risk"], rag="serie",
        url=SEC + "cgi-bin/browse-edgar?action=getcurrent&type=4", acceso="descarga", licencia_y_terminos=LICENCIA,
        cobertura_inicio=ini, cobertura_fin=fin, frecuencia=f"diaria (≈{med['form4_por_dia_medio']} Form 4/día medido)",
        retraso_publicacion=(f"transacción -> publicación (aceptación, corte 16:00 ET): mediana "
                             f"{med['lag_dias_mediana']} d, p90 {med['lag_dias_p90']} d (límite legal de presentación "
                             f"2 días hábiles desde 2002)"),
        formato="idx (anchos fijos) + txt SGML con XML ownershipDocument",
        volumen_estimado=(f"≈{form4_dia * 252 * anios / 1e6:.1f} M Form 4 desde 2003-06 × {bytes_med / 1e3:.0f} KB ≈ "
                          f"{form4_dia * 252 * anios * bytes_med / 1e9:.0f} GB (mejor el dataset trimestral DERA)"),
        limites="≤10 peticiones/s con User-Agent con contacto (aquí 5/s); sin clave",
        estado_sonda=est, mediciones=med, notas=[
            "fecha_publicacion = día de aceptación EDGAR si fue antes de las 16:00 ET; si no, el día hábil "
            "siguiente (la fecha del índice / FILED AS OF se guarda como fecha_presentacion). La fecha de "
            "transacción va como metadato.",
            "El XML (ownershipDocument) existe desde 2003-06-30; antes, Form 4 en texto libre.",
            "Señal típica: compras abiertas (código P), sobre todo varias de un mismo emisor (cluster buying); "
            "las ventas S son ruidosas (planes 10b5-1).",
            f"Muestra: {n_dia} por día testigo y, en la sonda completa, búsqueda de compras P en "
            f"{DIAS_EXTRA} días hábiles tras cada día testigo (techo {MAX_XML} XML).",
        ]))


def _sondear_dera(cfg: base.ConfigSonda, res: base.Resultado) -> None:
    """Datasets trimestrales Form 3/4/5: lista los zip publicados (no se descargan)."""
    datos = _bajar(res, cfg, F_DERA, "listado", URL_DERA, None, timeout=60, reintentos=2)
    zips = sorted(set(re.findall(r"(\d{4})q([1-4])_form345\.zip", datos.decode("utf-8", "replace")))) if datos else []
    ini = date(int(zips[0][0]), 3 * int(zips[0][1]) - 2, 1).isoformat() if zips else None
    if zips:
        a, q = int(zips[-1][0]), int(zips[-1][1])
        fin = (date(a + (q == 4), (3 * q) % 12 + 1, 1) - timedelta(days=1)).isoformat()
    else:
        fin = None
    med = dict(trimestres=len(zips), primero="q".join(zips[0]) if zips else None,
               ultimo="q".join(zips[-1]) if zips else None)
    est = base.estado(res.registro, F_DERA)
    res.cobertura.append(dict(fuente=F_DERA, familia="divulgaciones", tipo="evento", inicio=ini, fin=fin, hora=False,
                              idiomas="en", lag_mediana_h=None, destino="serie", backtest=True, estado=est,
                              nota=f"{len(zips)} zips trimestrales listados ({med['primero']}–{med['ultimo']}); no se bajan"))
    res.fichas.append(base.ficha(
        fuente=F_DERA, nombre="SEC Insider Transactions Data Sets (Forms 3/4/5 trimestrales)", agentes=["equity"],
        rag="serie", url=URL_DERA, acceso="descarga", licencia_y_terminos=LICENCIA, cobertura_inicio=ini,
        cobertura_fin=fin, frecuencia="trimestral",
        retraso_publicacion="dataset: tras el cierre del trimestre (documentado); por fila, FILING_DATE ≤2 días",
        formato="zip de TSV (SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, DERIV_TRANS…)",
        volumen_estimado=f"≈10-25 MB/trimestre (documentado) × {len(zips)} trimestres",
        limites="≤10 peticiones/s; descarga lenta desde IP compartida (fase2/datos midió ≈14 MB en ≈3 min)",
        estado_sonda=est, mediciones=med, notas=[
            "Vía recomendada para el backtest completo: FILING_DATE por fila (no la fecha de publicación del zip).",
            "Solo se lista la página de descargas: los zip no se bajan en la sonda.",
        ]))


def sondear(cfg: base.ConfigSonda) -> base.Resultado:
    """Sonda de insiders: índices diarios y Form 4 de los días testigo + ficha de los datasets DERA."""
    res = base.Resultado(MODULO)
    for fuente, paso in ((F_FORM4, _sondear_form4), (F_DERA, _sondear_dera)):
        t0 = base.cronometro()
        try:
            paso(cfg, res)
        except Exception as e:  # noqa: BLE001
            res.registrar(fuente, "sonda", "", False, error=f"{type(e).__name__}: {e}")
            if not any(c["fuente"] == fuente for c in res.cobertura):
                res.cobertura.append(dict(fuente=fuente, familia="divulgaciones", tipo="evento", inicio=None, fin=None,
                                          hora=False, idiomas="en", lag_mediana_h=None, destino="ninguno",
                                          backtest=False, estado="fallo", nota=f"{type(e).__name__}: {str(e)[:150]}"))
        cfg.log(f"[insiders] {fuente}: {base.segundos(t0)} s")
    return res
