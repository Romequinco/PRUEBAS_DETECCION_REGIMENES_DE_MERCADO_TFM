"""Sonda del Tema 3: prueba pequeña de punta a punta sobre 10 empresas.

Responde, con datos medidos, a lo que el equipo necesita para decidir universo y ventana:
qué hay, desde cuándo, con qué retraso se publica, cuánto ocupa y qué falla.

Pasos (cada uno se puede ejecutar por separado desde el notebook):

1. ``paso1_filings``       lista de todos los filings de cada empresa (fecha prudente).
   ``verificar_timegate``  hora oficial de los filings seleccionados -> ``fecha_publicacion`` definitiva.
2. ``medir_cobertura``     desde cuándo hay cada tipo de informe y con qué retraso.
3. ``contrastar_timegate`` desfase de la hora de la API frente a la oficial.
4. ``paso2_documentos``    descarga 10-K, 10-Q y 8-K de resultados; secciones -> jsonl.
5. ``paso3_fundamentales`` XBRL ``companyfacts`` -> parquet point-in-time.
6. ``paso4_sentimiento``   Loughran-McDonald por sección + ``risk_delta`` del Item 1A.
7. ``fichas_tema3``        fichas de fuente con lo medido (``results/ingesta/empresas/fichas``).

Corpus pesado (crudo, jsonl, parquet) en ``<INGESTA_CORPUS_DIR>/empresas``; tablas pequeñas de
medición en ``results/ingesta/empresas``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.comun.rutas import corpus_dir, resultados_dir
from ingesta.empresas import sec_edgar, secciones, sentimiento, xbrl
from ingesta.empresas.universo import MUESTRA_SONDA, Empresa

TEMA = "empresas"


@dataclass
class ConfigSonda:
    empresas: tuple[Empresa, ...] = MUESTRA_SONDA
    familias: tuple[str, ...] = ("10-K", "10-Q")
    resultados_8k: bool = True
    max_por_familia: int | None = None   # por empresa y familia (None = todos); útil para una prueba rápida
    desde: date | None = None
    usar_cache: bool = True
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def carpeta_documentos(self) -> str:
        """Subcarpeta del corpus: una prueba rápida no pisa el jsonl de una ejecución completa."""
        return "documentos" if self.max_por_familia is None else f"documentos_muestra{self.max_por_familia}"


def _log(cfg: ConfigSonda, msg: str) -> None:
    if cfg.verbose:
        print(msg, flush=True)


# --------------------------------------------------------------------------- 1. filings

def paso1_filings(cfg: ConfigSonda) -> pd.DataFrame:
    """Todos los filings (todas las formas) de las empresas de la muestra, con ``ticker``."""
    partes = []
    for e in cfg.empresas:
        t0 = time.monotonic()
        try:
            df = sec_edgar.tabla_filings(e.cik, usar_cache=cfg.usar_cache)
        except Exception as ex:  # noqa: BLE001 — una empresa que falla no para la sonda
            _log(cfg, f"  {e.ticker:6s} ERROR {type(ex).__name__}: {str(ex)[:150]}")
            continue
        df.insert(0, "ticker", e.ticker)
        df.insert(1, "activa", e.activa)
        partes.append(df)
        _log(cfg, f"  {e.ticker:6s} {len(df):5d} filings ({time.monotonic() - t0:.1f} s)")
    if not partes:
        raise RuntimeError("No se pudo leer la lista de filings de ninguna empresa (¿red o User-Agent?).")
    return pd.concat(partes, ignore_index=True)


def verificar_timegate(filings: pd.DataFrame, cfg: ConfigSonda) -> pd.DataFrame:
    """Lee la hora oficial de los filings que se van a descargar y les fija la ``fecha_publicacion``."""
    sel = seleccion(filings, cfg)
    t0 = time.monotonic()
    verificados = sec_edgar.verificar_aceptacion(filings.loc[filings["accession"].isin(sel["accession"])],
                                                 usar_cache=cfg.usar_cache)
    out = filings.copy()
    out = out.set_index("accession")
    out.update(verificados.set_index("accession")[["aceptacion", "aceptacion_verificada", "fecha_publicacion"]])
    out = out.reset_index()[filings.columns]
    out["aceptacion_verificada"] = out["aceptacion_verificada"].astype(bool)
    _log(cfg, f"  {int(verificados['aceptacion_verificada'].sum())} de {len(verificados)} filings con hora oficial "
              f"({time.monotonic() - t0:.1f} s)")
    return out


def medir_cobertura(filings: pd.DataFrame) -> pd.DataFrame:
    """Por empresa y familia: nº de informes, primero/último, retraso de publicación y % tras el cierre.

    ``pct_tras_cierre`` se calcula solo sobre los filings con hora oficial verificada.
    """
    df = filings.loc[filings["familia"].notna() & ~filings["enmienda"]].copy()
    df.loc[df["familia"] == "8-K", "familia"] = df["items"].map(lambda s: "8-K resultados" if sec_edgar.es_resultados(s) else "8-K otros")
    df["tras_cierre"] = (df["fecha_publicacion"] != df["filing_date"]).where(df["aceptacion_verificada"])
    g = df.groupby(["ticker", "familia"])
    out = g.agg(
        n=("accession", "size"),
        primero=("filing_date", "min"),
        ultimo=("filing_date", "max"),
        retraso_mediana_d=("retraso_dias", "median"),
        retraso_p10_d=("retraso_dias", lambda s: s.quantile(0.10)),
        retraso_p90_d=("retraso_dias", lambda s: s.quantile(0.90)),
        n_hora_oficial=("aceptacion_verificada", "sum"),
        pct_tras_cierre=("tras_cierre", lambda s: s.dropna().astype(float).mean() if s.notna().any() else None),
        pct_html=("documento", lambda s: s.str.lower().str.endswith((".htm", ".html")).mean()),
        pct_xbrl=("is_xbrl", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(bool).mean()),
    ).reset_index()
    return out


def contrastar_timegate(filings: pd.DataFrame) -> pd.DataFrame:
    """Desfase (horas) de la hora de la API frente a la oficial, en los filings verificados.

    No hace peticiones: usa lo leído por ``verificar_timegate``. Sirve para documentar el problema
    de la API y para saber si algún día se podría corregir sin leer las cabeceras.
    """
    v = filings.loc[filings["aceptacion_verificada"] & filings["aceptacion_api"].notna()].copy()
    v["desfase_h"] = [(a - o).total_seconds() / 3600 for a, o in zip(v["aceptacion_api"], v["aceptacion"])]
    v["mes"] = [o.month for o in v["aceptacion"]]
    v["hora_oficial"] = [o.strftime("%H:%M") for o in v["aceptacion"]]
    v["coinciden"] = v["desfase_h"] == 0
    return v[["ticker", "form", "accession", "filing_date", "aceptacion_api", "aceptacion", "hora_oficial",
              "desfase_h", "mes", "coinciden", "fecha_publicacion"]].reset_index(drop=True)


# --------------------------------------------------------------------------- 2. documentos

def seleccion(filings: pd.DataFrame, cfg: ConfigSonda) -> pd.DataFrame:
    sel = sec_edgar.filtrar(filings, familias=cfg.familias, resultados_8k=cfg.resultados_8k, desde=cfg.desde)
    if cfg.max_por_familia:
        sel = (sel.sort_values("filing_date", ascending=False)
               .groupby(["ticker", "familia"]).head(cfg.max_por_familia)
               .sort_values(["ticker", "filing_date"]).reset_index(drop=True))
    return sel


def _procesar_filing(f: pd.Series, cfg: ConfigSonda) -> tuple[list[dict], dict]:
    t0 = time.monotonic()
    reg = {"ticker": f["ticker"], "cik": f["cik"], "accession": f["accession"], "form": f["form"],
           "familia": f["familia"], "filing_date": f["filing_date"], "fecha_publicacion": f["fecha_publicacion"],
           "formato": f["url_documento"].rsplit(".", 1)[-1].lower(), "bytes": 0, "bytes_gz": 0,
           "secciones": "", "n_secciones": 0, "por_referencia": "", "ok": False, "error": None}
    try:
        if f["familia"] == "8-K":
            url, ficheros = sec_edgar.exhibit_resultados(f, usar_cache=cfg.usar_cache)
            if url is None:
                # Sin anexo 99 la nota de resultados no está separada: se registra (con los ficheros, para
                # poder revisarlo) y no se usa la portada del 8-K.
                reg["error"] = "8-K sin anexo 99 · ficheros: " + "; ".join(ficheros[:6])
                reg["segundos"] = round(time.monotonic() - t0, 2)
                return [], reg
            reg["formato"] = url.rsplit(".", 1)[-1].lower()
            contenido, ruta = sec_edgar.descargar_documento(f, url=url, usar_cache=cfg.usar_cache)
            texto = secciones.html_a_texto(contenido)
            docs = [secciones.documento_resultados(f, texto, url, f["ticker"])] if len(texto) > 200 else []
            reg["secciones"] = "Exhibit 99"
        else:
            contenido, ruta = sec_edgar.descargar_documento(f, usar_cache=cfg.usar_cache)
            texto = secciones.html_a_texto(contenido)
            secs = secciones.extraer_secciones(texto, f["familia"])
            docs = secciones.documentos_de_filing(f, secs, f["ticker"])
            reg["secciones"] = ",".join(k for k, v in secs.items() if not secciones.es_por_referencia(v))
            reg["por_referencia"] = ",".join(k for k, v in secs.items() if secciones.es_por_referencia(v))
        reg.update(bytes=len(contenido), bytes_gz=ruta.stat().st_size, n_secciones=len(docs), ok=bool(docs))
        if not docs and f["familia"] != "8-K" and len(texto) > 2000:
            # Sin Items reconocibles (p. ej. el 10-K de GE): el filing entero va al cajón para no perderlo.
            docs = [secciones.documento_completo(f, texto, f["ticker"])]
            reg.update(secciones="completo", n_secciones=1,
                       error="sin Items reconocibles: guardado entero en el cajón")
        elif not docs:
            reg["error"] = "sin secciones reconocibles"
    except Exception as e:  # noqa: BLE001 — una fuente rota no para la sonda
        docs, reg["error"] = [], f"{type(e).__name__}: {str(e)[:200]}"
    reg["segundos"] = round(time.monotonic() - t0, 2)
    return docs, reg


def paso2_documentos(filings: pd.DataFrame, cfg: ConfigSonda) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga y trocea los filings seleccionados.

    Escribe ``<corpus>/empresas/documentos/<cik>.jsonl`` (una sección o nota por línea) y devuelve
    ``(registro, documentos)``: una fila por filing (qué se bajó, cuánto ocupa, qué secciones salieron)
    y una fila por documento (metadatos, sin el texto).
    """
    sel = seleccion(filings, cfg)
    sel = sec_edgar.verificar_aceptacion(sel, usar_cache=cfg.usar_cache)   # cacheado: gratis si ya se verificó
    _log(cfg, f"  {len(sel)} filings seleccionados")
    carpeta = corpus_dir(TEMA) / cfg.carpeta_documentos
    carpeta.mkdir(parents=True, exist_ok=True)
    registros, metas = [], []
    for e in cfg.empresas:
        cik = e.cik
        grupo = sel.loc[sel["cik"] == cik]
        docs_empresa = []
        for i, (_, f) in enumerate(grupo.iterrows(), 1):
            docs, reg = _procesar_filing(f, cfg)
            docs_empresa.extend(docs)
            registros.append(reg)
            if cfg.verbose and (i % 25 == 0 or i == len(grupo)):
                ok = sum(r["ok"] for r in registros if r["cik"] == cik)
                print(f"  {f['ticker']:6s} {i}/{len(grupo)} filings · {ok} con secciones", flush=True)
        # Se reescribe siempre (también vacío) para no dejar documentos de una ejecución anterior.
        tmp = carpeta / f"{cik}.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            for d in docs_empresa:
                fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, carpeta / f"{cik}.jsonl")
        metas.extend({k: v for k, v in d.items() if k != "texto"} for d in docs_empresa)
    return pd.DataFrame(registros), pd.DataFrame(metas)


def leer_documentos(cfg: ConfigSonda | None = None, ciks: list[int] | None = None) -> list[dict]:
    """Lee los jsonl del corpus (con texto) de la ejecución descrita por ``cfg`` (por defecto, la completa)."""
    carpeta = corpus_dir(TEMA) / (cfg.carpeta_documentos if cfg else "documentos")
    rutas = [carpeta / f"{c}.jsonl" for c in ciks] if ciks else sorted(carpeta.glob("*.jsonl"))
    docs = []
    for r in rutas:
        if r.exists():
            docs.extend(json.loads(ln) for ln in r.read_text(encoding="utf-8").splitlines() if ln.strip())
    return docs


def tasa_extraccion(registro: pd.DataFrame) -> pd.DataFrame:
    """% de 10-K / 10-Q en los que se encontró cada sección clave, por década de presentación."""
    if registro.empty:
        return pd.DataFrame()
    df = registro.loc[registro["familia"].isin(["10-K", "10-Q"])].copy()
    if df.empty:
        return pd.DataFrame()
    df["decada"] = pd.to_datetime(df["filing_date"]).dt.year // 10 * 10
    claves = {"10-K": secciones.SECCIONES_BASE_10K, "10-Q": secciones.SECCIONES_BASE_10Q}
    filas = []
    for (fam, dec), g in df.groupby(["familia", "decada"]):
        fila = {"familia": fam, "decada": f"{dec}s", "n_filings": len(g), "pct_error": (~g["ok"]).mean()}
        for k in claves[fam]:
            fila[k] = g["secciones"].str.split(",").map(lambda s, k=k: k in s).mean()
            fila[f"{k}_por_ref"] = g["por_referencia"].str.split(",").map(lambda s, k=k: k in s).mean()
        filas.append(fila)
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- 3. fundamentales

def paso3_fundamentales(cfg: ConfigSonda, filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """``companyfacts`` de cada empresa -> ``<corpus>/empresas/fundamentales/<cik>.parquet``."""
    carpeta = corpus_dir(TEMA) / "fundamentales"
    carpeta.mkdir(parents=True, exist_ok=True)
    partes = []
    for e in cfg.empresas:
        try:
            cf = xbrl.descargar_companyfacts(e.cik, usar_cache=cfg.usar_cache)
        except Exception as ex:  # noqa: BLE001 — p. ej. 404: empresas sin XBRL (anteriores a 2009)
            _log(cfg, f"  {e.ticker:6s} sin companyfacts ({type(ex).__name__})")
            continue
        sub = None if filings is None else filings.loc[filings["cik"] == e.cik]
        t = xbrl.companyfacts_a_tabla(cf, e.ticker, sub)
        t.to_parquet(carpeta / f"{e.cik}.parquet", index=False)
        partes.append(t)
        _log(cfg, f"  {e.ticker:6s} {len(t):7d} cifras · {t['concepto'].nunique()} conceptos")
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=xbrl.COLUMNAS)


# --------------------------------------------------------------------------- 4. sentimiento

def paso4_sentimiento(docs: list[dict], lm: dict | None) -> pd.DataFrame:
    """Puntuación LM de cada documento y ``risk_delta`` del Item 1A de cada 10-K frente al anterior."""
    filas = []
    for d in docs:
        fila = {"ticker": d["tickers"][0] if d["tickers"] else None, "tipo": d["tipo_documento"],
                "fecha_publicacion": d["fecha_publicacion"], "seccion": d["seccion"], "doc_id": d["doc_id"]}
        if lm:
            fila.update(sentimiento.puntuar(d["texto"], lm))
        filas.append(fila)
    out = pd.DataFrame(filas)
    # risk_delta: Item 1A de 10-K, ordenado por fecha dentro de cada empresa.
    riesgos = sorted((d for d in docs if d["seccion"] == "Item 1A" and d["tipo_documento"].startswith("10-K")
                      and d.get("version") is None), key=lambda d: (d["tickers"][0], d["fecha_publicacion"]))
    deltas, previo = {}, {}
    for d in riesgos:
        tk = d["tickers"][0]
        deltas[d["doc_id"]] = sentimiento.risk_delta(d["texto"], previo.get(tk))
        previo[tk] = d["texto"]
    if not out.empty:
        extra = pd.DataFrame.from_dict(deltas, orient="index")
        if not extra.empty:
            out = out.merge(extra, left_on="doc_id", right_index=True, how="left")
    return out


# --------------------------------------------------------------------------- volumen y fichas

def extrapolar_volumen(registro: pd.DataFrame, n_empresas: int, universo: int = 500) -> pd.DataFrame:
    """Lo medido en la muestra, escalado a ``universo`` empresas (orden de magnitud)."""
    filas = []
    if registro.empty:
        return pd.DataFrame(filas)
    for fam, g in registro.groupby("familia"):
        factor = universo / n_empresas
        filas.append({"familia": fam, "filings_muestra": len(g), "mb_muestra": g["bytes"].sum() / 1e6,
                      "mb_gz_muestra": g["bytes_gz"].sum() / 1e6, "filings_universo": round(len(g) * factor),
                      "gb_universo": g["bytes"].sum() * factor / 1e9, "gb_gz_universo": g["bytes_gz"].sum() * factor / 1e9,
                      "horas_descarga_universo": g["segundos"].sum() * factor / 3600})
    return pd.DataFrame(filas)


def _estado(registro: pd.DataFrame) -> str:
    if registro.empty or not registro["ok"].any():
        return "fallo"
    return "ok" if registro["ok"].mean() >= 0.9 else "parcial"


def fichas_tema3(cobertura: pd.DataFrame, registro: pd.DataFrame, volumen: pd.DataFrame,
                 fundamentales: pd.DataFrame, lm_disponible: bool, n_empresas: int) -> list:
    """Construye y guarda las fichas de las cuatro fuentes con las mediciones de la sonda."""
    def cob(fam):
        c = cobertura.loc[cobertura["familia"] == fam]
        return (str(c["primero"].min()) if not c.empty else None, str(c["ultimo"].max()) if not c.empty else None,
                c["retraso_mediana_d"].median() if not c.empty else None)

    def vol(fam):
        if volumen.empty:
            return "sin medir"
        v = volumen.loc[volumen["familia"] == fam]
        return f"{v['gb_universo'].sum():.1f} GB crudo / {v['gb_gz_universo'].sum():.1f} GB gzip para 500 empresas" if not v.empty else "sin medir"

    ini_k, fin_k, ret_k = cob("10-K")
    _, _, ret_q = cob("10-Q")
    if registro.empty:
        registro = pd.DataFrame(columns=["familia", "ok"])
    reg_kq = registro.loc[registro["familia"].isin(["10-K", "10-Q"])]
    reg_8k = registro.loc[registro["familia"] == "8-K"]
    comun = dict(tema=TEMA, url="https://www.sec.gov/edgar", acceso="api", autenticacion="ninguna (User-Agent con email)",
                 coste="0 €", licencia_y_terminos="Documentos oficiales de dominio público; acceso según las normas de acceso justo de la SEC (≤10 peticiones/s)",
                 limites="10 peticiones/s por IP; User-Agent obligatorio")
    fichas = [
        FichaFuente(fuente="sec_10k_10q", nombre="SEC EDGAR · 10-K y 10-Q por secciones", agentes=["equity", "risk"],
                    rag="base+cajon", cobertura_inicio=ini_k, cobertura_fin=fin_k, frecuencia="anual / trimestral",
                    retraso_publicacion=f"10-K mediana {ret_k} d tras el cierre del ejercicio; 10-Q {ret_q} d",
                    formato="html (txt antes de ~2001)", volumen_estimado=f"{vol('10-K')} (10-K); {vol('10-Q')} (10-Q)",
                    estado_sonda=_estado(reg_kq),
                    mediciones={"n_empresas": n_empresas, "n_filings": len(reg_kq), "pct_ok": float(reg_kq["ok"].mean()) if len(reg_kq) else None},
                    notas=["TimeGate: filingDate, o día hábil siguiente si la aceptación es ≥16:00 ET (hora oficial de .hdr.sgml; "
                           "acceptanceDateTime de la API no es fiable)",
                           "Secciones: tramo más largo por Item; 'por_referencia' marca MD&A remitido al anexo 13"], **comun),
        FichaFuente(fuente="sec_8k_resultados", nombre="SEC EDGAR · 8-K de resultados (Item 2.02, anexo 99.1)", agentes=["equity"],
                    rag="cajon", cobertura_inicio=cob("8-K resultados")[0], cobertura_fin=cob("8-K resultados")[1],
                    frecuencia="trimestral", retraso_publicacion="0 d (mismo día del anuncio; suele aceptarse tras el cierre)",
                    formato="html", volumen_estimado=vol("8-K"), estado_sonda=_estado(reg_8k),
                    mediciones={"n_filings": len(reg_8k), "pct_ok": float(reg_8k["ok"].mean()) if len(reg_8k) else None},
                    notas=["Item 2.02 existe desde agosto de 2004 (antes, Item 12 desde marzo de 2003)",
                           "Sustituto legal de las transcripciones de calls"], **comun),
        FichaFuente(fuente="sec_xbrl_companyfacts", nombre="SEC XBRL · companyfacts (fundamentales point-in-time)",
                    agentes=["equity", "risk"], rag="serie",
                    cobertura_inicio=str(fundamentales["fecha_publicacion"].min()) if not fundamentales.empty else None,
                    cobertura_fin=str(fundamentales["fecha_publicacion"].max()) if not fundamentales.empty else None,
                    frecuencia="por filing", retraso_publicacion="el del filing que la contiene (10-K / 10-Q)",
                    formato="json -> parquet largo", volumen_estimado=f"{len(fundamentales)} cifras en {n_empresas} empresas",
                    estado_sonda="ok" if not fundamentales.empty else "fallo",
                    mediciones={"n_cifras": len(fundamentales), "n_conceptos": int(fundamentales["concepto"].nunique()) if not fundamentales.empty else 0},
                    notas=["Cada cifra se guarda una vez por publicación: las reexpresiones quedan visibles",
                           "Antes de 2009 no hay XBRL: hará falta Compustat (WRDS) o extraer de los textos"], **comun),
        FichaFuente(fuente="loughran_mcdonald", nombre="Diccionario Loughran-McDonald", tema=TEMA, agentes=["equity", "risk"],
                    rag="ninguno", url="https://sraf.nd.edu/loughranmcdonald-master-dictionary/", acceso="descarga manual",
                    autenticacion="ninguna", coste="0 €", licencia_y_terminos="Gratis para uso académico; no redistribuir",
                    cobertura_inicio=None, cobertura_fin=None, frecuencia="atemporal (versión anual)",
                    retraso_publicacion="no aplica", formato="csv", volumen_estimado="~9 MB",
                    limites="enlace de descarga en Google Drive (no automatizable de forma fiable)",
                    estado_sonda="ok" if lm_disponible else "no_probada",
                    notas=["Se aplica al ingerir: proporciones por categoría, tono y risk_delta del Item 1A"]),
    ]
    for f in fichas:
        guardar_ficha(f)
    return fichas


def guardar_tablas(**tablas: pd.DataFrame) -> None:
    """Tablas pequeñas de medición -> ``results/ingesta/empresas/<nombre>.csv`` (versionadas)."""
    carpeta = resultados_dir(TEMA)
    for nombre, t in tablas.items():
        if t is not None and not t.empty:
            t.to_csv(carpeta / f"{nombre}.csv", index=False)
