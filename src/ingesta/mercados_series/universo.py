"""Universo de instrumentos del Tema 1 (bloque 5): ``config/universo.yaml``.

``cargar_universo`` lee el YAML (grupos con ``defaults`` + ``instrumentos``), aplica los defaults
y valida cada instrumento:

- campos obligatorios y enumerados (``clase``, ``capa``, ``estado``); ids únicos con prefijo de
  fuente (``yf:``, ``coinbase:``); ``tickers.yahoo`` se deduce de ``yf:<ticker>`` si falta;
- ``mercado`` existe en ``config/mercados.yaml``; ``divisa`` ISO 4217 o subunidad (``GBp``, ``ZAc``, ``ILA``);
- ISIN con dígito de control correcto (Luhn sobre letras expandidas) y ``invertible_minorista_ue``
  coherente: un ETF/ETP invertible por un minorista de la UE necesita ISIN no estadounidense (los
  ETF de EE. UU. no publican KID PRIIPs);
- referencias cruzadas: ``equivalente_ucits`` y ``proxy_previo`` apuntan a ids del universo; el
  equivalente UCITS es invertible;
- fechas ``AAAA-MM-DD`` (o null).

Lanza ``ValueError`` con el id culpable. ``tabla_universo`` devuelve un DataFrame (una fila por
instrumento) para el notebook y las fichas.

Mediciones (red, solo metadatos al repositorio):

- ``verificar_ucits_justetf``: fecha de lanzamiento («Inception Date») y nombre de cada UCITS en su
  ficha de justETF (``robots_permite`` + ``ingesta.comun.http.get``, 1 petición/s) →
  ``results/ingesta/mercados_series/tablas/universo_ucits_verificacion.csv``.
- ``comprobar_yahoo``: primer y último día, número de velas y divisa de cada ticker de Yahoo
  (yfinance, <= 0,5 tickers/s, timeout 90 s, reintentos) →
  ``tablas/universo_yahoo.csv``. Sin precios: solo fechas y recuentos.

CLI: ``python -m ingesta.mercados_series.universo [--justetf] [--yahoo] [--ficha]``.
"""
from __future__ import annotations

import argparse
import html
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ingesta.comun.esquema import FichaFuente
from ingesta.comun.fichas import guardar_ficha
from ingesta.comun.rutas import resultados_dir

CONFIG = Path(__file__).resolve().parent / "config"
RUTA_UNIVERSO = CONFIG / "universo.yaml"
RUTA_MERCADOS = CONFIG / "mercados.yaml"
TEMA = "mercados_series"

CLASES = ("etf", "etp", "indice", "volatilidad", "divisa", "futuro", "cripto", "fondo")
CAPAS = ("estrategica", "tactica", "contexto", "baseline")
ESTADOS = ("activo", "proxy", "por_verificar", "inactivo")
PREFIJOS = ("yf", "coinbase", "binance")
SUBUNIDADES = {"GBp": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100), "USX": ("USD", 100)}
OBLIGATORIOS = ("id", "nombre", "clase", "bucket", "capa", "invertible_minorista_ue", "mercado", "divisa", "estado")
OPCIONALES = ("tickers", "isin", "indice_referencia", "inicio_esperado", "equivalente_ucits", "proxy_previo",
              "fondo_indexado_equivalente", "notas")
CAMPOS = (*OBLIGATORIOS, *OPCIONALES)
_PATRON_ID = re.compile(rf"^(?:{'|'.join(PREFIJOS)}):\S+$")
_PATRON_DIVISA = re.compile(r"^[A-Z]{3}$")
_PATRON_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")

URL_JUSTETF = "https://www.justetf.com/en/etf-profile.html?isin={isin}"
FICHERO_JUSTETF = "universo_ucits_verificacion.csv"
FICHERO_YAHOO = "universo_yahoo.csv"
NOTA_SUPERVIVENCIA = ("universo de hoy: los ETF cerrados o fusionados no están (sesgo de supervivencia "
                      "declarado; WRDS/CRSP pendiente de MIAX)")


# --- Validación -------------------------------------------------------------------------------------
def isin_valido(isin: str) -> bool:
    """Formato ISO 6166 y dígito de control (Luhn sobre las letras expandidas: A=10 ... Z=35)."""
    if not isinstance(isin, str) or not _PATRON_ISIN.match(isin):
        return False
    digitos = "".join(str(int(c, 36)) for c in isin[:-1])
    total = 0
    for i, c in enumerate(reversed(digitos)):
        d = int(c)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10 == int(isin[-1])


def divisa_base(divisa: str) -> tuple[str, int]:
    """``('GBP', 100)`` para ``GBp``; ``('USD', 1)`` para ``USD``."""
    return SUBUNIDADES.get(divisa, (divisa, 1))


def claves_mercados(ruta_mercados: Path | None = None) -> set[str]:
    """Claves de mercado conocidas (las de ``config/mercados.yaml``)."""
    claves = (yaml.safe_load((ruta_mercados or RUTA_MERCADOS).read_text(encoding="utf-8")) or {}).keys()
    return {str(c) for c in claves}


def _fecha(valor: Any, donde: str) -> str | None:
    if valor is None:
        return None
    if isinstance(valor, date):
        return valor.isoformat()
    try:
        return datetime.strptime(str(valor), "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ValueError(f"{donde}: fecha {valor!r} no es AAAA-MM-DD") from None


def _validar_instrumento(crudo: dict, donde: str, mercados: set[str]) -> dict[str, Any]:
    desconocidos = set(crudo) - set(CAMPOS)
    if desconocidos:
        raise ValueError(f"{donde}: campos desconocidos {sorted(desconocidos)}")
    faltan = [c for c in OBLIGATORIOS if crudo.get(c) is None]
    if faltan:
        raise ValueError(f"{donde}: faltan {faltan}")
    i = {c: crudo.get(c) for c in CAMPOS}
    if not _PATRON_ID.match(str(i["id"])):
        raise ValueError(f"{donde}: id {i['id']!r} sin prefijo de fuente {PREFIJOS}")
    for campo, validos in (("clase", CLASES), ("capa", CAPAS), ("estado", ESTADOS)):
        if i[campo] not in validos:
            raise ValueError(f"{donde}: {campo} {i[campo]!r} no está en {validos}")
    if not isinstance(i["invertible_minorista_ue"], bool):
        raise ValueError(f"{donde}: 'invertible_minorista_ue' debe ser true o false")
    if i["mercado"] not in mercados:
        raise ValueError(f"{donde}: mercado {i['mercado']!r} no está en mercados.yaml")
    if not (_PATRON_DIVISA.match(str(i["divisa"])) or i["divisa"] in SUBUNIDADES):
        raise ValueError(f"{donde}: divisa {i['divisa']!r} no es ISO 4217 ni subunidad {tuple(SUBUNIDADES)}")
    tickers = dict(i["tickers"] or {})
    prefijo, _, codigo = str(i["id"]).partition(":")
    if prefijo == "yf":
        tickers.setdefault("yahoo", codigo)
    if not tickers:
        raise ValueError(f"{donde}: sin tickers")
    i["tickers"] = tickers
    if i["isin"] is not None and not isin_valido(str(i["isin"])):
        raise ValueError(f"{donde}: ISIN {i['isin']!r} con formato o dígito de control incorrecto")
    if i["invertible_minorista_ue"] and i["clase"] in ("etf", "etp", "fondo"):
        if i["isin"] is None and i["estado"] != "por_verificar":
            raise ValueError(f"{donde}: invertible en la UE sin ISIN (o márcalo por_verificar)")
        if i["isin"] is not None and str(i["isin"]).startswith("US"):
            raise ValueError(f"{donde}: un ETF de EE. UU. no es invertible por un minorista de la UE (sin KID)")
    i["inicio_esperado"] = _fecha(i["inicio_esperado"], donde)
    notas = i["notas"] or []
    if not isinstance(notas, list):
        raise ValueError(f"{donde}: 'notas' debe ser una lista")
    i["notas"] = [str(n) for n in notas]
    return i


def validar_universo(crudo: Any, origen: str = "universo.yaml", mercados: set[str] | None = None) -> dict[str, Any]:
    """Valida el YAML ya leído y devuelve ``{version, fondos_por_que_no_entran, notas_generales,
    instrumentos: [dict...]}`` con los defaults aplicados."""
    if not isinstance(crudo, dict) or not isinstance(crudo.get("grupos"), list):
        raise ValueError(f"{origen}: se esperaba un diccionario con 'grupos'")
    mercados = claves_mercados() if mercados is None else mercados
    instrumentos: list[dict[str, Any]] = []
    vistos: set[str] = set()
    for g in crudo["grupos"]:
        nombre_g = g.get("nombre", "?")
        defaults = g.get("defaults") or {}
        for k, crudo_i in enumerate(g.get("instrumentos") or []):
            donde = f"{origen}: grupo {nombre_g!r}, instrumento {crudo_i.get('id', k)!r}"
            i = _validar_instrumento({**defaults, **crudo_i}, donde, mercados)
            if i["id"] in vistos:
                raise ValueError(f"{donde}: id duplicado")
            vistos.add(i["id"])
            i["grupo"] = nombre_g
            instrumentos.append(i)
    for i in instrumentos:
        for campo in ("equivalente_ucits", "proxy_previo"):
            ref = i[campo]
            if ref is not None and ref not in vistos:
                raise ValueError(f"{origen}: {i['id']!r}: {campo} {ref!r} no está en el universo")
        if i["equivalente_ucits"] is not None:
            eq = next(x for x in instrumentos if x["id"] == i["equivalente_ucits"])
            if not eq["invertible_minorista_ue"]:
                raise ValueError(f"{origen}: {i['id']!r}: el equivalente UCITS {eq['id']!r} no es invertible")
    if not str(crudo.get("fondos_por_que_no_entran") or "").strip():
        raise ValueError(f"{origen}: falta 'fondos_por_que_no_entran'")
    return {"version": crudo.get("version"), "fondos_por_que_no_entran": crudo["fondos_por_que_no_entran"],
            "notas_generales": list(crudo.get("notas_generales") or []), "instrumentos": instrumentos}


def cargar_universo(ruta: Path | None = None, mercados: set[str] | None = None) -> dict[str, Any]:
    """Lee y valida ``config/universo.yaml`` (o ``ruta``)."""
    r = Path(ruta) if ruta is not None else RUTA_UNIVERSO
    return validar_universo(yaml.safe_load(r.read_text(encoding="utf-8")), r.name, mercados)


def tabla_universo(universo: dict[str, Any] | None = None) -> pd.DataFrame:
    """Una fila por instrumento (tickers y notas aplanados) con ``ticker_yahoo`` y ``region``."""
    u = universo or cargar_universo()
    filas = []
    for i in u["instrumentos"]:
        f = {k: v for k, v in i.items() if k not in ("tickers", "notas")}
        f["ticker_yahoo"] = i["tickers"].get("yahoo")
        f["tickers"] = "; ".join(f"{k}={v}" for k, v in i["tickers"].items())
        f["notas"] = " | ".join(i["notas"])
        filas.append(f)
    df = pd.DataFrame(filas)
    df["inicio_esperado"] = pd.to_datetime(df["inicio_esperado"])
    df["region"] = df["bucket"].map(_region)
    return df


def _region(bucket: str) -> str:
    for clave, region in (("eeuu", "EE. UU."), ("europa", "Europa"), ("japon", "Japón"), ("china", "China"),
                          ("emergentes", "Emergentes"), ("asia_pacifico", "Asia-Pacífico"), ("pais_g7", "G7 ex EE. UU."),
                          ("global", "Global"), ("desarrollados", "Desarrollados ex EE. UU.")):
        if clave in bucket:
            return region
    return "Global / sin región"


# --- Mediciones con red ------------------------------------------------------------------------------
_PATRON_INCEPCION = re.compile(r"Inception Date.*?(\d{1,2} [A-Z][a-z]+ \d{4})", re.S)
_PATRON_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)


def leer_justetf(texto: str) -> dict[str, Any]:
    """Nombre y fecha de lanzamiento de una ficha HTML de justETF (None si no aparecen)."""
    m = _PATRON_H1.search(texto)
    nombre = html.unescape(re.sub(r"<[^>]+>|\s+", " ", m.group(1))).strip() if m else None
    f = _PATRON_INCEPCION.search(texto)
    lanzamiento = datetime.strptime(f.group(1), "%d %B %Y").date().isoformat() if f else None
    return {"nombre_justetf": nombre, "lanzamiento_justetf": lanzamiento}


def verificar_ucits_justetf(universo: dict[str, Any] | None = None, destino: Path | None = None) -> pd.DataFrame:
    """Lanzamiento de cada instrumento invertible con ISIN según justETF → CSV de metadatos."""
    from ingesta.comun.http import get, robots_permite

    u = universo or cargar_universo()
    filas = []
    for i in u["instrumentos"]:
        if not (i["invertible_minorista_ue"] and i["isin"]):
            continue
        url = URL_JUSTETF.format(isin=i["isin"])
        fila = {"id": i["id"], "isin": i["isin"], "url": url, "estado": None, "nombre_justetf": None,
                "lanzamiento_justetf": None, "consultado_utc": pd.Timestamp.now(tz="UTC").isoformat(), "error": None}
        if not robots_permite(url):
            fila["error"] = "robots.txt no lo permite"
        else:
            try:
                r = get(url, por_segundo=1.0, timeout=45)
                fila["estado"] = r.status_code
                fila.update(leer_justetf(r.text))
            except Exception as e:  # noqa: BLE001 - se documenta y se sigue
                fila["error"] = str(e)[:200]
        filas.append(fila)
    df = pd.DataFrame(filas)
    ruta = destino or resultados_dir(TEMA) / "tablas" / FICHERO_JUSTETF
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    return df


def comprobar_yahoo(universo: dict[str, Any] | None = None, destino: Path | None = None, *,
                    por_segundo: float = 0.5, reintentos: int = 3, ids: list[str] | None = None) -> pd.DataFrame:
    """Primer/último día, velas y divisa de cada ticker de Yahoo (sin guardar precios)."""
    import yfinance as yf

    u = universo or cargar_universo()
    filas = []
    for i in u["instrumentos"]:
        t = i["tickers"].get("yahoo")
        if t is None or (ids is not None and i["id"] not in ids):
            continue
        fila = {"id": i["id"], "ticker_yahoo": t, "primer_dia": None, "ultimo_dia": None, "n_velas": 0,
                "divisa_yahoo": None, "consultado_utc": pd.Timestamp.now(tz="UTC").isoformat(), "error": None}
        for k in range(reintentos):
            t0 = time.monotonic()
            try:
                tk = yf.Ticker(t)
                h = tk.history(period="max", interval="1d", auto_adjust=False, actions=False, timeout=90)
                if len(h):
                    fila.update(primer_dia=h.index.min().date().isoformat(), ultimo_dia=h.index.max().date().isoformat(),
                                n_velas=int(len(h)))
                    try:
                        fila["divisa_yahoo"] = (tk.history_metadata or {}).get("currency")
                    except Exception:  # noqa: BLE001
                        pass
                    fila["error"] = None
                    break
                fila["error"] = "sin datos"
            except Exception as e:  # noqa: BLE001
                fila["error"] = str(e)[:200]
            finally:
                espera = 1.0 / por_segundo - (time.monotonic() - t0)
                if espera > 0:
                    time.sleep(espera)
            if fila["error"] == "sin datos":
                break
            time.sleep(2.0 * (k + 1))
        filas.append(fila)
    df = pd.DataFrame(filas)
    ruta = destino or resultados_dir(TEMA) / "tablas" / FICHERO_YAHOO
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    return df


# --- Ficha ------------------------------------------------------------------------------------------
def _leer_csv(nombre: str) -> pd.DataFrame | None:
    p = resultados_dir(TEMA) / "tablas" / nombre
    return pd.read_csv(p) if p.exists() else None


def ficha_universo(universo: dict[str, Any] | None = None) -> Path:
    """Ficha 'universo' con recuentos y las mediciones de justETF y Yahoo (si existen)."""
    u = universo or cargar_universo()
    df = tabla_universo(u)
    med: dict[str, Any] = {
        "n_instrumentos": int(len(df)),
        "por_clase": df["clase"].value_counts().to_dict(),
        "por_capa": df["capa"].value_counts().to_dict(),
        "invertibles_ue": int(df["invertible_minorista_ue"].sum()),
        "con_equivalente_ucits": int(df["equivalente_ucits"].notna().sum()),
        "con_proxy_previo": int(df["proxy_previo"].notna().sum()),
    }
    ini = fin = None
    jt = _leer_csv(FICHERO_JUSTETF)
    if jt is not None:
        med["justetf"] = {"consultados": int(len(jt)), "con_fecha": int(jt["lanzamiento_justetf"].notna().sum()),
                          "errores": int(jt["error"].notna().sum())}
    yh = _leer_csv(FICHERO_YAHOO)
    retraso = "cierre de mercado + 30 min → primera decisión 16:30 NY (precios); el universo no tiene retraso propio"
    if yh is not None:
        ok = yh[yh["primer_dia"].notna()]
        med["yahoo"] = {"consultados": int(len(yh)), "con_datos": int(len(ok)),
                        "sin_datos": yh.loc[yh["primer_dia"].isna(), "ticker_yahoo"].tolist(),
                        "divisas": ok["divisa_yahoo"].value_counts(dropna=False).to_dict()}
        if len(ok):
            ini, fin = ok["primer_dia"].min(), ok["ultimo_dia"].max()
            ultimo = pd.to_datetime(ok["ultimo_dia"])
            consulta = pd.to_datetime(ok["consultado_utc"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            med["yahoo"]["dias_desde_ultima_vela_mediana"] = float((consulta - ultimo).dt.days.median())
            retraso += (f"; medido: mediana de {med['yahoo']['dias_desde_ultima_vela_mediana']:.0f} días entre la "
                        "última vela de Yahoo y el día de consulta (NY)")
    ficha = FichaFuente(
        fuente="universo", nombre="Universo de instrumentos del Tema 1 (ETF, UCITS, índices, volatilidad, FX, futuros, cripto)",
        tema=TEMA, agentes=["portfolio", "equity", "risk", "macro"], rag="serie",
        url="src/ingesta/mercados_series/config/universo.yaml", acceso="libreria",
        autenticacion="ninguna", coste="0 €",
        licencia_y_terminos="Yahoo Finance (yfinance): uso personal, sin redistribuir; justETF: consulta puntual de fichas públicas",
        cobertura_inicio=ini, cobertura_fin=fin, frecuencia="diaria", retraso_publicacion=retraso,
        formato="yaml + csv de metadatos", volumen_estimado=f"{len(df)} instrumentos",
        limites="yfinance <= 0,5 tickers/s, timeout 90 s; justETF 1 petición/s",
        estado_sonda="ok" if yh is not None and jt is not None else "parcial", mediciones=med,
        notas=[f"Fondos: por qué no entran. {u['fondos_por_que_no_entran']}", NOTA_SUPERVIVENCIA, *u["notas_generales"],
               "inicio_esperado de los UCITS = lanzamiento según justETF (tablas/universo_ucits_verificacion.csv); null si no se pudo verificar"],
    )
    return guardar_ficha(ficha)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--justetf", action="store_true", help="verificar lanzamientos de los UCITS en justETF")
    p.add_argument("--yahoo", action="store_true", help="medir primer/último día de cada ticker en Yahoo")
    p.add_argument("--ficha", action="store_true", help="escribir la ficha 'universo'")
    a = p.parse_args(argv)
    u = cargar_universo()
    print(f"universo: {len(u['instrumentos'])} instrumentos válidos")
    if a.justetf:
        print(verificar_ucits_justetf(u)[["id", "lanzamiento_justetf", "error"]].to_string())
    if a.yahoo:
        df = comprobar_yahoo(u)
        print(df[df["primer_dia"].isna()][["id", "error"]].to_string())
    if a.ficha:
        print(ficha_universo(u))


if __name__ == "__main__":
    main()
