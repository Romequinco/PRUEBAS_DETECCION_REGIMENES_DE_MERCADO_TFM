"""Fusiones fijas (ADR-011 §3, §5 y §6.1; notebook 15).

Reglas sin parámetros aprendidos sobre la **señal binaria** de los candidatos (§2.1), construidas
sobre todo el OOS de una vez (son causales) y recortadas después para puntuar cada tramo:

- **(a) parejas alerta→confirmación** ``PAR_<X>><Y>``: ``maquina.fuse_early_warning`` sin
  cambios, con la configuración de las fusiones antiguas D7+D8 y D2+D6 y el confirmador binarizado
  (``p_crisis`` := señal binaria de Y). Principal: «accionable» (vigilancia ∪ confirmado).
  Sensibilidad ``TRAS_<X>><Y>``: «confirmación tras aviso» (``maquina._gated_confirmation``).
  Referencia ``CONF_<Y>``: «confirmado» puro (Y con persistencia 3).
- **(b) votación** ``VOTO_2de3``, ``VOTO_3de5`` (top-n del orden de H1 de los individuales) y
  ``VOTO_mayoria`` (todos los candidatos, ⌈n/2⌉ de n).
- **(c) OR y AND de parejas** ``OR_<X>+<Y>`` y ``AND_<X>+<Y>`` (X < Y como cadenas).

F* es la primera de las fusiones principales en el orden de H1 (§5.2) entre las de nivel 0 en H1.
El criterio del §6.1 se evalúa por pista con la vara de :mod:`regimenes.fusion.vara` y el veredicto
global es «aporta» (las dos pistas), «indicio» (una) o «no aporta».

CLI (un solo proceso)::

    python -m regimenes.fusion.fijas ejecutar [--pista A B]   # calcula y escribe results/fusion/fijas/
    python -m regimenes.fusion.fijas leer                      # resumen de lo escrito

Las funciones ``leer_*`` devuelven las tablas escritas para el notebook (``None`` si faltan).
"""

from __future__ import annotations

import argparse
from itertools import combinations, permutations
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.fusion import vara
from regimenes.fusion.maquina import (
    FusionConfig,
    _gated_confirmation,
    fuse_early_warning,
    warning_ground_truth_table,
    warning_phase_base_rates,
)

SALIDA: Path = rutas.RESULTS_FUSION_FIJAS
PISTAS = vara.PISTAS
TRAMOS = ("oos", "H1", "H2")
# §3.1: configuración de las fusiones antiguas D7+D8 y D2+D6 (alerta por estado, sin persistencia;
# confirmación con umbral 0,50 y 3 sesiones; ventana de vigilancia de 63 sesiones).
HORIZONTE = 63
UMBRAL_CONFIRMACION = 0.50
PERSISTENCIA_CONFIRMACION = 3
FAMILIAS = {"PAR": "a", "VOTO": "b", "OR": "c", "AND": "c"}
# Grupos de señales: solo «principal» entra en la selección de F* (§5.2).
GRUPOS = ("principal", "individual", "tras_aviso", "confirmado")
VOTOS_TOP = {"VOTO_2de3": (3, 2), "VOTO_3de5": (5, 3)}
NO_APORTA = "no aporta: combinar no mejora al mejor detector solo"
COLUMNAS_ANTICIPACION = ["pista", "f_star", "fase", "n_avisos", "fraccion_avisos", "fraccion_base_azar"]


# --------------------------------------------------------------------------- #
# Familia (a) · parejas alerta→confirmación (§3.1)
# --------------------------------------------------------------------------- #
def config_pareja(n_states_alerta: int) -> FusionConfig:
    """``FusionConfig`` del §3.1 con la alerta en el estado ``n_states_X − 1``."""
    return FusionConfig(alert_source="state", alert_crisis_state=int(n_states_alerta) - 1,
                        alert_persistence=1, confirmation_threshold=UMBRAL_CONFIRMACION,
                        confirmation_persistence=PERSISTENCIA_CONFIRMACION,
                        warning_horizon=HORIZONTE)


def panel_confirmador(panel: pd.DataFrame, senal: pd.Series) -> pd.DataFrame:
    """Panel del confirmador con ``p_crisis`` sustituida por su señal binaria (0,0/1,0).

    Con el umbral 0,50, «confirma» = señal binaria de Y en 3 sesiones seguidas (§3.1); así el
    confirmador es la misma señal que se puntúa. No se toca ``maquina.py``."""
    out = panel[["state"]].copy()
    out["p_crisis"] = pd.Series(senal, index=panel.index).astype(bool).astype(float)
    return out


def fusionar_pareja(panel_alerta: pd.DataFrame, panel_conf: pd.DataFrame, senal_conf: pd.Series,
                    n_states_alerta: int) -> pd.DataFrame:
    """Salida de la máquina para X alerta → Y confirma, con la columna ``tras_aviso``.

    ``actionable`` es la binarización principal; ``tras_aviso`` la sensibilidad
    (``_gated_confirmation``); ``confirmation_active`` el «confirmado» puro."""
    config = config_pareja(n_states_alerta)
    fused = fuse_early_warning(panel_alerta, panel_confirmador(panel_conf, senal_conf), config)
    fused["tras_aviso"] = _gated_confirmation(fused, config.warning_horizon)
    return fused


def id_pareja(x: str, y: str, prefijo: str = "PAR") -> str:
    return f"{prefijo}_{x}>{y}"


def familia_a(pan: dict[str, pd.DataFrame], sen: pd.DataFrame, ns: dict[str, int],
              ids: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(principal ``PAR_*``, sensibilidad ``TRAS_*``, referencia ``CONF_*``) de todos los pares
    ordenados de ``ids`` (por defecto, las columnas de ``sen``)."""
    ids = list(ids or sen.columns)
    acc, tras = {}, {}
    for x, y in permutations(ids, 2):
        f = fusionar_pareja(pan[x], pan[y], sen[y], ns[x])
        acc[id_pareja(x, y)] = f["actionable"].astype(bool)
        tras[id_pareja(x, y, "TRAS")] = f["tras_aviso"].astype(bool)
    conf = {f"CONF_{y}": persistencia(sen[y], PERSISTENCIA_CONFIRMACION) for y in ids}
    return pd.DataFrame(acc), pd.DataFrame(tras), pd.DataFrame(conf)


def persistencia(senal: pd.Series, k: int) -> pd.Series:
    """Señal activa si lo estuvo las ``k`` últimas sesiones (``maquina._sustained``)."""
    s = pd.Series(senal).astype(int)
    return s.rolling(int(k), min_periods=int(k)).sum().eq(int(k))


# --------------------------------------------------------------------------- #
# Familias (b) y (c) (§3.2 y §3.3)
# --------------------------------------------------------------------------- #
def votacion(sen: pd.DataFrame, k: int) -> pd.Series:
    """Crisis el día t si al menos ``k`` columnas de ``sen`` marcan crisis (sin persistencia)."""
    return sen.astype(bool).sum(axis=1).ge(int(k))


def k_mayoria(n: int) -> int:
    """⌈n/2⌉: 8 de 15 en A y 9 de 17 en B."""
    return int(math.ceil(int(n) / 2))


def familia_b(sen: pd.DataFrame, orden_h1: Sequence[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """``VOTO_2de3``, ``VOTO_3de5`` (top-n de ``orden_h1``) y ``VOTO_mayoria`` (todas)."""
    out, miembros = {}, {}
    for nombre, (n, k) in VOTOS_TOP.items():
        top = list(orden_h1)[:n]
        out[nombre], miembros[nombre] = votacion(sen[top], k), top
    out["VOTO_mayoria"] = votacion(sen, k_mayoria(sen.shape[1]))
    miembros["VOTO_mayoria"] = list(sen.columns)
    return pd.DataFrame(out), miembros


def familia_c(sen: pd.DataFrame) -> pd.DataFrame:
    """``OR_<X>+<Y>`` y ``AND_<X>+<Y>`` de cada pareja no ordenada (X < Y como cadenas)."""
    out = {}
    for x, y in combinations(sorted(map(str, sen.columns)), 2):
        out[f"OR_{x}+{y}"] = sen[x] | sen[y]
        out[f"AND_{x}+{y}"] = sen[x] & sen[y]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Construcción de una pista
# --------------------------------------------------------------------------- #
def construir(pista: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Todas las señales de la pista y su catálogo.

    Devuelve ``(señales, catálogo)``: señales bool (individuales, principales, ``TRAS_*`` y
    ``CONF_*``) sobre todo el OOS; catálogo con ``id``, ``grupo``, ``familia``, ``x``, ``y``,
    ``k``, ``n`` y ``miembros``."""
    p = pista.upper()
    pan, sen, ns = vara.paneles(p), vara.senales(p), vara.n_states(p)
    h1, _ = vara.mitades(sen.index)
    orden_h1 = list(vara.evaluar(sen, p, h1).index)
    a, tras, conf = familia_a(pan, sen, ns)
    b, miembros = familia_b(sen, orden_h1)
    c = familia_c(sen)
    filas = [{"id": d, "grupo": "individual", "familia": "individual", "x": d, "y": None,
              "k": 1, "n": 1, "miembros": d} for d in sen.columns]
    for prefijo, grupo, tabla in (("PAR", "principal", a), ("TRAS", "tras_aviso", tras)):
        for col in tabla.columns:
            x, y = col.split("_", 1)[1].split(">")
            filas.append({"id": col, "grupo": grupo, "familia": "a", "x": x, "y": y, "k": None,
                          "n": 2, "miembros": f"{x}|{y}"})
    for col in b.columns:
        m = miembros[col]
        k = VOTOS_TOP[col][1] if col in VOTOS_TOP else k_mayoria(len(m))
        filas.append({"id": col, "grupo": "principal", "familia": "b", "x": None, "y": None, "k": k,
                      "n": len(m), "miembros": "|".join(m)})
    for col in c.columns:
        x, y = col.split("_", 1)[1].split("+")
        filas.append({"id": col, "grupo": "principal", "familia": "c", "x": x, "y": y,
                      "k": 1 if col.startswith("OR_") else 2, "n": 2, "miembros": f"{x}|{y}"})
    for col in conf.columns:
        y = col.split("_", 1)[1]
        filas.append({"id": col, "grupo": "confirmado", "familia": "a", "x": None, "y": y,
                      "k": PERSISTENCIA_CONFIRMACION, "n": 1, "miembros": y})
    todas = pd.concat([sen, a, b, c, tras, conf], axis=1).astype(bool)
    cat = pd.DataFrame(filas).assign(pista=p)
    if todas.columns.duplicated().any():
        raise ValueError(f"{p}: ids de señales repetidos.")
    return todas, cat


def recuento(cat: pd.DataFrame) -> pd.Series:
    """Señales por grupo y familia (§3.4: 423 principales en A y 547 en B)."""
    return cat.groupby(["grupo", "familia"]).size()


def evaluar_tramos(todas: pd.DataFrame, cat: pd.DataFrame, pista: str) -> pd.DataFrame:
    """Vara en ``oos``, ``H1`` y ``H2`` por grupo: una fila por (id, tramo) con su ``puesto``
    dentro del grupo (el orden del §5.2 del tramo)."""
    p = pista.upper()
    tr = vara.tramos(p)
    partes = []
    for grupo, ids in cat.groupby("grupo", sort=False)["id"]:
        for nombre in TRAMOS:
            e = vara.evaluar(todas[list(ids)], p, None if nombre == "oos" else tr[nombre])
            partes.append(e.reset_index().assign(grupo=grupo, tramo=nombre))
    out = pd.concat(partes, ignore_index=True).assign(pista=p)
    familia = cat.set_index("id")["familia"]
    out["familia"] = out["id"].map(familia)
    return out[["pista", "tramo", "grupo", "familia", "id", *[c for c in out if c not in
                                                              ("pista", "tramo", "grupo", "familia", "id")]]]


def _tabla(ev: pd.DataFrame, grupo: str, tramo: str) -> pd.DataFrame:
    return ev[(ev["grupo"] == grupo) & (ev["tramo"] == tramo)].set_index("id").sort_values("puesto")


def primera_nivel0(tabla: pd.DataFrame) -> str | None:
    """Primera fila (ya en el orden del §5.2) con nivel 0, o ``None``."""
    t = tabla[tabla["nivel"] == 0].sort_values("puesto")
    return str(t.index[0]) if len(t) else None


# --------------------------------------------------------------------------- #
# Selección y criterio (§5.2 y §6.1)
# --------------------------------------------------------------------------- #
def eleccion(ev: pd.DataFrame, cat: pd.DataFrame, pista: str) -> dict:
    """F* (H1, nivel 0), I* de todo el OOS, I*_H1 y los miembros de las votaciones."""
    p = pista.upper()
    f_star = primera_nivel0(_tabla(ev, "principal", "H1"))
    ind_oos, ind_h1 = _tabla(ev, "individual", "oos"), _tabla(ev, "individual", "H1")
    miembros = cat.set_index("id")["miembros"]
    fila = {"pista": p, "f_star": f_star,
            "f_star_familia": FAMILIAS[f_star.split("_", 1)[0]] if f_star else None,
            "f_star_miembros": miembros.get(f_star) if f_star else None,
            "n_principales": int((cat["grupo"] == "principal").sum()),
            "n_nivel0_H1": int((_tabla(ev, "principal", "H1")["nivel"] == 0).sum()),
            "i_star": str(ind_oos.index[0]), "i_star_h1": str(ind_h1.index[0])}
    for nombre in ("VOTO_2de3", "VOTO_3de5", "VOTO_mayoria"):
        fila[f"miembros_{nombre}"] = miembros.get(nombre)
    return fila


def _fila(ev: pd.DataFrame, grupo: str, tramo: str, i: str) -> pd.Series:
    return _tabla(ev, grupo, tramo).loc[i]


def criterio_pista(ev: pd.DataFrame, elec: dict, tabla_mcs: pd.DataFrame | None) -> dict:
    """(i), (ii) y (iii) del §6.1 para F*. Comparaciones de score sin redondeo y estrictas."""
    f, i_oos, i_h1 = elec["f_star"], elec["i_star"], elec["i_star_h1"]
    io, ih2 = _fila(ev, "individual", "oos", i_oos), _fila(ev, "individual", "H2", i_h1)
    out = {"pista": elec["pista"], "f_star": f, "i_star": i_oos,
           "score_i_star_oos": float(io["score_deteccion"]), "i_star_h1": i_h1,
           "score_i_star_h1_H2": float(ih2["score_deteccion"])}
    if f is None:
        return {**out, "cumple_i": False, "cumple_ii": False, "cumple_iii": False,
                "cumple_pista": False, "nota": "sin F*: ninguna fusión con nivel 0 en H1"}
    fo, fh1, fh2 = (_fila(ev, "principal", t, f) for t in TRAMOS)
    incl = bool(tabla_mcs.loc[f, "incluido"]) if tabla_mcs is not None else False
    i_ok = bool(fo["nivel"] == 0 and float(fo["score_deteccion"]) > float(io["score_deteccion"]))
    iii_ok = bool(fh2["nivel"] == 0 and float(fh2["score_deteccion"]) > float(ih2["score_deteccion"]))
    return {**out, "nivel_f_star_H1": int(fh1["nivel"]), "score_f_star_H1": float(fh1["score_deteccion"]),
            "nivel_f_star_oos": int(fo["nivel"]), "score_f_star_oos": float(fo["score_deteccion"]),
            "f_star_en_mcs": incl,
            "f_star_duplicado_de": tabla_mcs.loc[f, "duplicado_de"] if tabla_mcs is not None else None,
            "nivel_f_star_H2": int(fh2["nivel"]), "score_f_star_H2": float(fh2["score_deteccion"]),
            "cumple_i": i_ok, "cumple_ii": incl, "cumple_iii": iii_ok,
            "cumple_pista": bool(i_ok and incl and iii_ok), "nota": ""}


def veredicto(criterios: pd.DataFrame) -> str:
    """«aporta» si se cumple en las dos pistas, «indicio» en una y «no aporta» en ninguna."""
    if set(criterios["pista"]) != set(PISTAS):
        return "pendiente (faltan pistas)"
    n = int(criterios["cumple_pista"].astype(bool).sum())
    return "aporta" if n == 2 else "indicio (una sola pista)" if n == 1 else NO_APORTA


def senales_mcs_f(todas: pd.DataFrame, pista: str, f_star: str | None) -> pd.DataFrame:
    """Columnas del MCS-F: los individuales y F* (los baselines los añade :func:`vara.mcs`)."""
    cols = vara.candidatos(pista) + ([f_star] if f_star else [])
    return todas[cols]


# --------------------------------------------------------------------------- #
# Lecturas sin peso (§6.3)
# --------------------------------------------------------------------------- #
def ganadores_familia(ev: pd.DataFrame, pista: str) -> pd.DataFrame:
    """Ganador de cada familia en H1 (regla de F* dentro de la familia) y su puesto en H2."""
    h1, h2, oos = (_tabla(ev, "principal", t) for t in ("H1", "H2", "oos"))
    filas = []
    for fam in ("a", "b", "c"):
        g = primera_nivel0(h1[h1["familia"] == fam])
        fila = {"pista": pista, "familia": fam, "ganador_H1": g}
        if g is not None:
            fila.update({"score_H1": h1.loc[g, "score_deteccion"], "puesto_H1": int(h1.loc[g, "puesto"]),
                         "nivel_H2": int(h2.loc[g, "nivel"]), "score_H2": h2.loc[g, "score_deteccion"],
                         "puesto_H2": int(h2.loc[g, "puesto"]),
                         "puesto_H2_familia": int(h2[h2["familia"] == fam].index.get_loc(g)) + 1,
                         "score_oos": oos.loc[g, "score_deteccion"]})
        filas.append(fila)
    return pd.DataFrame(filas)


def optimismo(ev: pd.DataFrame, elec: dict) -> dict:
    """Score (todo el OOS) de la mejor fusión elegida con todo el OOS menos el de F*."""
    oos = _tabla(ev, "principal", "oos")
    mejor = primera_nivel0(oos)
    f = elec["f_star"]
    s_mejor = float(oos.loc[mejor, "score_deteccion"]) if mejor else np.nan
    s_f = float(oos.loc[f, "score_deteccion"]) if f else np.nan
    return {"pista": elec["pista"], "mejor_oos": mejor, "score_mejor_oos": s_mejor, "f_star": f,
            "score_f_star_oos": s_f, "optimismo": s_mejor - s_f}


def topn(ev: pd.DataFrame, pista: str) -> pd.DataFrame:
    """Top-3 y top-5 del ranking v3 ampliado de todo el OOS frente a los de H1 (§3.2)."""
    from regimenes.benchmark import pool_lectura as pl

    r = pl.ranking_v3(menus=("N", "U", "P", "R"))
    v3 = list(r[r["pista"] == pista].sort_values("puesto_deteccion")["id"])
    h1 = list(_tabla(ev, "individual", "H1").index)
    oos = list(_tabla(ev, "individual", "oos").index)
    filas = []
    for n in (3, 5):
        filas.append({"pista": pista, "n": n, "top_v3_oos": "|".join(v3[:n]),
                      "top_orden_oos": "|".join(oos[:n]), "top_H1": "|".join(h1[:n]),
                      "solo_en_v3": "|".join(sorted(set(v3[:n]) - set(h1[:n]))),
                      "solo_en_H1": "|".join(sorted(set(h1[:n]) - set(v3[:n]))),
                      "mismo_orden": v3[:n] == h1[:n]})
    return pd.DataFrame(filas)


def variantes(ev: pd.DataFrame, elec: dict) -> pd.DataFrame:
    """«Confirmación tras aviso» y «confirmado» puro: mejor de H1 (nivel 0) y de todo el OOS.

    Si F* es una pareja, se añade su sensibilidad ``TRAS_`` y el ``CONF_`` de su confirmador."""
    filas = []
    for grupo in ("tras_aviso", "confirmado"):
        h1, h2, oos = (_tabla(ev, grupo, t) for t in ("H1", "H2", "oos"))
        g, m = primera_nivel0(h1), primera_nivel0(oos)
        filas.append({"pista": elec["pista"], "grupo": grupo, "lectura": "mejor_H1", "id": g,
                      "score_H1": h1.loc[g, "score_deteccion"] if g else np.nan,
                      "nivel_H2": int(h2.loc[g, "nivel"]) if g else np.nan,
                      "score_H2": h2.loc[g, "score_deteccion"] if g else np.nan,
                      "score_oos": oos.loc[g, "score_deteccion"] if g else np.nan,
                      "nivel_oos": int(oos.loc[g, "nivel"]) if g else np.nan})
        filas.append({"pista": elec["pista"], "grupo": grupo, "lectura": "mejor_oos", "id": m,
                      "score_oos": oos.loc[m, "score_deteccion"] if m else np.nan,
                      "nivel_oos": int(oos.loc[m, "nivel"]) if m else np.nan})
    f = elec["f_star"]
    if f and f.startswith("PAR_"):
        x, y = f[4:].split(">")
        for grupo, i in (("tras_aviso", id_pareja(x, y, "TRAS")), ("confirmado", f"CONF_{y}")):
            h1, h2, oos = (_tabla(ev, grupo, t) for t in ("H1", "H2", "oos"))
            filas.append({"pista": elec["pista"], "grupo": grupo, "lectura": "de_f_star", "id": i,
                          "score_H1": h1.loc[i, "score_deteccion"], "nivel_H2": int(h2.loc[i, "nivel"]),
                          "score_H2": h2.loc[i, "score_deteccion"], "score_oos": oos.loc[i, "score_deteccion"],
                          "nivel_oos": int(oos.loc[i, "nivel"])})
    return pd.DataFrame(filas)


def anticipacion(pista: str, f_star: str | None) -> pd.DataFrame:
    """Fases de los avisos de F* (``warning_ground_truth_table``) frente a la base de azar
    (``warning_phase_base_rates``), solo si F* es de la familia (a)."""
    if not f_star or not f_star.startswith("PAR_"):
        return pd.DataFrame(columns=COLUMNAS_ANTICIPACION)
    p = pista.upper()
    x, y = f_star[4:].split(">")
    pan, sen, ns, win = vara.paneles(p), vara.senales(p), vara.n_states(p), vara.ventanas(p)
    fused = fusionar_pareja(pan[x], pan[y], sen[y], ns[x])
    avisos = warning_ground_truth_table(fused, HORIZONTE, win)
    base = warning_phase_base_rates(fused.index, HORIZONTE, win)
    fases = (avisos["phase"].value_counts(normalize=True).reindex(base.index, fill_value=0.0)
             if len(avisos) else pd.Series(0.0, index=base.index))
    cuenta = avisos["phase"].value_counts().reindex(base.index, fill_value=0) if len(avisos) \
        else pd.Series(0, index=base.index)
    return pd.DataFrame({"pista": p, "f_star": f_star, "fase": base.index, "n_avisos": cuenta.values,
                         "fraccion_avisos": fases.values, "fraccion_base_azar": base.values})


def utilidad(todas: pd.DataFrame, elec: dict) -> pd.DataFrame:
    """Utilidad operativa (``maquina.signal_scores``) de F*, V, I* e I*_H1 por tramo (§6.3).

    O y M son de la fusión online (notebook 17)."""
    p = elec["pista"]
    quien = {"F*": elec["f_star"], "V": "VOTO_mayoria", "I*": elec["i_star"], "I*_H1": elec["i_star_h1"]}
    quien = {k: v for k, v in quien.items() if v}
    tr = vara.tramos(p)
    partes = []
    for nombre in TRAMOS:
        u = vara.utilidad(todas[list(dict.fromkeys(quien.values()))], p,
                          None if nombre == "oos" else tr[nombre])
        for papel, i in quien.items():
            partes.append({"pista": p, "tramo": nombre, "papel": papel, "id": i, **u.loc[i].to_dict()})
    return pd.DataFrame(partes)


# --------------------------------------------------------------------------- #
# Ejecución y escritura
# --------------------------------------------------------------------------- #
def _csv(ruta: Path, df: pd.DataFrame, index: bool = False) -> Path:
    from regimenes.benchmark.cache import _write_atomic

    _write_atomic(ruta, lambda tmp: df.to_csv(tmp, index=index))
    return ruta


def _parquet(ruta: Path, df: pd.DataFrame) -> Path:
    from regimenes.benchmark.cache import _write_atomic

    _write_atomic(ruta, lambda tmp: df.to_parquet(tmp))
    return ruta


def ejecutar_pista(pista: str, salida: Path = SALIDA, *, mcs_kw: dict | None = None,
                   log=print) -> dict[str, pd.DataFrame]:
    """Construye, evalúa, elige F*, calcula los MCS-F (todo el OOS y H2) y escribe la pista."""
    p = pista.upper()
    salida = Path(salida)
    mcs_kw = mcs_kw or {}
    log(f"[{p}] construyendo señales")
    todas, cat = construir(p)
    log(f"[{p}] {todas.shape[1]} señales; recuento: {recuento(cat).to_dict()}")
    ev = evaluar_tramos(todas, cat, p)
    elec = eleccion(ev, cat, p)
    v3 = topn(ev, p)
    if v3.loc[0, "top_v3_oos"].split("|")[0] != elec["i_star"]:
        raise ValueError(f"{p}: I* del orden del §5.2 ({elec['i_star']}) no es el puesto 1 del v3.")
    log(f"[{p}] F* = {elec['f_star']}  I* = {elec['i_star']}  I*_H1 = {elec['i_star_h1']}")
    tablas_mcs = {}
    for nombre, tramo in (("oos", None), ("H2", vara.tramos(p)["H2"])):
        log(f"[{p}] MCS-F {nombre}")
        t = vara.mcs(senales_mcs_f(todas, p, elec["f_star"]), p, tramo=tramo, **mcs_kw)
        tablas_mcs[nombre] = t.assign(tramo=nombre)
    crit = criterio_pista(ev, elec, tablas_mcs["oos"])
    out = {
        f"catalogo_{p}": cat, f"evaluacion_{p}": ev, f"eleccion_{p}": pd.DataFrame([elec]),
        f"mcs_F_{p}": tablas_mcs["oos"], f"mcs_F_H2_{p}": tablas_mcs["H2"],
        f"criterio_{p}": pd.DataFrame([crit]), f"ganadores_familia_{p}": ganadores_familia(ev, p),
        f"optimismo_{p}": pd.DataFrame([optimismo(ev, elec)]), f"topn_{p}": v3,
        f"variantes_{p}": variantes(ev, elec), f"anticipacion_{p}": anticipacion(p, elec["f_star"]),
        f"utilidad_{p}": utilidad(todas, elec),
    }
    for nombre, df in out.items():
        _csv(salida / f"{nombre}.csv", df, index=nombre.startswith("mcs_F"))
    _parquet(salida / f"senales_{p}.parquet", todas)
    log(f"[{p}] criterio: (i) {crit['cumple_i']}  (ii) {crit['cumple_ii']}  (iii) {crit['cumple_iii']}")
    return out


def consolidar(salida: Path = SALIDA) -> pd.DataFrame | None:
    """``criterio.csv``: los ``criterio_<P>.csv`` presentes y el veredicto global (§6.1)."""
    salida = Path(salida)
    partes = [pd.read_csv(salida / f"criterio_{p}.csv") for p in PISTAS
              if (salida / f"criterio_{p}.csv").exists()]
    if not partes:
        return None
    crit = pd.concat(partes, ignore_index=True)
    crit["veredicto"] = veredicto(crit)
    _csv(salida / "criterio.csv", crit)
    return crit


def ejecutar(pistas: Iterable[str] = PISTAS, salida: Path = SALIDA, **kwargs) -> pd.DataFrame | None:
    for p in pistas:
        ejecutar_pista(p, salida, **kwargs)
    return consolidar(salida)


# --------------------------------------------------------------------------- #
# Lectura para el notebook
# --------------------------------------------------------------------------- #
def leer_tabla(nombre: str, salida: Path = SALIDA, **kwargs) -> pd.DataFrame | None:
    """CSV ``<salida>/<nombre>.csv`` o ``None`` si falta."""
    ruta = Path(salida) / (nombre if nombre.endswith(".csv") else f"{nombre}.csv")
    if not ruta.exists():
        return None
    try:
        t = pd.read_csv(ruta, **kwargs)
    except pd.errors.EmptyDataError:                      # tabla vacía (p. ej. anticipación sin F* de (a))
        t = pd.DataFrame()
    t.attrs["ruta"] = ruta
    return t


def _por_pista(nombre: str, salida: Path, **kwargs) -> pd.DataFrame | None:
    partes = [leer_tabla(f"{nombre}_{p}", salida, **kwargs) for p in PISTAS]
    partes = [t for t in partes if t is not None and len(t)]
    return pd.concat(partes) if partes else None


def leer_criterio(salida: Path = SALIDA) -> pd.DataFrame | None:
    return leer_tabla("criterio", salida)


def leer_eleccion(salida: Path = SALIDA) -> pd.DataFrame | None:
    return _por_pista("eleccion", salida)


def leer_catalogo(salida: Path = SALIDA) -> pd.DataFrame | None:
    return _por_pista("catalogo", salida)


def leer_evaluacion(pista: str | None = None, salida: Path = SALIDA) -> pd.DataFrame | None:
    return leer_tabla(f"evaluacion_{pista.upper()}", salida) if pista else _por_pista("evaluacion", salida)


def leer_mcs(pista: str, tramo: str = "oos", salida: Path = SALIDA) -> pd.DataFrame | None:
    nombre = f"mcs_F_{pista.upper()}" if tramo == "oos" else f"mcs_F_{tramo}_{pista.upper()}"
    return leer_tabla(nombre, salida, index_col="modelo")


def leer_lectura(nombre: str, salida: Path = SALIDA) -> pd.DataFrame | None:
    """Lecturas sin peso del §6.3: ``ganadores_familia``, ``optimismo``, ``topn``, ``variantes``,
    ``anticipacion`` o ``utilidad`` (las dos pistas juntas)."""
    return _por_pista(nombre, salida)


def leer_senales(pista: str, columnas: Sequence[str] | None = None,
                 salida: Path = SALIDA) -> pd.DataFrame | None:
    """Señales guardadas de la pista (``senales_<P>.parquet``), opcionalmente solo ``columnas``."""
    ruta = Path(salida) / f"senales_{pista.upper()}.parquet"
    if not ruta.exists():
        return None
    return pd.read_parquet(ruta, columns=list(columnas) if columnas else None)


def resumen(salida: Path = SALIDA) -> pd.DataFrame | None:
    """Una fila por pista: F*, scores frente a I* e I*_H1, (i)(ii)(iii) y veredicto."""
    crit = leer_criterio(salida)
    if crit is None:
        return None
    cols = ["pista", "f_star", "score_f_star_oos", "i_star", "score_i_star_oos", "f_star_en_mcs",
            "score_f_star_H2", "i_star_h1", "score_i_star_h1_H2", "cumple_i", "cumple_ii", "cumple_iii",
            "cumple_pista", "veredicto"]
    return crit[[c for c in cols if c in crit]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("orden", choices=["ejecutar", "leer"])
    parser.add_argument("--pista", nargs="+", default=list(PISTAS), choices=["A", "B", "a", "b"])
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    if args.orden == "ejecutar":
        ejecutar([p.upper() for p in args.pista], args.salida, log=lambda m: print(m, flush=True))
    r = resumen(args.salida)
    if r is None:
        print(f"Faltan resultados en {args.salida} (python -m regimenes.fusion.fijas ejecutar).")
        return
    print(r.to_string(index=False))
    print(f"\nVeredicto: {r['veredicto'].iloc[0]}")


if __name__ == "__main__":
    main()
