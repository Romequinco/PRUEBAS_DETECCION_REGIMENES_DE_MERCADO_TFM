"""Veredicto de admision por niveles (``validacion.niveles``) con umbrales fijados a priori.

Lee las tablas largas de las funciones de validacion (contrato en ``fidelidad``)
y aplica ``validacion.umbrales`` y ``validacion.reglas`` de ``configs/sinteticos.yaml``:

- una fila de fidelidad PASA si ``en_banda`` es cierto o su ``cociente`` cae en
  ``banda_cociente``; un criterio pasa si pasan TODAS sus filas clave
  (``reglas.marginal`` y ``reglas.dependencia``). En ``curtosis_condicional_tray``
  el cociente es NaN y decide ``en_banda`` (el REAL dentro de la banda por
  trayectoria del sintetico); la referencia de igual a igual de la dependencia
  ya viene aplicada en la tabla de ``fidelidad_dependencia``;
- ``correlaciones`` (informativo): frobenius por regimen <= ``correlaciones_factor_p95``
  x p95 de la distancia real frente a remuestreo del real;
- ``condicionamiento``: cociente de ``separacion_vol`` en ``banda_cociente``;
- ``discriminador``: AUC global (todas las columnas) <= ``discriminador_auc_max``;
- ``memorizacion``: ``cociente_nn`` en el espacio de las columnas modeladas,
  global y en crisis, >= ``memorizacion_cociente_min``;
- ``utilidad`` (informativo): por detector, TSTR >= TRTR - ``tstr_margen_trtr`` y
  TSTR > p95 del nulo aleatorio; pasa si lo cumplen MAS de
  ``tstr_fraccion_detectores`` de los detectores.

Los umbrales se fijaron a priori; los cambios posteriores de reglas y niveles
estan declarados y fechados en los comentarios de ``validacion`` del yaml.

Todos los criterios se calculan y se publican; solo deciden los que figuran en
``validacion.niveles`` (``apto_laboratorio``, ``laboratorio_condicionado``,
``apto_aumento``), cada nivel
como conjuncion de sus criterios. Los que no figuran en ningun nivel (hoy
``correlaciones`` y ``utilidad``) son informativos. Un criterio sin tabla queda
``<NA>`` y hace ``<NA>`` el nivel que lo use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIMENES = ("calma", "crisis")


def _pasa_fila(tabla: pd.DataFrame, banda: tuple[float, float]) -> pd.Series:
    en_banda = tabla["en_banda"].astype("boolean").fillna(False).astype(bool)
    cociente = tabla["cociente"].astype(float)
    return en_banda | cociente.between(*banda)


def _evaluar_filas(tabla: pd.DataFrame, mascara: pd.Series, banda, esperadas: int) -> tuple[bool, list[str]]:
    clave = tabla[mascara]
    if len(clave) < esperadas:
        raise ValueError(f"Faltan filas clave: {len(clave)} de {esperadas} esperadas.")
    pasa = _pasa_fila(clave, banda)
    fallos = [f"{r.regimen}/{r.columna}/{r.metrica}" for r in clave[~pasa].itertuples()]
    return not fallos, fallos


def _marginal(t: pd.DataFrame, reglas: dict, banda) -> tuple[bool, list[str]]:
    reg = t["regimen"].isin(REGIMENES)
    m = pd.Series(False, index=t.index)
    n = 0
    if reglas.get("desviacion_modeladas", True):
        sel = reg & (t["tipo"] == "modelada") & (t["metrica"] == "desviacion")
        m |= sel
        # esperadas: toda columna modelada presente en la tabla, en cada regimen
        n += t.loc[t["tipo"] == "modelada", "columna"].nunique() * len(REGIMENES)
    colas = list(reglas.get("colas_ret", []))
    sel = reg & (t["columna"] == "SP500_ret") & t["metrica"].isin(colas)
    m |= sel
    return _evaluar_filas(t, m, banda, n + len(colas) * len(REGIMENES))


def _dependencia(t: pd.DataFrame, reglas: dict, banda) -> tuple[bool, list[str]]:
    reg = t["regimen"].isin(REGIMENES)
    ret, pers = list(reglas.get("ret", [])), list(reglas.get("persistentes", []))
    m = reg & (((t["columna"] == "SP500_ret") & t["metrica"].isin(ret))
               | ((t["columna"] == "persistentes") & t["metrica"].isin(pers)))
    return _evaluar_filas(t, m, banda, (len(ret) + len(pers)) * len(REGIMENES))


def _correlaciones(t: pd.DataFrame, factor: float) -> tuple[bool, list[str]]:
    clave = t[t["regimen"].isin(REGIMENES) & (t["metrica"] == "frobenius")]
    if len(clave) < len(REGIMENES):
        raise ValueError("Faltan filas frobenius por regimen.")
    pasa = clave["sintetico"] <= factor * clave["banda_sup"]
    return bool(pasa.all()), [f"{r}/frobenius" for r in clave.loc[~pasa, "regimen"]]


def _condicionamiento(t: pd.DataFrame, metricas: list[str], banda) -> tuple[bool, list[str]]:
    clave = t[t["metrica"].isin(metricas)]
    if len(clave) < len(metricas):
        raise ValueError("Faltan filas de condicionamiento.")
    pasa = clave["cociente"].astype(float).between(*banda)
    return bool(pasa.all()), [f"{r.regimen}/{r.metrica}" for r in clave[~pasa].itertuples()]


def _discriminador(t: pd.DataFrame, auc_max: float) -> tuple[bool, list[str]]:
    fila = t[(t["regimen"] == "todos") & (t["metrica"] == "auc")]
    if len(fila) != 1:
        raise ValueError("El discriminador debe tener una fila auc global.")
    auc = float(fila["sintetico"].iloc[0])
    if not np.isfinite(auc):
        return pd.NA, ["auc no finito"]
    nota = [f"auc global {auc:.3f} < 0,5: indicio de copia (ver memorizacion)"] if auc < 0.45 else []
    return auc <= auc_max, ([] if auc <= auc_max else [f"auc global {auc:.3f}"]) + nota


def _memorizacion(t: pd.DataFrame, espacio: str, minimo: float) -> tuple[bool, list[str]]:
    clave = t[(t["columna"] == espacio) & (t["metrica"] == "cociente_nn") & t["regimen"].isin(["todos", "crisis"])]
    if len(clave) != 2:
        raise ValueError(f"memorizacion: faltan filas cociente_nn en el espacio {espacio!r}.")
    cociente = clave["cociente"].astype(float)
    if not np.isfinite(cociente).all():
        return pd.NA, ["cociente_nn no finito"]
    pasa = cociente >= minimo
    return bool(pasa.all()), [f"{r.regimen}: {r.cociente:.3f}" for r in clave[~pasa].itertuples()]


def _utilidad(t: pd.DataFrame, margen: float, fraccion: float) -> tuple[bool, list[str]]:
    fallos, n_ok, n = [], 0, 0
    for det, d in t.groupby("detector", sort=True):
        score = d[d["metrica"] == "score_deteccion"]
        nulo = d[d["metrica"] == "score_nulo_p95"]
        if len(score) != 1 or len(nulo) != 1:
            raise ValueError(f"utilidad: filas incompletas para {det}.")
        tstr, trtr = float(score["sintetico"].iloc[0]), float(score["real"].iloc[0])
        p95 = float(nulo["sintetico"].iloc[0])
        ok = bool(np.isfinite(tstr) and tstr >= trtr - margen and tstr > p95)
        n += 1
        n_ok += ok
        if not ok:
            fallos.append(f"{det}: TSTR {tstr:.3f} / TRTR {trtr:.3f} / nulo {p95:.3f}")
    if n == 0:
        raise ValueError("utilidad: tabla vacia.")
    return n_ok / n > fraccion, fallos


def veredicto(tablas: dict[str, dict[str, pd.DataFrame]], umbrales: dict, *, reglas: dict | None = None,
              niveles: dict | None = None, espacio_memorizacion: str = "modeladas") -> pd.DataFrame:
    """Tabla generador x criterio (bool) mas una columna por nivel de ``niveles``.

    ``tablas[generador][dimension]`` son las salidas de las funciones de validacion
    de una pista, con dimension en {marginal, dependencia, correlaciones,
    condicionamiento, discriminador, memorizacion, utilidad}; ``umbrales``,
    ``reglas`` y ``niveles`` son las secciones homonimas de ``validacion`` en
    ``configs/sinteticos.yaml``. Las columnas
    ``fallos_<criterio>`` listan las filas clave que no pasan (texto para el notebook).
    """
    if reglas is None or niveles is None:
        raise ValueError("Pasa `reglas` y `niveles` (configs/sinteticos.yaml, seccion validacion).")
    banda = tuple(float(x) for x in umbrales["banda_cociente"])
    criterios = {
        "marginal": lambda t: _marginal(t, reglas["marginal"], banda),
        "dependencia": lambda t: _dependencia(t, reglas["dependencia"], banda),
        "correlaciones": lambda t: _correlaciones(t, float(reglas["correlaciones_factor_p95"])),
        "condicionamiento": lambda t: _condicionamiento(t, list(reglas["condicionamiento"]), banda),
        "discriminador": lambda t: _discriminador(t, float(umbrales["discriminador_auc_max"])),
        "memorizacion": lambda t: _memorizacion(t, espacio_memorizacion, float(umbrales["memorizacion_cociente_min"])),
        "utilidad": lambda t: _utilidad(t, float(umbrales["tstr_margen_trtr"]), float(umbrales["tstr_fraccion_detectores"])),
    }
    filas = []
    for gen, dims in tablas.items():
        fila: dict = {"generador": gen}
        for crit, fn in criterios.items():
            tabla = dims.get(crit)
            if tabla is None or tabla.empty:
                fila[crit], fila[f"fallos_{crit}"] = pd.NA, "sin tabla"
                continue
            ok, fallos = fn(tabla)
            fila[crit], fila[f"fallos_{crit}"] = (pd.NA if ok is pd.NA else bool(ok)), "; ".join(fallos)
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("generador")
    for crit in criterios:
        tabla[crit] = tabla[crit].astype("boolean")
    for nivel, lista in niveles.items():
        # conjuncion de Kleene: False si algun criterio es False; <NA> si falta alguno y ninguno es False
        nivel_ok = pd.Series(pd.array([True] * len(tabla), dtype="boolean"), index=tabla.index)
        for crit in lista:
            nivel_ok = nivel_ok & tabla[crit]
        tabla[nivel] = nivel_ok
    orden = list(criterios) + list(niveles) + [f"fallos_{c}" for c in criterios]
    return tabla[orden]
