"""Generadores reentrenados por ancla anual, sin fuga (pre-registro ``aumento.anclas``).

Reglas (fijadas antes de ejecutar nada):

- Candidatas: ultima sesion de cada ano del panel de la pista.
- Para un pliegue cuyo train acaba en ``t``, ``ancla(t)`` = ultima candidata
  VALIDA ``tau <= t - margen_reconocimiento`` sesiones (:func:`limite_reconocimiento`
  y :func:`ancla_para`). Ninguna ancla valida -> el pliegue va sin aumento.
- Ajuste del ancla ``tau`` (:func:`corte_ajuste`): datos ``<= tau`` y etiquetas
  de crisis solo de ventanas cuyo suelo es ``<= tau`` (reconocidas). Si ``tau``
  cae dentro de una ventana aun abierta (``pico <= tau < suelo``), el ajuste se
  recorta a la sesion anterior al pico: el maximo previo ya es observable en
  ``t`` y la etiqueta de esos dias (calma) no depende del futuro.
- Ancla valida: al menos ``min_episodios_crisis`` episodios y ``min_dias_crisis``
  dias de crisis en el ajuste, el ajuste no falla y, salvo para el control, el
  generador pasa en ese ancla la memorizacion y el condicionamiento del 16 con
  los mismos umbrales (``validacion.umbrales``).

Salidas: ``data/sinteticos/aumento/generadores/<pista>/<fuente>/<tau>/generador.pkl``
(gitignored) y el registro ``results/sinteticos/aumento/anclas_<pista>.csv``.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from regimenes.sinteticos.aumento.config import (
    DIR_DATOS,
    DIR_RESULTADOS,
    config_aumento,
    config_completa,
    semilla,
)
from regimenes.sinteticos.datos import COL_REGIMEN, rachas, ventanas_crisis

OPCIONES_VALIDACION = {"inicial": "estacionaria", "duraciones": "empiricas"}


# --------------------------------------------------------------------------- fechas
def candidatas(indice: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ultima sesion de cada ano natural de ``indice``."""
    indice = pd.DatetimeIndex(indice).sort_values()
    ultimas = pd.Series(indice, index=indice).groupby(indice.year).max()
    return pd.DatetimeIndex(ultimas.to_numpy()).astype("datetime64[ns]")


def corte_ajuste(pista: str, tau: pd.Timestamp) -> tuple[pd.Timestamp, list[str], str | None]:
    """``(fin_ajuste, crisis_reconocidas, ventana_abierta)`` del ancla ``tau``.

    ``fin_ajuste`` es ``tau`` o, si ``tau`` cae dentro de una ventana abierta, el
    dia anterior a su pico (el ajuste toma las sesiones ``< pico``).
    """
    tau = pd.Timestamp(tau)
    fin = tau
    reconocidas, abierta = [], None
    for nombre, (pico, suelo) in ventanas_crisis(pista).items():
        if suelo <= tau:
            reconocidas.append(nombre)
        elif pico <= tau:
            abierta = nombre
            fin = min(fin, pico - pd.Timedelta(days=1))
    return fin, reconocidas, abierta


def limite_reconocimiento(indice_train: pd.DatetimeIndex, margen: int) -> pd.Timestamp | None:
    """Fecha ``t - margen`` sesiones, con ``t`` el ultimo dia del train (None si no cabe)."""
    if len(indice_train) <= margen:
        return None
    return pd.Timestamp(indice_train[-1 - int(margen)])


def ancla_para(validas: pd.DatetimeIndex, limite: pd.Timestamp | None) -> pd.Timestamp | None:
    """Ultima ancla valida ``<= limite`` (``None`` si no hay ninguna)."""
    if limite is None or not len(validas):
        return None
    posibles = validas[validas <= limite]
    return pd.Timestamp(posibles.max()) if len(posibles) else None


# --------------------------------------------------------------------------- rutas y registro
def ruta_generador(pista: str, fuente: str, tau, base: Path | None = None) -> Path:
    base = DIR_DATOS if base is None else Path(base)
    return base / "generadores" / pista.upper() / fuente / pd.Timestamp(tau).strftime("%Y-%m-%d")


def ruta_registro(pista: str, base: Path | None = None) -> Path:
    base = DIR_RESULTADOS if base is None else Path(base)
    return base / f"anclas_{pista.upper()}.csv"


def leer_registro(pista: str, base: Path | None = None) -> pd.DataFrame:
    """Registro de anclas de la pista (una fila por fuente y candidata)."""
    tabla = pd.read_csv(ruta_registro(pista, base), parse_dates=["tau", "fin_ajuste"])
    tabla["valida"] = tabla["valida"].astype(bool)
    return tabla


def anclas_validas(registro: pd.DataFrame, fuente: str) -> pd.DatetimeIndex:
    """Anclas validas de ``fuente`` (ordenadas)."""
    parte = registro[(registro["fuente"] == fuente) & registro["valida"]]
    return pd.DatetimeIndex(parte["tau"]).sort_values()


def cargar_generador(pista: str, fuente: str, tau, base: Path | None = None):
    from regimenes.sinteticos.registry import clase

    return clase(fuente).cargar(ruta_generador(pista, fuente, tau, base))


# --------------------------------------------------------------------------- validacion
def validar_generador(gen, panel: pd.DataFrame, reg: pd.Series, cfg_total: dict, n_tray: int,
                      semilla_val: int) -> dict[str, Any]:
    """Memorizacion y condicionamiento del 16 (mismos umbrales) en el ancla."""
    from regimenes.sinteticos.validacion.fidelidad import condicionamiento
    from regimenes.sinteticos.validacion.memorizacion import memorizacion
    from regimenes.sinteticos.validacion.veredicto import _condicionamiento, _memorizacion

    val = cfg_total["validacion"]
    tray = gen.sample(int(n_tray), int(cfg_total["muestreo"]["length"]), dict(OPCIONES_VALIDACION),
                      random_state=semilla_val)
    t_mem = memorizacion(panel, tray, regimen_real=reg,
                         largo_ventana=int(val["memorizacion"]["largo_ventana"]),
                         exclusion=int(val["memorizacion"]["exclusion"]), semilla=int(val["semilla"]))
    ok_mem, fallos_mem = _memorizacion(t_mem, val["memorizacion"]["espacio_veredicto"],
                                       float(val["umbrales"]["memorizacion_cociente_min"]))
    t_cond = condicionamiento(panel, tray, regimen_real=reg, semilla=int(val["semilla"]),
                              n_boot=int(val["n_boot"]))
    ok_cond, fallos_cond = _condicionamiento(t_cond, list(val["reglas"]["condicionamiento"]),
                                             tuple(val["umbrales"]["banda_cociente"]))
    memo = t_mem[(t_mem["columna"] == val["memorizacion"]["espacio_veredicto"])
                 & (t_mem["metrica"] == "cociente_nn")].set_index("regimen")["cociente"]
    sep = t_cond[t_cond["metrica"] == "separacion_vol"]["cociente"]
    return {
        "memorizacion": bool(ok_mem) if ok_mem is not pd.NA else False,
        "condicionamiento": bool(ok_cond),
        "memo_cociente_todos": float(memo.get("todos", np.nan)),
        "memo_cociente_crisis": float(memo.get("crisis", np.nan)),
        "separacion_vol_cociente": float(sep.iloc[0]) if len(sep) else np.nan,
        "fallos_validacion": "; ".join([*fallos_mem, *fallos_cond]),
    }


# --------------------------------------------------------------------------- un ancla
def _guardar_atomico(gen, destino: Path) -> None:
    """``gen.guardar`` en un directorio temporal y ``replace`` del fichero (un corte no deja un pkl a medias)."""
    import shutil

    from regimenes.sinteticos.comun import FICHERO_MODELO

    tmp = destino.with_name(destino.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    gen.guardar(tmp)
    destino.mkdir(parents=True, exist_ok=True)
    (tmp / FICHERO_MODELO).replace(destino / FICHERO_MODELO)
    shutil.rmtree(tmp, ignore_errors=True)


def _features(cfg_total: dict, pista: str) -> list[str] | None:
    """Columnas de ajuste de la pista, las mismas que en el 15 (vacio = nucleo + SP500_ret)."""
    for clave in ("entrenamiento", "entrenamiento_secundario"):
        if cfg_total[clave]["pista"].upper() == pista:
            return list(cfg_total[clave].get("features") or []) or None
    return None


def ajustar_ancla(pista: str, fuente: str, tau, *, cfg_total: dict | None = None,
                  historia: pd.Series | None = None, base: Path | None = None) -> dict[str, Any]:
    """Ajusta (y valida) el generador ``fuente`` en el ancla ``tau``; devuelve su fila de registro.

    Guarda el generador solo si el ancla es valida. Nunca lanza: un fallo queda en
    la fila (``valida=False`` y ``motivo``).
    """
    from regimenes.sinteticos.datos import cargar_entrenamiento
    from regimenes.sinteticos.espacio import cargar_historia_sp500
    from regimenes.sinteticos.registry import crear

    cfg_total = config_completa(cfg_total)
    cfg = config_aumento(cfg_total)
    reglas = cfg["anclas"]
    pista, tau = pista.upper(), pd.Timestamp(tau)
    fin, reconocidas, abierta = corte_ajuste(pista, tau)
    fila: dict[str, Any] = {
        "pista": pista, "fuente": fuente, "tau": tau.date().isoformat(),
        "fin_ajuste": fin.date().isoformat(), "ventana_abierta": abierta or "",
        "n_crisis_reconocidas": len(reconocidas), "valida": False, "motivo": "",
    }
    inicio = time.perf_counter()
    try:
        panel, reg = cargar_entrenamiento(pista, fin, _features(cfg_total, pista))
        episodios = rachas(reg)
        episodios = episodios[episodios["regimen"] == 1]
        # el primer y el ultimo tramo del ajuste no pueden ser crisis censuradas: las ventanas
        # reconocidas acaban <= tau y la abierta se recorta antes de su pico
        fila.update({"n_filas": len(panel), "inicio_ajuste": panel.index[0].date().isoformat(),
                     "n_episodios": int(len(episodios)), "dias_crisis": int(reg.sum())})
        if len(episodios) < int(reglas["min_episodios_crisis"]) or reg.sum() < int(reglas["min_dias_crisis"]):
            fila["motivo"] = "pocas crisis en el ajuste"
            return fila
        historia = cargar_historia_sp500() if historia is None else historia
        params = dict(cfg_total["generadores"][fuente]["params"])
        gen = crear(fuente, random_state=int(cfg["semilla"]), historia_sp500=historia, **params)
        gen.fit(panel, reg)
        fila["segundos_ajuste"] = time.perf_counter() - inicio
        if fuente != cfg["control"] and "validar" in reglas:
            t0 = time.perf_counter()
            fila.update(validar_generador(gen, panel, reg, cfg_total,
                                          int(reglas.get("validar_n_trayectorias", 20)),
                                          semilla("validacion", pista, fuente, fila["tau"], base=int(cfg["semilla"]))))
            fila["segundos_validacion"] = time.perf_counter() - t0
            fallan = [c for c in reglas["validar"] if not fila.get(c, False)]
            if fallan:
                fila["motivo"] = "no pasa " + " y ".join(fallan)
                return fila
        _guardar_atomico(gen, ruta_generador(pista, fuente, tau, base))
        fila["valida"] = True
    except Exception as exc:  # noqa: BLE001 - el registro debe quedar completo
        fila["motivo"] = f"error: {type(exc).__name__}: {exc}"
        fila["traza"] = traceback.format_exc(limit=3)
    finally:
        fila["segundos_total"] = time.perf_counter() - inicio
    return fila


def _ajustar_seguro(args) -> dict[str, Any]:
    pista, fuente, tau, base = args
    from regimenes.sinteticos.aumento.config import un_hilo
    from regimenes.sinteticos.espacio import cargar_historia_sp500

    with un_hilo():
        return ajustar_ancla(pista, fuente, tau, historia=cargar_historia_sp500(), base=base)


# --------------------------------------------------------------------------- todas las anclas
def candidatas_pista(pista: str) -> pd.DatetimeIndex:
    """Candidatas de la pista hasta la ultima que puede usar algun pliegue."""
    from regimenes.benchmark.ejecucion import common_track_index

    cfg = config_aumento()
    indice = common_track_index(pista)
    ultima = limite_reconocimiento(indice, int(cfg["anclas"]["margen_reconocimiento"]))
    todas = candidatas(indice)
    return todas[todas <= ultima]


def _escribir_registro(filas: list[dict], ruta: Path) -> pd.DataFrame:
    tabla = pd.DataFrame(filas)
    tabla["tau"] = pd.to_datetime(tabla["tau"])
    tabla["fin_ajuste"] = pd.to_datetime(tabla["fin_ajuste"])
    tabla = tabla.sort_values(["fuente", "tau"]).reset_index(drop=True)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(f".{ruta.stem}.tmp.csv")
    tabla.drop(columns=["traza"], errors="ignore").to_csv(tmp, index=False)
    tmp.replace(ruta)
    return tabla


def preparar_anclas(pistas: Iterable[str], fuentes_: Iterable[str] | None = None, *, n_jobs: int = 1,
                    base: Path | None = None, base_resultados: Path | None = None,
                    progreso: bool = True) -> dict[str, pd.DataFrame]:
    """Ajusta las anclas que falten y actualiza el registro tras CADA ancla (reanudable).

    Se rehace un ancla si no esta en el registro, si su ajuste fallo con error o
    si su ``huella`` (``config.huella_generador``) ya no coincide.
    """
    from regimenes.sinteticos.aumento import _pool
    from regimenes.sinteticos.aumento.config import fuentes as todas_fuentes, huella_generador

    cfg_total = config_completa()
    salida = {}
    for pista in pistas:
        pista = pista.upper()
        ruta = ruta_registro(pista, base_resultados)
        previo = leer_registro(pista, base_resultados) if ruta.exists() else pd.DataFrame()
        vigentes: dict = {}
        for r in ([] if previo.empty else previo.to_dict("records")):
            clave = (r["fuente"], pd.Timestamp(r["tau"]))
            error = str(r.get("motivo", "")).startswith("error")
            if not error and r.get("huella") == huella_generador(pista, r["fuente"], r["tau"], cfg_total):
                vigentes[clave] = r
        lista = [(pista, f, tau, base) for f in (fuentes_ or todas_fuentes())
                 for tau in candidatas_pista(pista) if (f, tau) not in vigentes]
        filas = dict(vigentes)
        hechos = [0]

        def al_terminar(a, fila, filas=filas, hechos=hechos, total=len(lista), pista=pista, ruta=ruta):
            _, f, tau, _ = a
            fila = {"pista": pista, "fuente": f, "tau": pd.Timestamp(tau).date().isoformat(),
                    "fin_ajuste": corte_ajuste(pista, tau)[0].date().isoformat(), "valida": False, **fila}
            fila["huella"] = huella_generador(pista, f, tau, cfg_total)
            filas[(f, pd.Timestamp(tau))] = fila
            _escribir_registro(list(filas.values()), ruta)
            hechos[0] += 1
            if progreso:
                print(f"[anclas {pista}] {hechos[0]}/{total} {f} {pd.Timestamp(tau).date()} "
                      f"{'valida' if fila.get('valida') else fila.get('motivo', '')}", flush=True)

        _pool.mapa(_ajustar_seguro, lista, n_jobs, al_terminar)
        salida[pista] = _escribir_registro(list(filas.values()), ruta) if filas else pd.DataFrame()
    return salida


__all__ = [
    "COL_REGIMEN", "ajustar_ancla", "ancla_para", "anclas_validas", "candidatas", "candidatas_pista",
    "cargar_generador", "corte_ajuste", "leer_registro", "limite_reconocimiento", "preparar_anclas",
    "ruta_generador", "ruta_registro", "validar_generador",
]
