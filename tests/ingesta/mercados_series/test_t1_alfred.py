"""Tests de los vintages exactos de ALFRED del Tema 1 (bloque 3): ``fuentes.alfred``.

Sin red: respuestas JSON SINTÉTICAS con el formato real de ``series/observations`` (``output_type=1``,
tiempo real completo, paginado por ``offset``) y un transporte falso. Los números que imitan casos
reales (UNRATE de abril de 2020, IPC de diciembre de 2019) son fixtures sintéticas, no datos de
FRED. ``@pytest.mark.live`` descarga una serie corta (solo con ``INGESTA_LIVE=1``); el de
``@pytest.mark.datos`` lee el archivo crudo local y se salta si no existe.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import catalogo
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, validar_series
from ingesta.mercados_series.fuentes import alfred as a
from ingesta.mercados_series.fuentes import fred as f
from ingesta.mercados_series.timegate import as_of_series

SECRETO = "abcdef0123456789abcdef0123456789"
ABIERTO = "9999-12-31"


# --- Respuestas sintéticas con el formato de la API ---------------------------------------------
def _obs(fecha: str, valor: str, rs: str, re: str = ABIERTO) -> dict:
    return {"realtime_start": rs, "realtime_end": re, "date": fecha, "value": valor}


def _respuesta(observaciones: list[dict], offset: int, count: int, limite: int = 100000) -> dict:
    return {"realtime_start": "1776-07-04", "realtime_end": ABIERTO, "observation_start": "1600-01-01",
            "observation_end": ABIERTO, "units": "lin", "output_type": 1, "file_type": "json",
            "order_by": "observation_date", "sort_order": "asc", "count": count, "offset": offset,
            "limit": limite, "observations": observaciones}


class Falso:
    """Transporte falso: ``series/observations`` de ``series[series_id]`` en páginas de ``pagina`` filas."""

    def __init__(self, series: dict[str, list[dict]], pagina: int = 100000, fallar: set[str] | None = None):
        self.series = series
        self.pagina = pagina
        self.fallar = fallar or set()
        self.llamadas: list[dict] = []

    def __call__(self, url: str, params: dict) -> dict:
        assert params["api_key"] == SECRETO and params["file_type"] == "json"
        assert url.endswith("/series/observations")
        assert params["realtime_start"] == "1776-07-04" and params["realtime_end"] == ABIERTO
        assert int(params["output_type"]) == 1
        self.llamadas.append(dict(params))
        sid = params["series_id"]
        if sid in self.fallar:
            raise RuntimeError(f"500 Server Error for url: https://x/?series_id={sid}&api_key={params['api_key']}")
        filas = self.series[sid]
        off = int(params.get("offset", 0))
        return _respuesta(filas[off:off + self.pagina], off, len(filas))


def _cliente(tmp_path, series: dict, hoy: str = "2026-10-13", **kw) -> tuple[f.ClienteFred, Falso]:
    falso = Falso(series, **kw)
    return a.cliente_alfred(clave=SECRETO, transporte=falso, carpeta=tmp_path, hoy=hoy), falso


def _entrada(sid: str, *, publicador: str = "bls", frecuencia: str = "mensual", previa: dict | None = None,
             unidad: str = "índice", **extra) -> dict:
    """Entrada mínima del catálogo (normalizada como la de ``cargar_catalogo``)."""
    e = {"alias": [], "nombre": sid, "fuente": "fred_alfred", "id_origen": sid.split(":", 1)[1],
         "fichero_fase1": [], "frecuencia": frecuencia, "unidad": unidad, "divisa": None,
         "metodo": "vintage_exacto", "publicador": publicador, "revisable": True, "rol_fase1": None,
         "licencia": "sintética", "estado": "activa", "verificado": False, "sucesor": None, "duplicado_de": None,
         "lag": None, "fin_periodo": None, "lag_habiles": None, "lag_n_habiles": None, "regla_previa_alfred": None,
         "proyecciones": None, "fuente_viva": "fred", "reestimada": False, "cobertura_variable": False,
         "unidad_por_vintage": False, "coherente_con": [], "notas": []}
    if previa:
        e["regla_previa_alfred"] = {"metodo": "lag_fijo", "lag_clave": "SINT", "lag_habiles": None,
                                    "lag_n_habiles": None, **previa}
    e.update(extra)
    return e


PREVIA_MES = {"lag": "1M1D", "fin_periodo": "mes", "primer_vintage": "2000-03-10"}


def _tabla(sid: str, filas: list[dict], entrada: dict) -> tuple[pd.DataFrame, dict]:
    t, info = a.tabla_serie_alfred(sid, a.parsear_vintages(filas), entrada)
    return validar_series(t), info


def _valor(tabla: pd.DataFrame, sid: str, fecha: str, ref: str) -> float | None:
    v = as_of_series(tabla, sid, fecha)
    fila = v[v["fecha_referencia"] == pd.Timestamp(ref)]
    return None if fila.empty else float(fila["valor"].iloc[0])


# --- Parseo ------------------------------------------------------------------------------------
def test_t1_alfred_parseo_vectorizado_igual_que_fred():
    filas = [_obs("2020-02-01", "1.25", "2020-03-10", "2020-04-09"), _obs("2020-01-01", ".", "2020-02-10"),
             _obs("2020-02-01", "1.30", "2020-04-10"), _obs("2020-03-01", "", "2020-04-10"),
             {"date": "2020-04-01", "realtime_start": "2020-05-08", "realtime_end": ABIERTO}]
    pd.testing.assert_frame_equal(a.parsear_vintages(filas), f.parsear_observaciones(filas))
    pd.testing.assert_frame_equal(a.parsear_vintages([]), f.parsear_observaciones([]))
    with pytest.raises(ValueError):
        a.parsear_vintages([_obs("2020-01-01", "n/d", "2020-02-10")])


# --- Hora del release ---------------------------------------------------------------------------
def test_t1_alfred_hora_publicacion_por_release():
    assert a.hora_publicacion({"publicador": "bls"})["mismo_dia"]                 # 08:30
    assert a.hora_publicacion({"publicador": "fed_g17"})["hora"] == "09:15"
    assert not a.hora_publicacion({"publicador": "fed_h8"})["mismo_dia"]          # 16:15 + 30 min
    assert not a.hora_publicacion({"publicador": "fred_calculo"})["mismo_dia"]    # conservador 23:59
    with pytest.raises(ValueError, match="publicador"):
        a.hora_publicacion({})


def test_t1_alfred_unrate_abril_2020_visible_el_8_de_mayo():
    sid = "fred:XUNRATE"
    filas = [_obs("2020-03-01", "4.4", "2020-04-03", "2020-05-07"), _obs("2020-03-01", "4.5", "2020-05-08"),
             _obs("2020-04-01", "14.0", "2020-05-08")]
    t, _ = _tabla(sid, filas, _entrada(sid))
    fila = t[t["fecha_referencia"] == pd.Timestamp("2020-04-01")].iloc[0]
    assert fila["fecha_publicacion"] == pd.Timestamp("2020-05-08") and fila["vintage"] == "2020-05-08"
    assert fila["disponible_utc"] == pd.Timestamp("2020-05-08 12:30", tz="UTC")       # 08:30 EDT
    assert _valor(t, sid, "2020-05-07", "2020-04-01") is None
    assert _valor(t, sid, "2020-05-08", "2020-04-01") == pytest.approx(14.0)
    assert _valor(t, sid, "2020-05-07", "2020-03-01") == pytest.approx(4.4)            # la revisión, aún no
    assert _valor(t, sid, "2020-05-08", "2020-03-01") == pytest.approx(4.5)


def test_t1_alfred_unrate_catalogo_real_mismo_dia():
    e = catalogo.cargar_catalogo()["fred:UNRATE"]
    filas = [_obs("2020-04-01", "14.0", "2020-05-08")]
    t, info = a.tabla_serie_alfred("fred:UNRATE", a.parsear_vintages(filas), e)
    assert t["fecha_publicacion"].iloc[0] == pd.Timestamp("2020-05-08")
    assert info["previas"] == 0                                    # 2020-04 no es anterior a ALFRED: llega después


def test_t1_alfred_ipc_diciembre_2019_revision_de_febrero():
    sid = "fred:XCPI"
    filas = [_obs("2019-11-01", "257.900", "2019-12-11"),
             _obs("2019-12-01", "258.501", "2020-01-14", "2020-02-10"),
             _obs("2019-12-01", "258.444", "2020-02-11")]
    t, info = _tabla(sid, filas, _entrada(sid))
    assert info["n_vintages"] == 3 and info["previas"] == 0
    assert _valor(t, sid, "2020-01-13", "2019-12-01") is None
    assert _valor(t, sid, "2020-01-14", "2019-12-01") == pytest.approx(258.501)
    assert _valor(t, sid, "2020-02-10", "2019-12-01") == pytest.approx(258.501)
    assert _valor(t, sid, "2020-02-11", "2019-12-01") == pytest.approx(258.444)
    dic = t[t["fecha_referencia"] == pd.Timestamp("2019-12-01")]
    assert list(dic["vintage"]) == ["2020-01-14", "2020-02-11"]
    assert set(dic["metodo_publicacion"]) == {"vintage_exacto"}


# --- Descarga: paginación, caché y reanudación --------------------------------------------------
def _serie_larga(n: int = 7) -> list[dict]:
    return [_obs(f"2020-{m:02d}-01", f"{m}.0", f"2020-{m + 1:02d}-10") for m in range(1, n + 1)]


def test_t1_alfred_paginacion_y_cache_por_pagina(tmp_path):
    filas = _serie_larga(7)
    cli, falso = _cliente(tmp_path, {"XPAG": filas}, pagina=3)
    r = a.descargar_vintages(["fred:XPAG"], cliente=cli, cat={"fred:XPAG": _entrada("fred:XPAG")})
    assert [int(p["offset"]) for p in falso.llamadas] == [0, 3, 6]
    assert r.loc[0, "estado"] == "descargada" and r.loc[0, "filas"] == 7 and r.loc[0, "peticiones"] == 3
    paginas = list((tmp_path / "series_observations" / "XPAG").glob("20261013_*.json"))
    assert len(paginas) == 3
    assert all(SECRETO not in p.read_text(encoding="utf-8") for p in paginas)
    dia, obs = a.archivo_completo(cli, "XPAG")
    assert dia == "20261013" and [o["date"] for o in obs] == [o["date"] for o in filas]


def test_t1_alfred_reanudable_otro_dia_sin_red(tmp_path):
    cat = {"fred:XPAG": _entrada("fred:XPAG")}
    cli, _ = _cliente(tmp_path, {"XPAG": _serie_larga(7)}, pagina=3)
    a.descargar_vintages(["fred:XPAG"], cliente=cli, cat=cat)
    cli2, falso2 = _cliente(tmp_path, {"XPAG": _serie_larga(7)}, hoy="2026-10-20", pagina=3)
    r = a.descargar_vintages(["fred:XPAG"], cliente=cli2, cat=cat)
    assert r.loc[0, "estado"] == "archivo" and r.loc[0, "dia"] == "20261013" and not falso2.llamadas
    r = a.descargar_vintages(["fred:XPAG"], cliente=cli2, cat=cat, refrescar=True)
    assert r.loc[0, "estado"] == "descargada" and len(falso2.llamadas) == 3


def test_t1_alfred_descarga_a_medias_no_cuenta(tmp_path):
    cat = {"fred:XPAG": _entrada("fred:XPAG")}
    cli, _ = _cliente(tmp_path, {"XPAG": _serie_larga(7)}, pagina=3)
    a.descargar_vintages(["fred:XPAG"], cliente=cli, cat=cat)
    carpeta = tmp_path / "series_observations" / "XPAG"
    for p in sorted(carpeta.glob("*.json")):          # simula un corte: solo queda una página
        if '"offset": "0"' not in p.read_text(encoding="utf-8"):
            p.unlink()
    assert a.archivo_completo(cli, "XPAG") is None
    cli2, falso2 = _cliente(tmp_path, {"XPAG": _serie_larga(7)}, hoy="2026-10-14", pagina=3)
    assert a.descargar_vintages(["fred:XPAG"], cliente=cli2, cat=cat).loc[0, "estado"] == "descargada"
    assert len(falso2.llamadas) == 3


def test_t1_alfred_error_sin_clave_y_limite_de_tiempo(tmp_path):
    cat = {s: _entrada(s) for s in ("fred:XMAL", "fred:XBIEN")}
    cli, _ = _cliente(tmp_path, {"XBIEN": _serie_larga(2), "XMAL": []}, fallar={"XMAL"})
    r = a.descargar_vintages(["fred:XMAL", "fred:XBIEN"], cliente=cli, cat=cat)
    assert list(r["estado"]) == ["error", "descargada"]
    assert SECRETO not in r.loc[0, "error"]
    cli2, falso2 = _cliente(tmp_path / "otra", {"XBIEN": _serie_larga(2)})
    r = a.descargar_vintages(["fred:XBIEN"], cliente=cli2, cat=cat, limite_segundos=-1)
    assert list(r["estado"]) == ["pendiente"] and not falso2.llamadas


# --- '.' y periodos -----------------------------------------------------------------------------
def test_t1_alfred_punto_nulo_con_nota():
    sid = "fred:XHUECO"
    filas = [_obs("2025-09-01", "4.3", "2025-11-20"), _obs("2025-10-01", ".", "2025-12-16"),
             _obs("2025-11-01", "4.6", "2025-12-16")]
    t, info = _tabla(sid, filas, _entrada(sid))
    hueco = t[t["fecha_referencia"] == pd.Timestamp("2025-10-01")].iloc[0]
    assert np.isnan(hueco["valor"]) and a.NOTA_PUNTO_ALFRED in hueco["nota"]
    assert info["puntos"] == 1
    v = as_of_series(t, sid, "2026-01-05")
    assert v["valor"].isna().sum() == 1


def test_t1_alfred_diaria_quita_puntos_iniciales():
    sid = "fred:XDIA"
    filas = [_obs("2026-01-01", ".", "2026-01-02"), _obs("2026-01-02", "1.5", "2026-01-05"),
             _obs("2026-01-05", ".", "2026-01-06", "2026-01-06"), _obs("2026-01-05", "1.7", "2026-01-07")]
    t, _ = _tabla(sid, filas, _entrada(sid, frecuencia="diaria", publicador="fred_calculo"))
    assert list(t["fecha_referencia"].dt.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-05"]
    assert t["valor"].notna().all()


def test_t1_alfred_funde_periodos_con_el_mismo_valor():
    sid = "fred:XFUNDE"
    filas = [_obs("2020-01-01", "1.0", "2020-02-10", "2020-03-09"), _obs("2020-01-01", "1.0", "2020-03-10",
                                                                          "2020-04-09"),
             _obs("2020-01-01", "1.2", "2020-04-10")]
    t, info = _tabla(sid, filas, _entrada(sid))
    assert list(t["vintage"]) == ["2020-02-10", "2020-04-10"] and info["fundidas"] == 1


def test_t1_alfred_observacion_retirada_deja_fila_nula():
    sid = "fred:XRET"
    filas = [_obs("1990-01-01", "50.0", "2020-01-10", "2023-09-27"),           # sale del vintage de 2023
             _obs("2020-01-01", "100.0", "2020-02-10", "2023-09-27"), _obs("2020-01-01", "101.0", "2023-09-28")]
    t, info = _tabla(sid, filas, _entrada(sid, cobertura_variable=True))
    assert info["retiradas"] == 1
    assert _valor(t, sid, "2023-09-27", "1990-01-01") == pytest.approx(50.0)
    v = as_of_series(t, sid, "2023-09-28")
    fila = v[v["fecha_referencia"] == pd.Timestamp("1990-01-01")].iloc[0]
    assert np.isnan(fila["valor"]) and a.NOTA_RETIRADA in fila["nota"]


# --- Vintage de arranque: filas anteriores a ALFRED ---------------------------------------------
def test_t1_alfred_filas_anteriores_al_primer_vintage():
    sid = "fred:XPRE"
    v0 = "2000-03-10"
    filas = ([_obs(f"1999-{m:02d}-01", f"{m}.0", v0) for m in range(1, 13)]
             + [_obs("2000-01-01", "13.0", v0), _obs("2000-02-01", "14.0", v0, "2000-04-06"),
                _obs("2000-02-01", "14.5", "2000-04-07"), _obs("2000-03-01", "15.0", "2000-04-07")])
    e = _entrada(sid, previa=PREVIA_MES)
    t, info = _tabla(sid, filas, e)
    pre = t[t["metodo_publicacion"] == "lag_fijo"]
    # Regla 1M1D: 1999-01 → 1999-03-01 ... 2000-01 → 2000-03-01 (antes del 2000-03-10); 2000-02 → 2000-04-01 no.
    assert len(pre) == 13 and info["previas"] == 13
    ene = pre[pre["fecha_referencia"] == pd.Timestamp("1999-01-01")].iloc[0]
    assert ene["fecha_publicacion"] == pd.Timestamp("1999-03-01") and ene["vintage"] == "1999-03-01"
    assert pd.isna(ene["disponible_utc"]) and ene["valor"] == pytest.approx(1.0)
    assert ene["nota"] == f"{catalogo.NOTA_PREVIA_ALFRED} (2000-03-10)" == catalogo.nota_previa_alfred(e)
    feb = t[t["fecha_referencia"] == pd.Timestamp("2000-02-01")]
    assert list(feb["metodo_publicacion"]) == ["vintage_exacto", "vintage_exacto"]
    assert list(feb["vintage"]) == [v0, "2000-04-07"]
    # Una sola fila por periodo del arranque (el valor del primer vintage no se repite).
    assert t[t["fecha_referencia"] == pd.Timestamp("1999-06-01")].shape[0] == 1
    assert _valor(t, sid, "1999-08-31", "1999-06-01") == pytest.approx(6.0)
    assert _valor(t, sid, "2000-03-09", "2000-02-01") is None
    assert _valor(t, sid, "2000-03-10", "2000-02-01") == pytest.approx(14.0)
    # Formato de vintage único por serie y métodos admitidos por el catálogo.
    assert t["vintage"].str.len().nunique() == 1
    catalogo.validar_contra_catalogo(t, {sid: e})


def test_t1_alfred_previa_no_se_ve_antes_que_su_primer_vintage():
    """Regresión: la regla daba el 2000-01 el mismo día del primer vintage (2000-03-01), que sale a las
    16:30 (decisión del día siguiente); se veía un día antes de existir (CFSI, RECPROUSM156N)."""
    sid = "fred:XFRONTERA"
    v0 = "2000-03-01"
    filas = [_obs("1999-12-01", "1.0", v0), _obs("2000-01-01", "2.0", v0)]
    e = _entrada(sid, publicador="fed_h41", previa=dict(PREVIA_MES, primer_vintage=v0))
    t, info = _tabla(sid, filas, e)
    assert info["previas"] == 1
    dic = t[t["fecha_referencia"] == pd.Timestamp("1999-12-01")].iloc[0]
    assert dic["metodo_publicacion"] == "lag_fijo" and dic["fecha_publicacion"] == pd.Timestamp("2000-02-01")
    ene = t[t["fecha_referencia"] == pd.Timestamp("2000-01-01")].iloc[0]
    assert ene["metodo_publicacion"] == "vintage_exacto" and ene["fecha_publicacion"] == pd.Timestamp("2000-03-02")
    assert pd.notna(ene["disponible_utc"])
    assert _valor(t, sid, v0, "2000-01-01") is None
    assert _valor(t, sid, "2000-03-02", "2000-01-01") == pytest.approx(2.0)
    # Toda fila previa queda estrictamente antes del día del vintage que cita su nota.
    previas = t[t["nota"].fillna("").str.contains(catalogo.NOTA_PREVIA_ALFRED, regex=False)]
    assert (previas["fecha_publicacion"] < pd.Timestamp(v0)).all()


def test_t1_alfred_sin_regla_previa_visible_desde_el_primer_vintage():
    sid = "fred:XWEI"
    filas = [_obs("2008-01-05", "1.0", "2020-04-16"), _obs("2020-04-11", "-6.0", "2020-04-16")]
    t, info = _tabla(sid, filas, _entrada(sid, frecuencia="semanal", publicador="nyfed_wei"))
    assert info["previas"] == 0 and set(t["fecha_publicacion"]) == {pd.Timestamp("2020-04-16")}


# --- Re-estimadas, proyecciones y truncado ------------------------------------------------------
def _reestimada(vintages: list[str], refs: list[str], semilla: int = 3) -> list[dict]:
    """Cada vintage re-estima TODOS los periodos (índice tipo NFCI)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i, ref in enumerate(refs):
        for j, v in enumerate(vintages):
            if pd.Timestamp(ref) + pd.Timedelta(days=5) > pd.Timestamp(v):
                continue
            fin = ABIERTO if j == len(vintages) - 1 else (pd.Timestamp(vintages[j + 1]) - pd.Timedelta(days=1)
                                                          ).date().isoformat()
            filas.append(_obs(ref, f"{rng.normal():.4f}", v, fin))
    return filas


def test_t1_alfred_reestimada_entera():
    sid = "fred:XNFCI"
    refs = pd.date_range("2020-01-03", periods=6, freq="W-FRI").strftime("%Y-%m-%d").tolist()
    vint = pd.date_range("2020-02-12", periods=4, freq="W-WED").strftime("%Y-%m-%d").tolist()
    t, info = _tabla(sid, _reestimada(vint, refs), _entrada(sid, frecuencia="semanal", publicador="chicago_fed",
                                                            reestimada=True))
    assert info["n_vintages"] == 4 and len(t) == 6 * 4
    assert t["nota"].str.contains(a.NOTA_REESTIMADA, regex=False).all()
    for j, v in enumerate(vint):
        vista = as_of_series(t, sid, v)
        assert len(vista) == 6 and set(vista["vintage"]) == {v}
        assert len(a.vista_vintage(t, sid, v)) == 6


def test_t1_alfred_proyecciones_visibles_desde_su_referencia():
    sid = "fred:XSEP"
    filas = [_obs("2025-01-01", "4.0", "2024-12-18", "2025-03-18"), _obs("2025-01-01", "3.9", "2025-03-19"),
             _obs("2026-01-01", "3.6", "2024-12-18", "2025-03-18"),                     # sustituida antes de 2026
             _obs("2026-01-01", "3.4", "2025-03-19"),
             _obs("2024-01-01", "4.5", "2024-12-18", "2025-03-18"), _obs("2024-01-01", ".", "2025-03-19")]
    e = _entrada(sid, frecuencia="anual", publicador="fed_sep", proyecciones="conservar")
    t, info = _tabla(sid, filas, e)
    assert info["proyecciones_descartadas"] == 1 and info["proyecciones_visibles"] == 2
    assert t["valor"].notna().all() and _valor(t, sid, "2025-06-30", "2024-01-01") == pytest.approx(4.5)
    assert (t["fecha_publicacion"] >= t["fecha_referencia"]).all()
    assert _valor(t, sid, "2024-12-31", "2025-01-01") is None
    assert _valor(t, sid, "2025-01-01", "2025-01-01") == pytest.approx(4.0)
    assert _valor(t, sid, "2025-03-19", "2025-01-01") == pytest.approx(3.9)
    assert _valor(t, sid, "2026-01-01", "2026-01-01") == pytest.approx(3.4)
    assert t["nota"].str.contains("proyección publicada el 2024-12-18", regex=False).any()


def test_t1_alfred_proyecciones_recortar_no_pasan_del_vintage():
    """Regresión (revisión): ``proyecciones: recortar`` (GDPPOT, NROU de la CBO) se trataba como
    ``conservar`` y la base llevaba proyecciones hasta 2036. Ahora de cada periodo solo queda lo que dan
    los vintages publicados desde su fecha de referencia."""
    sid = "fred:XCBO"
    filas = [_obs("2020-01-01", "4.6", "2020-01-28", "2021-02-10"), _obs("2020-01-01", "4.5", "2021-02-11"),
             # proyección de 2021-01 hecha en 2020: sigue vigente cuando sale el vintage de 2021-02-11
             _obs("2021-01-01", "4.4", "2020-01-28"),
             # proyección de 2022-01: sustituida antes de que exista un vintage posterior a su referencia
             _obs("2022-01-01", "4.3", "2020-01-28", "2021-02-10"), _obs("2022-01-01", "4.2", "2021-02-11")]
    e = _entrada(sid, frecuencia="anual", publicador="cbo", proyecciones="recortar")
    t, info = _tabla(sid, filas, e)
    assert t["fecha_referencia"].max() == pd.Timestamp("2021-01-01")       # nada más allá del último vintage
    assert info["proyecciones_recortadas"] == 2 and info["proyecciones_visibles"] == 0
    assert (t["fecha_publicacion"] >= t["fecha_referencia"]).all()
    assert _valor(t, sid, "2021-01-31", "2021-01-01") is None              # no visible desde su referencia
    assert _valor(t, sid, "2021-02-12", "2021-01-01") == pytest.approx(4.4)  # sí desde el vintage de 2021-02-11
    assert set(t.loc[t["fecha_referencia"] == pd.Timestamp("2021-01-01"), "vintage"]) == {"2021-02-11"}


def _truncar(filas: list[dict], corte: str) -> list[dict]:
    """Lo que ALFRED habría devuelto el día ``corte``: periodos empezados hasta entonces, abiertos si seguían."""
    out = []
    for o in filas:
        if o["realtime_start"] > corte:
            continue
        out.append({**o, "realtime_end": ABIERTO if o["realtime_end"] > corte else o["realtime_end"]})
    return out


def test_t1_alfred_truncado_vintages_posteriores_no_cambian_el_pasado():
    sid = "fred:XTRUNC"
    refs = pd.date_range("1999-01-01", "2001-12-01", freq="MS").strftime("%Y-%m-%d").tolist()
    vint = pd.date_range("2000-03-10", "2002-06-10", freq="MS").map(lambda d: d + pd.Timedelta(days=9))
    vint = vint.strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(7)
    filas = []
    for ref in refs:
        validas = [v for v in vint if v > (pd.Timestamp(ref) + pd.offsets.MonthEnd(1)).date().isoformat()]
        cambios = sorted({validas[0], *(str(x) for x in rng.choice(validas, size=min(3, len(validas)), replace=False))})
        for k, v in enumerate(cambios):
            fin = ABIERTO if k == len(cambios) - 1 else (pd.Timestamp(cambios[k + 1]) - pd.Timedelta(days=1)
                                                         ).date().isoformat()
            filas.append(_obs(ref, "." if rng.random() < 0.05 else f"{rng.normal(100, 5):.3f}", v, fin))
    e = _entrada(sid, previa={**PREVIA_MES, "primer_vintage": vint[0]})
    completa, _ = _tabla(sid, filas, e)
    for corte in ("2000-06-30", "2001-02-14", "2001-12-31"):
        truncada, _ = _tabla(sid, _truncar(filas, corte), e)
        for t in pd.date_range("1999-03-01", corte, freq="17D"):
            pd.testing.assert_frame_equal(as_of_series(truncada, sid, t), as_of_series(completa, sid, t))


# --- Coherencia de derivadas --------------------------------------------------------------------
def _cfnai(vintages: list[str], meses: int = 18, semilla: int = 11) -> tuple[list[dict], list[dict]]:
    """CFNAI sintético re-estimado en cada vintage y su media de 3 meses (redondeada) en el mismo vintage."""
    rng = np.random.default_rng(semilla)
    refs = pd.date_range("2024-01-01", periods=meses, freq="MS")
    base, media = [], []
    for j, v in enumerate(vintages):
        fin = ABIERTO if j == len(vintages) - 1 else (pd.Timestamp(vintages[j + 1]) - pd.Timedelta(days=1)
                                                      ).date().isoformat()
        vistos = refs[refs + pd.offsets.MonthEnd(1) < pd.Timestamp(v)]
        x = pd.Series(rng.normal(0, 0.5, len(vistos)).round(2), index=vistos)
        m3 = x.rolling(3).mean().round(2)
        base += [_obs(d.date().isoformat(), f"{x[d]:.2f}", v, fin) for d in vistos]
        media += [_obs(d.date().isoformat(), f"{m3[d]:.2f}", v, fin) for d in vistos if pd.notna(m3[d])]
    return base, media


def test_t1_alfred_coherencia_cfnaima3_en_el_mismo_vintage():
    vint = ["2025-01-23", "2025-02-24", "2025-03-24"]
    base, media = _cfnai(vint)
    e1 = _entrada("fred:CFNAI", publicador="chicago_fed", reestimada=True)
    e2 = _entrada("fred:CFNAIMA3", publicador="chicago_fed", reestimada=True, coherente_con=["fred:CFNAI"])
    t1, _ = a.tabla_serie_alfred("fred:CFNAI", a.parsear_vintages(base), e1)
    t2, _ = a.tabla_serie_alfred("fred:CFNAIMA3", a.parsear_vintages(media), e2)
    t = validar_series(pd.concat([t1, t2], ignore_index=True))
    r = a.coherencia_derivada(t, "fred:CFNAIMA3")
    assert list(r["vintage"]) == vint and r["ok"].all() and (r["n_comparadas"] >= 10).all()
    # Una media que no sale del CFNAI de su vintage (p. ej., la del vintage anterior) se detecta.
    roto = [dict(o, value=f"{float(o['value']) + 0.3:.2f}") if o["realtime_start"] == vint[1] else o for o in media]
    t3, _ = a.tabla_serie_alfred("fred:CFNAIMA3", a.parsear_vintages(roto), e2)
    r = a.coherencia_derivada(validar_series(pd.concat([t1, t3], ignore_index=True)), "fred:CFNAIMA3")
    assert list(r["ok"]) == [True, False, True]


def test_t1_alfred_transformaciones_conocidas():
    x = pd.Series([100.0, 101.0, 102.0], index=pd.date_range("2020-01-01", periods=3, freq="QS"))
    tasa = a._transformar(x, "tasa_trimestral_anualizada")
    assert tasa.iloc[1] == pytest.approx(((101 / 100) ** 4 - 1) * 100)
    # Sahm: media de 3 meses menos su mínimo en los 12 meses ANTERIORES (puede ser negativa).
    u = pd.Series([5.0] * 14 + [4.0, 4.3, 4.6, 4.9], index=pd.date_range("2020-01-01", periods=18, freq="MS"))
    sahm = a._transformar(u, "sahm")
    assert sahm.iloc[-1] == pytest.approx((4.3 + 4.6 + 4.9) / 3 - (4.0 + 4.3 + 4.6) / 3)
    assert sahm.iloc[14] == pytest.approx((5.0 + 5.0 + 4.0) / 3 - 5.0) and sahm.iloc[14] < 0
    assert sahm.iloc[:14].isna().all()
    with pytest.raises(KeyError, match="transformación"):
        a.coherencia_derivada(pd.DataFrame(columns=list(COLUMNAS_SERIES)), "fred:XNADA")


# --- Revisiones y comparación con la regla previa -----------------------------------------------
def test_t1_alfred_revisiones_y_comparar_con_lag():
    sid = "fred:XREV"
    v0 = "2000-03-10"
    filas = [_obs("2000-01-01", "10.0", v0),                                         # arranque (previa)
             _obs("2000-02-01", "20.0", v0, "2000-04-06"), _obs("2000-02-01", "21.0", "2000-04-07"),
             _obs("2000-03-01", "30.0", "2000-04-07", "2000-05-04"), _obs("2000-03-01", "29.4", "2000-05-05"),
             _obs("2000-04-01", "40.0", "2000-05-05")]
    e = _entrada(sid, previa=PREVIA_MES)
    t, _ = _tabla(sid, filas, e)
    r = a.revisiones(sid, t)
    assert list(r["n_revisiones"]) == [0, 1, 1, 0]
    assert list(r["previa_alfred"]) == [True, False, False, False]
    mar = r[r["fecha_referencia"] == pd.Timestamp("2000-03-01")].iloc[0]
    assert mar["revision"] == pytest.approx(-0.6) and mar["revision_pct"] == pytest.approx(-2.0)
    assert mar["primera_publicacion"] == pd.Timestamp("2000-04-07")
    c = a.comparar_con_lag(sid, t, cat={sid: e})
    det = c["detalle"].set_index("fecha_referencia")["dias"]
    # 2000-03: regla 2000-05-01, real 2000-04-07 → +24 (retrasa); 2000-04: regla 2000-06-01, real 2000-05-05 → +27.
    assert det.to_dict() == {pd.Timestamp("2000-03-01"): 24, pd.Timestamp("2000-04-01"): 27}
    assert c["n_periodos"] == 2 and c["n_adelanta"] == 0 and c["n_retrasa"] == 2
    # Regla de 1 día: 2000-03 → 2000-04-01 (real 2000-04-07: adelanta 6); 2000-04 → 2000-05-01 (adelanta 4).
    e_corta = _entrada(sid, previa={**PREVIA_MES, "lag": "1D"})
    c = a.comparar_con_lag(sid, t, cat={sid: e_corta})
    assert c["n_adelanta"] == 2 and c["dias_adelanta_max"] == 6 and c["n_retrasa"] == 0
    with pytest.raises(ValueError, match="regla_previa_alfred"):
        a.comparar_con_lag(sid, t, cat={sid: _entrada(sid)})


def test_t1_alfred_resumen_revisiones_y_vs_lag(tmp_path):
    sid = "fred:XREV"
    v0 = "2000-03-10"
    filas = [_obs("2000-01-01", "10.0", v0),                                         # arranque (previa)
             _obs("2000-02-01", "20.0", v0, "2000-04-06"), _obs("2000-02-01", "21.0", "2000-04-07"),
             _obs("2000-03-01", "30.0", "2000-04-07", "2000-05-04"), _obs("2000-03-01", "29.4", "2000-05-05"),
             _obs("2000-04-01", "40.0", "2000-05-05")]
    e = _entrada(sid, previa={**PREVIA_MES, "lag": "1D"}, grupo="empleo")
    t, _ = _tabla(sid, filas, e)
    cat = {sid: e}
    pu = a.primeras_y_ultimas(t).set_index("fecha_referencia")
    assert pu.loc[pd.Timestamp("2000-03-01"), "n_filas"] == 2 and pu.loc[pd.Timestamp("2000-01-01"), "primer_metodo"] == "lag_fijo"
    r = a.resumen_revisiones(t, cat)
    assert list(r.columns) == list(a.COLUMNAS_REVISIONES) and len(r) == 1
    fila = r.iloc[0]
    # Con la regla de 1 día, 2000-01 y 2000-02 pasan enteras a lag_fijo (todo el primer vintage): el primer
    # vintage se toma de su nota. Solo 2000-03 y 2000-04 tienen primera publicación real.
    assert a.primer_vintage(t).iloc[0] == pd.Timestamp(v0)
    assert fila["n_periodos"] == 4 and fila["n_periodos_medidos"] == 2 and fila["n_previas_alfred"] == 2
    assert fila["primer_vintage"] == v0 and fila["n_vintages"] == 3 and fila["grupo"] == "empleo"
    assert fila["pct_revisados"] == pytest.approx(50.0) and fila["revision_pct_max"] == pytest.approx(2.0)
    assert fila["revision_pct_mediana"] == pytest.approx(1.0)                      # mediana de 2 % y 0 %
    v = a.resumen_vs_lag(t, cat)
    assert list(v.columns) == list(a.COLUMNAS_VS_LAG)
    assert v.iloc[0]["n_adelanta"] == 2 and v.iloc[0]["pct_adelanta"] == pytest.approx(100.0)
    assert v.iloc[0]["dias_adelanta_max"] == 6 and v.iloc[0]["dias_adelanta_mediana"] == pytest.approx(5.0)
    assert v.iloc[0]["adelantos_bloque3"] == 0
    rutas = a.escribir_tablas_alfred(t, cat, carpeta=tmp_path)
    assert [x.name for x in rutas] == ["alfred_revisiones_resumen.csv", "alfred_vs_lag.csv"]
    texto = "".join(x.read_text(encoding="utf-8") for x in rutas)
    assert "29.4" not in texto and "21.0" not in texto                              # sin valores de nivel


def test_t1_alfred_relleno_de_historia_no_cuenta_como_publicacion():
    # Tipo USREC: el primer vintage trae 2000-2001 y un vintage posterior añade 1990 (relleno de historia).
    sid = "fred:XREC"
    v0 = "2002-02-01"
    filas = [_obs("2000-01-01", "0", v0), _obs("2001-12-01", "1", v0), _obs("2002-01-01", "1", "2002-03-01"),
             _obs("1990-01-01", "0", "2002-04-01"), _obs("2002-02-01", "0", "2002-04-01")]
    e = _entrada(sid, previa={**PREVIA_MES, "lag": "1D", "primer_vintage": v0})
    t, _ = _tabla(sid, filas, e)
    prim = a.primeras_y_ultimas(t)
    relleno = a.relleno_historico(prim)
    assert prim.loc[relleno, "fecha_referencia"].tolist() == [pd.Timestamp("1990-01-01")]
    c = a.comparar_con_lag(sid, t, cat={sid: e})
    assert c["n_relleno"] == 1 and c["n_periodos"] == 2                       # 2002-01 y 2002-02, no 1990
    assert c["dias_adelanta_max"] < 60                                       # el relleno no cuenta como fuga
    # En la base, el relleno se ve desde el vintage que lo añadió (conservador).
    assert _valor(t, sid, "2002-03-31", "1990-01-01") is None and _valor(t, sid, "2002-04-01", "1990-01-01") == 0.0


# --- Tabla completa (catálogo + archivo) y ficha ------------------------------------------------
def test_t1_alfred_tabla_alfred_desde_archivo(tmp_path):
    cat = {"fred:XCPI": _entrada("fred:XCPI", previa={**PREVIA_MES, "primer_vintage": "2019-12-11"}),
           "fred:XFALTA": _entrada("fred:XFALTA")}
    filas = [_obs("2019-10-01", "257.0", "2019-12-11"), _obs("2019-11-01", "257.9", "2019-12-11"),
             _obs("2019-12-01", "258.501", "2020-01-14", "2020-02-10"), _obs("2019-12-01", "258.444", "2020-02-11")]
    cli, falso = _cliente(tmp_path, {"XCPI": filas}, pagina=2)
    a.descargar_vintages(["fred:XCPI"], cliente=cli, cat=cat)
    n = len(falso.llamadas)
    informe: dict = {}
    t = a.tabla_alfred(["fred:XCPI", "fred:XFALTA"], cliente=cli, cat=cat, informe=informe)
    assert len(falso.llamadas) == n                                               # sin red
    assert list(t.columns) == list(COLUMNAS_SERIES) and set(t["serie"]) == {"fred:XCPI"}
    assert informe["fred:XCPI"]["previas"] == 1                                   # 2019-10: regla 2019-12-01
    assert "sin descarga completa" in informe["fred:XFALTA"]["avisos"][0]
    ficha, ruta = a.ficha_alfred(t, informe, cliente=cli, cat=cat, guardar=True, destino=tmp_path / "fichas")
    assert ruta.exists() and ficha.fuente == a.FICHA and ficha.estado_sonda == "parcial"
    assert ficha.cobertura_inicio == "2019-10-01" and ficha.cobertura_fin == "2019-12-01"
    assert "258.501" not in ruta.read_text(encoding="utf-8")                          # la ficha no lleva valores


def test_t1_alfred_metodo_distinto_rechazado():
    with pytest.raises(ValueError, match="vintage_exacto"):
        a.tabla_serie_alfred("fred:X", a.parsear_vintages([]), _entrada("fred:X", metodo="lag_fijo"))
    e = _entrada("fred:X")
    e.pop("publicador")
    with pytest.raises(ValueError, match="publicador"):
        a.tabla_serie_alfred("fred:X", a.parsear_vintages([]), e)


def test_t1_alfred_ids_del_catalogo_real():
    cat = catalogo.cargar_catalogo()
    ids = a.ids_alfred(cat)
    assert "fred:UNRATE" in ids and "fred:CPIAUCSL" in ids
    assert all(cat[s]["metodo"] == "vintage_exacto" and cat[s]["estado"] in catalogo.ESTADOS_CARGABLES for s in ids)
    for sid, (origen, _, _) in a.TRANSFORMACIONES.items():
        assert sid in cat and origen in cat[sid]["coherente_con"]
    for sid in ids:                                                                  # todo publicador tiene hora
        a.hora_publicacion(cat[sid])


def test_t1_alfred_entrada_real_no_se_modifica():
    cat = catalogo.cargar_catalogo()
    e = cat["fred:UNRATE"]
    antes = copy.deepcopy(e)
    a.tabla_serie_alfred("fred:UNRATE", a.parsear_vintages([_obs("1960-01-01", "5.2", "1960-03-15")]), e)
    assert e == antes


# --- Datos locales y red ------------------------------------------------------------------------
@pytest.mark.datos
def test_t1_alfred_archivo_local_valida():
    cli = a.cliente_alfred(clave="sin_clave")
    cat = catalogo.cargar_catalogo()
    ids = [s for s in a.ids_alfred(cat) if a.archivo_completo(cli, cat[s]["id_origen"]) is not None][:5]
    if not ids:
        pytest.skip("sin descargas de ALFRED en el corpus local")
    informe: dict = {}
    t = a.tabla_alfred(ids, cliente=cli, cat=cat, informe=informe)
    assert set(t["serie"]) <= set(ids) and len(t)
    assert set(t["metodo_publicacion"]) <= {"vintage_exacto", "lag_fijo"}


@pytest.mark.datos
def test_t1_alfred_coherencia_derivadas_reales():
    cli = a.cliente_alfred(clave="sin_clave")
    cat = catalogo.cargar_catalogo()
    pares = [(s, o) for s, (o, _, _) in a.TRANSFORMACIONES.items()
             if all(a.archivo_completo(cli, cat[x]["id_origen"]) is not None for x in (s, o))]
    if not pares:
        pytest.skip("sin descargas de ALFRED de una derivada y su origen en el corpus local")
    for sid, origen in pares:
        t = a.tabla_alfred([sid, origen], cliente=cli, cat=cat)
        r = a.coherencia_derivada(t, sid)
        # FRED no siempre recalcula la historia de la derivada el día de una revisión del origen
        # (SAHMCURRENT, 2020-01 a 2021-02: revisión estacional de UNRATE de 2017-2018 sin propagar).
        assert len(r) and r["ok"].iloc[-1] and r["ok"].mean() >= 0.75, (sid, r[~r["ok"]].head())


@pytest.mark.live
def test_t1_alfred_live_serie_corta(tmp_path):
    cat = catalogo.cargar_catalogo()
    cli = a.cliente_alfred(carpeta=tmp_path)
    r = a.descargar_vintages(["fred:FEDTARMD"], cliente=cli, cat=cat)
    assert r.loc[0, "estado"] == "descargada" and r.loc[0, "filas"] > 50
    informe: dict = {}
    t = a.tabla_alfred(["fred:FEDTARMD"], cliente=cli, cat=cat, informe=informe)
    assert len(t) and (t["fecha_publicacion"] >= t["fecha_referencia"]).all()
    textos = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert "api_key" not in textos
