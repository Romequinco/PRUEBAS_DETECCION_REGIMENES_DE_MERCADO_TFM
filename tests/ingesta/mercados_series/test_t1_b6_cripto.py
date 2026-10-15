"""Tests del bloque 6 (cripto): ``fuentes.cripto`` (Coinbase principal, Binance Vision control).

Sin red: velas SINTÉTICAS con el formato real (Coinbase ``[time, low, high, open, close, volume]``
descendente; CSV de Binance sin cabecera con ``open_time`` en ms hasta 2024 y en µs desde 2025) y
transportes falsos. Comprueban el orden de columnas, la paginación de 300 velas, los microsegundos,
la vela visible en D+1, el descarte de la vela en curso, el control fuera del contrato (USDT) y la
comparación. El ``@pytest.mark.live`` pide una semana de Coinbase y un mes de Binance (``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from ingesta.mercados_series import almacen
from ingesta.mercados_series.contrato import COLUMNAS_PRECIOS, ErrorContrato, validar_precios
from ingesta.mercados_series.fuentes import cripto as c

DIA = 86400
T0 = 1_704_067_200  # 2024-01-01 00:00 UTC


def _vela_cb(t: int, base: float) -> list[float]:
    # [time, low, high, open, close, volume] con low < open < close < high: delata un cambio de orden.
    return [t, base - 10, base + 20, base, base + 5, 1.5]


def _velas_cb(n: int, t0: int = T0) -> list[list[float]]:
    return [_vela_cb(t0 + i * DIA, 100.0 + i) for i in range(n)][::-1]   # descendente, como la API


class FalsoCoinbase:
    """Sirve velas sintéticas según ``start``/``end`` (ambos incluidos) y registra las peticiones."""

    def __init__(self, n: int, t0: int = T0):
        self.velas = _velas_cb(n, t0)
        self.llamadas: list[dict] = []

    def __call__(self, url: str, params: dict) -> list:
        self.llamadas.append(dict(params))
        ini = pd.Timestamp(params["start"]).timestamp()
        fin = pd.Timestamp(params["end"]).timestamp()
        filas = [v for v in self.velas if ini <= v[0] <= fin]
        assert len(filas) <= c.MAX_VELAS
        return filas


def _csv_binance(dias: pd.DatetimeIndex, micro: bool, base: float = 100.0) -> str:
    lineas = []
    for i, d in enumerate(dias):
        t = int(d.tz_localize("UTC").timestamp())
        f = 1_000_000 if micro else 1_000
        o, h, lo, cl = base + i, base + i + 20, base + i - 10, base + i + 5.05
        lineas.append(f"{t * f},{o:.8f},{h:.8f},{lo:.8f},{cl:.8f},2.0,{(t + DIA) * f - 1},200.0,10,1.0,100.0,0")
    return "\n".join(lineas) + "\n"


def _zip(nombre: str, contenido: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(nombre, contenido)
    return b.getvalue()


# --- Coinbase -----------------------------------------------------------------------------------
def test_t1_b6_cripto_orden_columnas_coinbase():
    velas = c.parsear_velas_coinbase(_velas_cb(3))
    assert velas["sesion"].is_monotonic_increasing
    primera = velas.iloc[0]
    assert primera["sesion"] == pd.Timestamp("2024-01-01")
    assert (primera["open"], primera["high"], primera["low"], primera["close"]) == (100.0, 120.0, 90.0, 105.0)


def test_t1_b6_cripto_coinbase_orden_cambiado_falla():
    # Si la API devolviera OHLC (open en la 2.ª columna), low > open y se detecta.
    malas = [[T0, 100.0, 120.0, 90.0, 105.0, 1.0]]
    with pytest.raises(ValueError, match="orden de columnas"):
        c.parsear_velas_coinbase(malas)
    with pytest.raises(ValueError, match="6 campos"):
        c.parsear_velas_coinbase([[T0, 1, 2, 3, 4]])
    with pytest.raises(ValueError, match="medianoche"):
        c.parsear_velas_coinbase([[T0 + 3600, 90, 120, 100, 105, 1]])


def test_t1_b6_cripto_tabla_coinbase_d_mas_1_y_vela_en_curso():
    velas = c.parsear_velas_coinbase(_velas_cb(5))
    # Descarga el 2024-01-05 a las 12:00 UTC: la vela del 05 está en curso y se descarta.
    tabla = c.tabla_coinbase(velas, "BTC-USD", pd.Timestamp("2024-01-05 12:00", tz="UTC"))
    assert list(tabla.columns) == list(COLUMNAS_PRECIOS)
    assert len(tabla) == 4 and tabla["sesion"].max() == pd.Timestamp("2024-01-04")
    assert (tabla["ticker"] == "cb:BTC-USD").all() and (tabla["mercado"] == "CRYPTO").all()
    assert (tabla["divisa"] == "USD").all() and (tabla["fuente"] == "coinbase").all()
    assert (tabla["dividendo"] == 0).all() and (tabla["split"] == 1).all()
    assert (tabla["fecha_publicacion"] == tabla["sesion"] + pd.Timedelta(days=1)).all()
    assert (tabla["disponible_utc"] == (tabla["sesion"] + pd.Timedelta(days=1, minutes=30)).dt.tz_localize("UTC")).all()
    validar_precios(tabla)


def test_t1_b6_cripto_paginacion_300_y_escritura(tmp_path):
    falso = FalsoCoinbase(650)
    velas, meta = c.descargar_coinbase("BTC-USD", "2024-01-01", "2025-12-31", get=falso, archivar=False)
    assert len(velas) == 650 and meta["peticiones"] == len(falso.llamadas) == 3
    assert all(p["granularity"] == 86400 for p in falso.llamadas)
    # Incremental: con lo guardado, solo se pide desde la sesión siguiente.
    tabla = c.tabla_coinbase(velas.iloc[:100], "BTC-USD", pd.Timestamp("2026-01-01", tz="UTC"))
    almacen.escribir_precios(tabla, "coinbase", ruta=tmp_path)
    falso2 = FalsoCoinbase(120)
    res = c.actualizar_coinbase(["BTC-USD"], get=falso2, ruta=tmp_path, archivar=False)
    assert falso2.llamadas[0]["start"].startswith("2024-04-10")
    assert res["BTC-USD"]["filas_nuevas"] == 20
    assert len(almacen.leer_precios("cb:BTC-USD", "coinbase", ruta=tmp_path)) == 120


# --- Binance ------------------------------------------------------------------------------------
def test_t1_b6_cripto_binance_ms_y_microsegundos():
    dic = pd.date_range("2024-12-30", periods=2, freq="D")
    ene = pd.date_range("2025-01-01", periods=2, freq="D")
    velas = c.parsear_klines_binance(_csv_binance(dic, micro=False) + _csv_binance(ene, micro=True))
    assert list(velas["sesion"]) == list(dic) + list(ene)
    con_cabecera = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ignore\n"
    assert len(c.parsear_klines_binance(con_cabecera + _csv_binance(ene, micro=True))) == 2
    # Microsegundos leídos como milisegundos darían el año ~56000: se exige medianoche UTC.
    with pytest.raises(ValueError, match="medianoche"):
        c.parsear_klines_binance("1735693200000,1,2,0.5,1.5,1,0,0,0,0,0,0\n")


def test_t1_b6_cripto_control_usdt_fuera_del_contrato(tmp_path):
    velas = c.parsear_klines_binance(_csv_binance(pd.date_range("2025-01-01", periods=3), micro=True))
    tabla = c.tabla_binance(velas, "BTCUSDT", pd.Timestamp("2025-02-01", tz="UTC"))
    assert (tabla["divisa"] == "USDT").all() and (tabla["ticker"] == "bn:BTCUSDT").all()
    with pytest.raises(ErrorContrato):
        validar_precios(tabla)          # USDT no es divisa del contrato: por eso va al control
    c.guardar_control(tabla, tmp_path)
    leida = c.leer_control("bn:BTCUSDT", tmp_path)
    pd.testing.assert_frame_equal(leida, tabla)
    with pytest.raises(ErrorContrato, match="USDT"):
        c.validar_control(tabla.assign(divisa="USD"))


def test_t1_b6_cripto_descarga_binance_reanudable(tmp_path):
    pedidas: list[str] = []

    def falso(url: str):
        pedidas.append(url)
        if "2025-02" in url:
            raise RuntimeError("404 simulado")
        dias = pd.date_range("2025-01-01", periods=31) if "2025-01" in url else pd.date_range("2024-12-01", periods=31)
        nombre = url.rsplit("/", 1)[1].replace(".zip", ".csv")
        return _zip(nombre, _csv_binance(dias, micro="2025-01" in url)), {"Last-Modified": "Wed, 08 Jan 2025 00:50:27 GMT"}

    velas, metas = c.descargar_binance("BTCUSDT", "2024-12", "2025-02", get=falso, carpeta=tmp_path)
    assert len(velas) == 62 and [m["estado"] for m in metas] == ["ok", "ok", "error"]
    velas2, metas2 = c.descargar_binance("BTCUSDT", "2024-12", "2025-01", get=falso, carpeta=tmp_path)
    assert len(pedidas) == 3 and all(m["archivado"] for m in metas2)   # reutiliza los ZIP
    r = c.retraso_binance(metas)
    assert r["n"] == 2 and r["max_dias"] == pytest.approx(7.04, abs=0.01)


def test_t1_b6_cripto_retraso_y_huecos():
    metas = [{"mes": "2024-12", "estado": "ok", "last_modified": "2025-01-08T00:50:27+00:00"}]
    assert c.retraso_binance(metas)["mediana_dias"] == pytest.approx(7.04, abs=0.01)
    velas = c.parsear_velas_coinbase(_velas_cb(5))
    tabla = c.tabla_coinbase(velas.drop(index=2), "ETH-USD", pd.Timestamp("2024-02-01", tz="UTC"))
    assert c.huecos(tabla) == ["2024-01-03"]
    assert c.ultimo_mes_cerrado(pd.Timestamp("2026-10-13 10:00", tz="UTC")) == "2026-09"


def test_t1_b6_cripto_comparar():
    cb = c.tabla_coinbase(c.parsear_velas_coinbase(_velas_cb(10)), "BTC-USD", pd.Timestamp("2024-02-01", tz="UTC"))
    bn = cb.copy()
    bn["close"] = bn["close"] * 1.001          # +10 pb
    bn = bn.drop(index=[9])                    # una sesión menos al final (fuera del periodo común)
    bn = bn.drop(index=[4])                    # y un hueco dentro
    res = c.comparar(cb, bn)
    assert res["sesiones_comunes"] == 8
    assert res["solo_coinbase_en_periodo_comun"] == 1 and res["solo_binance_en_periodo_comun"] == 0
    assert res["close"]["mediana_abs_pb"] == pytest.approx(10.0, abs=0.01)
    assert res["close"]["p95_abs_pb"] == pytest.approx(10.0, abs=0.01)
    assert res["open"]["mediana_abs_pb"] == 0.0


def test_t1_b6_cripto_fichas_sin_guardar():
    med = {"pares": {"BTC-USD": {"cobertura": ["2015-07-20", "2026-10-12"]}}}
    f = c.ficha_coinbase(med)
    assert f.fuente == "coinbase" and f.cobertura_inicio == "2015-07-20" and f.estado_sonda == "ok"
    fb = c.ficha_binance({"simbolos": {}, "retraso_last_modified": {"n": 0}})
    assert fb.fuente == "binance_vision" and fb.estado_sonda == "fallo" and "CC BY-NC-SA" in fb.licencia_y_terminos


@pytest.mark.live
def test_t1_b6_cripto_live_semana_y_mes(tmp_path):
    velas, meta = c.descargar_coinbase("BTC-USD", "2024-01-01", "2024-01-07", archivar=False)
    assert len(velas) == 7 and meta["peticiones"] == 1
    tabla = c.tabla_coinbase(velas, "BTC-USD", pd.Timestamp.now(tz="UTC"))
    assert (tabla["fecha_publicacion"] == tabla["sesion"] + pd.Timedelta(days=1)).all()
    bn, metas = c.descargar_binance("BTCUSDT", "2024-01", "2024-01", carpeta=tmp_path)
    assert len(bn) == 31 and metas[0]["estado"] == "ok" and metas[0]["last_modified"]
    res = c.comparar(tabla, c.tabla_binance(bn, "BTCUSDT", pd.Timestamp.now(tz="UTC")))
    assert res["sesiones_comunes"] == 7 and res["close"]["mediana_abs_pb"] < 50
