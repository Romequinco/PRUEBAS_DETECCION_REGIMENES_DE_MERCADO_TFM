"""Tests de visibilidad del Tema 1 (``ingesta.mercados_series.visibilidad``): solo pandas y PyYAML.

Hora de decisión 16:30 America/New_York; cierres de ``config/mercados.yaml`` (+30 min) y horas de
``config/publicaciones.yaml``. La propiedad con hypothesis se salta si no está instalado.
"""
from __future__ import annotations

import zlib
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import visibilidad as v
from ingesta.mercados_series.contrato import primera_decision_vec

UN_DIA = pd.Timedelta(days=1)
MISMO_DIA = ("XNYS", "XNAS", "XTSE", "XLON", "XETR", "XPAR", "XAMS", "XMAD", "XMIL", "XSWX",
             "XTKS", "XHKG", "XSHG", "XNSE", "XBOM", "XKRX", "XTAI", "XASX", "XTAE", "XJSE")
DIA_SIGUIENTE = ("CBOE_IDX", "ICE_DXY", "CME", "CRYPTO")
OBLIGATORIOS = ("XNYS", "XNAS", "CBOE_IDX", "ICE_DXY", "CME", "XLON", "XETR", "XPAR", "XAMS", "XMAD",
                "XMIL", "XSWX", "XTKS", "XHKG", "XSHG", "XNSE", "XKRX", "XTAI", "XASX", "BVMF", "XTSE",
                "XJSE", "CRYPTO")
PUBLICADORES = ("bls", "bea", "census", "fed_g17", "fed_h15", "fed_h10", "treasury_curva", "ecb_fx",
                "ecb_estr", "eurostat", "umich", "jolts")
# Laborables de 2024 que cubren las semanas de desfase DST EE. UU./Europa y ambos regímenes.
DIAS_PRUEBA = pd.bdate_range("2024-03-04", "2024-04-05").append(pd.bdate_range("2024-10-21", "2024-11-08"))


def _utc(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto, tz="UTC")


def _ny(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto, tz="America/New_York")


def _yaml(tmp_path, texto: str):
    ruta = tmp_path / "cfg.yaml"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


# --- momento_decision ---------------------------------------------------------------------------
@pytest.mark.parametrize("fecha, esperado", [
    ("2024-01-15", "2024-01-15 21:30"),   # invierno (EST)
    ("2024-07-15", "2024-07-15 20:30"),   # verano (EDT)
    ("2024-03-08", "2024-03-08 21:30"),   # viernes antes del cambio de EE. UU. (10-mar)
    ("2024-03-11", "2024-03-11 20:30"),   # desfase de marzo: EE. UU. en verano, Europa aún no
    ("2024-03-29", "2024-03-29 20:30"),   # último día del desfase (Europa cambia el 31-mar)
    ("2024-10-28", "2024-10-28 20:30"),   # desfase de otoño: Europa ya en invierno, EE. UU. no
    ("2024-11-01", "2024-11-01 20:30"),
    ("2024-11-04", "2024-11-04 21:30"),   # EE. UU. ya en invierno (3-nov)
    ("2024-03-10", "2024-03-10 20:30"),   # el propio domingo del cambio: 16:30 ya es EDT
    ("2024-11-03", "2024-11-03 21:30"),
])
def test_t1_momento_decision_invierno_verano_y_desfases(fecha, esperado):
    assert v.momento_decision(fecha) == _utc(esperado)


def test_t1_momento_decision_es_siempre_1630_en_nueva_york():
    dias = pd.date_range("2023-01-01", "2026-12-31", freq="D")
    locales = {v.momento_decision(d).tz_convert("America/New_York").strftime("%H:%M") for d in dias}
    assert locales == {"16:30"}


def test_t1_momento_decision_tipos_de_entrada_e_ignora_la_hora():
    esperado = _utc("2024-01-15 21:30")
    assert v.momento_decision(date(2024, 1, 15)) == esperado
    assert v.momento_decision(pd.Timestamp("2024-01-15 23:59")) == esperado
    assert v.momento_decision(pd.Timestamp("2024-01-15 09:00", tz="Asia/Tokyo")) == esperado
    assert str(esperado.tz) == "UTC"


def test_t1_un_sabado_es_fecha_de_decision():
    sabado = pd.Timestamp("2024-03-09")
    assert sabado.dayofweek == 5
    assert v.momento_decision(sabado) == _utc("2024-03-09 21:30")
    # La vela cripto del viernes cierra a medianoche UTC: la ve la decisión del sábado.
    assert v.fecha_publicacion_cierre("2024-03-08", "CRYPTO") == sabado
    # Lo que se publica el viernes tras las 16:30 NY se ve el sábado.
    assert v.primera_decision(_ny("2024-03-08 18:00")) == sabado
    # Y una sesión de sábado (cripto) se ve el domingo.
    assert v.fecha_publicacion_cierre(sabado, "CRYPTO") == pd.Timestamp("2024-03-10")


# --- primera_decision ---------------------------------------------------------------------------
@pytest.mark.parametrize("instante, esperado", [
    ("2024-01-15 16:30:00", "2024-01-15"),              # exactamente la hora de decisión: mismo día
    ("2024-01-15 16:30:01", "2024-01-16"),              # un segundo después: día siguiente
    ("2024-07-15 16:30:00", "2024-07-15"),
    ("2024-07-15 16:30:00.000000001", "2024-07-16"),   # un nanosegundo después
    ("2024-07-15 00:00:00", "2024-07-15"),
    ("2024-07-15 23:59:59", "2024-07-16"),
])
def test_t1_primera_decision_en_la_frontera(instante, esperado):
    assert v.primera_decision(_ny(instante)) == pd.Timestamp(esperado)


def test_t1_primera_decision_devuelve_fecha_naive_y_exige_huso():
    r = v.primera_decision(_utc("2024-01-15 21:30"))
    assert r == pd.Timestamp("2024-01-15") and r.tzinfo is None
    assert v.primera_decision(_utc("2024-01-15 21:30:01")) == pd.Timestamp("2024-01-16")
    with pytest.raises(ValueError, match="huso"):
        v.primera_decision(pd.Timestamp("2024-01-15 12:00"))
    assert v.primera_decision(pd.NaT) is pd.NaT


def test_t1_primera_decision_coincide_con_la_vectorizada_del_contrato():
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2023-01-01", tz="UTC").value
    ns = base + rng.integers(0, 4 * 365 * 86_400, 3000) * 1_000_000_000
    instantes = pd.Series(pd.to_datetime(ns, utc=True))
    # Añadir fronteras exactas (16:30 NY y +1 s) de días de invierno y de verano.
    fronteras = [v.momento_decision(d) for d in pd.date_range("2024-03-01", "2024-11-30", freq="7D")]
    instantes = pd.concat([instantes, pd.Series(fronteras),
                           pd.Series(fronteras) + pd.Timedelta(seconds=1)], ignore_index=True)
    vec = primera_decision_vec(instantes)
    esc = instantes.map(v.primera_decision)
    assert (vec == esc.astype("datetime64[ns]")).all()


def test_t1_primera_decision_de_momento_decision_es_la_propia_fecha():
    for d in pd.date_range("2024-01-01", "2024-12-31", freq="D"):
        assert v.primera_decision(v.momento_decision(d)) == d
        assert v.primera_decision(v.momento_decision(d) + pd.Timedelta(seconds=1)) == d + UN_DIA


# --- cierres de mercado -------------------------------------------------------------------------
@pytest.mark.parametrize("mercado", MISMO_DIA)
def test_t1_cierre_visible_el_mismo_dia(mercado):
    pub = v.fecha_publicacion_cierre_serie(pd.Series(DIAS_PRUEBA), mercado)
    assert (pub == pd.Series(DIAS_PRUEBA)).all()


@pytest.mark.parametrize("mercado", DIA_SIGUIENTE)
def test_t1_cierre_visible_al_dia_siguiente(mercado):
    pub = v.fecha_publicacion_cierre_serie(pd.Series(DIAS_PRUEBA), mercado)
    assert (pub == pd.Series(DIAS_PRUEBA) + UN_DIA).all()


@pytest.mark.parametrize("sesion, mercado, esperado_utc, pub", [
    ("2024-01-15", "XNYS", "2024-01-15 21:30", "2024-01-15"),     # 16:00 + 30 = 16:30 NY exacto
    ("2024-03-11", "XNYS", "2024-03-11 20:30", "2024-03-11"),
    ("2024-01-15", "CBOE_IDX", "2024-01-15 21:45", "2024-01-16"),  # 16:15 + 30 = 16:45 NY
    ("2024-07-15", "CBOE_IDX", "2024-07-15 20:45", "2024-07-16"),
    ("2024-01-15", "ICE_DXY", "2024-01-15 22:30", "2024-01-16"),
    ("2024-07-15", "CME", "2024-07-15 21:30", "2024-07-16"),
    ("2024-01-15", "CRYPTO", "2024-01-16 00:30", "2024-01-16"),
    ("2024-03-15", "XLON", "2024-03-15 17:00", "2024-03-15"),     # 16:30 GMT (Londres aún en invierno)
    ("2024-04-02", "XLON", "2024-04-02 16:00", "2024-04-02"),     # 16:30 BST
    ("2024-03-15", "XETR", "2024-03-15 17:00", "2024-03-15"),     # 17:30 CET
    ("2024-07-15", "XETR", "2024-07-15 16:00", "2024-07-15"),     # 17:30 CEST
    ("2024-10-29", "XETR", "2024-10-29 17:00", "2024-10-29"),     # desfase de otoño: CET
    ("2024-03-15", "XTKS", "2024-03-15 06:30", "2024-03-15"),
])
def test_t1_disponible_cierre_por_mercado(sesion, mercado, esperado_utc, pub):
    assert v.disponible_cierre(sesion, mercado) == _utc(esperado_utc)
    assert v.fecha_publicacion_cierre(sesion, mercado) == pd.Timestamp(pub)


def test_t1_b3_depende_del_dst_de_nueva_york():
    # 18:00 Sao Paulo (sin DST desde 2019) + 30 min = 21:30 UTC: igual a la decisión en invierno de
    # EE. UU. (mismo día) y una hora después en verano (día siguiente).
    assert v.fecha_publicacion_cierre("2024-01-15", "BVMF") == pd.Timestamp("2024-01-15")
    assert v.fecha_publicacion_cierre("2024-07-15", "BVMF") == pd.Timestamp("2024-07-16")


def test_t1_cboe_visible_a_las_1645_nueva_york():
    local = v.disponible_cierre("2024-03-12", "CBOE_IDX").tz_convert("America/New_York")
    assert local.strftime("%H:%M") == "16:45"


def test_t1_tokio_antes_y_despues_del_cambio_de_2024_11_05():
    assert v.hora_cierre("XTKS", "2024-11-04") == (time(15, 0), "Asia/Tokyo")
    assert v.hora_cierre("XTKS", "2024-11-05") == (time(15, 30), "Asia/Tokyo")
    assert v.disponible_cierre("2024-11-01", "XTKS") == _utc("2024-11-01 06:30")
    assert v.disponible_cierre("2024-11-04", "XTKS") == _utc("2024-11-04 06:30")
    assert v.disponible_cierre("2024-11-05", "XTKS") == _utc("2024-11-05 07:00")
    sesiones = pd.Series(pd.bdate_range("2024-10-28", "2024-11-08"))
    pub = v.fecha_publicacion_cierre_serie(sesiones, "XTKS")
    assert (pub == sesiones).all()
    disp = v.disponible_cierre_serie(sesiones, "XTKS").dt.strftime("%H:%M")
    assert set(disp[sesiones <= "2024-11-04"]) == {"06:30"}
    assert set(disp[sesiones >= "2024-11-05"]) == {"07:00"}


def test_t1_hora_cierre_errores():
    with pytest.raises(KeyError, match="XXXX"):
        v.hora_cierre("XXXX", "2024-01-15")
    with pytest.raises(ValueError, match="ningún tramo"):
        v.hora_cierre("XNYS", "1899-12-31")
    assert v.hora_cierre("CRYPTO", "2024-01-15") == ("24:00", "UTC")


# --- vectorizadas = escalares -------------------------------------------------------------------
@pytest.mark.parametrize("mercado", OBLIGATORIOS + ("XBOM", "XTAE"))
def test_t1_vectorizada_igual_a_escalar_por_mercado(mercado):
    rng = np.random.default_rng(zlib.crc32(mercado.encode()))
    dias = pd.Timestamp("2014-01-01") + pd.to_timedelta(rng.integers(0, 13 * 365, 150), unit="D")
    dias = dias.append(pd.DatetimeIndex(["2016-07-22", "2016-07-25", "2016-07-29", "2016-08-01",
                                         "2024-11-04", "2024-11-05", "2024-03-10", "2024-03-31"]))
    sesiones = pd.Series(dias, index=np.arange(len(dias))[::-1] * 3)   # índice no trivial
    disp = v.disponible_cierre_serie(sesiones, mercado)
    pub = v.fecha_publicacion_cierre_serie(sesiones, mercado)
    assert disp.index.equals(sesiones.index) and pub.index.equals(sesiones.index)
    assert str(disp.dtype) == "datetime64[ns, UTC]" and str(pub.dtype) == "datetime64[ns]"
    for i, s in sesiones.items():
        assert disp[i] == v.disponible_cierre(s, mercado)
        assert pub[i] == v.fecha_publicacion_cierre(s, mercado)
    assert (pub == primera_decision_vec(disp)).all()


def test_t1_vectorizada_con_mercado_por_fila_nulos_y_tipos():
    sesiones = pd.Series(["2024-03-15", None, "2024-11-05", "2024-01-15"], index=list("abcd"))
    mercado = pd.Series(["XLON", "XNYS", "XTKS", "CRYPTO"], index=list("abcd"))
    disp = v.disponible_cierre_serie(sesiones, mercado)
    assert pd.isna(disp["b"])
    assert disp["a"] == _utc("2024-03-15 17:00") and disp["c"] == _utc("2024-11-05 07:00")
    assert disp["d"] == _utc("2024-01-16 00:30")
    pub = v.fecha_publicacion_cierre_serie(sesiones, ["XLON", "XNYS", "XTKS", "CRYPTO"])
    assert pub.tolist()[0] == pd.Timestamp("2024-03-15") and pd.isna(pub["b"])
    assert pub["d"] == pd.Timestamp("2024-01-16")
    with pytest.raises(ValueError, match="mismo índice"):
        v.disponible_cierre_serie(sesiones, mercado.reset_index(drop=True))
    with pytest.raises(ValueError, match="elementos"):
        v.disponible_cierre_serie(sesiones, ["XLON"])
    with pytest.raises(KeyError, match="XXXX"):
        v.disponible_cierre_serie(sesiones, "XXXX")
    vacia = v.disponible_cierre_serie(pd.Series([], dtype="datetime64[ns]"), "XNYS")
    assert len(vacia) == 0 and str(vacia.dtype) == "datetime64[ns, UTC]"


def test_t1_coherente_con_los_sinteticos_multimercado():
    from ingesta.mercados_series import sinteticos as sint   # módulo propio: un ImportError es un fallo

    multi = sint.precios_multimercado()
    disp = v.disponible_cierre_serie(multi["sesion"], multi["mercado"])
    assert (disp == multi["disponible_utc"]).all()
    pub = v.fecha_publicacion_cierre_serie(multi["sesion"], multi["mercado"])
    assert (pub == multi["fecha_publicacion"]).all()


# --- publicaciones ------------------------------------------------------------------------------
def test_t1_disponible_publicacion_bce_y_cambio_de_hora():
    assert v.disponible_publicacion("2024-03-15", "16:00", "Europe/Berlin") == _utc("2024-03-15 15:00")
    assert v.disponible_publicacion("2024-07-15", time(16, 0), "Europe/Berlin") == _utc("2024-07-15 14:00")
    assert v.disponible_publicacion("2024-03-15", "08:30", "America/New_York",
                                    pd.Timedelta(minutes=5)) == _utc("2024-03-15 12:35")
    with pytest.raises(ValueError, match="inexistente o ambigua"):
        v.disponible_publicacion("2024-03-31", "02:30", "Europe/Berlin")     # no existe
    with pytest.raises(ValueError, match="inexistente o ambigua"):
        v.disponible_publicacion("2024-11-03", "01:30", "America/New_York")  # ocurre dos veces
    with pytest.raises(ValueError, match="HH:MM"):
        v.disponible_publicacion("2024-03-15", "8:30", "America/New_York")
    with pytest.raises(ValueError, match="no existe"):
        v.disponible_publicacion("2024-03-15", "08:30", "America/Nueva_York")


@pytest.mark.parametrize("clave, fecha, esperado_utc, pub", [
    ("bls", "2024-03-08", "2024-03-08 13:30", "2024-03-08"),
    ("bea", "2024-07-26", "2024-07-26 12:30", "2024-07-26"),
    ("census", "2024-03-14", "2024-03-14 12:30", "2024-03-14"),
    ("fed_g17", "2024-03-15", "2024-03-15 13:15", "2024-03-15"),
    ("jolts", "2024-03-12", "2024-03-12 14:00", "2024-03-12"),
    ("umich", "2024-03-15", "2024-03-15 14:00", "2024-03-15"),
    ("eurostat", "2024-03-19", "2024-03-19 10:00", "2024-03-19"),
    ("ecb_fx", "2024-03-15", "2024-03-15 15:00", "2024-03-15"),
    ("ecb_fx", "2024-10-29", "2024-10-29 15:00", "2024-10-29"),          # desfase de otoño
    ("treasury_curva", "2024-03-15", "2024-03-15 22:00", "2024-03-16"),  # 18:00 NY > 16:30
    ("treasury_curva", "2024-01-16", "2024-01-16 23:00", "2024-01-17"),
    ("fed_h15", "2024-07-03", "2024-07-05 20:15", "2024-07-05"),         # salta el 4 de julio
    ("fed_h15", "2024-03-15", "2024-03-18 20:15", "2024-03-18"),         # viernes → lunes
    ("fed_h10", "2024-03-13", "2024-03-18 20:15", "2024-03-18"),         # lunes siguiente
    ("fed_h10", "2024-02-14", "2024-02-20 21:15", "2024-02-20"),         # lunes festivo → martes
    ("ecb_estr", "2024-03-28", "2024-04-02 06:00", "2024-04-02"),        # Viernes Santo y Lunes de Pascua
    ("ecb_estr", "2024-01-15", "2024-01-16 07:00", "2024-01-16"),
])
def test_t1_disponible_publicador(clave, fecha, esperado_utc, pub):
    disp = v.disponible_publicador(clave, fecha)
    assert disp == _utc(esperado_utc)
    assert v.primera_decision(disp) == pd.Timestamp(pub)


def test_t1_disponible_publicador_festivos_extra_y_errores():
    # Cierre extraordinario declarado a mano (p. ej. luto nacional): se salta.
    disp = v.disponible_publicador("fed_h15", "2025-01-08", festivos=["2025-01-09"])
    assert disp == _utc("2025-01-10 21:15")
    with pytest.raises(KeyError, match="nadie"):
        v.disponible_publicador("nadie", "2024-01-15")


@pytest.mark.parametrize("clave, fecha, esperado_utc", [
    ("fed_h15", "2025-01-08", "2025-01-10 21:15"),   # 2025-01-09: luto nacional por Carter
    ("fed_h15", "2018-12-04", "2018-12-06 21:15"),   # 2018-12-05: luto nacional por Bush padre
    ("fed_h15", "2024-12-23", "2024-12-26 21:15"),   # 2024-12-24 (orden ejecutiva) y Navidad
    ("fed_h10", "2025-01-01", "2025-01-06 21:15"),   # el lunes siguiente no se ve afectado
])
def test_t1_disponible_publicador_cierres_extraordinarios_por_defecto(clave, fecha, esperado_utc):
    """Regresión: H.15 del 2025-01-08 salía el 2025-01-09 (día de cierre federal), un día antes de lo real."""
    assert v.disponible_publicador(clave, fecha) == _utc(esperado_utc)
    assert not v.es_habil("2025-01-09", "us_federal") and v.es_habil("2025-01-09", "target")
    assert not v.es_habil("2018-12-05", "us_federal")


def test_t1_es_habil():
    assert v.es_habil("2024-07-05", "us_federal") and not v.es_habil("2024-07-04", "us_federal")
    assert not v.es_habil("2024-03-29", "target") and v.es_habil("2024-03-29", "us_federal")
    assert not v.es_habil("2024-12-26", "target") and not v.es_habil("2024-03-09", "semana")
    assert not v.es_habil("2024-03-11", "semana", festivos=[date(2024, 3, 11)])
    with pytest.raises(ValueError, match="calendario"):
        v.es_habil("2024-03-11", "nyse")


# --- configuración ------------------------------------------------------------------------------
def test_t1_mercados_yaml_completo_y_valido():
    m = v.cargar_mercados()
    assert set(OBLIGATORIOS) <= set(m)
    for clave, info in m.items():
        assert isinstance(info["verificado"], bool) and info["fuente"], clave
        assert info["cierres"][-1]["hasta"] is None, clave            # vigente hoy
        assert info["cierres"][0]["desde"] <= date(1990, 1, 1), clave  # cubre el histórico
    assert v.hora_cierre("XHKG", "2016-07-22")[0] == time(16, 0)
    assert v.hora_cierre("XHKG", "2016-07-25")[0] == time(16, 10)
    assert v.hora_cierre("XKRX", "2016-08-01")[0] == time(15, 30)


def test_t1_cargar_mercados_devuelve_copia():
    m = v.cargar_mercados()
    m["XNYS"]["huso"] = "Europe/Madrid"
    assert v.cargar_mercados()["XNYS"]["huso"] == "America/New_York"
    assert v.disponible_cierre("2024-01-15", "XNYS") == _utc("2024-01-15 21:30")


def test_t1_mercados_pasados_a_mano():
    propios = {"ZZZ": {"huso": "Europe/Madrid", "cierres": [
        {"desde": "2000-01-01", "hasta": "2009-12-31", "hora": "17:35"},
        {"desde": "2010-01-01", "hasta": None, "hora": "18:00"}]}}
    assert v.disponible_cierre("2009-07-01", "ZZZ", propios) == _utc("2009-07-01 16:05")
    assert v.disponible_cierre("2010-07-01", "ZZZ", propios) == _utc("2010-07-01 16:30")
    s = pd.Series(pd.to_datetime(["2009-07-01", "2010-07-01"]))
    assert v.disponible_cierre_serie(s, "ZZZ", propios).tolist() == [_utc("2009-07-01 16:05"),
                                                                     _utc("2010-07-01 16:30")]
    with pytest.raises(ValueError, match="ningún tramo"):
        v.disponible_cierre_serie(pd.Series(pd.to_datetime(["1999-12-31"])), "ZZZ", propios)


_MERCADO_OK = """
ZZZ:
  nombre: Prueba
  huso: Europe/Madrid
  cierres:
    - {desde: "2000-01-01", hasta: "2009-12-31", hora: "17:35"}
    - {desde: "2010-01-01", hasta: null, hora: "18:00"}
  fuente: prueba
  verificado: false
"""


@pytest.mark.parametrize("cambio, mensaje", [
    (('hasta: "2009-12-31"', 'hasta: "2010-01-05"'), "solapan"),
    (('hasta: "2009-12-31"', 'hasta: "2009-12-20"'), "hueco"),
    (("Europe/Madrid", "Europe/Nowhere"), "no existe"),
    (('hora: "17:35"', 'hora: "25:00"'), "HH:MM"),
    (('hora: "17:35"', "hora: 17:35"), "entero"),
    (("verificado: false", "verificado: quizas"), "verificado"),
    (("verificado: false", "verificado: false\n  horario: x"), "desconocidas"),
    (('hasta: "2009-12-31"', "hasta: null"), "abierto"),
    (('desde: "2000-01-01", hasta: "2009-12-31"', 'desde: "2000-01-01", hasta: "1999-12-31"'), "anterior"),
])
def test_t1_cargar_mercados_rechaza_configuraciones_malas(tmp_path, cambio, mensaje):
    assert v.cargar_mercados(_yaml(tmp_path, _MERCADO_OK))["ZZZ"]["cierres"][1]["hora"] == time(18, 0)
    malo = _MERCADO_OK.replace(*cambio)
    with pytest.raises(ValueError, match=mensaje) as e:
        v.cargar_mercados(_yaml(tmp_path, malo))
    assert "ZZZ" in str(e.value)


def test_t1_publicaciones_yaml_completo_y_valido():
    p = v.cargar_publicaciones()
    assert set(PUBLICADORES) <= set(p["publicadores"])
    for clave, pub in p["publicadores"].items():
        assert pub["dia"] in v.DIAS_PUBLICACION and isinstance(pub["verificado"], bool), clave
    # Bloque 2: H.15 y H.10 medidos en ALFRED y contrastados con la página oficial (sondas_horas_bloque2.csv).
    assert p["publicadores"]["fed_h15"]["verificado"] is True
    assert p["publicadores"]["fed_h10"]["verificado"] is True
    assert p["publicadores"]["treasury_curva"]["hora"] == time(18, 0)
    assert p["alias"]["VIX"] == "fred:VIXCLS"
    assert p["series"]["fred:UNRATE"]["publicador"] == "bls"


_PUB_OK = """
alias: {}
publicadores:
  org:
    hora: "08:00"
    huso: Europe/Berlin
    dia: siguiente_habil
    habiles: target
    fuente: prueba
    verificado: false
series:
  "sint:X":
    fuente: prueba
    metodo: calendario
    publicador: org
    hora_local: "08:00"
    huso: Europe/Berlin
    verificado: false
"""


@pytest.mark.parametrize("cambio, mensaje", [
    (("dia: siguiente_habil", "dia: cuando_sea"), "dia"),
    (("    habiles: target\n", ""), "habiles"),
    (('hora: "08:00"', 'hora: "8h"'), "HH:MM"),
    (("metodo: calendario", "metodo: adivinado"), "metodo"),
    (("publicador: org", "publicador: otro"), "publicador"),
    (("alias: {}", "alias: {}\nextra: 1"), "desconocidas"),
    (("    huso: Europe/Berlin\n    verificado: false", "    huso: Europe/Berlin\n    verificado: false\n"
      "    sinsentido: 1"), "desconocidas"),                         # clave desconocida en una serie
    (("metodo: calendario", "metodo: lag_fijo"), "lag"),
    (("metodo: calendario", "metodo: lag_fijo\n    lag: 35 dias"), "lag"),
    (("metodo: calendario", "metodo: cierre_mercado\n    mercado: XXXX"), "mercado"),
    (("metodo: calendario", "metodo: derivada"), "componentes"),
    (("alias: {}", "alias: {}\ncierres_extraordinarios:\n  nyse: []"), "calendario desconocido"),
    (("alias: {}", 'alias: {}\ncierres_extraordinarios:\n  target: [{fecha: "2024-13-01", motivo: x}]'),
     "AAAA-MM-DD"),
])
def test_t1_cargar_publicaciones_rechaza_configuraciones_malas(tmp_path, cambio, mensaje):
    ok = v.cargar_publicaciones(_yaml(tmp_path, _PUB_OK))
    assert v.disponible_publicador("org", "2024-12-24", ok) == _utc("2024-12-27 07:00")
    with pytest.raises(ValueError, match=mensaje):
        v.cargar_publicaciones(_yaml(tmp_path, _PUB_OK.replace(*cambio)))


# --- propiedad (hypothesis opcional) ------------------------------------------------------------
def test_t1_propiedad_primera_decision_minima():
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies

    @hypothesis.settings(max_examples=300, deadline=None, database=None)
    @hypothesis.given(st.integers(min_value=pd.Timestamp("1990-01-01", tz="UTC").value,
                                  max_value=pd.Timestamp("2040-01-01", tz="UTC").value))
    def propiedad(ns: int) -> None:
        x = pd.Timestamp(ns, tz="UTC")
        d = v.primera_decision(x)
        assert v.momento_decision(d) >= x
        assert v.momento_decision(d - UN_DIA) < x

    propiedad()


def test_t1_cargar_publicaciones_metodos_validos_y_cierres(tmp_path):
    ok = _PUB_OK.replace("metodo: calendario", "metodo: lag_fijo\n    lag: 35D")
    assert v.cargar_publicaciones(_yaml(tmp_path, ok))["series"]["sint:X"]["lag"] == "35D"
    ok = _PUB_OK.replace("metodo: calendario", "metodo: cierre_mercado\n    mercado: XNYS")
    assert v.cargar_publicaciones(_yaml(tmp_path, ok))["series"]["sint:X"]["mercado"] == "XNYS"
    ok = _PUB_OK.replace("metodo: calendario", "metodo: derivada\n    componentes: [\"sint:A\", \"sint:B\"]")
    assert v.cargar_publicaciones(_yaml(tmp_path, ok))["series"]["sint:X"]["componentes"] == ["sint:A", "sint:B"]
    cierres = _PUB_OK.replace("alias: {}", 'alias: {}\ncierres_extraordinarios:\n  target:\n'
                                           '    - {fecha: "2024-12-27", motivo: prueba}')
    cfg = v.cargar_publicaciones(_yaml(tmp_path, cierres))
    assert cfg["cierres_extraordinarios"]["target"][0]["fecha"] == date(2024, 12, 27)
    # el org publica el siguiente hábil TARGET: el 27 ya no lo es
    assert v.disponible_publicador("org", "2024-12-24", cfg) == _utc("2024-12-30 07:00")
    real = v.cargar_publicaciones()
    fechas = {e["fecha"] for e in real["cierres_extraordinarios"]["us_federal"]}
    assert {date(2025, 1, 9), date(2018, 12, 5)} <= fechas
    assert real["series"]["fred:UNRATE"]["revisable"] is True


def test_t1_h6_semanal_hasta_2021_se_ve_al_dia_siguiente():
    """Regresión: el H.6/H.3 semanal salía los jueves a las 16:30 NY (no a las 13:00 de hoy)."""
    jueves = v.disponible_publicador("fed_h6", pd.Timestamp("2020-02-27"))
    assert jueves == _ny("2020-02-27 16:35")
    assert primera_decision_vec(pd.Series([jueves]))[0] == pd.Timestamp("2020-02-28")
    hoy = v.disponible_publicador("fed_h6", pd.Timestamp("2026-09-22"))
    assert primera_decision_vec(pd.Series([hoy]))[0] == pd.Timestamp("2026-09-22")

