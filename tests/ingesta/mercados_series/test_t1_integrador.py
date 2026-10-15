"""Tests del integrador de los bloques 4 a 6: fragmentos fundidos, cargadores oficiales y CLI (sin red)."""
from __future__ import annotations

import pandas as pd
import pytest

from ingesta.mercados_series import __main__ as cli
from ingesta.mercados_series import almacen, base, visibilidad
from ingesta.mercados_series import catalogo as c
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, validar_series

PREFIJOS_NUEVOS = {"ecb", "eurostat", "bis", "imf", "wb", "oecd", "philly", "boc", "boj", "bbk", "rba", "ust",
                   "mof", "boe"}


def test_t1_integrador_catalogo_fundido():
    cat = c.cargar_catalogo()
    assert PREFIJOS_NUEVOS <= set(c.PREFIJOS)
    prefijos = {s.split(":", 1)[0] for s in cat}
    assert PREFIJOS_NUEVOS <= prefijos
    # FX del BCE: una entrada por divisa (el patrón ecb:EXR.D.*.EUR de publicaciones.yaml se quitó).
    assert cat["ecb:EXR.D.USD.EUR"]["fuente"] == "bce"
    pub = visibilidad.cargar_publicaciones()
    assert "ecb:EXR.D.*.EUR" not in pub["series"] and pub["alias"]["EURUSD"] == "ecb:EXR.D.USD.EUR"
    for clave in ("ecb_tipos_oficiales", "imf_weo", "oecd_edicion", "philly_rtdsm", "ecb_yc", "mof_jgb"):
        assert clave in pub["publicadores"], clave
    assert {"ARCX", "NAV_US", "STOXX_IDX", "ICE_MOVE", "FX_YAHOO"} <= set(visibilidad.cargar_mercados())
    lags = c.cargar_lags()
    assert {"BIS_WS_CBPOL", "BIS_WS_CREDIT_GAP", "BIS_WS_DSR", "BIS_WS_EER", "WB_ANUAL"} <= set(lags)
    # Regla previa en las revisables del BCE con historial de versiones (VALID_FROM).
    for sid, e in cat.items():
        if sid.startswith("ecb:") and e["metodo"] == "vintage_exacto":
            assert e["regla_previa_alfred"] and e["regla_previa_alfred"]["primer_vintage"] >= "2026-01-01", sid


def test_t1_integrador_patrones_banco_mundial():
    cat = c.cargar_catalogo()
    wb = [s for s in cat if s.startswith("wb:")]
    assert len(wb) == 594 and "wb:USA.NY.GDP.MKTP.KD.ZG" in cat
    assert cat["wb:USA.NY.GDP.MKTP.KD.ZG"]["nombre"].endswith(" - USA")
    assert cat["wb:USA.NY.GDP.MKTP.KD.ZG"]["lag"] == "12M"
    pat = {"wb:*.X": {"nombre": "Indicador X - todos los países", "paises": ["ESP", "FRA"], "fuente": "worldbank"}}
    out = c.expandir_patrones(pat)
    assert set(out) == {"wb:ESP.X", "wb:FRA.X"} and out["wb:ESP.X"]["nombre"] == "Indicador X - ESP"
    assert "paises" not in out["wb:ESP.X"]
    with pytest.raises(ValueError, match="repetido"):
        c.expandir_patrones(pat, existentes={"wb:ESP.X": {}})
    with pytest.raises(ValueError, match="paises"):
        c.expandir_patrones({"wb:*.X": {"nombre": "x"}})


def test_t1_integrador_fecha_previa_invalida():
    with pytest.raises(ValueError, match="AAAA-MM-DD"):
        c._fecha_iso("NaT", "prueba", "primer_vintage")


def test_t1_integrador_origenes_y_prioridad():
    cat = c.cargar_catalogo()
    esperado = {"ecb:ESTR": "bce", "bis:WS_CBPOL.M.US": "bis", "imf:WEO.USA.LUR": "imf",
                "wb:USA.NY.GDP.MKTP.KD.ZG": "worldbank", "oecd:LI_AA.USA": "oecd", "philly:ROUTPUT": "philly_rtdsm",
                "boc:V39079": "bancos_centrales", "ust:PAR.10Y": "curvas", "ecb:YC.AAA.10Y": "curvas"}
    for sid, cargador in esperado.items():
        assert c.origenes(cat[sid])[0] == cargador, sid
        assert cargador in base.CARGADORES
    p = base.PRIORIDAD
    assert p["alfred"] > p["academicos"] > p["fred"] > p["bce"] > p["curvas"] > p["philly_rtdsm"] > p["fase1"]
    assert tuple(p) == base.CARGADORES and set(base.CARGADORES_OFICIALES) < set(base.CARGADORES)
    # Toda serie cargable de los bloques 4-6 tiene un cargador.
    nuevas = [e for s, e in cat.items() if s.split(":", 1)[0] in PREFIJOS_NUEVOS]
    assert all(c.origenes(e) for e in nuevas)


def _filas(sid: str, fuente: str, n: int = 3) -> pd.DataFrame:
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "serie": sid, "fecha_referencia": fechas, "fecha_publicacion": fechas + pd.offsets.MonthEnd(2),
        "disponible_utc": (fechas + pd.offsets.MonthEnd(2)).tz_localize("UTC") + pd.Timedelta(hours=12),
        "vintage": "unico", "valor": [1.0, None, 3.0][:n], "unidad": "% anual", "divisa": None, "fuente": fuente,
        "metodo_publicacion": "lag_fijo", "nota": [None, "hueco de la fuente", None][:n]})[list(COLUMNAS_SERIES)]


def test_t1_integrador_cargador_bloque4_desde_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(almacen, "corpus_dir", lambda tema: tmp_path / tema)
    avisos: list[str] = []
    assert base.tabla_oficial("bis", registro=avisos.append).empty and "falta" in avisos[0]
    carpeta = tmp_path / "mercados_series" / "bloque4"
    carpeta.mkdir(parents=True)
    validar_series(_filas("bis:SINT.X", "bis")).to_parquet(carpeta / "bis.parquet", index=False)
    t = base.tabla_oficial("bis")
    assert set(t["serie"]) == {"bis:SINT.X"} and len(t) == 3 and t["valor"].isna().sum() == 1
    with pytest.raises(ValueError, match="desconocido"):
        base.tabla_oficial("nada")


def test_t1_integrador_union_con_prioridad():
    t_fase1 = _filas("fred:X", "fred_alfred")
    t_bce = _filas("fred:X", "fred_alfred").assign(valor=9.0)
    tabla, elegido, conflictos = base.unir({"fase1": t_fase1, "bce": t_bce})
    assert elegido["fred:X"] == "bce" and conflictos["fred:X"]["cargadores"] == ["fase1", "bce"]
    assert (tabla["valor"] == 9.0).all()


def test_t1_integrador_cli_ordenes(monkeypatch):
    llamadas = []

    class Falso:
        @staticmethod
        def main(argv):
            llamadas.append(argv)
            return 0

    monkeypatch.setattr("importlib.import_module", lambda nombre: Falso)
    assert cli.main(["descargar-oecd"]) == 0 and llamadas[-1] == ["--descargar", "--ficha"]
    assert cli.main(["descargar-bis", "--sin-red"]) == 0 and llamadas[-1] == ["--sin-red"]
    assert set(cli.FUENTES_CLI) == {"bce", "eurostat", "bis", "imf", "worldbank", "oecd", "philly", "bancos",
                                    "curvas", "cripto"}
    from ingesta.mercados_series.proveedores import yahoo

    monkeypatch.setattr(yahoo, "main", lambda argv: llamadas.append(argv))
    assert cli.main(["descargar-precios", "--nivel", "ampliado", "--fichero", "x.csv"]) == 0
    assert llamadas[-1][:2] == ["ampliado", "--fichero"]
