"""Tests del extractor de FRED/ALFRED del Tema 1 (bloque 2): ``fuentes.fred``.

Sin red: respuestas JSON SINTÉTICAS con el formato real de la API (ningún valor real de FRED/ICE) y
un transporte falso. Los ``@pytest.mark.live`` piden una serie corta (solo con ``INGESTA_LIVE=1``);
el que lee el archivo crudo local se salta si no existe.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import catalogo
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, validar_series
from ingesta.mercados_series.fuentes import fred as f

SECRETO = "abcdef0123456789abcdef0123456789"


# --- Respuestas sintéticas con el formato de la API ---------------------------------------------
def _obs(fecha: str, valor: str, rs: str = "2026-10-12", re: str = "9999-12-31") -> dict:
    return {"realtime_start": rs, "realtime_end": re, "date": fecha, "value": valor}


def _respuesta_obs(observaciones: list[dict], offset: int = 0, count: int | None = None) -> dict:
    return {"realtime_start": "2026-10-12", "realtime_end": "2026-10-12", "observation_start": "1600-01-01",
            "observation_end": "9999-12-31", "units": "lin", "output_type": 1, "file_type": "json",
            "order_by": "observation_date", "sort_order": "asc",
            "count": len(observaciones) if count is None else count, "offset": offset, "limit": 100000,
            "observations": observaciones}


class Falso:
    """Transporte falso: registra las peticiones y responde con ``respuestas[(endpoint, series_id)]``.

    Las observaciones se sirven en páginas de ``pagina`` filas según ``offset``. Con ``vintage_dates``
    filtra las filas cuyo ``realtime_start`` está entre las fechas pedidas.
    """

    def __init__(self, respuestas: dict, pagina: int = 100000):
        self.respuestas = respuestas
        self.pagina = pagina
        self.llamadas: list[tuple[str, dict]] = []

    def __call__(self, url: str, params: dict) -> dict:
        assert params["api_key"] == SECRETO and params["file_type"] == "json"
        endpoint = url.split("/fred/", 1)[1]
        self.llamadas.append((endpoint, dict(params)))
        datos = self.respuestas[(endpoint, params.get("series_id") or params.get("release_id"))]
        if endpoint != "series/observations":
            return datos
        filas = datos["observations"]
        if "vintage_dates" in params:
            pedidas = set(params["vintage_dates"].split(","))
            filas = [o for o in filas if o["realtime_start"] in pedidas]
        off = int(params.get("offset", 0))
        return _respuesta_obs(filas[off:off + self.pagina], offset=off, count=len(filas))


def _cliente(tmp_path, respuestas: dict, **kw) -> tuple[f.ClienteFred, Falso]:
    falso = Falso(respuestas, **{k: kw.pop(k) for k in ("pagina",) if k in kw})
    return f.ClienteFred(SECRETO, transporte=falso, carpeta=tmp_path, hoy=kw.pop("hoy", "2026-10-12"), **kw), falso


# --- Parseo -------------------------------------------------------------------------------------
def test_t1_fred_parseo_valores_punto_y_realtime_abierto():
    df = f.parsear_observaciones([_obs("2026-10-06", "3.10"), _obs("2026-10-05", "."),
                                  _obs("2026-10-07", "2.95", rs="2026-10-08", re="2026-10-09")])
    assert list(df.columns) == ["fecha_referencia", "valor", "ausente", "realtime_start", "realtime_end"]
    assert list(df["fecha_referencia"].dt.strftime("%Y-%m-%d")) == ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert np.isnan(df.loc[0, "valor"]) and df.loc[0, "ausente"]
    assert df.loc[1, "valor"] == pytest.approx(3.10) and not df.loc[1, "ausente"]
    assert pd.isna(df.loc[0, "realtime_end"])                      # 9999-12-31 → abierto
    assert df.loc[2, "realtime_end"] == pd.Timestamp("2026-10-09")
    assert str(df["fecha_referencia"].dtype) == "datetime64[ns]" and str(df["valor"].dtype) == "float64"


def test_t1_fred_parseo_vacio_y_valor_vacio():
    vacio = f.parsear_observaciones([])
    assert vacio.empty and str(vacio["realtime_start"].dtype) == "datetime64[ns]"
    assert f.parsear_observaciones([_obs("2026-10-05", "")])["ausente"].iloc[0]


def test_t1_fred_last_updated_a_nueva_york():
    assert f.last_updated_ny("2026-10-09 08:08:04-05") == "2026-10-09 09:08"     # CDT → EDT
    assert f.last_updated_ny("2022-01-28 16:01:02-06") == "2022-01-28 17:01"     # CST → EST
    assert f.last_updated_ny(None) is None and f.last_updated_ny("ayer") is None


# --- Troceado de vintages -----------------------------------------------------------------------
def test_t1_fred_trozos_sin_perder_ni_repetir():
    fechas = [f"v{i:05d}" for i in range(4501)]
    grupos = f.trozos(fechas)
    assert [len(g) for g in grupos] == [500] * 9 + [1]
    assert sum(grupos, []) == fechas
    assert len(f.trozos(fechas, f.MAX_VINTAGES)) == 3
    assert f.trozos([], 10) == []
    for n in (0, f.MAX_VINTAGES + 1):
        with pytest.raises(ValueError, match="2000"):
            f.trozos(fechas, n)


def test_t1_fred_vintages_mas_de_2000_en_varias_peticiones(tmp_path):
    vint = pd.bdate_range("2015-01-01", periods=2300).strftime("%Y-%m-%d").tolist()
    # Una fila por vintage (misma fecha de referencia, valor distinto en cada vintage).
    filas = [_obs("2015-01-01", f"{i / 100:.2f}", rs=v, re=v) for i, v in enumerate(vint)]
    cli, falso = _cliente(tmp_path, {("series/observations", "XTEST"): {"observations": filas}})
    obs = cli.observaciones("XTEST", vintage_dates=list(reversed(vint)) + vint[:3])   # desordenadas y repetidas
    pedidas = [p["vintage_dates"].split(",") for _, p in falso.llamadas]
    assert len(pedidas) == 5 and all(len(g) <= f.VINTAGES_POR_PETICION for g in pedidas)
    assert sum(pedidas, []) == vint                                                     # en orden, sin repetir
    assert len(obs) == 2300 and len({o["realtime_start"] for o in obs}) == 2300


def test_t1_fred_paginado_por_offset(tmp_path):
    filas = [_obs(f"2026-09-{d:02d}", f"{d}.0") for d in range(1, 8)]
    cli, falso = _cliente(tmp_path, {("series/observations", "XPAG"): {"observations": filas}}, pagina=3)
    assert [o["date"] for o in cli.observaciones("XPAG")] == [o["date"] for o in filas]
    assert [int(p["offset"]) for _, p in falso.llamadas] == [0, 3, 6]
    # Una página por fichero, con la fecha de descarga; archivadas() las vuelve a juntar sin red.
    assert len(list((tmp_path / "series_observations" / "XPAG").glob("20261012_*.json"))) == 3
    [(dia, juntas)] = cli.archivadas("XPAG")
    assert dia == "20261012" and [o["date"] for o in juntas] == [o["date"] for o in filas]


# --- Clave, errores y caché ---------------------------------------------------------------------
def test_t1_fred_error_sin_clave_en_el_mensaje(tmp_path):
    def roto(url, params):
        raise RuntimeError(f"400 Client Error: Bad Request for url: {url}?series_id=X&api_key={params['api_key']}")

    cli = f.ClienteFred(SECRETO, transporte=roto, carpeta=tmp_path)
    with pytest.raises(f.ErrorFred) as e:
        cli.observaciones("X")
    assert SECRETO not in str(e.value) and "api_key=***" in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__              # sin la excepción original
    assert not list(tmp_path.rglob("*.json"))                                       # el fallo no se cachea


def test_t1_fred_falta_clave(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "env", lambda *a, **k: None)
    with pytest.raises(f.ErrorFred, match="FRED_API_KEY"):
        f.ClienteFred(transporte=lambda u, p: {}, carpeta=tmp_path).serie("DBAA")


def test_t1_fred_cache_por_dia_sin_clave(tmp_path):
    resp = {("series", "DBAA"): {"seriess": [{"id": "DBAA", "last_updated": "2026-10-09 10:31:25-05"}]}}
    cli, falso = _cliente(tmp_path, resp)
    assert cli.serie("DBAA")["id"] == "DBAA" and cli.serie("DBAA")["id"] == "DBAA"
    assert cli.peticiones == 1 and len(falso.llamadas) == 1                         # segunda: de la caché
    [fichero] = list((tmp_path / "series" / "DBAA").glob("*.json"))
    texto = fichero.read_text(encoding="utf-8")
    assert fichero.name.startswith("20261012_") and SECRETO not in texto and "api_key" not in texto
    envoltorio = json.loads(texto)
    assert envoltorio["params"] == {"series_id": "DBAA"} and envoltorio["endpoint"] == "series"
    # Otro día: nuevo fichero (archivo propio). cache=False: descarga aunque exista.
    otro, _ = _cliente(tmp_path, resp, hoy="2026-10-13")
    otro.serie("DBAA")
    assert len(list((tmp_path / "series" / "DBAA").glob("*.json"))) == 2
    forzado, falso2 = _cliente(tmp_path, resp, cache=False)
    forzado.serie("DBAA")
    assert len(falso2.llamadas) == 1


def test_t1_fred_vintagedates_y_release_dates(tmp_path):
    resp = {("series/vintagedates", "WEI"): {"count": 2, "vintage_dates": ["2026-10-01", "2026-10-08"]},
            ("release/dates", 17): {"count": 2, "release_dates": [{"release_id": 17, "date": "2026-10-05"},
                                                                  {"release_id": 17, "date": "2026-10-13"}]}}
    cli, falso = _cliente(tmp_path, resp)
    assert cli.vintages("WEI") == ["2026-10-01", "2026-10-08"]
    assert cli.fechas_release(17, realtime_start="2026-10-01", realtime_end="2026-10-31") == ["2026-10-05",
                                                                                              "2026-10-13"]
    assert falso.llamadas[0][1]["realtime_start"] == f.INICIO_ALFRED
    assert (tmp_path / "release_dates" / "release_17").is_dir()


def test_t1_fred_union_del_archivo_ventana_movil():
    dia1 = [_obs("2026-10-01", "3.00"), _obs("2026-10-02", "."), _obs("2026-10-05", "3.20")]
    dia2 = [_obs("2026-10-02", "3.10"), _obs("2026-10-05", "3.25"), _obs("2026-10-06", "3.30")]
    u = f.unir_archivo([("20261006", dia1), ("20261007", dia2)])
    assert list(u["fecha_referencia"].dt.day) == [1, 2, 5, 6]          # la fecha que salió de la ventana sigue
    assert list(u["valor"]) == [3.00, 3.10, 3.20, 3.30]                # primer valor no nulo visto
    assert list(u["descarga"]) == ["20261006", "20261007", "20261006", "20261007"]
    assert f.unir_archivo([]).empty


# --- Catálogo → contrato: fechas de publicación -------------------------------------------------
@pytest.fixture(scope="module")
def cat():
    return catalogo.cargar_catalogo()


def _fila(tabla: pd.DataFrame, fecha: str) -> pd.Series:
    return tabla.set_index("fecha_referencia").loc[pd.Timestamp(fecha)]


@pytest.mark.parametrize("sid, fecha, disponible, decision", [
    ("fred:DBAA", "2026-10-08", "2026-10-09 16:00", "2026-10-09"),          # Moody's: hábil siguiente 12:00
    ("fred:DBAA", "2026-10-09", "2026-10-13 16:00", "2026-10-13"),          # viernes → martes (Columbus Day)
    ("fred:DEXUSEU", "2026-09-29", "2026-10-05 20:15", "2026-10-05"),       # H.10: lunes siguiente 16:15
    ("fred:DTWEXBGS", "2026-10-06", "2026-10-13 20:15", "2026-10-13"),      # lunes festivo → martes
    ("fred:BAMLH0A0HYM2", "2026-10-08", "2026-10-09 13:15", "2026-10-09"),  # ICE en FRED: D+1 09:15
    ("fred:CPFF", "2026-10-06", "2026-10-07 21:05", "2026-10-08"),          # FRED la calcula en D+1 17:05
    ("fred:DCPF3M", "2026-10-06", "2026-10-07 20:15", "2026-10-07"),        # papel comercial: D+1 16:15
])
def test_t1_fred_fechas_de_publicacion_calendario(cat, sid, fecha, disponible, decision):
    obs = f.parsear_observaciones([_obs(fecha, "1.23")])
    t = validar_series(f.tabla_serie(sid, obs, cat[sid]))
    fila = t.iloc[0]
    assert fila["disponible_utc"] == pd.Timestamp(disponible, tz="UTC")
    assert fila["fecha_publicacion"] == pd.Timestamp(decision)
    assert fila["metodo_publicacion"] == "calendario" and fila["vintage"] == "unico"
    assert fila["unidad"] == cat[sid]["unidad"] and fila["fuente"] == "fred_alfred"


def test_t1_fred_punto_suelto_de_diaria_no_es_hueco(cat):
    """Regresión: cada '.' de un festivo era una fila nula con aviso (DBAA: 407 avisos en get_serie)."""
    obs = f.parsear_observaciones([_obs("2026-10-07", "5.80"), _obs("2026-10-08", "."), _obs("2026-10-09", "5.81"),
                                   _obs("2026-10-12", ".")])
    t = validar_series(f.tabla_serie("fred:DBAA", obs, cat["fred:DBAA"]))
    assert t["fecha_referencia"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-07", "2026-10-09"]
    dato = _fila(t, "2026-10-07")
    assert dato["valor"] == pytest.approx(5.80) and pd.isna(dato["nota"])
    assert list(t.columns) == list(COLUMNAS_SERIES)


def test_t1_fred_racha_larga_de_puntos_deja_un_hueco(cat):
    dias = pd.bdate_range("2016-09-01", "2016-12-30")
    valores = ["." if pd.Timestamp("2016-10-03") <= d <= pd.Timestamp("2016-12-15") else "4.0" for d in dias]
    obs = f.parsear_observaciones([_obs(d.date().isoformat(), v) for d, v in zip(dias, valores)])
    t = validar_series(f.tabla_serie("fred:DBAA", obs, cat["fred:DBAA"]))
    nulos = t[t["valor"].isna()]
    assert len(nulos) == 1 and nulos["fecha_referencia"].iloc[0] == pd.Timestamp("2016-10-03")
    assert nulos["nota"].iloc[0] == f.NOTA_PUNTO_LARGO


def test_t1_fred_punto_de_mensual_es_nulo_con_nota(cat):
    obs = f.parsear_observaciones([_obs("2024-01-01", "1.0"), _obs("2024-02-01", "."), _obs("2024-03-01", "1.2")])
    t = validar_series(f.tabla_serie("fred:UMCSENT", obs, cat["fred:UMCSENT"]))
    hueco = _fila(t, "2024-02-01")
    assert np.isnan(hueco["valor"]) and f.NOTA_PUNTO in hueco["nota"]


def test_t1_fred_vintage_exacto_wei(cat):
    # Semana del sábado 2026-09-26: aún '.' en la vintage del jueves 09-24 (no existía), dato el 10-01,
    # revisado el 10-08. Semana del 10-03: dato el 10-08.
    filas = [_obs("2026-09-26", ".", rs="2026-09-24", re="2026-09-30"),
             _obs("2026-09-26", "1.50", rs="2026-10-01", re="2026-10-07"),
             _obs("2026-09-26", "1.60", rs="2026-10-08", re="9999-12-31"),
             _obs("2026-10-03", "1.70", rs="2026-10-08", re="9999-12-31")]
    t = validar_series(f.tabla_serie("fred:WEI", f.parsear_observaciones(filas), cat["fred:WEI"]))
    assert len(t) == 3 and t["valor"].notna().all()                     # el '.' previo se descarta
    assert list(t["vintage"]) == ["2026-10-01", "2026-10-08", "2026-10-08"]
    assert list(t["fecha_publicacion"].dt.strftime("%Y-%m-%d")) == ["2026-10-01", "2026-10-08", "2026-10-08"]
    assert (t["disponible_utc"].dt.tz_convert("America/New_York").dt.hour == 12).all()
    assert (t["metodo_publicacion"] == "vintage_exacto").all()


def test_t1_fred_metodo_no_admitido(cat):
    with pytest.raises(ValueError, match="metodo"):
        f.tabla_serie("deriv:DBAA-DAAA", f.parsear_observaciones([_obs("2026-10-08", "1")]), cat["deriv:DBAA-DAAA"])


def test_t1_fred_ids_sustitutos(cat):
    ids = f.ids_sustitutos(cat)
    for s in ("fred:DBAA", "fred:DAAA", "fred:DEXUSEU", "fred:DTWEXBGS", "fred:DTWEXAFEGS", "fred:TEDRATE",
              "fred:DCPF3M", "fred:CPFF", "fred:WEI", "fred:BAMLH0A0HYM2", "fred:BAMLC0A0CM",
              "fred:BAMLH0A0HYM2EY", "fred:BAMLC0A4CBBB", "fred:BAMLC0A1CAAA"):
        assert s in ids, s
    for s in ("fred:SOFR", "fred:EFFR", "fred:DED3", "fred:DGS10"):        # ya vienen de data/raw
        assert s not in ids, s
    assert all(cat[s]["fuente"] == "fred_alfred" and not cat[s]["fichero_fase1"] for s in ids)


def test_t1_fred_tabla_sustitutos_con_transporte_falso(tmp_path, cat):
    resp = {
        ("series/observations", "DBAA"): {"observations": [_obs("2026-10-07", "5.80"), _obs("2026-10-08", ".")]},
        ("series/observations", "BAMLH0A0HYM2"): {"observations": [_obs("2026-10-08", "2.90")]},
        ("series/observations", "WEI"): {"observations": [_obs("2026-10-03", "1.70", rs="2026-10-08")]},
    }
    cli, falso = _cliente(tmp_path, resp)
    t = f.tabla_fred_sustitutos(["fred:DBAA", "fred:BAMLH0A0HYM2", "fred:WEI"], cliente=cli, cat=cat)
    assert sorted(t["serie"].unique()) == ["fred:BAMLH0A0HYM2", "fred:DBAA", "fred:WEI"]
    assert len(t) == 3 and len(falso.llamadas) == 3   # el '.' final de DBAA se descarta
    wei = [p for e, p in falso.llamadas if p["series_id"] == "WEI"][0]
    assert wei["realtime_start"] == f.INICIO_ALFRED and wei["realtime_end"] == f.FIN_ALFRED
    # Sin red: el mismo resultado desde el archivo crudo.
    sin_red = f.ClienteFred(SECRETO, transporte=lambda u, p: pytest.fail("no debe pedir"), carpeta=tmp_path,
                            hoy="2026-10-12")
    t2 = f.tabla_fred_sustitutos(["fred:DBAA", "fred:BAMLH0A0HYM2", "fred:WEI"], cliente=sin_red, cat=cat,
                                 descargar=False)
    pd.testing.assert_frame_equal(t.reset_index(drop=True), t2.reset_index(drop=True))
    with pytest.raises(KeyError, match="catálogo"):
        f.tabla_fred_sustitutos(["fred:NO_EXISTE"], cliente=cli, cat=cat)


def test_t1_fred_medir_publicacion(tmp_path):
    filas = [_obs("2026-10-01", "1", rs="2026-09-01"),                      # ya existía: truncada, se descarta
             _obs("2026-10-08", ".", rs="2026-10-08", re="2026-10-08"),
             _obs("2026-10-08", "2", rs="2026-10-09"),                      # jueves → viernes: D+1
             _obs("2026-10-09", "3", rs="2026-10-13")]                      # viernes → martes (Columbus): D+1
    cli, _ = _cliente(tmp_path, {("series/observations", "XLAG"): {"observations": filas}})
    m = f.medir_publicacion(cli, "XLAG", dias=35, hoy="2026-10-14")
    assert list(m["fecha_referencia"]) == ["2026-10-08", "2026-10-09"]
    assert list(m["dias_habiles_us"]) == [1, 1] and list(m["dias_naturales"]) == [1, 4]
    assert "valor" not in m.columns


def test_t1_fred_historial_vintage_registra_errores(tmp_path):
    def transporte(url, params):
        if params.get("vintage_dates") == "2019-01-02" or params.get("realtime_start") == "2019-01-02":
            raise RuntimeError(f"400 Client Error for url {url}?api_key={params['api_key']}")
        return _respuesta_obs([_obs("2023-10-10", "4.0", rs="2023-10-10"), _obs("2023-10-11", "4.1")])

    cli = f.ClienteFred(SECRETO, transporte=transporte, carpeta=tmp_path)
    ok = f.historial_vintage(cli, "XOAS", "2026-03-31")
    assert ok["vintage_dates"] == {"n_obs": 2, "inicio": "2023-10-10", "fin": "2023-10-11"}
    err = f.historial_vintage(cli, "XOAS", "2019-01-02")
    assert "error" in err["vintage_dates"] and SECRETO not in err["realtime"]["error"]


# --- Archivo crudo local y red ------------------------------------------------------------------
def _hay_crudo() -> bool:
    try:
        return any((f.ruta_crudo() / "series_observations" / "DBAA").glob("*.json"))
    except OSError:
        return False


@pytest.mark.datos
@pytest.mark.skipif(not _hay_crudo(), reason="falta el archivo crudo de FRED (local)")
def test_t1_fred_archivo_crudo_local_valida():
    ids = ["fred:DBAA", "fred:DAAA", "fred:BAMLH0A0HYM2"]
    t = f.tabla_fred_sustitutos(ids, cliente=f.ClienteFred("sin-uso", transporte=lambda u, p: pytest.fail("red")),
                                descargar=False)
    assert set(t["serie"]) == set(ids)
    assert (t.loc[t["valor"].isna(), "nota"] == f.NOTA_PUNTO_LARGO).all()
    oas = t[t["serie"] == "fred:BAMLH0A0HYM2"]
    assert oas["fecha_referencia"].min() <= pd.Timestamp("2023-10-10")       # ventana de 3 años archivada
    assert (oas["fecha_publicacion"] > oas["fecha_referencia"]).all()


@pytest.mark.live
def test_t1_fred_live_serie_corta(tmp_path):
    cli = f.ClienteFred(carpeta=tmp_path, cache=False)
    hoy = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    obs = f.parsear_observaciones(cli.observaciones(
        "DBAA", observation_start=(hoy - pd.Timedelta(days=20)).date().isoformat()))
    assert 5 <= len(obs) <= 20 and obs["valor"].notna().any()
    assert cli.serie("DBAA")["id"] == "DBAA"
    textos = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert "api_key" not in textos
