"""Tests del extractor de FRED/ALFRED del Tema 1 (bloques 2 y 3): ``fuentes.fred``.

Sin red: respuestas JSON SINTÉTICAS con el formato real de la API (ningún valor real de FRED/ICE) y
un transporte falso. Bloque 3: regla de la fuente viva, descarga reanudable (páginas interrumpidas,
incrementales, tandas con tiempo máximo), derivadas (fecha = máxima de los componentes, valor A − B,
contraste con la serie de FRED) y fichas sin guardar. Los ``@pytest.mark.live`` piden una serie corta
y la sonda de 3 series de la fuente viva (solo con ``INGESTA_LIVE=1``); el que lee el archivo crudo
local se salta si no existe.
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
        if "observation_start" in params:
            filas = [o for o in filas if o["date"] >= params["observation_start"]]
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
# Bloque 3: las revisables de ALFRED (vintage_exacto) en su forma del bloque 2 (regla previa, lag_fijo):
# estos tests ejercitan las reglas de publicación con entradas reales del catálogo.
@pytest.fixture(scope="module")
def cat():
    return catalogo.vista_regla_previa()


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
              "fred:DCPF3M", "fred:CPFF", "fred:BAMLH0A0HYM2", "fred:BAMLC0A0CM",
              "fred:BAMLH0A0HYM2EY", "fred:BAMLC0A4CBBB", "fred:BAMLC0A1CAAA"):
        assert s in ids, s
    for s in ("fred:SOFR", "fred:EFFR", "fred:DED3", "fred:DGS10"):        # ya vienen de data/raw
        assert s not in ids, s
    # Bloque 3: las vintage_exacto (WEI y las 128 nuevas de ALFRED) las carga ALFRED, no este cargador.
    assert "fred:WEI" not in ids and "fred:UNRATE" not in ids
    assert all(cat[s]["fuente"] == "fred_alfred" and not cat[s]["fichero_fase1"] for s in ids)
    assert all(cat[s]["metodo"] != "vintage_exacto" for s in ids)


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


# --- Bloque 3: fuente viva ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def cat_real():
    return catalogo.cargar_catalogo()


SIN_EXC = pd.DataFrame({"serie": pd.Series(dtype=object), "fecha_referencia": pd.Series(dtype="datetime64[ns]"),
                        "fecha_publicacion": pd.Series(dtype="datetime64[ns]")})


def _vivo(sid: str, cat: dict, valores: dict[str, str]) -> pd.DataFrame:
    """Filas del contrato de ``sid`` (regla de su entrada) con valores SINTÉTICOS."""
    return f.tabla_serie(sid, f.parsear_observaciones([_obs(d, v) for d, v in valores.items()]), cat[sid])


def test_t1_fred_ids_fred_vivo_regla(cat_real):
    ids = set(f.ids_fred_vivo(cat_real))
    for sid, e in cat_real.items():
        esperado = (e["fuente_viva"] == "fred" and e["metodo"] != "vintage_exacto"
                    and e["estado"] in catalogo.ESTADOS_CARGABLES)
        assert (sid in ids) == esperado, sid
    for s in ("fred:DGS10", "fred:DFF", "fred:SOFR", "fred:VIXCLS", "fred:NASDAQ100", "fred:GS10", "fred:DEXJPUS",
              "fred:BAA10Y", "fred:CPFF", "fred:BAMLH0A0HYM2", "fred:USRECD", "fred:TEDRATE"):
        assert s in ids, s                       # H.15, NY Fed, Cboe/NASDAQ, mensuales, duplicadas, capadas, retiradas
    for s in ("fred:UNRATE", "fred:WEI", "fred:GDPC1", "yf:^GSPC", "deriv:DBAA-DGS10", "ff:F3D.SMB"):
        assert s not in ids, s                   # ALFRED, otras fuentes y derivadas propias
    assert all(s.startswith("fred:") and cat_real[s]["metodo"] in f.METODOS_VIVO for s in ids)
    assert set(f.ids_sustitutos(cat_real)) <= ids
    # fuente_viva null (FRED ya no la sirve) o estado no cargable: fuera.
    otro = {s: dict(e) for s, e in cat_real.items()}
    otro["fred:DGS10"]["fuente_viva"] = None
    otro["fred:DFF"]["estado"] = "pendiente_alfred"
    ids2 = f.ids_fred_vivo(otro)
    assert "fred:DGS10" not in ids2 and "fred:DFF" not in ids2 and "fred:DGS5" in ids2


def test_t1_fred_vivo_rechaza_series_ajenas(tmp_path, cat_real):
    cli, _ = _cliente(tmp_path, {})
    for sid in ("fred:UNRATE", "yf:^GSPC", "BAA10Y"):            # BAA10Y: alias de la derivada propia
        with pytest.raises(ValueError, match="fuente viva"):
            f.tabla_fred_vivo([sid], cliente=cli, cat=cat_real, sin_red=True)


def test_t1_fred_descarga_interrumpida_se_reanuda(tmp_path):
    filas = [_obs(f"2026-09-{d:02d}", f"{d}.0") for d in range(1, 8)]
    falso = Falso({("series/observations", "XVIVO"): {"observations": filas}}, pagina=3)
    cortes = []

    def transporte(url, params):
        if int(params.get("offset", 0)) == 3 and not cortes:
            cortes.append(1)
            raise RuntimeError(f"corte de red en {url}?api_key={params['api_key']}")
        return falso(url, params)

    cli = f.ClienteFred(SECRETO, transporte=transporte, carpeta=tmp_path, hoy="2026-10-12")
    with pytest.raises(f.ErrorFred) as e:
        f.descargar_vigente(cli, "XVIVO")
    assert SECRETO not in str(e.value)
    assert cli.descargas_vigentes("XVIVO") == []                       # la página suelta no cuenta
    assert len(list((tmp_path / "series_observations" / "XVIVO").glob("*.json"))) == 1
    r = f.descargar_vigente(cli, "XVIVO")
    assert r["modo"] == "completa" and r["peticiones"] == 2            # solo las páginas que faltaban
    assert [int(p["offset"]) for _, p in falso.llamadas] == [0, 3, 6]
    [(dia, params, obs)] = cli.descargas_vigentes("XVIVO")
    assert dia == "20261012" and params == {"series_id": "XVIVO"} and len(obs) == 7
    assert f.descargar_vigente(cli, "XVIVO")["modo"] == "cache" and len(falso.llamadas) == 3


def test_t1_fred_descarga_incremental_y_union(tmp_path):
    dia1 = [_obs("2026-07-01", "1.0"), _obs("2026-09-30", "2.0"), _obs("2026-10-01", ".")]
    cli1, _ = _cliente(tmp_path, {("series/observations", "XINC"): {"observations": dia1}}, hoy="2026-10-01")
    assert f.descargar_vigente(cli1, "XINC")["modo"] == "completa"
    # Día siguiente: FRED rellena el '.', añade un dato y ya no sirve la fecha más antigua (ventana móvil).
    dia2 = [_obs("2026-09-30", "2.0"), _obs("2026-10-01", "2.5"), _obs("2026-10-02", "3.0")]
    cli2, falso2 = _cliente(tmp_path, {("series/observations", "XINC"): {"observations": dia2}}, hoy="2026-10-02")
    r = f.descargar_vigente(cli2, "XINC")
    desde = (pd.Timestamp("2026-09-30") - pd.Timedelta(days=f.SOLAPE_DIAS)).date().isoformat()
    assert r["modo"] == "incremental" and r["desde"] == desde       # desde el último dato con valor, no el '.'
    assert falso2.llamadas[0][1]["observation_start"] == desde
    u = f.unir_archivo([(d, o) for d, _, o in cli2.descargas_vigentes("XINC")])
    assert u["fecha_referencia"].dt.strftime("%Y-%m-%d").tolist() == ["2026-07-01", "2026-09-30", "2026-10-01",
                                                                      "2026-10-02"]
    assert u["valor"].tolist() == [1.0, 2.0, 2.5, 3.0]
    assert f.descargar_vigente(cli2, "XINC")["modo"] == "cache" and len(falso2.llamadas) == 1
    assert f.descargar_vigente(cli2, "XINC", completa=True)["modo"] == "completa"      # forzada: sin observation_start
    assert "observation_start" not in falso2.llamadas[-1][1]
    # Las peticiones con realtime u output_type (ALFRED, excepciones) no son lo vigente.
    cli2.observaciones("XINC", realtime_start=f.INICIO_ALFRED, realtime_end=f.FIN_ALFRED, output_type=4)
    assert len(cli2.descargas_vigentes("XINC")) == 3


def test_t1_fred_descargar_vivo_por_tandas_y_errores(tmp_path, cat_real):
    resp = {("series/observations", "DGS10"): {"observations": [_obs("2026-10-08", "4.10")]},
            ("series/observations", "DFF"): {"observations": [_obs("2026-10-08", "3.90")]}}
    falso = Falso(resp)

    def transporte(url, params):
        if params["series_id"] == "DFF":
            raise RuntimeError(f"500 en {url}?api_key={params['api_key']}")
        return falso(url, params)

    cli = f.ClienteFred(SECRETO, transporte=transporte, carpeta=tmp_path, hoy="2026-10-12")
    ids = ["fred:DGS10", "fred:DFF"]
    tanda = f.descargar_fred_vivo(ids, cliente=cli, cat=cat_real, max_segundos=-1)   # sin tiempo: todo pendiente
    assert tanda["pendientes"] == ids and tanda["peticiones"] == 0 and not falso.llamadas
    rep = f.descargar_fred_vivo(ids, cliente=cli, cat=cat_real)
    assert rep["hechas"] == ["fred:DGS10"] and list(rep["errores"]) == ["fred:DFF"]
    assert SECRETO not in rep["errores"]["fred:DFF"]
    assert f.descargar_fred_vivo(ids, cliente=cli, cat=cat_real)["cache"] == ["fred:DGS10"]   # reanudable
    with pytest.raises(f.ErrorFred, match="fred:DFF") as e:
        f.tabla_fred_vivo(ids, cliente=cli, cat=cat_real, excepciones=SIN_EXC)
    assert SECRETO not in str(e.value)
    t = f.tabla_fred_vivo(ids, cliente=cli, cat=cat_real, excepciones=SIN_EXC, estricto=False)
    assert set(t["serie"]) == {"fred:DGS10"}                            # DFF sin archivo: no aparece


def test_t1_fred_tabla_vivo_reglas_excepciones_y_sin_red(tmp_path, cat_real):
    dias = {"2026-10-05": "4.10", "2026-10-06": "4.12", "2026-10-07": "4.15", "2026-10-08": "."}
    resp = {("series/observations", "DGS10"): {"observations": [_obs(d, v) for d, v in dias.items()]},
            ("series/observations", "VIXCLS"): {"observations": [_obs("2026-10-08", "17.5")]},
            ("series/observations", "GS10"): {"observations": [_obs("2026-08-01", "4.2"), _obs("2026-09-01", "4.1")]}}
    cli, falso = _cliente(tmp_path, resp)
    exc = pd.DataFrame({"serie": ["fred:DGS10"], "fecha_referencia": [pd.Timestamp("2026-10-06")],
                        "fecha_publicacion": [pd.Timestamp("2026-10-09")]})
    ids = ["fred:DGS10", "fred:VIXCLS", "fred:GS10"]
    t = f.tabla_fred_vivo(ids, cliente=cli, cat=cat_real, excepciones=exc)
    assert set(t["serie"]) == set(ids) and len(falso.llamadas) == 3
    dgs = t[t["serie"] == "fred:DGS10"].set_index("fecha_referencia")
    assert len(dgs) == 3                                                    # el '.' final se descarta
    assert dgs.loc[pd.Timestamp("2026-10-05"), "fecha_publicacion"] == pd.Timestamp("2026-10-06")   # H.15: D+1
    assert dgs.loc[pd.Timestamp("2026-10-06"), "fecha_publicacion"] == pd.Timestamp("2026-10-09")   # excepción medida
    assert catalogo.NOTA_EXCEPCION in dgs.loc[pd.Timestamp("2026-10-06"), "nota"]
    assert (t.loc[t["serie"] == "fred:VIXCLS", "metodo_publicacion"] == "cierre_mercado").all()
    gs10 = t[t["serie"] == "fred:GS10"]
    assert (gs10["metodo_publicacion"] == "lag_fijo").all() and gs10["disponible_utc"].isna().all()
    assert (t["vintage"] == "unico").all() and (t["fuente"] == "fred_alfred").all()
    # Sin red: lo mismo desde el archivo crudo.
    sin_red = f.ClienteFred(SECRETO, transporte=lambda u, p: pytest.fail("no debe pedir"), carpeta=tmp_path,
                            hoy="2026-10-13")
    t2 = f.tabla_fred_vivo(ids, sin_red=True, cliente=sin_red, cat=cat_real, excepciones=exc)
    pd.testing.assert_frame_equal(t, t2)


# --- Bloque 3: derivadas ------------------------------------------------------------------------
def test_t1_fred_pares_derivadas(cat_real):
    assert f.pares_derivadas(cat_real) == {"deriv:DBAA-DGS10": "fred:BAA10Y", "deriv:DCPF3M-DFF": "fred:CPFF",
                                           "deriv:DGS10-DFF": "fred:T10YFF", "deriv:DGS5-DFF": "fred:T5YFF"}
    for d, s in f.pares_derivadas(cat_real).items():
        assert cat_real[d]["metodo"] == "derivada" and cat_real[s]["duplicado_de"] == d
        a, b = cat_real[d]["componentes"]
        assert {a, b} <= set(f.ids_fred_vivo(cat_real))                   # sus componentes son de la fuente viva


def test_t1_fred_derivada_fecha_maxima_de_componentes(cat_real):
    dbaa = _vivo("fred:DBAA", cat_real, {"2026-10-05": "5.80", "2026-10-06": "5.82", "2026-10-07": "5.79",
                                         "2026-10-08": "5.81"})
    dgs10 = _vivo("fred:DGS10", cat_real, {"2026-10-05": "4.10", "2026-10-06": "4.12", "2026-10-07": "4.15",
                                           "2026-10-09": "4.20"})
    exc = pd.DataFrame({"serie": ["fred:DBAA"], "fecha_referencia": [pd.Timestamp("2026-10-07")],
                        "fecha_publicacion": [pd.Timestamp("2026-10-13")]})
    comp, n = catalogo.aplicar_excepciones(pd.concat([dbaa, dgs10], ignore_index=True), exc, catalogo=cat_real)
    comp = validar_series(comp)
    assert n == 1
    d = f.tabla_derivadas(comp, ["deriv:DBAA-DGS10"], cat=cat_real).set_index("fecha_referencia")
    assert d.index.strftime("%Y-%m-%d").tolist() == ["2026-10-05", "2026-10-06", "2026-10-07"]   # solo fechas comunes
    for ref, fila in d.iterrows():
        pubs = comp[comp["fecha_referencia"] == ref]
        assert fila["fecha_publicacion"] == pubs["fecha_publicacion"].max()
        assert fila["disponible_utc"] == pubs["disponible_utc"].max()
    assert d.loc[pd.Timestamp("2026-10-07"), "fecha_publicacion"] == pd.Timestamp("2026-10-13")   # hereda la excepción
    assert d["valor"].tolist() == pytest.approx([1.70, 1.70, 1.64])
    assert (d["metodo_publicacion"] == "derivada").all() and (d["vintage"] == "unico").all()
    assert (d["fuente"] == "calculo_propio").all() and d["nota"].isna().all()
    assert (d["unidad"] == cat_real["deriv:DBAA-DGS10"]["unidad"]).all()


def test_t1_fred_derivadas_errores(cat_real):
    dgs10 = validar_series(_vivo("fred:DGS10", cat_real, {"2026-10-05": "4.10"}))
    with pytest.raises(KeyError, match="fred:DBAA"):
        f.tabla_derivadas(dgs10, ["deriv:DBAA-DGS10"], cat=cat_real)
    assert f.tabla_derivadas(dgs10, ["deriv:DBAA-DGS10"], cat=cat_real, omitir_incompletas=True).empty
    with pytest.raises(ValueError, match="diferencia"):
        f.tabla_derivadas(dgs10, ["deriv:VOLREAL21_GSPC"], cat=cat_real)


def test_t1_fred_medir_derivadas_contra_fred(cat_real):
    dbaa = _vivo("fred:DBAA", cat_real, {"2026-10-05": "5.80", "2026-10-06": "5.82", "2026-10-07": "5.79"})
    dgs10 = _vivo("fred:DGS10", cat_real, {"2026-10-05": "4.10", "2026-10-06": "4.12", "2026-10-07": "4.15"})
    # Serie de FRED: igual salvo una fecha con 0,02 de diferencia y una fecha que no tiene la derivada.
    baa10y = _vivo("fred:BAA10Y", cat_real, {"2026-10-02": "1.60", "2026-10-05": "1.70", "2026-10-06": "1.72",
                                             "2026-10-07": "1.64"})
    tabla = validar_series(pd.concat([dbaa, dgs10, baa10y], ignore_index=True))
    m = f.medir_derivadas(tabla, cat=cat_real)
    assert m["derivada"].tolist() == ["deriv:DBAA-DGS10"]                 # las demás no tienen componentes
    fila = m.iloc[0]
    assert fila["serie_fred"] == "fred:BAA10Y" and fila["fechas_comunes"] == 3
    assert fila["valores_distintos"] == 1 and fila["dif_max"] == pytest.approx(0.02)
    assert fila["fecha_dif_max"] == "2026-10-06" and fila["solo_fred"] == 1 and fila["solo_derivada"] == 0
    assert fila["adelanto_mediano_dias"] == 1 and fila["derivada_posterior"] == 0     # FRED la calcula en D+1 17:05
    assert "valor" not in m.columns


def test_t1_fred_contrastar_fase1(cat_real):
    vivo = validar_series(_vivo("fred:DGS10", cat_real, {"2026-10-05": "4.10", "2026-10-06": "4.12",
                                                         "2026-10-07": "4.15"}))
    copia = validar_series(_vivo("fred:DGS10", cat_real, {"2026-10-02": "4.00", "2026-10-05": "4.10",
                                                          "2026-10-06": "4.13"}))
    c = f.contrastar_fase1(vivo, copia)
    fila = c.iloc[0]
    assert fila["serie"] == "fred:DGS10" and fila["fechas_comunes"] == 2 and fila["valores_distintos"] == 1
    assert fila["solo_fred"] == 1 and fila["solo_fase1"] == 1 and fila["fin_fred"] == "2026-10-07"
    assert fila["inicio_fase1"] == "2026-10-02" and fila["dif_max"] == pytest.approx(0.01)
    assert f.contrastar_fase1(vivo, copia.iloc[0:0]).empty


def test_t1_fred_fichas_vivo_sin_guardar(tmp_path, cat_real):
    resp = {("series/observations", "DGS10"): {"observations": [_obs("2026-10-05", "4.10", rs="2026-10-06"),
                                                                 _obs("2026-10-06", "4.12", rs="2026-10-07")]},
            ("series/observations", "DFF"): {"observations": [_obs("2026-10-05", "3.90", rs="2026-10-06"),
                                                               _obs("2026-10-06", "3.91", rs="2026-10-07")]},
            ("series/observations", "T10YFF"): {"observations": [_obs("2026-10-05", "0.20", rs="2026-10-06"),
                                                                  _obs("2026-10-06", "0.21", rs="2026-10-07")]}}
    for s in ("DGS10", "DFF", "T10YFF"):
        resp[("series", s)] = {"seriess": [{"id": s, "observation_start": "1962-01-02",
                                            "observation_end": "2026-10-06", "last_updated": "2026-10-07 15:20:01-05"}]}
    cli, _ = _cliente(tmp_path, resp)
    t = f.tabla_fred_vivo(["fred:DGS10", "fred:DFF", "fred:T10YFF"], cliente=cli, cat=cat_real, excepciones=SIN_EXC)
    deriv = f.medir_derivadas(t, cat=cat_real)
    assert deriv["valores_distintos"].tolist() == [0]
    r = f.fichas_fred(cli, t, series_lag=("DGS10",), medir_oas=False, derivadas=deriv, cat=cat_real, guardar=False)
    ficha, oas = r["fichas"]
    assert r["rutas"] == {} and ficha.fuente == "fred_alfred" and oas.fuente == "ice_bofa_oas"
    assert ficha.cobertura_inicio == "2026-10-05" and ficha.cobertura_fin == "2026-10-06"
    assert set(ficha.mediciones["series"]) == {"fred:DGS10", "fred:DFF", "fred:T10YFF"}
    assert ficha.mediciones["existen_en_fred"]["fred:DGS10"]["last_updated_ny"] == "2026-10-07 16:20"
    assert ficha.mediciones["derivadas"][0]["derivada"] == "deriv:DGS10-DFF" and "diaria 3" in ficha.frecuencia
    assert "valor" not in r["lags"].columns and SECRETO not in json.dumps(ficha.mediciones)


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


@pytest.mark.live
def test_t1_fred_live_vivo_sonda(tmp_path):
    """Sonda de la fuente viva: DGS10, DFF y T10YFF completas; la derivada reproduce la serie de FRED."""
    cat = catalogo.cargar_catalogo()
    cli = f.ClienteFred(carpeta=tmp_path)
    ids = ["fred:DGS10", "fred:DFF", "fred:T10YFF"]
    t = f.tabla_fred_vivo(ids, cliente=cli, cat=cat)
    assert set(t["serie"]) == set(ids) and t["fecha_referencia"].max() > pd.Timestamp("2026-07-17")
    m = f.medir_derivadas(t, cat=cat).set_index("derivada").loc["deriv:DGS10-DFF"]
    assert m["fechas_comunes"] > 15000 and m["valores_distintos"] <= 5 and m["derivada_posterior"] == 0
    assert f.descargar_fred_vivo(ids, cliente=cli, cat=cat)["cache"] == ids     # lo de hoy no se repite
    assert "api_key" not in "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
