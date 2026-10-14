"""Catálogo navegable del estado del arte: una ficha por obra con su resumen y dónde se integra.

Cruza (1) las referencias numeradas de estado_del_arte.md, (2) las fichas de los agentes buscadores
(qué hace, diferencia con el TFM, fase P1–P5, rasgos; fichas.json), (3) las citas [n] del cuerpo del
documento, los hitos y la matriz de posicionamiento (§9) y (4) la pasada extra de competidores de
junio–octubre de 2026 (competidores_2026.json). Escribe catalogo.html en esta misma carpeta (no se versiona).
Uso: python docs/estado_del_arte/catalogo.py
"""
import html
import json
import re
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DOCUMENTO = AQUI / "estado_del_arte.md"
FICHAS = AQUI / "fichas.json"
EXTRA = AQUI / "competidores_2026.json"
SALIDA = AQUI / "catalogo.html"

# Hitos de la línea temporal y sistemas de la matriz (§9) → patrón para reconocer la obra por título/autores
HITOS = {
    "hito régimen 1989": r"Hamilton.*1989|Nonstationary Time Series and the Business Cycle",
    "hito régimen 2012": r"^Regime Shifts: Implications",
    "hito régimen 2020-24 (jump models)": r"persistent states by penalizing jumps|Downside Risk Reduction Using Regime-Switching",
    "hito régimen 2026": r"Geometric Observables for Financial Regime",
    "hito agentes 2020 (RAG)": r"Retrieval-Augmented Generation for Knowledge-Intensive",
    "hito agentes 2024": r"^FinCon|^TradingAgents",
    "hito agentes 2025": r"MarketSenseAI 2\.0",
    "hito agentes 2026 (look-ahead)": r"Can LLM-based Financial Investing Strategies|Look-Ahead-Bench",
    "matriz de posicionamiento": r"Self Driving Portfolio|^OpenPM|Market Regime Council|^TradingAgents|^CLQT",
}

# Correcciones del verificador adversarial de la matriz (2026-10-13), que el buscador no tenía
VERIFICADOR = {
    "Market Regime Council": "El régimen sí está definido: regla tanh(rentabilidad/volatilidad a 30 días) con umbrales ±0,30 (alcista, bajista, volátil), sin validar fuera de muestra. Usa Polymarket, unos 27.000 tuits y datos on-chain; decide con datos hasta t−1, sin vintages.",
    "OpenPM": "Tiene un «regime analyst» que puntúa amplitud, pendiente de la curva y un risk-on score con FRED/ALFRED (no es un modelo de régimen). Puerta point-in-time con la hora de aceptación de EDGAR y vintages ALFRED, certificado de contaminación y crítico determinista que proyecta los pesos del LLM sobre límites. Usa GDELT y 8-K. Sin memo con citas.",
    "The Self Driving Portfolio": "Régimen por un marco de puntuación (4 dimensiones, 4 regímenes del ciclo), sin validar fuera de muestra. 21 métodos de cartera, incluidos risk parity, mínima varianza y máxima diversificación; el CIO-LLM elige bajo restricciones duras de la IPS (tracking error ≤ 6 %). Board memo sin cita por afirmación. No afirma eliminar el look-ahead.",
    "MarketSenseAI 2.0": "Guarda la procedencia de los documentos pero la salida no cita fuentes; el filtro por fecha solo se aplica a los documentos macro.",
    "FinCon": "Los pesos salen de media-varianza con shrinkage; el gestor-LLM solo fija la dirección (comprar, vender, mantener).",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def leer_fichas(ruta: Path = FICHAS):
    obras = {}
    for o in json.loads(ruta.read_text(encoding="utf-8")):
        for k in (norm(o.get("url", "")), norm(o.get("titulo", ""))[:60]):
            obras.setdefault(k, o)
    return obras


def leer_documento():
    texto = DOCUMENTO.read_text(encoding="utf-8")
    cuerpo, _, refs_txt = texto.partition("\n## Referencias")
    refs = {}
    for m in re.finditer(r"^(\d+)\.\s+(.*)$", refs_txt, re.M):
        refs[int(m.group(1))] = m.group(2).strip()
    # citas [n] por sección y frases que las contienen
    citas, frases = defaultdict(set), defaultdict(list)
    seccion = "Introducción"
    for linea in cuerpo.splitlines():
        h = re.match(r"^#{2,3}\s+(.*)", linea)
        if h:
            seccion = h.group(1).strip()
            continue
        for m in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", linea):
            nums = set()
            for parte in re.split(r"\s*,\s*", m.group(1)):
                if re.search(r"[–-]", parte):
                    a, b = re.split(r"\s*[–-]\s*", parte)
                    nums.update(range(int(a), int(b) + 1))
                else:
                    nums.add(int(parte))
            for n in nums:
                citas[n].add(seccion)
                if seccion.startswith("10.") or seccion.startswith("10 "):
                    frase = re.sub(r"\s+", " ", linea.strip(" -*"))
                    if frase not in frases[n]:
                        frases[n].append(frase)
    return refs, citas, frases


def titulo_de_ref(ref: str) -> str:
    m = re.search(r"\*(.+?)\*", ref)
    return m.group(1) if m else ref


def url_de_ref(ref: str) -> str:
    m = re.search(r"https?://\S+", ref)
    return m.group(0).rstrip(".)") if m else ""


def uso_hito(titulo: str, autores: str) -> list:
    return [k for k, pat in HITOS.items() if re.search(pat, titulo, re.I) or re.search(pat, f"{autores} {titulo}", re.I)]


def chip_rasgos(r: dict) -> str:
    nombres = [("regimen", "régimen"), ("multiagente", "multi-agente"), ("rag_citas", "RAG/citas"),
               ("anti_lookahead", "point-in-time"), ("cartera", "cartera"), ("datos_alt", "datos alt.")]
    out = []
    for k, n in nombres:
        v = (r or {}).get(k)
        v = {True: "si", False: "no"}.get(v, v or "no")
        if v in ("si", "parcial"):
            out.append(f'<span class="r {v}">{n}{" ◐" if v == "parcial" else ""}</span>')
    return "".join(out)


def main() -> None:
    obras_j = leer_fichas()
    refs, citas, frases = leer_documento()
    fichas = []
    for n, ref in sorted(refs.items()):
        t, u = titulo_de_ref(ref), url_de_ref(ref)
        o = obras_j.get(norm(u)) or obras_j.get(norm(t)[:60]) or {}
        autores_anio = ref.split("*")[0].strip(" .")
        fichas.append(dict(
            n=n, titulo=t, cabecera=autores_anio, url=u or o.get("url", ""), bloque=o.get("bloque", "(ya citada en el repo)"),
            que_hace=o.get("que_hace", ""), diferencia=o.get("diferencia_con_nosotros", ""),
            fase=o.get("fase", ""), rasgos=o.get("rasgos", {}), seminal=o.get("seminal"),
            secciones=sorted(citas.get(n, [])), adopta=frases.get(n, [])[:3], hitos=uso_hito(t, autores_anio),
            correcciones=(o.get("correcciones", "") + " " + next((v for k, v in VERIFICADOR.items() if t.startswith(k)), "")).strip(),
            extra=False))
    if EXTRA.exists():
        for i, o in enumerate(json.loads(EXTRA.read_text(encoding="utf-8")), start=1):
            fichas.append(dict(
                n=f"E{i}", titulo=o["titulo"], cabecera=f'{o["autores"]} ({o["fecha"]})', url=o["url"],
                bloque="9.bis Competidores jun–oct 2026", que_hace=o["que_hace"], diferencia=o.get("amenaza_al_hueco", ""),
                fase="", rasgos=o, seminal=False, secciones=["9.bis Pasada extra de competidores (junio–octubre 2026)"],
                adopta=[o.get("nota", "")], hitos=uso_hito(o["titulo"], o["autores"]), correcciones="", extra=True))

    bloques = sorted({f["bloque"] for f in fichas})
    n_hitos = sum(1 for f in fichas if f["hitos"])
    n_sin = sum(1 for f in fichas if not f["que_hace"])
    e = html.escape
    tarjetas = []
    for f in fichas:
        donde = []
        if f["hitos"]:
            donde.append("<b>Hito o matriz:</b> " + e(", ".join(f["hitos"])))
        if f["secciones"]:
            donde.append("<b>Estado del arte:</b> " + e(" · ".join(f["secciones"])))
        if f["adopta"]:
            donde.append("<b>Qué tomamos:</b> " + e(" ".join(f["adopta"]))[:700])
        if not donde:
            donde.append("Solo en la lista de referencias y en obras.csv (contexto del frente).")
        busca = norm(" ".join([f["titulo"], f["cabecera"], f["bloque"], f["que_hace"], f["fase"]]))
        tarjetas.append(f'''
<article class="card" data-bloque="{e(f["bloque"])}" data-hito="{1 if f["hitos"] else 0}" data-busca="{busca}">
  <div class="top"><span class="num">{e(str(f["n"]))}</span><span class="blk">{e(f["bloque"][:70])}</span>
    {'<span class="sem">seminal</span>' if f["seminal"] else ''}{'<span class="sl">hito o matriz</span>' if f["hitos"] else ''}</div>
  <h3>{e(f["titulo"])}</h3>
  <div class="meta">{e(f["cabecera"])}{(' · <a href="' + e(f["url"]) + '" target="_blank" rel="noopener">fuente</a>') if f["url"] else ''}
    {(' · <span class="fase">' + e(f["fase"]) + '</span>') if f["fase"] else ''}</div>
  <p><b>Qué hace.</b> {e(f["que_hace"]) or '<i>Sin ficha del buscador (obra que ya citaba el repo o añadida por el redactor).</i>'}</p>
  {('<p><b>Frente a nuestro TFM.</b> ' + e(f["diferencia"]) + '</p>') if f["diferencia"] else ''}
  <div class="rasgos">{chip_rasgos(f["rasgos"])}</div>
  <div class="donde">{"<br>".join(donde)}</div>
  {('<div class="corr">Corregido por el verificador: ' + e(f["correcciones"]) + '</div>') if f["correcciones"] else ''}
</article>''')

    opciones = "".join(f'<option value="{e(b)}">{e(b[:80])}</option>' for b in bloques)
    pagina = f'''<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catálogo del estado del arte</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {{ --ink:#16140f; --text:#3a352d; --muted:#7b7468; --rule:#e3dbcf; --cream:#f4f0e7; --card:#fff; --red:#a32d2d; --gold:#b98b2e; --green:#1f6f54; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ink:#f4efe6; --text:#d6cfc2; --muted:#9a9284; --rule:#3a342b; --cream:#16130e; --card:#211c16; }} }}
:root[data-theme="dark"] {{ --ink:#f4efe6; --text:#d6cfc2; --muted:#9a9284; --rule:#3a342b; --cream:#16130e; --card:#211c16; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--cream); color:var(--text); font:15px/1.5 Inter, system-ui, sans-serif; }}
header {{ max-width:1100px; margin:0 auto; padding:28px 16px 8px; }}
h1 {{ font-family:Fraunces, Georgia, serif; color:var(--ink); font-size:34px; margin:4px 0 8px; }}
.k {{ font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--red); }}
.intro {{ max-width:900px; }} .intro li {{ margin:4px 0; }}
.bar {{ position:sticky; top:0; background:var(--cream); border-bottom:1px solid var(--rule); z-index:2; }}
.bar div {{ max-width:1100px; margin:0 auto; padding:10px 16px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
input, select {{ font:inherit; padding:7px 10px; border:1px solid var(--rule); border-radius:8px; background:var(--card); color:var(--text); }}
input[type=search] {{ flex:1; min-width:180px; }} #cuenta {{ color:var(--muted); font-size:13px; }}
main {{ max-width:1100px; margin:0 auto; padding:14px 16px 60px; display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 500px), 1fr)); gap:14px; }}
.card {{ background:var(--card); border:1px solid var(--rule); border-radius:12px; padding:14px 16px; }}
.card h3 {{ font-family:Fraunces, Georgia, serif; color:var(--ink); font-size:18px; line-height:1.25; margin:6px 0 4px; }}
.top {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center; font-size:11px; }}
.num {{ font-weight:700; color:var(--red); }} .blk {{ color:var(--muted); }}
.sem, .sl {{ font-weight:700; padding:1px 7px; border-radius:20px; }} .sem {{ background:#f6ecd6; color:#7a5a17; }} .sl {{ background:#f3dede; color:var(--red); }}
.meta {{ font-size:13px; color:var(--muted); margin-bottom:6px; }} .meta a {{ color:var(--red); }}
.fase {{ font-weight:600; color:var(--green); }}
.card p {{ margin:6px 0; font-size:14px; }}
.rasgos {{ display:flex; gap:5px; flex-wrap:wrap; margin:6px 0; }} .r {{ font-size:11px; padding:1px 7px; border-radius:20px; border:1px solid var(--rule); }}
.r.si {{ background:#e3f0e9; color:var(--green); border-color:#bfdccc; }} .r.parcial {{ color:var(--gold); }}
.donde {{ font-size:13px; border-top:1px dashed var(--rule); padding-top:7px; margin-top:6px; }}
.corr {{ font-size:12px; color:var(--gold); margin-top:6px; }}
</style></head><body>
<header><div class="k">TFM MIAX · Estado del arte</div><h1>Catálogo de obras y cómo se integran</h1>
<div class="intro"><p>{len(fichas)} obras: {len(refs)} de estado_del_arte.md (verificadas una a una por un segundo agente) y {len(fichas) - len(refs)} de la pasada extra de competidores de junio–octubre 2026. {n_hitos} son hitos de la línea temporal o filas de la matriz de posicionamiento (§9). Los resúmenes son de los agentes, a partir del resumen y la página de cada obra: no son lecturas completas.</p>
<ul><li><b>Qué hace</b> y <b>Frente a nuestro TFM</b>: ficha del agente buscador (1–3 frases).</li>
<li><b>Estado del arte</b>: secciones de estado_del_arte.md donde se cita. <b>Qué tomamos</b>: frase de la §10, la adopción por fase.</li>
<li><b>Hito o matriz</b>: si es un hito de la línea temporal o una fila de la matriz (§9). La fase P1–P5 es la del TFM donde se usa. Los rasgos marcan lo que la obra tiene (◐ = parcial).</li></ul></div></header>
<div class="bar"><div><input type="search" id="q" placeholder="Buscar título, autor, tema…">
<select id="b"><option value="">Todos los frentes</option>{opciones}</select>
<label><input type="checkbox" id="s"> solo hitos y matriz</label><span id="cuenta"></span></div></div>
<main id="lista">{"".join(tarjetas)}</main>
<script>
const q=document.getElementById('q'), b=document.getElementById('b'), s=document.getElementById('s'), c=document.getElementById('cuenta');
const cards=[...document.querySelectorAll('.card')];
const norm=t=>t.toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g,'').replace(/[^a-z0-9]+/g,'');
function filtra(){{ const t=norm(q.value); let n=0;
  cards.forEach(x=>{{ const ok=(!t||x.dataset.busca.includes(t))&&(!b.value||x.dataset.bloque===b.value)&&(!s.checked||x.dataset.hito==='1');
    x.style.display=ok?'':'none'; if(ok) n++; }}); c.textContent=n+' obras'; }}
[q,b,s].forEach(x=>x.addEventListener('input',filtra)); filtra();
</script></body></html>'''
    SALIDA.write_text(pagina, encoding="utf-8")
    print(f"{SALIDA.name}: {len(fichas)} fichas ({len(refs)} del estado del arte + {len(fichas) - len(refs)} extra); "
          f"{n_hitos} hitos o filas de la matriz; {n_sin} sin ficha de buscador")


if __name__ == "__main__":
    main()
