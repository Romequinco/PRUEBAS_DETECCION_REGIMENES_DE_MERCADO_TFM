"""TANDA 2 — Re-ejecuta los detectores 02-12 para propagar las métricas aditivas
(silhouette + IC bootstrap de cobertura) a sus CSV, preservando la narrativa markdown.

Orden rápido->lento (01 ya hecho aparte; 00 no es detector; 13 se hace después).
Cada notebook se escribe al terminar -> resumible ante apagón. Continúa ante fallo.
Salida: _method_report.md + log por stdout.
"""
import time
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"

# orden aproximado rápido->lento (según baseline: 02~17s, 03~227s, 06~109s; 04~19min,
# 05>40min; 07/08/10/11/12 desconocidos -> colocados según coste esperado).
ORDER = [
    "01_rule_vix_threshold.ipynb",
    "02_rule_composite_riskoff.ipynb",
    "06_garch_t_vol.ipynb",
    "03_clustering_gmm.ipynb",
    "07_changepoint_online.ipynb",
    "10_turbulence_mahalanobis.ipynb",
    "09_jump_model.ipynb",
    "12_deep_ae_regime.ipynb",
    "08_hmm_tstudent.ipynb",
    "04_hmm_gaussian_2s.ipynb",
    "11_msgarch_regime.ipynb",
    "05_markov_switching_var.ipynb",
]


def run_nb(name: str) -> tuple[int, float]:
    t0 = time.time()
    nb = nbf.read(NB_DIR / name, as_version=4)
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    nbf.write(nb, NB_DIR / name)
    errs = sum(1 for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error")
    return errs, time.time() - t0


def main() -> None:
    report = ["# TANDA 2 — Re-ejecución con métricas aditivas\n"]
    total_err = 0
    for name in ORDER:
        try:
            errs, dt = run_nb(name)
            total_err += errs
            report.append(f"- **{name}** — {dt:6.1f}s, errores={errs}")
            print(f"[OK] {name} {dt:.1f}s err={errs}", flush=True)
        except Exception as e:  # noqa: BLE001
            total_err += 1
            report.append(f"- **{name}** — FALLO: {type(e).__name__}: {e}")
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        (ROOT / "_method_report.md").write_text("\n".join(report), encoding="utf-8")
    report.append(f"\n## Total errores de celda: {total_err}")
    (ROOT / "_method_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n=== DONE total_err={total_err} ===", flush=True)


if __name__ == "__main__":
    main()
