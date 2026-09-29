# Ablación HSMM: D8 frente a D13

Esta carpeta está separada del banco histórico D1–D12 y de
`../metrics_master.csv`.

- `metrics_d8_reference.csv`: D8 HMM t-Student reejecutado como referencia de
  la ablación.
- `metrics_d13_hsmm.csv`: D13 HSMM t-Student evaluado sobre la misma ventana.
- `d13_*.png`: figuras regenerables de `A1_hsmm_ablation.ipynb` (ignoradas por
  Git).

Ambas filas deben compartir `oos_start`, `oos_end` y `n_oos`. El notebook A1
comprueba estas tres condiciones antes de compararlas.

Resultado: la duración explícita no reduce el *switching* ni aumenta la
duración media; D8 se mantiene por parsimonia.
