# 00 — Estado del arte de detección de regímenes de mercado

> **Pendiente de FASE 2.** Este fichero se rellenará con la investigación
> profunda de la literatura (familias, supuestos, fortalezas/debilidades, papers
> seminales, aplicaciones a finanzas) y terminará con la PROPUESTA de lista
> definitiva de detectores a implementar, ordenada de baseline a avanzado.
> Todas las referencias se añadirán a `docs/references.bib`.

## Familias candidatas a cubrir (de la consigna del proyecto)
- Reglas / umbrales (sobre VIX, drawdown, spreads).
- Clustering (k-means / GMM sobre features).
- HMM (gaussiano, t-Student, GMM-HMM).
- Markov-Switching (statsmodels).
- Change-point (CUSUM, bayesiano, `ruptures`).
- Redes neuronales (autoencoder + clustering, LSTM).
- Econométricos: ARIMA, ARCH, GARCH y variantes regime-switching (RS-GARCH).

## Semilla bibliográfica (de la propuesta TFM, ya en references.bib)
Hamilton (1989), Ang & Bekaert (2002), Guidolin & Timmermann (2007),
Kritzman, Page & Turkington (2012), Estrella & Mishkin (1998), Gulko (2002),
Lopez de Prado (2018). Se ampliará con literatura específica de cada familia.
