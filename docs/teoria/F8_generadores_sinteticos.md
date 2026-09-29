# F8 — Generadores de datos sintéticos (ESQUELETO)

> **Estado: esqueleto.** Ficha nueva, sin equivalente en la Capa 1. Fija la
> estructura del estado del arte de la fase de sintéticos (notebooks
> `15_sinteticos_generadores` a `18_sinteticos_aumento`) y la conecta con el código ya
> previsto en `src/regimenes/sinteticos/` (interfaz `Generador`, registro,
> `validacion.py` con firmas) y con `configs/sinteticos.yaml`. **Nada de lo que sigue es
> todavía un resultado**; las secciones marcadas *(pendiente)* deben rellenarse con
> literatura verificada y claves BibTeX reales antes de citarse en la memoria.

## Para qué sirven en este TFM

Con 17 crisis evaluables en la pista A y 9 en la pista B, el banco de pruebas tiene muy poca
potencia estadística para separar detectores (ver la limitación de significancia de la Capa 1
en `docs/historia/capa1/memoria/99_conclusions.md`). Los generadores sintéticos se usan para:

1. **Laboratorio con verdad conocida**: trayectorias cuya cadena de regímenes se conoce, para
   medir recall/precisión de los detectores sin la ambigüedad del etiquetado histórico.
2. **Aumento de datos**: entrenar detectores con más episodios de crisis (TSTR: *train on
   synthetic, test on real*) sin tocar el test real.
3. **Pruebas de estrés** de la máquina de fusión y del *pseudo-live* (notebook `20_pseudolive`).

## Familias de generadores *(pendiente de desarrollar)*

| Familia | Idea | Verdad de régimen | Riesgo principal | Carpeta de código |
|---|---|---|---|---|
| Bootstrap por bloques / estacionario | Remuestrear bloques contiguos del panel real | Hereda la del bloque (etiqueta histórica) | No crea episodios nuevos; copia el pasado | `sinteticos/parametricos/` |
| Bootstrap condicionado a régimen | Remuestrear bloques dentro de cada régimen y encadenarlos con una cadena de Markov estimada | Sí (cadena simulada) | Discontinuidades en las uniones | `sinteticos/parametricos/` |
| Paramétricos con cambio de régimen (MS-VAR, MS-GARCH, HMM-t) | Ajustar un modelo de F3/F4/F5 y simular de él | Sí (exacta) | Solo reproduce lo que el modelo supone (circularidad con los detectores de la misma familia) | `sinteticos/parametricos/` |
| Procesos con saltos / volatilidad estocástica | Difusiones con saltos, Heston, Hawkes | Parcial | Calibración frágil; multivariante costoso | `sinteticos/parametricos/` |
| Neuronales (GAN, VAE, difusión, neural SDE) | Aprender la distribución conjunta de trayectorias | No (salvo condicionamiento explícito) | Memorización, colapso de modos, pocos datos | `sinteticos/neuronales/` |

Para cada familia completar: definición y supuestos · ecuaciones clave · qué hechos
estilizados reproduce y cuáles no · coste · implementación disponible · referencias.

## Criterios de validación

Alineados con las funciones de `regimenes.sinteticos.validacion` y la lista
`validacion.metricas` de `configs/sinteticos.yaml`:

| Criterio | Función | Qué compara *(detalle pendiente)* |
|---|---|---|
| Fidelidad marginal | `fidelidad_marginal` | Distribución por feature: momentos, colas, test KS |
| Fidelidad de dependencia | `fidelidad_dependencia` | ACF de retornos y de \|r\| (agrupamiento de volatilidad), correlaciones cruzadas |
| Fidelidad de regímenes | `fidelidad_regimenes` | Duraciones de régimen y matriz de transición real vs sintética |
| Utilidad (TSTR) | `utilidad_tstr` | Detector ajustado en sintético y evaluado en real con el protocolo walk-forward y el ranking ADR-003 |
| Memorización / privacidad | `memorizacion` | Distancia al vecino real más cercano: detecta copias del entrenamiento |

Checklist de hechos estilizados a exigir *(pendiente de cuantificar)*: colas pesadas,
agrupamiento de volatilidad, efecto apalancamiento, ausencia de autocorrelación lineal en
retornos, correlaciones que suben en crisis.

## Riesgos metodológicos

- **Fuga de información**: el generador solo puede ajustarse con datos anteriores a
  `entrenamiento.fin_train`; nunca con el tramo que luego se usa para evaluar.
- **Circularidad**: un generador MS-VAR favorece al detector D05 (misma familia de supuestos).
  El laboratorio debe usar varios generadores de familias distintas y declararlo.
- **Verdad de régimen artificial**: la etiqueta simulada no es la taxonomía histórica de
  crisis (p. ej. el punto ciego de 2013); las conclusiones del laboratorio no sustituyen al
  benchmark real.
- **Memorización** en generadores neuronales con pocos datos (reproducir la GFC casi literal).
- **Sobreconfianza por volumen**: miles de trayectorias no añaden información sobre crisis
  que el generador no sabe producir.

## Bibliografía *(pendiente)*

Ninguna clave de esta familia está aún en `docs/references.bib`. Candidatas a verificar y
añadir en un `F8_generadores_sinteticos.bib` propio (prefijo de clave sugerido `synth_`):

- Hechos estilizados de los retornos financieros (Cont, 2001).
- Bootstrap por bloques móviles (Künsch, 1989) y estacionario (Politis y Romano, 1994).
- Simulación desde modelos de cambio de régimen: reutilizar `hamilton1989`,
  `vol_haasmittnikpaolella2004`, `hmm_bulla2011` (ya en `docs/references.bib`).
- Generadores neuronales de series temporales: TimeGAN (Yoon et al., 2019), Quant GANs
  (Wiese et al., 2020), *market generators* (Buehler et al., 2020).
- Evaluación TSTR (Esteban et al., 2017).
