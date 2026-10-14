"""Tablas y gráficos por institución/familia, solo sobre mediciones obtenidas."""

import re

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from .temporal import vista_a_fecha


def cobertura(frame):
    rows = []
    for (source, family), group in frame.groupby(['fuente', 'tipo'], sort=False):
        ok = group.loc[group['estado'].eq('OK')]
        lag = ok['retraso_evento_dias'].dropna()
        dates = pd.to_datetime(ok.loc[ok['precision_publicacion'].isin(['día', 'hora', 'hora_sin_zona']),
                                     'fecha_publicacion'], errors='coerce', utc=True, format='mixed')
        rows.append({'fuente': source, 'tipo': family, 'intentos': len(group), 'OK': len(ok),
                     'primera_muestra': dates.min(), 'ultima_muestra': dates.max(),
                     'sin_fecha_dia': len(ok) - dates.notna().sum(), 'n_retrasos': len(lag),
                     'retraso_p10_d': lag.quantile(.1), 'retraso_mediana_d': lag.median(),
                     'retraso_p90_d': lag.quantile(.9),
                     'aplicacion_mediana_d': ok['retraso_aplicacion_dias'].median(),
                     'palabras_mediana': ok['palabras'].median(), 'texto_completo': int(ok['texto_completo'].sum()),
                     'versiones_acreditadas': int(ok['disponible_desde_utc'].notna().sum()),
                     'MB': ok.drop_duplicates('sha256')['bytes'].sum() / 1e6})
    return pd.DataFrame(rows)


def graficos(frame, title):
    summary = cobertura(frame)
    labels = (summary['fuente'] + ' · ' + summary['tipo']).tolist()
    # A separate point/series for every family; absence of lag is never zero.
    height = max(4, len(summary) * .37)
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=True)
    y = list(range(len(labels)))
    dated = summary['OK'] - summary['sin_fecha_dia']
    axes[0].barh(y, dated, label='Publicación con día/hora', color='#3274a1')
    axes[0].barh(y, summary['sin_fecha_dia'], left=dated, label='Sin fecha precisa', color='#e5a134')
    axes[0].barh(y, summary['intentos'] - summary['OK'], left=summary['OK'], label='Fallo/pendiente', color='#c9c9c9')
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].set_xlabel('Muestras (denominador explícito)')
    axes[0].legend(fontsize=8)
    for row_number, row in summary.iterrows():
        if row['n_retrasos']:
            axes[1].plot([row['retraso_p10_d'], row['retraso_p90_d']], [row_number, row_number], color='#3274a1')
            axes[1].scatter(row['retraso_mediana_d'], row_number, color='#3274a1', s=25)
        else:
            axes[1].text(.02, row_number, 'sin par de fechas', transform=axes[1].get_yaxis_transform(), fontsize=7)
        if pd.notna(row['aplicacion_mediana_d']):
            axes[2].scatter(row['aplicacion_mediana_d'], row_number, color='#438b59', s=25)
        else:
            axes[2].text(.02, row_number, 'no consta / no procede', transform=axes[2].get_yaxis_transform(), fontsize=7)
    axes[1].set_xlabel('Publicación − evento/periodo/corte (días)\nMediana y P10–P90; n en tabla')
    axes[2].set_xlabel('Publicación − aplicación (días)\nNegativo: anuncio anterior a la aplicación')
    for ax in axes[1:]:
        ax.axvline(0, color='gray', linewidth=.6)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle(title + ' · cobertura temporal y retrasos de la muestra')
    fig.tight_layout()
    plt.show()
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, height), sharey=True)
    for pos, (source, family) in enumerate(zip(summary['fuente'], summary['tipo'])):
        group = frame.loc[frame['fuente'].eq(source) & frame['tipo'].eq(family) & frame['estado'].eq('OK')]
        group = group.loc[group['precision_publicacion'].isin(['día', 'hora', 'hora_sin_zona'])]
        dates = pd.to_datetime(group['fecha_publicacion'], utc=True, errors='coerce', format='mixed')
        axes2[0].scatter(dates, [pos] * len(group), s=20, color='#3274a1')
    axes2[0].set_yticks(y, labels, fontsize=8)
    axes2[0].set_xlabel('Fecha de publicación registrada (no acredita toda la versión)')
    axes2[1].barh(y, summary['palabras_mediana'].fillna(0), color='#438b59')
    axes2[1].set_xlabel('Palabras extraídas: mediana\n0 en metadatos/tablas no equivale a documento vacío')
    axes2[0].invert_yaxis()
    fig2.suptitle(title + ' · publicaciones y volumen textual observado')
    fig2.tight_layout()
    plt.show()
    plt.close(fig)
    plt.close(fig2)


def cambios_texto(documents):
    """Cambio léxico descriptivo; no mide sentimiento ni predice regímenes."""
    rows = []
    groups = {}
    for doc in documents:
        if doc.get('texto') and doc.get('texto_completo') and doc.get('fecha_publicacion'):
            groups.setdefault((doc['fuente'], doc['tipo']), []).append(doc)
    for (source, family), group in groups.items():
        group = sorted(group, key=lambda d: d['fecha_publicacion'])
        for previous, current in zip(group, group[1:]):
            # Use bounded word sets rather than a costly character-level alignment.
            old = set(re.findall(r'\b\w+\b', previous['texto'].lower()))
            new = set(re.findall(r'\b\w+\b', current['texto'].lower()))
            change = 1 - len(old & new) / len(old | new) if old | new else None
            rows.append({'fuente': source, 'tipo': family, 'fecha_publicacion': current['fecha_publicacion'],
                         'cambio_vocabulario_jaccard': change, 'version_anterior': previous['version_id'],
                         'version_actual': current['version_id']})
    return pd.DataFrame(rows)


def mostrar_apartado(sonda, apartado, title):
    frame = sonda.results[apartado]
    print(f'{len(frame)} intentos · {frame["estado"].eq("OK").sum()} muestras obtenidas. '
          'La cobertura describe la muestra; no el archivo histórico completo.')
    display(cobertura(frame).round(2))
    display(frame[['fuente', 'tipo', 'estado', 'fecha_evento', 'fecha_publicacion', 'precision_publicacion',
                   'fecha_aplicacion', 'retraso_evento_dias', 'retraso_aplicacion_dias',
                   'evidencia_publicacion', 'estado_historico', 'url']])
    graficos(frame, title)
    errors = frame.loc[~frame['estado'].eq('OK'), ['fuente', 'tipo', 'estado', 'error', 'url']]
    if not errors.empty:
        print('Fallos y pendientes de esta fuente:')
        display(errors)
    documents = sonda.documents[apartado]
    if documents:
        example = next((d for d in documents if d.get('texto')), documents[0])
        print('Ejemplo ·', example['titulo'], '\nURL:', example['url'], '\nHash:', example['sha256'])
        print((example.get('texto') or str(example.get('ejemplo_datos')))[:1200])
    changes = cambios_texto(documents)
    if not changes.empty:
        print('Cambio de vocabulario entre documentos consecutivos de la muestra (sin interpretar tono):')
        display(changes.round(3))
    historical = vista_a_fecha(frame, sonda.config.corte_utc)
    print(f'TimeGate al {sonda.config.corte_utc}: {len(historical)} textos completos con versión acreditada. '
          'El resto conserva su fecha conocida para la biblioteca general.')
    return cobertura(frame)
