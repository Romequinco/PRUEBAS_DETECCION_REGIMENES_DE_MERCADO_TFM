"""Tablas y gráficos por institución/familia, solo sobre mediciones obtenidas."""

import re
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from .temporal import add_metrics, vista_a_fecha


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


def datos_retraso_evento(frame):
    """Pares observados por documento; un corte analítico no es un evento causal."""
    measured = add_metrics(frame)
    valid = measured['estado'].eq('OK') & measured['retraso_evento_dias'].notna()
    # El apartado 14 asigna 2008-09-15 como corte ilustrativo a todos los papers.
    # Las etiquetas retrospectivas de bases de crisis tampoco fechan su creación.
    if 'apartado' in measured:
        valid &= ~measured['apartado'].isin([12, 14])
    return measured.loc[valid].copy()


def grafico_retraso_evento(frame, title):
    """Una fila por publicación: evento → publicación y diferencia en días."""
    measured = datos_retraso_evento(frame)
    if measured.empty:
        print('Retraso entre evento y publicación: sin gráfico; no hay pares de fechas '
              'verificados de un evento y su publicación. Un corte de análisis '
              'o una fecha de aplicación no sustituyen al evento.')
        return None

    fig, ax = plt.subplots(figsize=(13, max(3.2, len(measured) * .78 + 1.2)))
    labels = []
    for pos, (_, row) in enumerate(measured.iterrows()):
        lag = row['retraso_evento_dias']
        color = '#3274a1' if lag >= 0 else '#c0782a'
        ax.barh(pos, lag, color=color, height=.45, alpha=.8)
        # El punto permite ver las publicaciones del mismo día (barra de ancho 0).
        ax.scatter(lag, pos, color=color, s=30, zorder=3)
        annotation = '0 d · mismo día' if lag == 0 else f'{lag:g} d'
        ax.annotate(annotation, (lag, pos), xytext=(7 if lag >= 0 else -7, 0),
                    textcoords='offset points', ha='left' if lag >= 0 else 'right',
                    va='center', fontsize=9)
        event = str(row['fecha_evento'])[:10]
        publication = str(row['fecha_publicacion'])[:10]
        labels.append(fill(f'{row["fuente"]} · {row["tipo"]}', width=48)
                      + f'\n{event} → {publication}')
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', linewidth=.8)
    ax.grid(axis='x', alpha=.2)
    left, right = measured['retraso_evento_dias'].min(), measured['retraso_evento_dias'].max()
    padding = max(1, (max(0, right) - min(0, left)) * .22)
    ax.set_xlim(min(0, left) - padding, max(0, right) + padding)
    ax.set_xlabel('Días de calendario desde el evento hasta la publicación\n'
                  '0 = mismo día · positivo = después · negativo = antes')
    ax.set_title(title + '\nRetraso de publicación respecto al evento')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    omitted = frame['estado'].eq('OK').sum() - len(measured)
    if omitted:
        print(f'{omitted} muestras obtenidas quedan fuera del gráfico de retraso: '
              'sin par de fechas preciso o sin evento asociado verificable; véase la tabla.')
    return fig


def graficos(frame, title):
    grafico_retraso_evento(frame, title)
    summary = cobertura(frame)
    labels = (summary['fuente'] + ' · ' + summary['tipo']).tolist()
    height = max(4, len(summary) * .37)
    y = list(range(len(labels)))
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
    coverage = cobertura(frame)
    # Solo las columnas numéricas: redondear las de fecha no tiene efecto y avisa.
    display(coverage.round({column: 2 for column in coverage.select_dtypes('number').columns}))
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
