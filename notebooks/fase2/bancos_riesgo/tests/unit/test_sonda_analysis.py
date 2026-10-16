"""El gráfico compara eventos y publicaciones, sin imputar fechas ausentes."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from macro_riesgo.sonda.analysis import datos_retraso_evento, grafico_retraso_evento


class PublicationLagTests(unittest.TestCase):
    def record(self, **changes):
        row = dict(apartado=1, fuente='Fed', tipo='Actas', estado='OK',
                   fecha_evento='2026-01-28', fecha_publicacion='2026-02-18',
                   precision_publicacion='día', fecha_aplicacion='2026-03-01')
        row.update(changes)
        return row

    def test_only_precise_event_publication_pairs_are_used(self):
        rows = [self.record(), self.record(fecha_evento=None),
                self.record(fecha_evento='2026-01'),
                self.record(fecha_publicacion='2026-02', precision_publicacion='mes'),
                self.record(estado='PENDIENTE'),
                self.record(apartado=14, fecha_evento='2008-09-15'),
                self.record(apartado=12, fecha_evento='2008-09-15')]
        measured = datos_retraso_evento(pd.DataFrame(rows))
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured.iloc[0]['retraso_evento_dias'], 21)

    def test_each_publication_keeps_its_dates_and_zero_day_is_visible(self):
        rows = [self.record(), self.record(tipo='Decisión', fecha_publicacion='2026-01-28')]
        with patch('matplotlib.pyplot.show'):
            fig = grafico_retraso_evento(pd.DataFrame(rows), 'Fed')
        ax = fig.axes[0]
        self.assertEqual([bar.get_width() for bar in ax.patches], [21, 0])
        self.assertIn('2026-01-28 → 2026-02-18', ax.get_yticklabels()[0].get_text())
        self.assertIn('0 d · mismo día', [text.get_text() for text in ax.texts])
        self.assertEqual(len(ax.collections), 2)
        self.assertNotIn(fig.number, plt.get_fignums())

    def test_missing_event_omits_plot_even_when_application_is_known(self):
        frame = pd.DataFrame([self.record(fecha_evento=None)])
        with patch('matplotlib.pyplot.show') as show:
            self.assertIsNone(grafico_retraso_evento(frame, 'Sin evento'))
        show.assert_not_called()

    def test_publication_before_event_is_not_clipped_to_zero(self):
        with patch('matplotlib.pyplot.show'):
            fig = grafico_retraso_evento(pd.DataFrame([
                self.record(fecha_publicacion='2026-01-26')]), 'Anuncio previo')
        self.assertEqual(fig.axes[0].patches[0].get_width(), -2)


if __name__ == '__main__':
    unittest.main()
