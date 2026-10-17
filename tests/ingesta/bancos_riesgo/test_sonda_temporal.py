"""Pruebas de reglas que podrían introducir sesgo de anticipación."""

import unittest

import pandas as pd

from ingesta.bancos_riesgo.sonda.parsing import application_date, event_date, publication_clock, publication_html, soup
from ingesta.bancos_riesgo.sonda.runner import _dia_primero
from ingesta.bancos_riesgo.sonda.temporal import add_metrics, anuncio_arxiv, available_at, parse_date, vista_a_fecha


class TemporalTests(unittest.TestCase):
    def record(self, **changes):
        result = dict(estado='OK', texto_completo=True, version_acreditada=True,
                      fecha_publicacion='2026-10-14', precision_publicacion='día',
                      zona_fuente='Europe/Madrid', publicacion_utc=None)
        result.update(changes)
        return result

    def test_partial_dates_do_not_get_a_day(self):
        for raw in ('2026', '2026-10', 'October 2026'):
            normalized, precision, utc = parse_date(raw)
            self.assertNotEqual(precision, 'día')
            self.assertIsNone(utc)
            self.assertEqual(normalized, raw)

    def test_clock_requires_timezone(self):
        self.assertEqual(parse_date('2026-10-14T14:00:00')[1], 'hora_sin_zona')
        self.assertEqual(parse_date('2026-10-14T14:00:00-04:00')[2], '2026-10-14T18:00:00+00:00')

    def test_ambiguous_numeric_date_follows_source_convention(self):
        self.assertEqual(parse_date('03/04/2026')[0], '2026-03-04')
        self.assertEqual(parse_date('03/04/2026', dayfirst=True)[0], '2026-04-03')
        self.assertEqual(parse_date('03/04/2026', dayfirst=None), ('03/04/2026', 'ambigua', None))
        self.assertEqual(parse_date('13/04/2026', dayfirst=None)[0], '2026-04-13')
        self.assertFalse(_dia_primero('Fed'))
        self.assertTrue(_dia_primero('Banco de España'))
        self.assertIsNone(_dia_primero('Fuente sin zona'))

    def test_arxiv_is_available_from_announcement_not_submission(self):
        # Lunes antes de las 14:00 ET -> anuncio el lunes a las 20:00 ET (+1 día de margen).
        self.assertEqual(anuncio_arxiv('2009-05-04T12:00:00Z'), '2009-05-06T00:00:00+00:00')
        # Jueves por la tarde y viernes por la mañana -> anuncio del domingo.
        self.assertEqual(anuncio_arxiv('2009-05-07T19:00:00Z', margen_dias=0), '2009-05-11T00:00:00+00:00')
        self.assertEqual(anuncio_arxiv('2009-05-08T12:00:00Z', margen_dias=0), '2009-05-11T00:00:00+00:00')
        # Viernes por la tarde y fin de semana -> anuncio del lunes.
        self.assertEqual(anuncio_arxiv('2009-05-09T12:00:00Z', margen_dias=0), '2009-05-12T00:00:00+00:00')
        sent = '2009-05-04T12:00:00+00:00'
        record = self.record(precision_publicacion='hora', publicacion_utc=sent, anuncio_utc=anuncio_arxiv(sent))
        frame = pd.DataFrame([dict(record, disponible_desde_utc=available_at(record))])
        self.assertTrue(vista_a_fecha(frame, '2009-05-05T12:00:00+00:00').empty)
        self.assertEqual(len(vista_a_fecha(frame, '2009-05-06T00:00:00+00:00')), 1)

    def test_published_label_preserves_day(self):
        self.assertEqual(parse_date('Published on 30 July 2026'), ('2026-07-30', 'día', None))

    def test_day_is_usable_only_next_local_day(self):
        self.assertEqual(available_at(self.record()), '2026-10-14T22:00:00+00:00')
        self.assertEqual(available_at(self.record(fecha_publicacion='2026-03-29')),
                         '2026-03-29T22:00:00+00:00')

    def test_unknown_version_partial_text_and_unknown_zone_excluded(self):
        for changes in ({'version_acreditada': False}, {'texto_completo': False},
                        {'zona_fuente': None}, {'precision_publicacion': 'mes'}):
            self.assertIsNone(available_at(self.record(**changes)))

    def test_history_filters_future_and_unverified_rows(self):
        rows = [self.record(), self.record(version_acreditada=False),
                self.record(fecha_publicacion='2026-10-15'), self.record(texto_completo=False)]
        for row in rows:
            row['disponible_desde_utc'] = available_at(row)
        frame = pd.DataFrame(rows)
        self.assertEqual(len(vista_a_fecha(frame, '2026-10-14T21:59:59Z')), 0)
        self.assertEqual(len(vista_a_fecha(frame, '2026-10-14T22:00:00Z')), 1)
        with self.assertRaises(ValueError):
            vista_a_fecha(frame, '2026-10-14')

    def test_negative_application_lag_is_kept(self):
        row = self.record(fecha_evento='2026-10-13', fecha_aplicacion='2026-10-20')
        result = add_metrics(pd.DataFrame([row])).iloc[0]
        self.assertEqual(result['retraso_evento_dias'], 1)
        self.assertEqual(result['retraso_aplicacion_dias'], -6)

    def test_missing_or_partial_date_is_not_zero_lag(self):
        rows = [self.record(fecha_evento=None, fecha_aplicacion=None),
                self.record(fecha_publicacion='2026-10', precision_publicacion='mes',
                            fecha_evento='2026-10-01', fecha_aplicacion='2026-10-01')]
        result = add_metrics(pd.DataFrame(rows))
        self.assertTrue(result['retraso_evento_dias'].isna().all())
        self.assertTrue(result['retraso_aplicacion_dias'].isna().all())

    def test_lag_uses_local_calendar_not_utc_day(self):
        row = self.record(fecha_publicacion='2026-10-14T23:30:00-04:00', precision_publicacion='hora',
                          fecha_evento='2026-10-14', fecha_aplicacion=None)
        self.assertEqual(add_metrics(pd.DataFrame([row])).iloc[0]['retraso_evento_dias'], 0)

    def test_partial_reference_date_does_not_become_first_day(self):
        rows = [self.record(fecha_evento=partial, fecha_aplicacion=partial)
                for partial in ('2026', '2026-10')]
        result = add_metrics(pd.DataFrame(rows))
        self.assertTrue(result['retraso_evento_dias'].isna().all())
        self.assertTrue(result['retraso_aplicacion_dias'].isna().all())

    def test_meeting_and_publication_are_separate(self):
        event, _ = event_date('https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm',
                              '', 'Fed', 'Actas FOMC')
        self.assertEqual(event, '2026-01-28')
        page = soup('<meta name="dateModified" content="2026-02-18"><h1>January 28, 2026</h1>')
        self.assertIsNone(publication_html(page, 'Fed')[0])

    def test_ecb_accounts_end_of_meeting(self):
        text = 'Meeting of 9-10 September 2026\n8 October 2026'
        self.assertEqual(event_date('', text, 'BCE', 'Monetary policy accounts')[0], '2026-09-10')

    def test_application_needs_explicit_phrase(self):
        self.assertEqual(application_date('with effect from 18 September 2024')[0], '2024-09-18')
        self.assertIsNone(application_date('The report covers September 2024')[0])

    def test_release_clock_respects_est_and_edt(self):
        self.assertEqual(publication_clock('January 28, 2026', 'For release at 2:00 p.m. EST', 'Fed'),
                         '2026-01-28T14:00:00-05:00')
        self.assertEqual(publication_clock('July 29, 2026', 'For release at 2:00 p.m. EDT', 'Fed'),
                         '2026-07-29T14:00:00-04:00')


if __name__ == '__main__':
    unittest.main()
