"""Fallos de acceso y versiones de arXiv no deben producir evidencia ficticia."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import requests

from ingesta.bancos_riesgo.sonda import ConfigSonda, Sonda
from ingesta.bancos_riesgo.sonda.runner import INVENTARIO
from ingesta.bancos_riesgo.sonda.analysis import cobertura, graficos
from ingesta.bancos_riesgo.sonda.collectors import Collector
from ingesta.bancos_riesgo.sonda.http import BloqueadaPorRobots, HttpClient, Original


def robots(permite):
    return patch('ingesta.comun.http.robots_permite', return_value=permite)


class AcquisitionTests(unittest.TestCase):
    def test_all_failed_family_remains_in_table_and_charts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = dict(apartado=1, fuente='Fed', tipo='Actas FOMC', adaptador='documento',
                        url='https://example.test/minutes.htm')
            (root / 'inventario.json').write_text(json.dumps([spec]), encoding='utf-8')
            probe = Sonda(ConfigSonda(raiz=root, directorio_corpus=root / 'corpus',
                                      inventario=root / 'inventario.json'))
            probe.http.get = Mock(side_effect=requests.ConnectionError('network down'))
            frame = probe.ejecutar(1)
            self.assertEqual(frame.iloc[0]['estado'], 'ERROR_ACCESO')
            self.assertTrue(frame['retraso_evento_dias'].isna().all())
            self.assertEqual(cobertura(frame).iloc[0]['OK'], 0)
            with patch('matplotlib.pyplot.show'):
                graficos(frame, 'Acceso fallido')
            self.assertEqual(probe.resumen().iloc[0]['fallos_pendientes'], 1)
            self.assertTrue((root / 'corpus/local/sonda/apartado_01.jsonl').exists())
            self.assertEqual(probe.guardar_tablas(), root / 'corpus/exports/sonda')
            self.assertFalse((root / 'data').exists())

    def test_arxiv_rejects_latest_version_when_v1_was_requested(self):
        atom = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/0905.0220v2</id><published>2009-05-02T12:00:00Z</published>
        <updated>2010-01-01T00:00:00Z</updated></entry></feed>'''
        http = Mock()
        http.get.return_value = Original(atom, 'https://export.arxiv.org/api/query', '2026-10-14', 'hash')
        collector = Collector(http, ConfigSonda(raiz=Path('.')))
        with self.assertRaisesRegex(ValueError, 'no confirma'):
            collector.api(dict(adaptador='arxiv', identificador='0905.0220', tipo='Preprint'))
        self.assertEqual(http.get.call_args.kwargs['params']['id_list'], '0905.0220v1')
        self.assertEqual(http.get.call_count, 1)

    def test_refresh_preserves_previous_original_and_first_observation(self):
        with TemporaryDirectory() as temporary, robots(True):
            client = HttpClient(Path(temporary), use_cache=False)
            response = Mock(content=b'old version', url='https://example.test/doc')
            patch('ingesta.comun.http.get', return_value=response).start()
            self.addCleanup(patch.stopall)
            first = client.get('https://example.test/doc')
            repeated = client.get('https://example.test/doc')
            self.assertEqual(first.retrieved_at, repeated.retrieved_at)
            response.content = b'new version'
            revised = client.get('https://example.test/doc')
            self.assertNotEqual(first.sha256, revised.sha256)
            self.assertTrue((Path(temporary) / 'objects' / first.sha256).exists())
            self.assertTrue((Path(temporary) / 'objects' / revised.sha256).exists())

    def test_personal_requests_are_not_persisted(self):
        with TemporaryDirectory() as temporary, robots(True):
            client = HttpClient(Path(temporary))
            response = Mock(content=b'{}', url='https://example.test/?email=personal@example.test')
            with patch('ingesta.comun.http.get', return_value=response):
                original = client.get('https://example.test/', params={'email': 'personal@example.test'}, api=True)
            self.assertNotIn('email', original.url)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_robots_disallow_blocks_download_without_network(self):
        with TemporaryDirectory() as temporary, robots(False), patch('ingesta.comun.http.get') as get:
            client = HttpClient(Path(temporary))
            with self.assertRaises(BloqueadaPorRobots):
                client.get('https://example.test/doc.pdf')
            get.assert_not_called()

    def test_documented_api_is_not_gated_by_robots(self):
        with TemporaryDirectory() as temporary, robots(False) as permite:
            response = Mock(content=b'{}', url='https://api.example.test/x')
            with patch('ingesta.comun.http.get', return_value=response) as get:
                HttpClient(Path(temporary)).get('https://api.example.test/x', api=True, por_segundo=0.5)
            permite.assert_not_called()
            self.assertEqual(get.call_args.kwargs['por_segundo'], 0.5)

    def test_robots_block_and_exclusion_are_recorded_not_downloaded(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [dict(apartado=5, fuente='PBoC (China)', tipo='Informe', adaptador='documento',
                          url='https://example.test/a.pdf', excluida='robots.txt con Disallow: /'),
                     dict(apartado=5, fuente='RBA (Australia)', tipo='Actas', adaptador='documento',
                          url='https://example.test/b.pdf')]
            (root / 'inventario.json').write_text(json.dumps(specs), encoding='utf-8')
            probe = Sonda(ConfigSonda(raiz=root, directorio_corpus=root / 'corpus',
                                      inventario=root / 'inventario.json'))
            with robots(False), patch('ingesta.comun.http.get') as get:
                frame = probe.ejecutar(5)
            get.assert_not_called()
            self.assertEqual(frame['estado'].tolist(), ['EXCLUIDA', 'BLOQUEADA_ROBOTS'])

    def test_http_error_without_response_keeps_status_only(self):
        error = requests.HTTPError('404 Client Error: Not Found for url: https://x.test/?email=personal@example.test')
        record = Sonda._error(dict(apartado=1, fuente='Fed', tipo='Actas', adaptador='documento'),
                              'https://x.test/', error, 0)
        self.assertEqual(record['error'], 'HTTP 404; revisar acceso oficial')

    def test_shipped_inventory_excludes_plan_blocked_sources(self):
        inventory = json.loads(INVENTARIO.read_text(encoding='utf-8'))
        excluded = {spec['url'] for spec in inventory if spec.get('excluida')}
        for spec in inventory:
            if 'pbc.gov.cn' in spec.get('url', '') or 'gfsr' in spec.get('url', '').lower():
                self.assertIn(spec['url'], excluded)


if __name__ == '__main__':
    unittest.main()
