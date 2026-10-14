"""Fallos de acceso y versiones de arXiv no deben producir evidencia ficticia."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from macro_riesgo.sonda import ConfigSonda, Sonda
from macro_riesgo.sonda.analysis import cobertura, graficos
from macro_riesgo.sonda.collectors import Collector
from macro_riesgo.sonda.http import HttpClient, Original


class AcquisitionTests(unittest.TestCase):
    def test_all_failed_family_remains_in_table_and_charts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / 'config/sources'
            inventory.mkdir(parents=True)
            spec = dict(apartado=1, fuente='Fed', tipo='Actas FOMC', adaptador='documento',
                        url='https://example.test/minutes.htm')
            (inventory / 'sonda_fuentes.json').write_text(json.dumps([spec]), encoding='utf-8')
            probe = Sonda(ConfigSonda(raiz=root))
            probe.http.get = Mock(side_effect=requests.ConnectionError('network down'))
            frame = probe.ejecutar(1)
            self.assertEqual(frame.iloc[0]['estado'], 'ERROR_ACCESO')
            self.assertTrue(frame['retraso_evento_dias'].isna().all())
            self.assertEqual(cobertura(frame).iloc[0]['OK'], 0)
            with patch('matplotlib.pyplot.show'):
                graficos(frame, 'Acceso fallido')
            self.assertEqual(probe.resumen().iloc[0]['fallos_pendientes'], 1)

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
        with TemporaryDirectory() as temporary:
            client = HttpClient(Path(temporary), use_cache=False, pause=0)
            response = Mock(content=b'old version', url='https://example.test/doc')
            response.raise_for_status.return_value = None
            client.session.get = Mock(return_value=response)
            first = client.get('https://example.test/doc')
            repeated = client.get('https://example.test/doc')
            self.assertEqual(first.retrieved_at, repeated.retrieved_at)
            response.content = b'new version'
            revised = client.get('https://example.test/doc')
            self.assertNotEqual(first.sha256, revised.sha256)
            self.assertTrue((Path(temporary) / 'objects' / first.sha256).exists())
            self.assertTrue((Path(temporary) / 'objects' / revised.sha256).exists())

    def test_personal_requests_are_not_persisted(self):
        with TemporaryDirectory() as temporary:
            client = HttpClient(Path(temporary), pause=0)
            response = Mock(content=b'{}', url='https://example.test/?email=personal@example.test')
            response.raise_for_status.return_value = None
            client.session.get = Mock(return_value=response)
            original = client.get('https://example.test/', params={'email': 'personal@example.test'})
            self.assertNotIn('email', original.url)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
