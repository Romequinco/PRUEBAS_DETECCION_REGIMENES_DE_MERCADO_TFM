"""HTTP limitado y caché de originales: ninguna credencial se escribe a disco."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import requests


@dataclass(frozen=True)
class Original:
    content: bytes
    url: str
    retrieved_at: str
    sha256: str
    cache: bool = False


class HttpClient:
    def __init__(self, directory: Path, use_cache=True, timeout=(8, 18), pause=0.35):
        self.directory = directory
        self.use_cache = use_cache
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'TFM-source-probe/1.0 (small academic samples)'
        self.memory = {}

    def get(self, url, params=None, headers=None):
        # Authenticated/personal requests remain in memory; do not persist email/token.
        private = bool(headers) or 'email' in (params or {})
        key = sha256(json.dumps([url, params or {}, headers or {}], sort_keys=True).encode()).hexdigest()
        if self.use_cache and key in self.memory:
            return self.memory[key]
        folder = self.directory / key
        if self.use_cache and not private and (folder / 'metadata.json').exists():
            meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            body = (self.directory / 'objects' / meta['sha256']).read_bytes()
            if sha256(body).hexdigest() != meta['sha256']:
                raise ValueError('Hash de caché inválido: vuelve a descargar este original')
            result = Original(body, meta['url'], meta['retrieved_at'], meta['sha256'], True)
        else:
            time.sleep(self.pause)
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            if not response.content:
                raise ValueError('Respuesta vacía: revisar el enlace oficial')
            digest = sha256(response.content).hexdigest()
            observed_at = datetime.now(timezone.utc).isoformat()
            public_url = url if private else response.url
            result = Original(response.content, public_url, observed_at, digest)
            if not private:
                objects = self.directory / 'objects'
                objects.mkdir(parents=True, exist_ok=True)
                (objects / digest).write_bytes(result.content)
                version_meta = objects / (digest + '.json')
                if version_meta.exists():
                    first_seen = json.loads(version_meta.read_text(encoding='utf-8'))['retrieved_at']
                    result = Original(response.content, public_url, first_seen, digest)
                else:
                    version_meta.write_text(json.dumps({
                        'url': public_url, 'retrieved_at': observed_at, 'sha256': digest,
                    }), encoding='utf-8')
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'metadata.json').write_text(json.dumps({
                    'url': public_url, 'retrieved_at': result.retrieved_at, 'sha256': digest,
                }), encoding='utf-8')
                with (folder / 'observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'observed_at': observed_at, 'sha256': digest}) + '\n')
        self.memory[key] = result
        return result
