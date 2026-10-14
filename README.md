# Bancos centrales, riesgo sistémico y literatura

Repositorio para descubrir, recopilar y conservar documentos de bancos
centrales, estabilidad financiera y literatura económica; preparar un archivo
amplio, consultable y trazable para que los agentes recuperen literatura y
evidencias al preparar sus decisiones.

La prioridad es la cobertura, la calidad del texto y la facilidad de consulta.
Se recopilará por lotes, sin requerir ingesta inmediata. Las fechas y versiones
se conservan como contexto; TimeGate será opcional para evaluaciones históricas.

La referencia inicial es la imagen «Bancos centrales, riesgo sistémico y
literatura» aportada por el usuario el 11 de octubre de 2026. Las cantidades,
etiquetas OK/MIXTO/BLOQUEADA y retrasos de esa imagen son objetivos o referencias
por verificar, no resultados de una recolección ejecutada.

## Estado

Hay documentación y una sonda ejecutable de fuentes, con módulos Python para
descarga, extracción, auditoría temporal y gráficos. La sonda conserva muestras
locales fuera de Git. No hay recolección masiva, SQL, infraestructura desplegada
ni integración Cloudflare activa.
El repositorio existente utiliza `main` y está conectado a
`feraltauste1/Bancos-centrales-riesgo-sistem-tico-y-literatura` en GitHub.
El notebook, a cargo de **Fernando Dapena Tauste**, mantiene sus 14 apartados
independientes y consulta 94 familias de documentos, filas de datos y metadatos.
Muestra cobertura de la muestra, retrasos, fechas de publicación/aplicación,
calidad de extracción, volumen y errores. No representa el histórico completo.

Para probarlo, ejecuta la configuración común y después el apartado que te
interese. `MODO_RAPIDO=True` limita las muestras descubiertas a tres por familia
y la extracción a 40 páginas por PDF; el modo ampliado también tiene límites.
`HF_TOKEN` y `UNPAYWALL_EMAIL` se leen del entorno cuando hacen falta. Las demás
consultas no requieren API key. Cada apartado muestra sus gráficos, fechas,
precisión, evidencia y URL. Los fallos y las credenciales pendientes se registran
sin detener las demás fuentes. El TimeGate histórico exige acreditar la versión
textual; una fecha de publicación conocida no basta por sí sola.

El código está en `src/macro_riesgo/sonda/` y el inventario en
`config/sources/sonda_fuentes.json`. Los originales, JSONL y CSV se guardan en
`data/cache/sonda/`, `data/local/sonda/` y `data/exports/sonda/`, ignorados por Git.
La [lectura del notebook de Daniel](docs/ANALISIS_NOTEBOOK_REFERENCIA.md) explica
la estructura adaptada y las decisiones temporales.

## Entorno Python y ejecución

El entorno virtual local está en `.venv/` y está excluido de Git.
Las dependencias del notebook y de Jupyter están en `requirements.txt`.
Desde la raíz del repositorio, abre JupyterLab en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

Abre `01_fuentes_prioridad_gratuita.ipynb` y selecciona el kernel
`Python (.venv - Bancos centrales TFM)`. Puedes comprobar el intérprete con
`import sys; print(sys.executable)`:
debe apuntar a `.venv\Scripts\python.exe` dentro de este repositorio.
En VS Code, selecciona ese mismo intérprete como kernel del notebook.

Para activar el entorno en una terminal PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell no permite la activación, puedes seguir usando directamente
`.\.venv\Scripts\python.exe`, como en el comando de JupyterLab.
Para recrear el entorno en otra instalación con Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name bancos-centrales-tfm --display-name "Python (.venv - Bancos centrales TFM)"
```

## Diseño elegido

- Python para recolectores, descargas, PDF/HTML y normalización por lotes.
- Cloudflare R2 para originales, texto extraído, JSONL y resultados derivados.
- Cloudflare D1 para catálogo de documentos, versiones, procedencia y ejecuciones.
- GitHub Actions como primer ejecutor de lotes pequeños, manuales y programados.
- Catálogo y búsqueda de texto para todo el corpus validado; acceso al documento completo.
- RAG progresivo sobre las familias prioritarias, ampliable al medir utilidad y coste.
- Refrescos semanales o mensuales como punto de partida, según familia de fuentes.
- Worker de consulta e índice vectorial opcionales cuando exista un corpus validado.

R2 es almacenamiento de objetos. D1 aporta las consultas SQL del catálogo.
La recolección y el procesamiento requieren un ejecutor: R2 no ejecuta los jobs.
Esta selección es una propuesta de arquitectura, no servicios ya contratados.

## Estructura

```text
Bancos-centrales-riesgo-sistem-tico-y-literatura/
├── docs/                       Diseño y guías de puesta en marcha
├── notebooks/                  Fuentes recomendadas: notebook, Markdown y HTML
├── config/
│   ├── sources/                Fuentes, URLs y tipos de documento
│   ├── policies/               Calidad, acceso, retención y consulta histórica opcional
│   └── environments/           Parámetros públicos de dev y prod
├── src/macro_riesgo/
│   ├── collectors/             Descubrir documentos por institución
│   ├── fetching/               Descargar con reintentos y límites
│   ├── parsing/                Extraer HTML, PDF y tablas
│   ├── normalization/          Metadatos y documento común
│   ├── storage/                Acceso R2 y registro D1
│   ├── temporal/               Fechas, versiones y TimeGate opcional
│   ├── chunking/               Párrafos, Q&A y secciones con padre
│   ├── retrieval/              Búsqueda, archivo y RAG para agentes
│   ├── analysis/               Tono, diferencias y derivados
│   └── orchestration/          Jobs, checkpoints y recuperación
├── schemas/                    Futuros contratos y esquema del catálogo
├── migrations/                 Futuras migraciones SQL D1
├── jobs/                       Definiciones de lotes y calendarios
├── infra/cloudflare/           Documentación de recursos y permisos
├── tests/
│   ├── unit/                   Pruebas por módulo
│   ├── integration/            Pruebas R2/D1 con entorno de desarrollo
│   └── fixtures/               Ejemplos pequeños que se puedan redistribuir
├── data/                       Copias locales ignoradas y ejemplos permitidos
└── .github/workflows/          Lugar reservado; no hay workflows todavía
```

Los módulos Python de la sonda están implementados. `pyproject.toml`, lockfile,
esquemas de producción, migraciones SQL y workflows del pipeline siguen pendientes.

Para ejecutar todas las celdas y sincronizar las versiones de lectura:

```powershell
.\.venv\Scripts\python.exe jobs/ejecutar_notebook_sonda.py
.\.venv\Scripts\python.exe jobs/exportar_notebook.py
```

Las pruebas de reglas temporales se ejecutan con:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/unit -v
```

## Qué leer primero

1. [Notebook: fuentes de calidad, priorizando gratuitas](notebooks/01_fuentes_prioridad_gratuita.ipynb).
   También disponible en [HTML para abrir en el navegador](notebooks/01_fuentes_prioridad_gratuita.html)
   y [Markdown](notebooks/01_fuentes_prioridad_gratuita.md).
2. [Objetivo del corpus y consulta por agentes](docs/CORPUS_PARA_AGENTES.md).
3. [Conectar Cloudflare desde cero](docs/CLOUDFLARE.md).
4. [Arquitectura y flujo de datos](docs/ARQUITECTURA.md).
5. [Inventario de fuentes de la imagen](docs/FUENTES.md).
6. [Modelo de datos y TimeGate opcional](docs/MODELO_Y_TIMEGATE.md).
7. [Orden de implementación](docs/ROADMAP.md).

El repositorio de Mi app calendario es un proyecto independiente. Su preparación
de despliegue no forma parte de este pipeline.
