# Bancos centrales, riesgo sistémico y literatura

Repositorio para descubrir, recopilar y conservar documentos de bancos
centrales, estabilidad financiera y literatura económica; preparar un archivo
consultable y un subconjunto para RAG con control de disponibilidad temporal.

La referencia inicial es la imagen «Bancos centrales, riesgo sistémico y
literatura» aportada por el usuario el 11 de octubre de 2026. Las cantidades,
etiquetas OK/MIXTO/BLOQUEADA y retrasos de esa imagen son objetivos o referencias
por verificar, no resultados de una recolección ejecutada.

## Estado

Solo hay documentación y carpetas reservadas con `.gitkeep`. No hay scripts,
dependencias instaladas, SQL, workflows ejecutables, infraestructura desplegada,
credenciales, documentos descargados ni integración Cloudflare activa.
El repositorio existente utiliza `main` y está conectado a
`feraltauste1/Bancos-centrales-riesgo-sistem-tico-y-literatura` en GitHub.
Esta estructura está preparada localmente; todavía no se ha hecho commit ni push
de estos archivos.

## Diseño elegido

- Python para recolectores, descargas, PDF/HTML, normalización y lotes históricos.
- Cloudflare R2 para originales, texto extraído, JSONL y resultados derivados.
- Cloudflare D1 para catálogo de documentos, versiones, procedencia y ejecuciones.
- GitHub Actions como primer ejecutor de lotes pequeños, manuales y programados.
- RAG selectivo en una fase posterior; el resto se conserva en el archivo.
- Worker de consulta e índice vectorial opcionales cuando exista un corpus validado.

R2 es almacenamiento de objetos. D1 aporta las consultas SQL del catálogo.
La recolección y el procesamiento requieren un ejecutor: R2 no ejecuta los jobs.
Esta selección es una propuesta de arquitectura, no servicios ya contratados.

## Estructura

```text
Bancos-centrales-riesgo-sistem-tico-y-literatura/
├── docs/                       Diseño y guías de puesta en marcha
├── config/
│   ├── sources/                Fuentes, URLs y tipos de documento
│   ├── policies/               TimeGate, acceso, retención y selección RAG
│   └── environments/           Parámetros públicos de dev y prod
├── src/macro_riesgo/
│   ├── collectors/             Descubrir documentos por institución
│   ├── fetching/               Descargar con reintentos y límites
│   ├── parsing/                Extraer HTML, PDF y tablas
│   ├── normalization/          Metadatos y documento común
│   ├── storage/                Acceso R2 y registro D1
│   ├── temporal/               Disponibilidad histórica y versiones
│   ├── chunking/               Párrafos, Q&A y secciones con padre
│   ├── retrieval/              Archivo y RAG con filtro temporal
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

`pyproject.toml`, lockfile, módulos Python, esquemas JSON, migraciones SQL y
workflows se crearán al empezar la implementación. No se han añadido archivos
vacíos que puedan confundirse con una aplicación ejecutable.

## Qué leer primero

1. [Conectar Cloudflare desde cero](docs/CLOUDFLARE.md).
2. [Arquitectura y flujo de datos](docs/ARQUITECTURA.md).
3. [Inventario de fuentes de la imagen](docs/FUENTES.md).
4. [Modelo de datos y TimeGate](docs/MODELO_Y_TIMEGATE.md).
5. [Orden de implementación](docs/ROADMAP.md).

El repositorio de Mi app calendario es un proyecto independiente. Su preparación
de despliegue no forma parte de este pipeline.
