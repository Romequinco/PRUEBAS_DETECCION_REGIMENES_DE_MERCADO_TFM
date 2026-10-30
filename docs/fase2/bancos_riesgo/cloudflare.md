# Conectar el pipeline a Cloudflare desde cero

Estado: guía de preparación; ninguna de estas acciones se ha ejecutado en la nube.
Documentación y precios consultados el 11 de octubre de 2026.

## Lo que debes hacer tú primero

1. Crea tu cuenta en [Cloudflare](https://dash.cloudflare.com/sign-up), verifica
   el correo e inicia sesión. No necesitas comprar un dominio para empezar.
2. Abre **Storage & databases → R2 → Overview** y completa la activación de R2.
   Cloudflare requiere una suscripción R2 y un proceso de checkout, aunque haya
   uso gratuito incluido. Si pide un método de pago, revísalo tú antes de aceptar.
   No es necesario contratar un plan de dominio Pro para este diseño.
3. Decide el presupuesto y revisa el consumo/facturación. No presupongas que
   una alerta de consumo detiene automáticamente todas las operaciones.
4. Cuando empecemos a implementar, autoriza la conexión de la cuenta o configura
   los secretos en el ejecutor. No envíes claves por el chat.

La creación de la cuenta, verificación y aceptación de facturación requieren
tu intervención. Una vez haya acceso autorizado, los recursos y la configuración
del pipeline se podrán preparar con herramientas; no tienes que crearlos todos
manualmente antes de eso.

Referencia: [primeros pasos R2](https://developers.cloudflare.com/r2/get-started/).

## Recursos que configuraremos

| Recurso | Nombre propuesto | Función |
|---|---|---|
| Bucket R2 desarrollo | `macro-riesgo-dev` | Piloto pequeño y pruebas |
| Bucket R2 producción | `macro-riesgo-prod` | Corpus validado y versiones originales |
| D1 desarrollo | `macro-riesgo-catalog-dev` | Catálogo de prueba |
| D1 producción | `macro-riesgo-catalog-prod` | Metadatos y trazabilidad |
| GitHub | Repositorio del TFM, tema `bancos_riesgo` | Código, documentación y ejecutores integrados en el proyecto común |

Primero crearemos solo los recursos de desarrollo. Producción, Worker e índice
vectorial se incorporarán después de validar el piloto y su consumo.
Los buckets se mantendrán privados, sin acceso público `r2.dev`.
Usaremos almacenamiento **Standard** al empezar.

Si quieres configurar los recursos desde la interfaz:

- Bucket: **R2 → Overview → Create bucket**, introduce el nombre y elige Standard.
- Base de datos: **Storage & databases → D1 → Create database**, introduce el nombre.
- Decide la ubicación/jurisdicción antes de crear los buckets. Si eliges
  jurisdicción UE, utiliza después el endpoint S3 específico de UE. Una
  preferencia de ubicación no equivale a una restricción de jurisdicción.

Referencias: [crear buckets](https://developers.cloudflare.com/r2/buckets/create-buckets/),
[ubicación R2](https://developers.cloudflare.com/r2/reference/data-location/)
y [D1](https://developers.cloudflare.com/d1/get-started/).

## Cómo accederá el recolector Python

El job utilizará la API compatible con S3 para R2. En **R2 → Overview → Account
Details → API Tokens → Manage**, crea un token con **Object Read & Write**,
limitado al bucket del entorno correspondiente. Guarda el **Access Key ID** y
el **Secret Access Key**: el secreto solo se muestra al crearlo. El acceso de
lectura para consultas será otro token, de solo lectura, cuando haga falta.

R2 y D1 utilizan credenciales distintas. Para el catálogo D1, se necesitará un
token Cloudflare con los permisos mínimos de D1 para la cuenta elegida; no se
usará la Global API Key. El job podrá registrar metadatos vía la API de D1.

| Parámetro previsto | Dónde guardarlo |
|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | Variable de entorno; identificador de cuenta |
| `R2_ENDPOINT_URL` | Variable pública de configuración |
| `R2_BUCKET_NAME` | Variable pública de configuración |
| `R2_ACCESS_KEY_ID` | Secret del ejecutor |
| `R2_SECRET_ACCESS_KEY` | Secret del ejecutor |
| `D1_DATABASE_ID` | Variable de configuración |
| `CLOUDFLARE_API_TOKEN` | Secret del ejecutor para D1 |

Endpoint normal: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.
Endpoint para bucket con jurisdicción UE:
`https://<ACCOUNT_ID>.eu.r2.cloudflarestorage.com`.
Los placeholders son documentación y no credenciales reales.

En desarrollo, los secretos se guardarán en un `.env` ignorado por Git.
Para GitHub Actions: **repositorio → Settings → Secrets and variables → Actions**;
usa **Secrets** para claves y **Variables** para los identificadores públicos.
No se creará ningún workflow que suba archivos ni opere con cuentas en esta fase.

Referencia: [S3 y credenciales R2](https://developers.cloudflare.com/r2/get-started/s3/)
y [autenticación R2](https://developers.cloudflare.com/r2/api/tokens/).

## Si se añade un Worker después

El Worker permitirá consultar el catálogo y recuperar documentos autorizados.
Podrá usar bindings `ARCHIVE` para R2 y `CATALOG` para D1, sin incluir claves S3
en un frontend. Un Worker ligero será la capa de consulta; las descargas históricas,
PDF/OCR y procesamiento por lotes seguirán en el ejecutor Python.
No hará falta configurar CORS en R2 para jobs Python de servidor a servidor.
Si se incorporan accesos desde navegador, se diseñarán autenticación y CORS para
esa ruta concreta; CORS por sí solo no protege los documentos.

Referencias: [R2 desde Workers](https://developers.cloudflare.com/r2/get-started/workers-api/)
y [binding D1](https://developers.cloudflare.com/d1/get-started/).

## Costes y límites que revisar

R2 Standard incluye cada mes 10 GB-mes, un millón de operaciones clase A y diez
millones clase B. Fuera de ese uso incluido, almacenamiento cuesta 0,015 USD por
GB-mes; las operaciones se cobran aparte. R2 no cobra transferencia de salida,
pero el ejecutor, embeddings u otros servicios sí pueden tener sus propias tarifas.
La capa gratuita indicada no se aplica a Infrequent Access.

D1 y Workers tienen límites gratuitos independientes. En D1 Free, una base
individual admite hasta 500 MB; los 5 GB de almacenamiento gratuito total de
la cuenta no equivalen a 5 GB por base. Por eso los textos completos y los
fragmentos irán a R2 y el catálogo D1 tendrá solo metadatos e índices necesarios.

No se puede calcular el coste del corpus con los recuentos de la imagen: faltan
tamaños reales, frecuencia de refresco, número de operaciones y volumen de OCR/RAG.
El piloto medirá bytes, peticiones, tiempo y tokens antes de ampliar la carga.

Referencias: [precio R2](https://developers.cloudflare.com/r2/pricing/),
[precio D1](https://developers.cloudflare.com/d1/platform/pricing/),
[límites D1](https://developers.cloudflare.com/d1/platform/limits/)
y [precio Workers](https://developers.cloudflare.com/workers/platform/pricing/).

## Verificación cuando se implemente

1. Autenticar y comprobar que el job ve únicamente el entorno previsto.
2. Subir y recuperar un archivo pequeño; comparar su SHA-256.
3. Registrar y consultar sus metadatos en D1.
4. Repetir el mismo lote y comprobar que no crea duplicados.
5. Probar una versión revisada del mismo documento y conservar ambas versiones.
6. Consultar pasajes en modo general y abrir originales con citas; comprobar que
   la falta de una fecha pública exacta no bloquea un texto validado. Cuando se
   active el modo histórico, verificar además la exclusión de versiones posteriores.
7. Ejecutar el job desde GitHub con el ordenador apagado y revisar el checkpoint.

Nada de esto se presenta como verificado todavía.


## Infraestructura pendiente

Recursos previstos: buckets R2 y catálogos D1 independientes para desarrollo y
producción. Worker e índice vectorial opcionales en fases posteriores.

Se añadirán archivos de infraestructura reproducible al implementar el tema,
cuando haya cuenta, permisos y presupuesto definidos.
No hay Terraform, Wrangler ni despliegues activos en esta estructura.
