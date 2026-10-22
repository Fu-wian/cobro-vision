# cobro-vision

Prototipo de visión artificial clásica para identificar alimentos en una foto
cenital y calcular un cobro que revisa el cajero. Interfaz prevista: escritorio
con CustomTkinter.

Esta entrega completa la **Etapa 2: esqueleto de punta a punta**, además del
dominio, configuración, catálogo y cobro de la Etapa 1. Una foto pasa por
preprocesamiento, detección del plato, segmentación, características,
clasificación falsa, consolidación y cobro. Incluye fábrica, tiempos,
trazabilidad y `demo --debug`. El clasificador por defecto es `ClasificadorFalso`;
no se entrena ningún modelo. La interfaz y las correcciones quedan para la Etapa 3.

## Instalación

Se requiere Python 3.10 o superior. Desde la raíz del repositorio:

```text
python -m venv .venv
```

Activar el entorno en PowerShell (Windows):

```powershell
.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar el proyecto y las herramientas de desarrollo:

```text
pip install -e ".[dev]"
```

El extra opcional de etiquetado se instala con `pip install -e ".[etiquetado]"`.
Tkinter viene con el instalador de Python de python.org en Windows y macOS;
en Linux puede requerir el paquete del sistema `python3-tk` para la futura GUI.

## Verificación de la Etapa 2

```text
python tests/helpers.py
pytest -q
ruff check src tests
python -m cobro_vision demo --imagen tests/fixtures/plato_sintetico.png --debug
```

`python tests/helpers.py` regenera `tests/fixtures/plato_sintetico.png` usando
`semilla` de `config.yaml`. Abrir ese PNG permite inspeccionar el plato blanco
con fondo gris y sectores artificiales rojo, verde, amarillo y marrón, con ruido
leve y rayas. Las etiquetas del generador son índices de prueba (0 = fondo;
1 a 4 = esos colores), sin asociarlos al menú real.

El comando imprime un recibo ficticio y guarda en
`salidas/debug/plato_sintetico/` las imágenes `01_procesada.png`,
`02_mascara_plato.png`, `03_regiones_crudas.png`, `04_clasificadas.png`,
`05_consolidadas.png`, `recibo.txt` y `traza.json`. Sin `--debug` solo imprime el
recibo. La traza contiene tiempos para las siete etapas y `total`, decisiones de
consolidación e ids de origen de las regiones finales. Los errores de detección
devuelven una transacción en estado `error`, recibo vacío y código de CLI 1.

Las pruebas completas sustituyen el clasificador falso por
`ClasificadorPorColorDominante`, exclusivo de `tests/helpers.py`. Sobre el PNG
de cuatro sectores verifican exactamente **500 centavos (B/. 5.00)**:
pollo (rojo), ensalada (verde), arroz amarillo (amarillo) y frijoles (marrón).
Con tres sectores verifican **425 centavos (B/. 4.25)**. Son asociaciones de
prueba con precios de ejemplo; `demo` conserva las asignaciones ficticias de
`ClasificadorFalso` y puede producir otro total.

Las pruebas de USB usan un dispositivo simulado. La captura con hardware y la
ejecución de `demo` sobre una foto real de `data/raw/` quedan pendientes del
equipo. Para esta última, sustituir la ruta de imagen del comando anterior por
la de una foto real existente.

Los constructores de los segmentadores reciben el diccionario `segmentacion`;
los extractores reciben su propio diccionario de `caracteristicas`.
`CorrectorColor` acepta el diccionario `imagen.correccion_color` (por defecto
`gray_world`). `ConstructorVector` recibe extractores ya creados en el orden
deseado; no selecciona clases concretas.

Decisiones de esta parte marcadas como `TODO(equipo)` en el código:

- `CamaraArchivo` incluye las subcarpetas de sesiones en su recorrido alfabético.
- El ROI de tarjeta gris se interpreta en la imagen ya redimensionada.
- Con ventanas solapadas, cada píxel pertenece a la primera celda aceptada en
  orden por filas y columnas. Un paso mayor que el tamaño deja huecos.
- Los momentos de color se dividen por 180 (H) o 256 (resto), igual que los
  límites superiores de sus histogramas.
- GLCM usa matrices simétricas con distancia 1 y ángulo 0 por defecto.
  `glcm_distancias` y `glcm_angulos` (radianes) permiten ajustar esos valores;
  si hay varios, cada propiedad se promedia entre sus combinaciones.
- `sectores` del generador indica cuántos sectores iguales crear, de 0 a 4.

Decisiones de la parte B marcadas como `TODO(equipo)`:

- La fábrica comunica los nombres de características al clasificador falso por
  `nombres_caracteristicas`, sin cambiar su constructor. Usa la media H o la
  aproxima con el histograma H; sin HSV usa la media del vector como sustituto
  ficticio. Las predicciones no dependen del orden ni del tamaño del lote.
- El posprocesador expone `eventos` y `origenes` de su última llamada, que el
  controlador copia a la traza de la transacción, sin añadir argumentos al contrato.
- La absorción procesa primero el componente menor y vuelve a fusionar tras
  cada cambio. Suma fronteras por clase y desempata por `id_clase`; las proteínas
  de igual área también se priorizan por `id_clase`.
- Si el catálogo no existe, la fábrica lo inicializa con la semilla de ejemplo
  para permitir el `demo` directo. Si ya existe conserva sus precios.

El controlador conserva las regiones crudas y devuelve regiones consolidadas
independientes. También ajusta `area_frac` con la máscara real del plato que ya
conoce, conservando la firma de `CalculadoraCosto.calcular`.

## Verificación de la Etapa 1

```text
pytest -q
ruff check src tests
python -m cobro_vision --help
python -m cobro_vision inicializar-catalogo
python -m cobro_vision inicializar-catalogo
```

La ayuda debe enumerar los nueve comandos. Se puede consultar la ayuda de cada
uno con `python -m cobro_vision <comando> --help`.

La inicialización crea o actualiza `data/catalogo.db` y escribe
`data/clases.txt` en orden alfabético, con `fondo` incluido. Repetirla conserva
diez filas con la semilla de ejemplo; vuelve a aplicar sus precios.

Comprobar a mano el importe de ejemplo desde el catálogo generado:

```text
python -c "from pathlib import Path; from cobro_vision.negocio.catalogo import CatalogoAlimentos; from cobro_vision.dominio.utilidades import formatear_dinero; catalogo = CatalogoAlimentos(Path('data/catalogo.db')); precios = [catalogo.obtener(clase).precio_centavos for clase in ('pollo', 'arroz_amarillo', 'ensalada')]; assert precios == [250, 100, 75]; print(precios, formatear_dinero(sum(precios)))"
```

Salida esperada: `[250, 100, 75] B/. 4.25`. Las pruebas de cobro comprueban además
que dos regiones de pollo producen un único ítem y no duplican su importe.

Si un sandbox impide crear temporales en la carpeta del sistema, se pueden
dirigir las pruebas a una carpeta ignorada del proyecto:

```text
pytest -q --basetemp=salidas/pruebas_tmp_manual
```

Usar esa carpeta exclusivamente para temporales: pytest vacía su `basetemp`.

## Comandos previstos

Todos aceptan `--config RUTA`, por defecto `config.yaml`. La opción se puede
escribir antes o después del subcomando. Las rutas se indican desde el directorio
de trabajo; ejecutar estos ejemplos desde la raíz del repositorio.

| Comando | Etapa de implementación |
|---|---|
| `python -m cobro_vision inicializar-catalogo` | 1 |
| `python -m cobro_vision demo --imagen foto.jpg --debug` | 2 |
| `python -m cobro_vision app` | 3 |
| `python -m cobro_vision validar-etiquetas` | 4 |
| `python -m cobro_vision construir-manifiesto` | 4 |
| `python -m cobro_vision dividir --prueba 0.2` | 4 |
| `python -m cobro_vision entrenar --modo poligonos` | 5 |
| `python -m cobro_vision evaluar clasificador` | 6 |
| `python -m cobro_vision ablacion` | 7 |

`entrenar --modo` admite `poligonos`, `segmentacion` y `ambos`; el argumento es
opcional. `evaluar` admite `clasificador`, `segmentacion` y `sistema`.
`demo --debug` es opcional; `--imagen` y `dividir --prueba` son obligatorios.

`inicializar-catalogo` y `demo` terminan con código **0** al completar o **1** si
hay un error de configuración, entrada, procesamiento o escritura. Los comandos de etapas
posteriores imprimen `No implementado aún (etapa N)` y terminan con código **2**.
Consultar la ayuda termina con código **0**.

## Configuración y datos

`config.yaml` contiene todos los valores por defecto de la sección 5.4 de la
especificación. La carga exige el esquema completo, valida tipos y rangos y
produce `ConfiguracionInvalida` con el campo afectado. El hash usa SHA256 de
12 caracteres; el orden de las claves YAML no lo altera. Las rutas relativas
se interpretan desde el directorio de trabajo.
`data/catalogo_seed.csv` contiene las diez clases de ejemplo de la sección 5.3;
los precios se expresan en centavos enteros. El nombre de `fondo` se escribe
entre comillas porque contiene comas, conforme al formato CSV.

<!-- # TODO(equipo): Confirmar el menú, los id_clase y los precios de ejemplo de data/catalogo_seed.csv. -->

**Los precios de la semilla son ejemplos y deben ser reemplazados por el equipo.**
El catálogo SQLite se genera con `inicializar-catalogo` o al construir el
pipeline si aún no existe; los datos reales y los modelos corresponden al
equipo y a etapas posteriores. Las salidas de `demo --debug` se generan localmente.
Git ignora `data/*` salvo `data/catalogo_seed.csv`, además
de `models/` y `salidas/`; `config.yaml` se versiona.

Decisiones pendientes marcadas también como `TODO(equipo)` en el código:

- `CalculadoraCosto.calcular` no recibe la máscara del plato. `area_frac` usa
  como referencia la unión de las máscaras recibidas, incluidas las de fondo y
  eliminadas; aproxima el área del plato cuando faltan regiones.
- Con varias proteínas tarifadas, `CobroPorPlato` usa la tarifa mayor y
  desempata por `id_clase`. Sin tarifa aplicable usa la suma de componentes.
- Para conservar el retorno `dict` del contrato, `seccion("caracteristicas")`
  devuelve `{"extractores": [...]}`. El hash conserva la lista original del YAML.
- El ROI de tarjeta gris se especifica como `imagen.correccion_color.roi` con
  `[x, y, ancho, alto]`; la corrección usa las coordenadas de la imagen reducida.
- `clases.txt` se guarda junto al catálogo configurado, pues no tiene una ruta
  propia en el esquema; con los valores por defecto es `data/clases.txt`.

## Estructura

```text
src/cobro_vision/
    __init__.py
    __main__.py
    cli.py
    dominio/
    adquisicion/
    preprocesamiento/
    segmentacion/
    caracteristicas/
    clasificacion/
    negocio/
    explicacion/
    controlador/
    vista/
    evaluacion/
tests/
    conftest.py
    test_humo.py
    test_cli.py
    fixtures/
experimentos/
notebooks/
```

Cada subpaquete contiene su `__init__.py`. Los módulos funcionales se añadirán
en las etapas correspondientes.

Documentación recibida: [instrucciones](AGENTS.md),
[especificación técnica](docs/ESPECIFICACION_TECNICA.md) y
[plan de etapas](docs/PLAN_DE_ETAPAS.md).
