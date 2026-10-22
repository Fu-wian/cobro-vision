# cobro-vision

Prototipo de visión artificial clásica para identificar alimentos en una foto
cenital y calcular un cobro que revisa el cajero. Interfaz prevista: escritorio
con CustomTkinter.

Esta entrega llega hasta la **Etapa 1: dominio, configuración, catálogo y cobro**.
Incluye los modelos de datos, validación YAML, catálogo SQLite y políticas de
cobro por componente o por plato. La visión y la interfaz corresponden a etapas
posteriores.

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

`inicializar-catalogo` termina con código **0** al completar o **1** si hay un
error de configuración, de semilla o de escritura. Los comandos de etapas
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
El catálogo SQLite se genera con `inicializar-catalogo`; los datos de captura,
modelos y salidas de visión corresponden a etapas posteriores.
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
  `[x, y, ancho, alto]`; esta etapa solo valida esos parámetros.
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
