# AGENTS.md

Instrucciones para agentes de código (Codex u otros) que trabajen en este repositorio.

## Proyecto

`cobro-vision` es un prototipo en Python que recibe una **foto cenital de un plato de comida para llevar**, identifica los alimentos y calcula el costo total. Lo usará el cajero de la cafetería del Edificio 3 (UTP) para validar el resultado antes de cobrar. No reemplaza el sistema de pago.

Técnica: visión artificial **clásica**. Preprocesamiento, segmentación en superpíxeles (SLIC), histogramas de color + textura, clasificador SVM. Las etiquetas de entrenamiento se hacen a mano con **LabelMe** (polígonos).

**No usar** redes neuronales, modelos preentrenados, descargas de modelos ni servicios externos.

Interfaz del cajero: aplicación de escritorio con **CustomTkinter** (no web).

## Lee esto antes de escribir código

1. `docs/ESPECIFICACION_TECNICA.md`: qué construir (contratos, formatos de datos, algoritmos, CLI). **Es la fuente de verdad.**
2. `docs/PLAN_DE_ETAPAS.md`: en qué orden construirlo y cómo saber que una etapa terminó.

Si el código existente contradice la especificación, gana la especificación. Si la especificación es ambigua, elige la opción más simple, déjala anotada con `# TODO(equipo): <duda>` y continúa sin bloquearte.

## Comandos

```
pip install -e ".[dev]"          # instalar en modo editable
pytest -q                        # pruebas: deben pasar siempre
ruff check src tests             # estilo
python -m cobro_vision --help    # CLI
```

## Convenciones de código

- Python 3.10 o superior. Type hints en todas las firmas públicas. `dataclasses` para datos.
- Nombres de clases, funciones y variables **en español, sin tildes ni ñ** (`SegmentadorSuperpixeles`, `construir_vector`). Docstrings breves en español.
- Rutas con `pathlib.Path`. Debe funcionar igual en Windows, macOS y Linux: nada de scripts `.sh` obligatorios ni comandos específicos de un sistema operativo.
- Sin `print()` en el código de la librería: usar `logging`. Solo la CLI imprime al usuario.
- Toda aleatoriedad usa la `semilla` de `config.yaml`. Mismos datos + misma configuración = mismos resultados.
- Dinero: enteros en **centavos** (`int`) y `decimal.Decimal` para mostrar. Nunca `float` para precios.
- Imágenes: `numpy.ndarray` `uint8` en formato **BGR** (convención de OpenCV) salvo que la especificación diga otra cosa.
- Los parámetros ajustables viven en `config.yaml`, no en el código.

## Reglas obligatorias

1. **No cambiar las firmas de la sección 4 de la especificación** sin pedir confirmación. Son los contratos entre módulos.
2. **Una etapa a la vez.** Implementa solo la etapa que se te pida. No adelantes etapas posteriores.
3. **Pruebas siempre.** Cada módulo nuevo trae sus pruebas en `tests/`. Una etapa no está terminada si `pytest -q` falla.
4. **No inventes datos reales.** No fabriques fotos, etiquetas ni precios "reales". Para pruebas usa imágenes sintéticas generadas por código (en `tests/`). Los precios de `data/catalogo_seed.csv` son de ejemplo y deben quedar marcados como tales.
5. **No hagas commit de** `data/`, `models/`, `salidas/` ni de ningún archivo mayor a 5 MB. Revisa `.gitignore`.
6. **Sin fuga de datos.** Todas las fotos de un mismo `plato_id` van al mismo conjunto (entrenamiento o prueba). Nunca dividas por foto ni por región.
7. **No agregues dependencias** fuera de las de la especificación (sección 2) sin explicar por qué en el resumen final.
8. Los componentes intercambiables (`Camara`, `Segmentador`, `ExtractorCaracteristicas`, `Clasificador`, `PoliticaCobro`) se eligen **solo** en `FabricaPipeline`, a partir de `config.yaml`. Ningún otro módulo instancia variantes concretas.
9. `VistaCajero` no contiene lógica de visión ni de precios: solo llama a `ControladorTransaccion`. Los widgets de Tk se tocan **solo desde el hilo principal**: el procesamiento corre en un hilo aparte y devuelve el resultado por una cola.
10. Si falta información del equipo (menú real, precios, política de cobro), usa el valor de ejemplo, márcalo con `TODO(equipo)` y sigue.

## Al terminar cada etapa

Responde con este resumen, breve:

- Qué archivos creaste o modificaste.
- Resultado de `pytest -q` y de `ruff check` (pega la última línea de cada uno).
- Los comandos que el equipo debe correr para verificar a mano.
- Decisiones que tomaste donde la especificación era ambigua (las mismas que dejaste como `TODO(equipo)`).
- Cualquier criterio de aceptación de la etapa que **no** se cumplió, y por qué.

## Lo que NO es responsabilidad del agente

Capturar fotos, etiquetar en LabelMe, definir el menú y los precios reales, y calibrar umbrales con datos reales. Eso lo hace el equipo; el plan indica cuándo.
