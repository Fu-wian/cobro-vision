# Plan de etapas: cobro-vision

Orden de construcción para Codex y tareas que corresponden al equipo. Lee primero `AGENTS.md` y `docs/ESPECIFICACION_TECNICA.md` (en adelante, "la especificación").

---

## 1. Cómo repartir el trabajo

| Quién | Qué |
|---|---|
| **Codex** | Todo el código, las pruebas con datos sintéticos, la CLI, la interfaz y los scripts de evaluación y ablación |
| **Equipo** | Menú y precios reales, espacio de captura, fotos, etiquetado en LabelMe, cronometraje del cobro manual, calibración del umbral, análisis de resultados |

Codex puede dejar **todo el código listo y probado con datos sintéticos** sin ninguna foto real. El rendimiento real del clasificador depende de las fotos y las etiquetas que produzca el equipo; ningún código lo sustituye.

## 2. Cómo trabajar con Codex

1. Copien `AGENTS.md` a la raíz del repositorio y `docs/` tal cual. Hagan un commit inicial antes de empezar.
2. **Una etapa por sesión.** Creen una rama por etapa (`etapa-2-esqueleto`), pidan la etapa, revisen el diff, corran `pytest -q` ustedes mismos y hagan merge antes de pasar a la siguiente.
3. Usen esta plantilla de prompt, cambiando `N` y añadiendo las notas de la etapa:

```
Lee AGENTS.md, docs/ESPECIFICACION_TECNICA.md y la sección "Etapa N" de
docs/PLAN_DE_ETAPAS.md. Implementa únicamente la Etapa N. No adelantes
etapas posteriores. Al terminar ejecuta `pytest -q` y `ruff check src tests`
y responde con el resumen que pide AGENTS.md.
```

4. Revisen a mano los puntos donde estos proyectos suelen fallar sin que las pruebas lo noten:
   - **BGR vs RGB:** `slic` recibe RGB; OpenCV trabaja en BGR.
   - **Escala de polígonos:** las etiquetas están en la imagen original y el sistema trabaja a 640 px.
   - **Orientación EXIF** de las fotos de celular (las dimensiones del JSON deben coincidir).
   - **Fuga de datos:** ningún `plato_id` en `train` y `test` a la vez.
   - **Dinero** en centavos enteros, nunca `float`.
   - **Los mismos** `Preprocesador` y `ConstructorVector` en entrenamiento y operación.
   - **Hilos en la interfaz:** el procesamiento no corre en el hilo principal de Tk y ningún widget se toca desde otro hilo.

---

## 3. Etapas

### Etapa 0: repositorio base

**Entregables**
- `pyproject.toml`, `.gitignore`, `README.md` (instalación y comandos), `config.yaml` completo (sección 5.4), `data/catalogo_seed.csv` (sección 5.3).
- Estructura de paquetes de la sección 3 con sus `__init__.py`.
- `cli.py` con todos los subcomandos de la sección 8. Los que aún no existen imprimen `No implementado aún (etapa N)` y salen con código 2.
- `tests/conftest.py` y una prueba de humo que importa el paquete.

**Criterios de aceptación**
- `pip install -e ".[dev]"` termina sin errores.
- `pytest -q` y `ruff check src tests` pasan.
- `python -m cobro_vision --help` lista los comandos.

---

### Etapa 1: dominio, configuración, catálogo y cobro

**Entregables**
- `dominio/` completo (`modelos.py`, `errores.py`, `utilidades.py` con `mapa_desde_regiones` y formato de dinero).
- `Configuracion` con validación y `hash`.
- `CatalogoAlimentos`, `PoliticaCobro`, `CobroPorComponente`, `CobroPorPlato`, `CalculadoraCosto`.
- Comando `inicializar-catalogo`: crea `data/catalogo.db` desde la semilla **y escribe `data/clases.txt`** (una clase por línea, para el flag `--labels` de LabelMe).

**Criterios de aceptación**
- Pruebas de configuración, catálogo y cobro (tabla de la sección 11).
- Con la semilla de ejemplo, `pollo + arroz_amarillo + ensalada` suma **B/. 4.25** (250 + 100 + 75 centavos).
- Dos regiones de la misma clase generan **un solo ítem**.
- `inicializar-catalogo` ejecutado dos veces no duplica filas.

---

### Etapa 2: esqueleto de punta a punta (núcleo de visión)

El objetivo es que una foto entre y salga un recibo, aunque el clasificador sea falso.

**Entregables**
- `CamaraArchivo`, `Preprocesador`, `CorrectorColor`, `DetectorPlato` (los tres métodos).
- `SegmentadorSuperpixeles` y `SegmentadorVentanas`.
- `ContextoImagen`, `ExtractorHistogramaColor`, `ExtractorTextura`, `ConstructorVector`.
- `Clasificador` (abstracta) y `ClasificadorFalso`.
- `Posprocesador`, `Trazabilidad`, `MedidorTiempos`, `ControladorTransaccion.procesar`, `FabricaPipeline`.
- Comando `demo` con `--debug`.
- `tests/helpers.py` con `generar_plato_sintetico` y `ClasificadorPorColorDominante`; una imagen `tests/fixtures/plato_sintetico.png`.

**Criterios de aceptación**
- `python -m cobro_vision demo --imagen tests/fixtures/plato_sintetico.png --debug` imprime un recibo y escribe en `salidas/debug/` las imágenes `01` a `05`, `recibo.txt` y `traza.json`.
- Prueba de pipeline completo con `ClasificadorPorColorDominante` sobre el plato sintético: el **total calculado es el esperado**.
- `tiempos` contiene las 7 etapas y `total`.
- Pruebas del Posprocesador: fusión, absorción de componentes pequeñas, filtro por clase y regla de proteínas.
- Verificación manual (equipo): `demo` sobre cualquier foto real en `data/raw/` no lanza excepciones (el resultado será aleatorio porque el clasificador es falso).

---

### Etapa 3: interfaz del cajero (CustomTkinter) y correcciones

**Entregables**
- `corregir_region`, `corregir_item`, `quitar_item`, `agregar_item` y `confirmar` en el controlador (con `salidas/transacciones.csv`).
- `vista/render.py`, `vista/presentador.py` y `vista/vista_cajero.py` (`VistaCajero`), y el comando `app`.

**Criterios de aceptación**
- Pruebas de las correcciones: el recibo se recalcula, quedan `Ajuste` registrados y el estado cambia correctamente (incluida `corregir_item` sobre un ítem con varias regiones).
- Pruebas de las funciones puras de `presentador.py` y `render.py`, sin abrir ninguna ventana.
- Prueba de humo de `VistaCajero` (crear y destruir la ventana), omitida automáticamente si no hay pantalla.
- Verificación manual (equipo): `python -m cobro_vision app`; abrir una foto; comprobar que **la ventana no se congela** mientras procesa; ver regiones y total; cambiar un ítem, quitarlo y agregar otro; confirmar y comprobar la fila nueva en el CSV.

---

### Etapa 4: dataset con LabelMe

**Entregables**
- `LectorLabelMe`, `MapaEtiquetas`, `GestorDataset`, `ValidadorEtiquetas`.
- Comandos `validar-etiquetas`, `construir-manifiesto` y `dividir`.
- `escribir_dataset_sintetico` en `tests/helpers.py` (imágenes y JSON de LabelMe coherentes).

**Criterios de aceptación**
- Pruebas de lector, mapa, validador y gestor (sección 11), incluida la **escala de polígonos** y la **ausencia de `plato_id` en ambos splits**.
- Con el dataset sintético, `validar-etiquetas` termina con código 0. Al introducir errores (etiqueta fuera del catálogo, dimensiones distintas, polígono de 2 puntos), termina con código distinto de 0 y un mensaje por error.
- `construir-manifiesto` es reproducible y conserva los `split` ya asignados.
- Verificación manual (equipo): `validar-etiquetas` sobre el primer lote real de 20 platos etiquetados.

---

### Etapa 5: entrenamiento del SVM

**Entregables**
- `ConstructorInstancias` (modos `poligonos`, `segmentacion` y `ambos`), `EntrenadorModelo`, `ClasificadorSVM` (guardar y cargar).
- Comando `entrenar [--modo]`.
- `FabricaPipeline` con soporte para `clasificador.tipo: svm`.

**Criterios de aceptación**
- En el dataset sintético, exactitud ≥ 0.95 sobre el `split` de prueba en los modos `poligonos` y `segmentacion`.
- Guardar y cargar devuelve las mismas predicciones; un modelo con extractores distintos lanza `ModeloIncompatible`.
- Misma semilla, mismos datos y misma configuración producen las mismas métricas (prueba de determinismo).
- Se escriben `models/svm_<fecha>.joblib`, el `.json` hermano y `models/modelo_actual.joblib`.
- Verificación manual (equipo): `python -m cobro_vision entrenar` con datos reales; revisar el `.json` generado.

---

### Etapa 6: evaluación

**Entregables**
- `EvaluadorClasificador` (con tabla de umbral), `EvaluadorSegmentacion`, `EvaluadorSistema`, `RegistroExperimentos`.
- Comando `evaluar clasificador|segmentacion|sistema`.

**Criterios de aceptación**
- Pruebas de las claves de métricas, la forma de la matriz de confusión y que `RegistroExperimentos` agrega una fila por llamada.
- Sobre el dataset sintético se generan los CSV, JSON y PNG en `salidas/evaluacion/<fecha>/`.
- `evaluar sistema` informa `plato_exacto`, precisión y recall de ítems, `total_exacto`, `error_total_medio`, fracción en revisión y tiempos (media y p95).

---

### Etapa 7: ablación y experimentos

**Entregables**
- Comando `ablacion` con las variantes de la sección 10.4, incluidos kNN y Random Forest como línea comparativa.
- Gráficos de barras (matplotlib) de F1 macro y de exactitud del sistema por variante.

**Criterios de aceptación**
- Corre de principio a fin sobre el dataset sintético y agrega **una fila por variante** a `experimentos/resultados.csv`.
- Cada variante usa los mismos `split` y la misma semilla.

---

### Etapa 8: pulido y entrega

**Entregables**
- `README.md` completo: instalación, flujo de trabajo del equipo (capturar, etiquetar, validar, dividir, entrenar, evaluar, usar), solución de problemas comunes.
- Reporte de tiempo por etapa (perfilado simple) y lista de cuellos de botella.
- `docs/PENDIENTES.md` con todos los `TODO(equipo)` encontrados en el código.
- **Opcional, solo si el equipo lo decide:** guardar las correcciones del cajero (imagen y regiones corregidas en formato LabelMe) en `data/correcciones/` para reentrenar.

**Criterios de aceptación**
- `pytest --cov=cobro_vision -q` y `ruff check src tests` limpios.
- Siguiendo solo el README en un clon limpio, se llega a correr `demo` y `app`.

---

## 4. Tareas del equipo (en paralelo con las etapas)

| # | Tarea | Cuándo | Detalle |
|---|---|---|---|
| H1 | Menú, clases y precios | Antes de la etapa 4 (idealmente antes de la 1) | Editar `data/catalogo_seed.csv`. Si cambian los `id_clase` después de etiquetar, hay que reetiquetar. Confirmar con la cafetería cómo cobra |
| H2 | Espacio de captura | En paralelo con las etapas 0 a 3 | Fondo mate gris o negro, luz difusa, cámara fija cenital, marca de posición del plato, exposición y balance de blancos fijos. Las fotos actuales sirven solo para probar el `demo` |
| H3 | Captura de fotos | Antes de la etapa 5 | Mínimo 120 platos, ideal 200; cada clase en al menos 25 platos; varias sesiones y días; combinaciones distintas; incluir platos con oclusión y mezclas reales. Nombres `<SESION>/<PLATO>_<n>.jpg` |
| H4 | Etiquetado en LabelMe | Desde que exista la etapa 4 | Seguir la sección 6 de la especificación. Empezar con 20 platos y correr `validar-etiquetas` antes de seguir |
| H5 | Línea base manual | Antes de la evaluación final | Cronometrar 30 a 50 transacciones del cobro actual (con permiso). Sin esto no se demuestra la reducción de tiempo |
| H6 | Calibración y análisis | Tras las etapas 5 a 7 | Elegir el umbral con la tabla de `evaluar clasificador`; repetir `evaluar sistema`; decidir el modo de entrenamiento; interpretar la matriz de confusión y los errores por tipo (oclusión, mezcla, iluminación) |

## 5. Si algo no rinde

| Síntoma | Medida |
|---|---|
| `evaluar segmentacion` da un techo de exactitud bajo | Ajustar `n_segmentos` y `compacidad`; probar `SegmentadorVentanas`; revisar las imágenes de `--debug` |
| El clasificador confunde alimentos de color parecido | Activar `glcm`; probar Lab; ablación de características; más fotos de esas clases |
| Rinde bien en `poligonos` pero mal en el sistema | Cambiar `entrenamiento.modo` a `segmentacion` o `ambos` |
| Muchas regiones caen en revisión | Recalibrar `umbral_confianza`; revisar si hay clases con pocas muestras |
| El plato no se detecta | Cambiar `plato.metodo`; ajustar `radio_min_frac` y `radio_max_frac`; revisar la iluminación |
| El rendimiento cae al cambiar de día | Revisar la corrección de color; capturar más sesiones; fijar mejor exposición y balance de blancos |
