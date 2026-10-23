# Perfil de `Posprocesador.consolidar`

Medición local del 23 de octubre de 2026. No cambian las reglas de la sección
7.10 ni las firmas públicas de la sección 4.

## Método

- Cinco fotos existentes: `data/raw/S1/P001_a.jpg` a `P005_a.jpg`, en orden.
- `config.yaml` sin modificar: hash `224e72d1cd81`, semilla 42, imagen procesada
  de 640 × 360 píxeles, SLIC con `n_segmentos: 100`. SLIC produjo entre 86 y 93
  regiones tras su filtro de área; 100 es su parámetro de entrada.
- Se usa el `ClasificadorFalso` seleccionado por esa configuración. Estos tiempos
  describen sus predicciones, sin entrenar ni cargar otro modelo.
- Python 3.12.10, NumPy 2.5.2, SciPy 1.18.1 y OpenCV 5.0.0 en Windows.
- Cada versión procesa cada foto con el pipeline. Luego se mide `consolidar`
  tres veces sobre las mismas regiones crudas y se informa la mediana, sin
  incluir serialización ni perfilado. La media final promedia esas cinco medianas.
- Por separado, `cProfile` acumula una llamada a `consolidar` por foto. Los
  tiempos completos del pipeline provienen de su medidor, una ejecución por foto;
  son orientativos y no tienen la misma repetición que el posprocesamiento.
- Se guardó una copia del módulo original antes de editarlo, junto con las
  salidas de referencia. Todos los perfiles y resultados locales están en
  `salidas/perfilado/`, excluido de Git.

## Cuello de botella y cambio

El perfil original acumuló **8,276 s** en cinco llamadas a `consolidar`.
`_fusionar` consumió **6,823 s** (82,4% del total): cada absorción reconstruía
las componentes de **todas** las clases. Se ejecutaron 700 etiquetados y se
crearon 3.504 regiones intermedias. Dentro de esa fusión, `_crear_region`
consumió 4,761 s: recorría la máscara completa con `nonzero` para obtener la caja
y con `sum` para el área de cada componente. Otros costos fueron 487 consultas
al catálogo (0,789 s) y 70 construcciones del mapa de regiones (0,332 s).
Los tiempos de funciones anidadas están incluidos en sus llamadoras.

La implementación ahora:

1. Conserva las componentes por clase durante una llamada y solo invalida las
   clases de origen y destino después de cada absorción. Sigue reasignando los
   ids en el mismo orden de clase y etiqueta antes de elegir la siguiente pequeña.
2. Obtiene las cajas con `scipy.ndimage.find_objects` y las áreas con
   `numpy.bincount`, una vez por etiquetado. Las máscaras siguen siendo booleanas
   del tamaño completo de la imagen. Mantiene literalmente la selección de
   píxeles y `mean` en float64 para conservar los scores bit a bit.
3. Cuenta fronteras directamente en el mapa de clases, dentro de la caja de la
   componente más un halo de un píxel. Esto suma exactamente los mismos bordes
   de vecindad 4 por clase sin reconstruir el mapa de ids en cada absorción.
4. Consulta cada clase del catálogo una vez por llamada. Tanto esta información
   como las componentes se descartan entre fotos.

El perfil optimizado acumuló **1,455 s**, con 180 etiquetados, 927 regiones
intermedias, 50 consultas al catálogo y cinco construcciones del mapa inicial.

## Tiempos sin cProfile

| Foto | Regiones SLIC | Antes (ms) | Después (ms) |
|---|---:|---:|---:|
| P001_a.jpg | 93 | 1.724,776 | 301,676 |
| P002_a.jpg | 88 | 1.696,273 | 276,004 |
| P003_a.jpg | 88 | 1.741,701 | 292,352 |
| P004_a.jpg | 88 | 1.398,132 | 248,473 |
| P005_a.jpg | 86 | 1.231,120 | 254,877 |
| **Media de medianas** | | **1.558,400** | **274,677** |

`consolidar` es **5,67 veces más rápido**, una reducción del **82,4%**.
La medición local anterior fue de 1,23 a 1,74 s por foto; no reprodujo los 2,6 s
reportados inicialmente. La comparación usa ambas versiones en este entorno.

El pipeline completo pasó de una media de **2,703 s** a **1,307 s**. Dos de las
cinco ejecuciones optimizadas todavía superaron el objetivo general de 1,5 s
(1,555 y 1,668 s); este cambio se limita al posprocesador.

## Identidad de resultados

El script compara en cada foto y repetición: orden e ids de las regiones, SHA-256
de cada máscara completa, cajas, áreas, vectores, scores mediante `float.hex`,
clases, alternativas, revisión, estado, eventos y procedencia. Las cinco fotos
coincidieron exactamente con la referencia previa.

`tests/test_posprocesador.py` permanece sin modificaciones.
`tests/test_posprocesador_regresion.py` compara contra el JSON fijo
`tests/fixtures/posprocesador_referencia.json`, capturado con el código original.
Usa el plato sintético con celdas fijas, márgenes distintos, pequeñas predicciones
erróneas, una componente aislada, una clase de área insuficiente y dos proteínas.
También comprueba que las entradas no cambian y que repetir la llamada conserva
la salida. Las celdas evitan depender de cambios de SLIC entre versiones.

Además se contrastaron ambas implementaciones en 104 casos sintéticos con la
semilla configurada: solapes, ids no consecutivos, recortes, máscaras vacías,
distintos umbrales y límites de proteínas. Coincidieron los datos anteriores y
se compararon también las matrices de máscaras directamente.

## Repetir la comprobación

Desde la raíz del repositorio:

```text
python experimentos/perfilar_posprocesador.py --salida salidas/perfilado/nueva
pytest -q
ruff check src tests
```

Para repetir la comparación con la copia original guardada en esta sesión:

```text
python experimentos/perfilar_posprocesador.py --implementacion salidas/perfilado/posprocesador_original.py --salida salidas/perfilado/referencia
python experimentos/perfilar_posprocesador.py --salida salidas/perfilado/comparacion --comparar salidas/perfilado/referencia/resumen.json
```

Cada carpeta contiene `perfil.prof`, `perfil.txt` ordenado por tiempo acumulado
y `resumen.json` con tiempos individuales y huellas exactas. La copia original
y los resultados locales no se versionan; el script y la referencia sintética sí.

En el sandbox de esta sesión, pytest no pudo crear su carpeta temporal por
defecto. Se redirigió con `PYTEST_ADDOPTS` a una carpeta dentro de
`salidas/perfilado/`; no fue necesario modificar las pruebas.

Una primera ejecución completa dio 386 pruebas aprobadas y un fallo ajeno a
este cambio en `test_registro_agrega_filas_y_conserva_metricas[False]`: el hash
`3229e2618093`, escrito correctamente en el CSV, fue interpretado por pandas
como notación científica y convertido a infinito. El hash depende de las rutas
temporales de la configuración. Las tres pruebas de registro pasaron al usar
otra carpeta temporal; no se modificó el registro ni sus pruebas.

La repetición completa con `PYTEST_ADDOPTS` apuntando a
`--basetemp=salidas/perfilado/pytest-verificacion` terminó correctamente:

```text
pytest -q
387 passed, 192 warnings in 242.93s (0:04:02)

ruff check src tests
All checks passed!
```

Las 192 advertencias son de Joblib por una deprecación de NumPy 2.5, durante
las pruebas de entrenamiento y evaluación. El script de perfilado también
pasó Ruff y `git diff --check` no detectó errores de espacios.

No se agregaron dependencias, parámetros de configuración ni decisiones
ambiguas que requieran nuevos `TODO(equipo)`.
