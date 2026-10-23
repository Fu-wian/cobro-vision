# Especificación técnica: cobro-vision

Fuente de verdad para implementar el prototipo. Complementa al documento "Estructura de clases del proyecto" y al Hito 2 (arquitectura MVC). Donde este documento difiera de ellos, **gana este documento** (ver sección 1.2).

---

## 0. Alcance

**Entrada:** una foto cenital de un plato de comida para llevar, tomada en un espacio de captura controlado (cámara fija, iluminación difusa).
**Salida:** lista de alimentos identificados, precio de cada uno y total en B/. Un cajero revisa, corrige y confirma.

**Dentro del alcance**
- Pipeline clásico: preprocesamiento, detección del plato, segmentación en superpíxeles, características (histogramas de color + textura), clasificación SVM, posprocesamiento, cobro.
- Herramientas de dataset a partir de etiquetas LabelMe, entrenamiento, evaluación y ablación.
- Interfaz del cajero de escritorio (CustomTkinter) y CLI.

**Fuera del alcance**
- Deep learning, estimación de porciones o peso, integración con la caja real, pagos, video en tiempo real (se procesa **una foto por transacción**).

---

## 1. Decisiones asumidas y diferencias con el PDF

### 1.1 Decisiones asumidas (el equipo puede cambiarlas)

| Tema | Valor asumido | Dónde se cambia |
|---|---|---|
| Interfaz | CustomTkinter (escritorio) | `vista/` y sección `ui` |
| Unidad física | Plato redondo desechable, vista cenital | `plato.metodo` en `config.yaml` |
| Política de cobro | Suma de precios por componente | `cobro.politica` |
| Menú y precios | Semilla de ejemplo (`TODO(equipo)`) | `data/catalogo_seed.csv` |
| Cámara | Archivos de imagen; USB opcional | `camara.clase` |
| Python | 3.10 o superior, multiplataforma (el equipo probablemente usa Windows) | n/a |
| Etiquetado | LabelMe, polígonos | sección 6 |

### 1.2 Diferencias respecto al PDF "Estructura de clases"

1. El código vive en el paquete `src/cobro_vision/` (instalable con `pip install -e .`), con los mismos subpaquetes del PDF.
2. `Preprocesador` devuelve la imagen en **BGR**. La conversión a HSV o Lab se hace dentro de los extractores, a través de `ContextoImagen` (cache por imagen).
3. Los extractores reciben `ContextoImagen`, no la imagen cruda: `extraer(ctx, region)`.
4. Se agrega la interfaz abstracta `Clasificador` con dos variantes: `ClasificadorSVM` y `ClasificadorFalso` (para el esqueleto y las pruebas).
5. Se agregan `LectorLabelMe`, `MapaEtiquetas` y `ConstructorInstancias` (paquete `adquisicion/` y `clasificacion/`).
6. Precios en **centavos** (`int`). `Alimento.precio` devuelve `Decimal`.
7. `ResultadoClasificacion` tiene el campo opcional `alternativa` (segunda clase más probable) para sugerir correcciones en la interfaz.
8. `Posprocesador` incluye absorción de componentes pequeños y filtro de área mínima por clase (sección 7.10).
9. `PoliticaCobro.calcular` recibe `list[ItemRecibo]` y devuelve `Recibo` (así conserva las regiones de origen de cada ítem).
10. La interfaz es **CustomTkinter** (el PDF decía "Streamlit o Tkinter"). Se agrega `vista/presentador.py` con la lógica de presentación sin Tk, para poder probarla sin pantalla.
11. El controlador agrega `corregir_item` (corrige todas las regiones de un ítem de una vez).

---

## 2. Entorno y dependencias

`pyproject.toml`:

- `name = "cobro-vision"`, `requires-python = ">=3.10"`, build con `setuptools`, layout `src/`.
- Dependencias: `numpy>=1.24`, `opencv-python-headless>=4.8`, `scikit-image>=0.21`, `scikit-learn>=1.3`, `scipy>=1.10`, `pandas>=2.0`, `pyyaml>=6`, `joblib>=1.3`, `pillow>=10`, `matplotlib>=3.7`, `customtkinter>=5.2`.
- `tkinter` viene con Python en Windows y macOS (instalador de python.org). En Linux puede requerir el paquete del sistema `python3-tk`.
- Extra `dev`: `pytest`, `pytest-cov`, `ruff`.
- Extra `etiquetado`: `labelme`.
- Punto de entrada: `python -m cobro_vision` (`src/cobro_vision/__main__.py`).

`.gitignore` debe excluir: `data/`, `models/`, `salidas/`, `__pycache__/`, `.venv/`, `*.joblib`, `.pytest_cache/`. Se versiona `data/catalogo_seed.csv` (excepción explícita con `!data/catalogo_seed.csv`) y `config.yaml`.

---

## 3. Estructura del repositorio

```
cobro-vision/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── config.yaml
├── .gitignore
├── docs/
│   ├── ESPECIFICACION_TECNICA.md
│   └── PLAN_DE_ETAPAS.md
├── data/                          # fuera de git, salvo catalogo_seed.csv
│   ├── catalogo_seed.csv
│   ├── catalogo.db
│   ├── manifest.csv
│   └── raw/<SESION>/<PLATO>_<n>.jpg + .json
├── models/                        # modelos entrenados (fuera de git)
├── salidas/                       # debug y evaluaciones (fuera de git)
├── experimentos/                  # resultados.csv y scripts de ablación
├── src/cobro_vision/
│   ├── __init__.py
│   ├── __main__.py                # entrada de la CLI
│   ├── cli.py
│   ├── configuracion.py
│   ├── dominio/                   # modelos.py, errores.py, utilidades.py
│   ├── adquisicion/               # camara.py, lector_labelme.py, mapa_etiquetas.py,
│   │                              # gestor_dataset.py, validador_etiquetas.py
│   ├── preprocesamiento/          # preprocesador.py, corrector_color.py, detector_plato.py
│   ├── segmentacion/              # base.py, superpixeles.py, ventanas.py
│   ├── caracteristicas/           # contexto.py, base.py, color.py, textura.py, constructor.py
│   ├── clasificacion/             # base.py, svm.py, falso.py, instancias.py, entrenador.py
│   ├── negocio/                   # catalogo.py, posprocesador.py, politicas.py, calculadora.py
│   ├── explicacion/               # trazabilidad.py
│   ├── controlador/               # controlador.py, fabrica.py, medidor.py
│   ├── vista/                     # vista_cajero.py, presentador.py, render.py
│   └── evaluacion/                # evaluador_clasificador.py, evaluador_segmentacion.py,
│                                  # evaluador_sistema.py, registro.py
├── tests/
│   ├── conftest.py
│   ├── helpers.py                 # generadores sintéticos
│   └── fixtures/                  # imágenes sintéticas pequeñas
└── notebooks/
```

---

## 4. Contratos de código

Estas firmas son contratos entre módulos (regla 1 de `AGENTS.md`). Los métodos privados y detalles internos son libres.

### 4.1 Dominio (`dominio/modelos.py`, `dominio/errores.py`)

```python
ID_FONDO = "fondo"

class CobroVisionError(Exception): ...
class ConfiguracionInvalida(CobroVisionError): ...
class EtiquetaInvalida(CobroVisionError): ...
class PlatoNoDetectado(CobroVisionError): ...
class ClaseDesconocida(CobroVisionError): ...
class ModeloIncompatible(CobroVisionError): ...

@dataclass(frozen=True)
class Alimento:
    id_clase: str
    nombre: str
    categoria: str            # base | proteina | acompanante | ensalada | fondo
    precio_centavos: int
    activo: bool = True
    @property
    def precio(self) -> Decimal: ...          # precio_centavos / 100

@dataclass
class ResultadoClasificacion:
    id_clase: str
    score: float              # margen de decisión (mejor menos segunda)
    requiere_revision: bool
    alternativa: str | None = None

@dataclass
class Region:
    id: int
    mascara: np.ndarray       # bool, mismo alto y ancho que la imagen procesada
    bbox: tuple[int, int, int, int]   # (x0, y0, x1, y1) estilo slicing: imagen[y0:y1, x0:x1]
    area: int                 # píxeles
    vector: np.ndarray | None = None
    resultado: ResultadoClasificacion | None = None
    estado: str = "pendiente" # pendiente | aceptada | revisar | corregida | eliminada

@dataclass
class ItemRecibo:
    alimento: Alimento
    origen: str                       # "auto" | "manual"
    ids_region: list[int]
    area_frac: float | None           # fracción del plato que ocupa; None si es manual
    requiere_revision: bool
    precio_cobrado_centavos: int      # lo fija la PoliticaCobro

@dataclass
class Recibo:
    items: list[ItemRecibo]
    @property
    def total_centavos(self) -> int: ...      # suma de precio_cobrado_centavos
    @property
    def total(self) -> Decimal: ...
    @property
    def requiere_revision(self) -> bool: ...  # algún ítem con revisión pendiente

@dataclass
class EntradaTraza:
    etapa: str
    descripcion: str
    datos: dict
    duracion_ms: float | None = None

@dataclass
class Ajuste:
    tipo: str                 # "corregir" | "quitar" | "agregar"
    id_clase: str
    id_region: int | None = None

@dataclass
class Transaccion:
    id: str                   # "T-YYYYMMDD-HHMMSS-xxxx"
    imagen_original: np.ndarray
    imagen_procesada: np.ndarray | None = None
    mascara_plato: np.ndarray | None = None
    regiones_crudas: list[Region] = field(default_factory=list)   # salida del clasificador
    regiones: list[Region] = field(default_factory=list)          # tras posprocesar y ajustes
    recibo: Recibo | None = None
    ajustes: list[Ajuste] = field(default_factory=list)
    trazas: list[EntradaTraza] = field(default_factory=list)
    tiempos: dict[str, float] = field(default_factory=dict)       # ms por etapa
    estado: str = "nueva"     # nueva | procesada | requiere_revision | confirmada | error
    mensaje_error: str | None = None
```

Tipos del dataset (`dominio/modelos.py`):

```python
@dataclass
class Forma:
    etiqueta: str
    tipo: str                         # "polygon" | "rectangle"
    puntos: list[tuple[float, float]] # coordenadas en la imagen ORIGINAL

@dataclass
class Etiquetado:
    ruta_imagen: Path
    ruta_json: Path
    ancho: int
    alto: int
    formas: list[Forma]

@dataclass
class Muestra:                        # una foto lista para usar
    id_foto: str                      # "S03/P017_a"
    plato_id: str
    sesion: str
    split: str                        # "train" | "test"
    ruta_imagen: Path
    etiquetado: Etiquetado

@dataclass
class Instancia:                      # una fila de entrenamiento
    vector: np.ndarray
    id_clase: str
    plato_id: str
    id_foto: str
    origen: str                       # "poligono" | "parche" | "region"
```

### 4.2 Interfaces abstractas

```python
class Camara(ABC):
    @abstractmethod
    def capturar(self) -> np.ndarray: ...             # BGR uint8
    def liberar(self) -> None: ...

class Segmentador(ABC):
    @abstractmethod
    def segmentar(self, imagen: np.ndarray, mascara_plato: np.ndarray) -> list[Region]: ...

class ExtractorCaracteristicas(ABC):
    @abstractmethod
    def extraer(self, ctx: "ContextoImagen", region: Region) -> np.ndarray: ...   # 1-D float32
    @abstractmethod
    def nombres(self) -> list[str]: ...               # un nombre por dimensión

class Clasificador(ABC):
    @abstractmethod
    def predecir_lote(self, vectores: np.ndarray) -> list[ResultadoClasificacion]: ...
    def predecir(self, vector: np.ndarray) -> ResultadoClasificacion: ...  # usa predecir_lote

class PoliticaCobro(ABC):
    @abstractmethod
    def calcular(self, items: list[ItemRecibo]) -> Recibo: ...    # fija precio_cobrado_centavos
```

### 4.3 Clases concretas (firmas públicas)

```python
# configuracion.py
class Configuracion:
    @classmethod
    def cargar(cls, ruta: Path = Path("config.yaml")) -> "Configuracion": ...   # valida
    def seccion(self, nombre: str) -> dict: ...
    def hash(self, secciones: list[str] | None = None) -> str: ...   # sha256 corto (12 hex) de las secciones pedidas

# adquisicion/camara.py
class CamaraArchivo(Camara):
    def __init__(self, carpeta: Path, extensiones=(".jpg", ".jpeg", ".png")): ...
    @staticmethod
    def leer(ruta: Path) -> np.ndarray: ...          # cv2.imread (aplica orientación EXIF)
class CamaraUSB(Camara):
    def __init__(self, indice: int = 0, ancho: int = 1280, alto: int = 720, frames_calentamiento: int = 5): ...

# adquisicion/lector_labelme.py
class LectorLabelMe:
    def leer(self, ruta_json: Path) -> Etiquetado: ...

# adquisicion/mapa_etiquetas.py
class MapaEtiquetas:
    def __init__(self, clases: list[str]): ...       # índice = posición + 1; 0 = sin etiqueta
    def rasterizar(self, etiquetado: Etiquetado, shape_destino: tuple[int, int], escala: float) -> np.ndarray: ...
    # devuelve int16 (alto, ancho). Las formas se dibujan en orden; la última gana.
    def clase_de(self, indice: int) -> str | None: ...

# adquisicion/gestor_dataset.py
class GestorDataset:
    def __init__(self, config: Configuracion, catalogo: "CatalogoAlimentos"): ...
    def construir_manifiesto(self) -> pd.DataFrame: ...                  # escanea data/raw, conserva splits existentes
    def dividir(self, fraccion_prueba: float, semilla: int) -> None: ... # por plato_id
    def cargar_muestras(self, split: str | None = None) -> list[Muestra]: ...
    def contenido_esperado(self, muestra: Muestra) -> set[str]: ...      # override del manifiesto o etiquetas sin fondo

# adquisicion/validador_etiquetas.py
class ValidadorEtiquetas:
    def validar(self, muestras: list[Muestra]) -> "InformeValidacion": ...   # .errores, .advertencias, .ok

# preprocesamiento/
class Preprocesador:
    def __init__(self, config_imagen: dict): ...
    def procesar(self, imagen: np.ndarray) -> np.ndarray: ...
    def escala(self, imagen_original: np.ndarray) -> float: ...      # lado_mayor / max(alto, ancho), tope 1.0
class CorrectorColor:
    def corregir(self, imagen: np.ndarray) -> np.ndarray: ...
class DetectorPlato:
    def __init__(self, config_plato: dict): ...
    def detectar(self, imagen: np.ndarray) -> np.ndarray: ...        # máscara bool; PlatoNoDetectado

# segmentacion/
class SegmentadorSuperpixeles(Segmentador): ...
class SegmentadorVentanas(Segmentador): ...

# caracteristicas/
class ContextoImagen:
    def __init__(self, bgr: np.ndarray): ...
    bgr: np.ndarray
    @cached_property
    def hsv(self) -> np.ndarray: ...       # cv2.COLOR_BGR2HSV (H 0-179)
    @cached_property
    def lab(self) -> np.ndarray: ...       # cv2.COLOR_BGR2LAB (uint8)
    @cached_property
    def gris(self) -> np.ndarray: ...
    def lbp(self, radio: int, puntos: int) -> np.ndarray: ...   # método 'uniform', cacheado
class ExtractorHistogramaColor(ExtractorCaracteristicas): ...
class ExtractorTextura(ExtractorCaracteristicas): ...
class ConstructorVector:
    def __init__(self, extractores: list[ExtractorCaracteristicas]): ...
    def construir(self, ctx: ContextoImagen, region: Region) -> np.ndarray: ...
    def construir_lote(self, imagen: np.ndarray, regiones: list[Region]) -> np.ndarray: ...  # (n, d)
    def dimension(self) -> int: ...
    def nombres(self) -> list[str]: ...

# clasificacion/
class ClasificadorSVM(Clasificador):
    def __init__(self, pipeline, clases: list[str], umbral: float, meta: dict): ...
    def guardar(self, ruta: Path) -> None: ...
    @classmethod
    def cargar(cls, ruta: Path, constructor: ConstructorVector, umbral: float | None = None) -> "ClasificadorSVM": ...
class ClasificadorFalso(Clasificador):
    def __init__(self, clases: list[str], semilla: int = 0): ...
class ConstructorInstancias:
    def __init__(self, config: Configuracion, preprocesador: Preprocesador, detector: DetectorPlato,
                 segmentador: Segmentador, constructor: ConstructorVector, mapa: MapaEtiquetas): ...
    def generar(self, muestras: list[Muestra], modo: str) -> list[Instancia]: ...   # poligonos | segmentacion | ambos
class EntrenadorModelo:
    def __init__(self, config: Configuracion, constructor_instancias: ConstructorInstancias,
                 constructor: ConstructorVector): ...
    def entrenar(self, muestras: list[Muestra]) -> ClasificadorSVM: ...
    def validacion_cruzada(self, instancias: list[Instancia]) -> dict: ...

# negocio/
class CatalogoAlimentos:
    def __init__(self, ruta_db: Path): ...
    def inicializar(self, ruta_semilla: Path) -> None: ...          # crea tabla y hace upsert del CSV
    def obtener(self, id_clase: str) -> Alimento: ...               # ClaseDesconocida si no existe o inactivo
    def listar(self, solo_cobrables: bool = False) -> list[Alimento]: ...
    def clases_validas(self) -> list[str]: ...                      # activas, incluye "fondo", orden alfabético
    def agregar(self, alimento: Alimento) -> None: ...
    def actualizar_precio(self, id_clase: str, precio_centavos: int) -> None: ...
class Posprocesador:
    def __init__(self, config_posproceso: dict, catalogo: CatalogoAlimentos, umbral_confianza: float): ...
    def consolidar(self, regiones: list[Region], mascara_plato: np.ndarray) -> list[Region]: ...
class CobroPorComponente(PoliticaCobro): ...
class CobroPorPlato(PoliticaCobro): ...
class CalculadoraCosto:
    def __init__(self, catalogo: CatalogoAlimentos, politica: PoliticaCobro): ...
    def calcular(self, regiones: list[Region], ajustes: list[Ajuste] = ()) -> Recibo: ...

# explicacion/trazabilidad.py
class Trazabilidad:
    def registrar(self, transaccion: Transaccion, etapa: str, descripcion: str,
                  datos: dict | None = None, duracion_ms: float | None = None) -> None: ...
    def explicar_how(self, transaccion: Transaccion, id_region: int) -> str: ...
    def explicar_why(self, transaccion: Transaccion, id_clase: str) -> str: ...

# controlador/
class MedidorTiempos:
    def medir(self, etapa: str): ...            # context manager; acumula ms
    def resumen(self) -> dict[str, float]: ...  # incluye "total"
    def reiniciar(self) -> None: ...
class ControladorTransaccion:
    def procesar(self, imagen: np.ndarray) -> Transaccion: ...
    def capturar_y_procesar(self) -> Transaccion: ...           # usa la Camara
    def corregir_region(self, t: Transaccion, id_region: int, id_clase: str) -> Transaccion: ...
    def corregir_item(self, t: Transaccion, id_clase_actual: str, id_clase_nueva: str) -> Transaccion: ...
    def quitar_item(self, t: Transaccion, id_clase: str) -> Transaccion: ...
    def agregar_item(self, t: Transaccion, id_clase: str) -> Transaccion: ...
    def confirmar(self, t: Transaccion) -> Transaccion: ...
class FabricaPipeline:
    @staticmethod
    def crear(config: Configuracion) -> ControladorTransaccion: ...

# vista/render.py  (funciones puras, sin Tk)
def paleta_clases(clases: list[str]) -> dict[str, tuple[int, int, int]]: ...          # BGR estable por clase
def dibujar_regiones(imagen_bgr: np.ndarray, regiones: list[Region],
                     paleta: dict[str, tuple[int, int, int]]) -> np.ndarray: ...     # contorno + relleno translúcido
def ajustar_a_caja(imagen: np.ndarray, ancho_max: int, alto_max: int) -> np.ndarray: ...  # conserva proporción

# vista/presentador.py  (funciones puras, sin Tk)
@dataclass
class FilaRecibo:
    id_clase: str; nombre: str; precio_texto: str; estado: str; origen: str   # estado: "ok" | "revisar"
def filas_recibo(recibo: Recibo, moneda: str) -> list[FilaRecibo]: ...
def texto_total(recibo: Recibo | None, moneda: str) -> str: ...                  # "B/. 4.25"
def puede_confirmar(t: Transaccion, exigir_revision: bool) -> bool: ...
def opciones_correccion(catalogo: CatalogoAlimentos, t: Transaccion, id_clase: str) -> list[str]: ...
def mensajes_alerta(t: Transaccion) -> list[str]: ...

# vista/vista_cajero.py
class VistaCajero(ctk.CTk):
    def __init__(self, controlador: ControladorTransaccion, config: Configuracion): ...
    def mostrar_previsualizacion(self, imagen_bgr: np.ndarray) -> None: ...
    def mostrar_regiones(self, t: Transaccion) -> None: ...
    def mostrar_recibo(self, t: Transaccion) -> None: ...
    def mostrar_alerta(self, mensaje: str) -> None: ...
    def al_escanear(self) -> None: ...                       # eventos de los botones
    def al_corregir(self, id_clase_actual: str, id_clase_nueva: str) -> None: ...
    def al_quitar(self, id_clase: str) -> None: ...
    def al_agregar(self, id_clase: str) -> None: ...
    def al_confirmar(self) -> None: ...
```

---

## 5. Formatos de datos

### 5.1 Archivos de etiquetas LabelMe (`.json`, uno por foto, junto a la imagen)

Campos que se usan (el resto se ignora):

```json
{
  "imagePath": "P017_a.jpg",
  "imageHeight": 3000,
  "imageWidth": 4000,
  "shapes": [
    {"label": "arroz_amarillo", "shape_type": "polygon",
     "points": [[120.5, 300.0], [260.0, 310.2], [250.0, 420.0]]},
    {"label": "carne_guisada", "shape_type": "rectangle",
     "points": [[300, 280], [420, 380]]}
  ]
}
```

Reglas del lector:
- Aceptar solo `polygon` (mínimo 3 puntos) y `rectangle` (2 puntos, se convierte a 4). Otro tipo: `EtiquetaInvalida`.
- Ignorar `imageData` aunque venga (el equipo etiqueta con `--nodata`).
- `imageWidth` y `imageHeight` se comparan con las dimensiones de la imagen leída con `cv2.imread` (que aplica la orientación EXIF, igual que LabelMe). Si no coinciden: `EtiquetaInvalida` con un mensaje claro (suele ser una foto de celular rotada).
- Las coordenadas están en la imagen **original**. Para trabajar en la imagen procesada se multiplican por `Preprocesador.escala()`.

### 5.2 `data/manifest.csv`

| Columna | Descripción |
|---|---|
| `id_foto` | `<SESION>/<archivo sin extensión>`, por ejemplo `S03/P017_a` |
| `ruta_imagen` | ruta relativa a `data/raw` |
| `plato_id` | prefijo del nombre de archivo antes del primer `_` (`P017`) |
| `sesion` | nombre de la carpeta de sesión (`S03`) |
| `fecha` | fecha del archivo (`YYYY-MM-DD`) |
| `split` | `train`, `test` o vacío si aún no se ha dividido |
| `contenido_esperado_override` | opcional; `id_clase` separados por `;` cuando hay alimentos ocultos que las etiquetas no muestran |

`construir_manifiesto` agrega fotos nuevas sin tocar el `split` de las existentes. `dividir` asigna `split` **por `plato_id`** de forma reproducible con la semilla, y avisa si alguna clase queda sin ejemplos en `train` o en `test`.

### 5.3 Catálogo SQLite (`data/catalogo.db`) y semilla

```sql
CREATE TABLE IF NOT EXISTS alimentos (
  id_clase        TEXT PRIMARY KEY,
  nombre          TEXT NOT NULL,
  categoria       TEXT NOT NULL CHECK (categoria IN ('base','proteina','acompanante','ensalada','fondo')),
  precio_centavos INTEGER NOT NULL DEFAULT 0 CHECK (precio_centavos >= 0),
  activo          INTEGER NOT NULL DEFAULT 1
);
```

`data/catalogo_seed.csv` (columnas `id_clase,nombre,categoria,precio_centavos,activo`). **Los precios son de ejemplo (`TODO(equipo)`); el equipo debe reemplazarlos.**

```
id_clase,nombre,categoria,precio_centavos,activo
arroz_amarillo,Arroz amarillo con zanahoria,base,100,1
arroz_guandu,Arroz con guandú,base,100,1
frijoles,Frijoles,base,75,1
pollo,Pollo,proteina,250,1
pescado_frito,Pescado frito,proteina,275,1
carne_guisada,Carne guisada,proteina,275,1
platano_frito,Plátano maduro frito,acompanante,75,1
pepino,Pepino,acompanante,50,1
ensalada,Ensalada,ensalada,75,1
fondo,Fondo (plato, mesa, sombras),fondo,0,1
```

`inicializar` es idempotente (upsert por `id_clase`). `fondo` nunca se cobra.

### 5.4 `config.yaml` (esquema completo con valores por defecto)

```yaml
semilla: 42
rutas:
  imagenes: data/raw
  manifiesto: data/manifest.csv
  catalogo: data/catalogo.db
  semilla_catalogo: data/catalogo_seed.csv
  modelos: models
  modelo_actual: models/modelo_actual.joblib
  salidas: salidas
etiquetas:
  alias: {}                                 # etiqueta_origen: id_clase_destino
imagen:
  lado_mayor: 640
  filtro: {tipo: mediana, tamano: 3}        # mediana | gaussiano | ninguno
  correccion_color: {metodo: gray_world}    # gray_world | tarjeta_gris | ninguna
plato:
  metodo: hough                             # hough | umbral | imagen_completa
  radio_min_frac: 0.20                      # relativos al lado menor de la imagen procesada
  radio_max_frac: 0.60
  reducir_radio: 0.97
  area_minima_frac: 0.10
  umbral: {s_max: 60, v_min: 140}           # usado por el método 'umbral' y como respaldo de 'hough'
segmentacion:
  clase: SegmentadorSuperpixeles            # SegmentadorSuperpixeles | SegmentadorVentanas
  n_segmentos: 100
  compacidad: 10
  sigma: 1.0
  area_minima_px: 150
  ventanas: {tamano: 40, paso: 40}
caracteristicas:
  - clase: ExtractorHistogramaColor
    espacio: hsv                            # hsv | lab
    bins: [16, 8, 8]
    momentos: true
  - clase: ExtractorTextura
    lbp: {radio: 1, puntos: 8}
    gradiente: true
    glcm: false
clasificador:
  tipo: falso                               # falso | svm  (cambiar a svm tras la etapa 5)
  kernel: rbf
  C: 10
  gamma: scale
  class_weight: balanced
  umbral_confianza: 0.5                     # PROVISIONAL: calibrar con 'evaluar clasificador'
  busqueda: {C: [1, 10, 100], gamma: [scale, 0.01, 0.1], folds: 5}
entrenamiento:
  modo: poligonos                           # poligonos | segmentacion | ambos
  parches_por_poligono: 8
  radio_parche: 22
  pureza_minima: 0.8
  min_instancias_por_clase: 20
posproceso:
  componente_minimo_frac: 0.01
  clase_minima_frac: 0.03
  max_proteinas: 1                          # null = sin límite
cobro:
  politica: CobroPorComponente              # CobroPorComponente | CobroPorPlato
  por_plato:
    precios_por_proteina_centavos: {}       # {pollo: 450, pescado_frito: 475, ...}; vacío = usar suma
camara:
  clase: CamaraArchivo                      # CamaraArchivo | CamaraUSB
  carpeta: data/raw
  indice: 0
ui:
  exigir_revision: true                     # el botón Confirmar se bloquea con ítems en revisión
  moneda: "B/."
  tema: system                              # system | light | dark
  tamano_ventana: [1200, 720]
  vista_previa_ms: 33                       # intervalo de la vista previa en vivo (solo CamaraUSB)
```

`Configuracion.cargar` valida tipos y valores permitidos, y lanza `ConfiguracionInvalida` con el nombre del campo. `hash(["imagen","plato","segmentacion","caracteristicas"])` identifica la configuración de visión que se guarda con el modelo.

`etiquetas.alias` es un diccionario de textos no vacíos; por ejemplo, `{arroz: arroz_amarillo, "Pollo asado": pollo}`. Si faltan `etiquetas` o `alias` en una configuración anterior, se usa `{}`. `LectorLabelMe` y `ValidadorEtiquetas` sustituyen cada etiqueta una sola vez, antes de comprobar las clases o contar ejemplos. Las etiquetas sin alias se conservan; los alias no se encadenan y nunca se reescriben los JSON. Al preparar el lector o el validador con la configuración, cada destino debe existir entre las clases activas del catálogo; en otro caso se lanza `ConfiguracionInvalida`, incluso si el alias no aparece en ningún JSON. Esta comprobación se hace al usar el dataset para permitir inicializar primero el catálogo. `GestorDataset` entrega las muestras con etiquetas normalizadas a entrenamiento y a las demás herramientas. Los overrides del manifiesto siguen usando `id_clase` canónicos.

El constructor opcional `LectorLabelMe(config, catalogo)` permite proporcionar los alias sin cambiar `leer(ruta_json)`; `LectorLabelMe()` conserva el comportamiento sin alias. Si se proporciona solo `config`, se consulta el catálogo de `rutas.catalogo`.

### 5.5 Salidas

- `salidas/debug/<id_foto>/`: `01_procesada.png`, `02_mascara_plato.png`, `03_regiones_crudas.png` (bordes de regiones), `04_clasificadas.png` (colores por clase), `05_consolidadas.png`, `recibo.txt`, `traza.json`.
- `models/svm_<YYYYMMDD_HHMMSS>.joblib` y `.json` (métricas de validación cruzada, hash de configuración, clases, fecha). Tras entrenar se copia a `models/modelo_actual.joblib` (copia, no enlace simbólico, por Windows).
- `salidas/evaluacion/<fecha>/`: CSV, JSON y PNG de cada evaluación.
- `experimentos/resultados.csv`: una fila por experimento.

---

## 6. Convención de etiquetado (para el equipo y para el validador)

Lanzar LabelMe: `labelme data/raw --nodata --autosave --labels data/clases.txt` (una clase por línea, las del catálogo; el flag `--validatelabel exact` impide etiquetas fuera de la lista si la versión instalada lo soporta).

1. **Una foto = un JSON** junto a la imagen, con el mismo nombre. No renombrar, recortar ni redimensionar las fotos después de etiquetarlas.
2. **Nombres de archivo:** `data/raw/<SESION>/<PLATO>_<n>.jpg`, por ejemplo `S03/P017_a.jpg`. Varias fotos del mismo plato comparten `P017`. Cada foto usada debe tener su JSON; las fotos sin JSON se ignoran con una advertencia.
3. **Etiquetas:** exactamente los `id_clase` del catálogo, más `fondo`.
4. **Polígonos:** trazar cada región visible de cada alimento, **unos 3 a 5 px por dentro del borde** (zonas puras; evita mezclar colores vecinos). Un alimento puede tener varios polígonos.
5. **`fondo`:** marcar también la superficie visible del plato, sombras y reflejos dentro del plato. Lo que no esté etiquetado se ignora en el entrenamiento.
6. **Salsa:** la salsa sobre un alimento se etiqueta como ese alimento. `arroz_guandu` es una sola clase (no "arroz" + "guandú").
7. **Oclusión:** etiquetar solo lo visible. Si el plato lleva algo oculto que debe cobrarse, anotarlo en `contenido_esperado_override` del manifiesto.
8. **Superposiciones:** si dos polígonos se solapan, gana el último dibujado. El validador avisa si el solape es mayor al 5% del polígono menor.

El `ValidadorEtiquetas` debe detectar: JSON sin imagen o imagen sin JSON, etiquetas fuera del catálogo, tipos de forma no soportados, polígonos con menos de 3 puntos, puntos fuera de la imagen, dimensiones que no coinciden con la imagen, fotos sin ninguna forma, solapes grandes y clases con menos de `min_instancias_por_clase` ejemplos. Errores = código de salida distinto de cero; advertencias = solo se reportan.

---

## 7. Especificación por módulo

### 7.1 Cámaras
- `CamaraArchivo.capturar()` devuelve la siguiente imagen de la carpeta (orden alfabético, cíclico). Carpeta vacía: `FileNotFoundError`.
- `CamaraUSB`: abre `cv2.VideoCapture(indice)`, descarta `frames_calentamiento` cuadros y devuelve el último. Si falla: `RuntimeError` con mensaje claro.

### 7.2 Preprocesador
1. Redimensionar de modo que el lado mayor sea `lado_mayor` (`cv2.INTER_AREA`); no ampliar imágenes ya menores.
2. Filtro: `cv2.medianBlur` (tamaño impar), `cv2.GaussianBlur` o ninguno.
3. Corrección de color. `gray_world`: ganancia por canal = media global / media del canal, limitada a [0.5, 2.0]; aplicar y recortar a [0, 255]. `tarjeta_gris`: ganancias calculadas sobre un ROI `(x, y, w, h)` dado en la configuración.
4. Devuelve BGR `uint8`.

### 7.3 DetectorPlato
- `hough`: gris, `medianBlur(5)`, `cv2.HoughCircles(HOUGH_GRADIENT, dp=1.2, minDist=min(h,w)/2, param1=120, param2=50, minRadius=radio_min_frac*min(h,w), maxRadius=radio_max_frac*min(h,w))`. Tomar el primer círculo (mayor votación). Máscara = círculo con radio × `reducir_radio`. Si no hay círculo, usar `umbral`.
- `umbral`: HSV con `S < s_max` y `V > v_min`; cierre morfológico (elipse 15×15) y apertura (5×5); componente conexa mayor; casco convexo relleno.
- `imagen_completa`: máscara toda en `True` (pruebas y espacios con fondo uniforme).
- Si el área de la máscara es menor que `area_minima_frac × alto × ancho`: `PlatoNoDetectado`.

### 7.4 SegmentadorSuperpixeles
1. Convertir a RGB y llamar `skimage.segmentation.slic(rgb, n_segments=n_segmentos, compactness=compacidad, sigma=sigma, start_label=1, mask=mascara_plato, enforce_connectivity=True)`. Fuera de la máscara la etiqueta es 0.
2. Para cada etiqueta `l > 0`: máscara booleana, bbox, área. Descartar áreas menores que `area_minima_px`.
3. Asignar `id` consecutivos desde 1 en orden de etiqueta (determinista). `estado = "pendiente"`.
4. **No fusionar por color aquí.** La fusión se hace después de clasificar (sección 7.10).

### 7.5 SegmentadorVentanas
Cuadrícula de celdas `tamano × tamano` con paso `paso`. Cada celda es una región con máscara = celda ∩ máscara del plato; descartar si el área es menor que `area_minima_px`. Misma interfaz y salida que el anterior.

### 7.6 Extractores y `ConstructorVector`
- **`ExtractorHistogramaColor`**: con el espacio elegido (`ctx.hsv` o `ctx.lab`), un histograma por canal calculado **solo con los píxeles de la máscara de la región**, con `bins[i]` bins y rango completo del canal (H: 0–180; resto: 0–256); cada histograma normalizado a suma 1. Si `momentos`: media y desviación estándar de cada canal dividida por su rango máximo (6 valores). Dimensión HSV por defecto: 16+8+8+6 = 38.
- **`ExtractorTextura`**: histograma LBP uniforme (`ctx.lbp(radio, puntos)`, `puntos+2` bins, normalizado) sobre los píxeles de la máscara. Si `gradiente`: media y desviación de la magnitud del gradiente de Sobel dentro de la máscara y desviación del gris. Si `glcm`: `skimage.feature.graycomatrix` sobre el recorte del bbox cuantizado a 16 niveles (píxeles fuera de la máscara = media de la región) y `graycoprops` (contraste, homogeneidad, energía, correlación).
- **`ConstructorVector`**: concatena los extractores en el orden de `config.caracteristicas`, `float32`. `nombres()` devuelve nombres estables (`hsv_h_00`, `lbp_03`, ...). Debe ser determinista y sin NaN (región vacía → ceros).
- Los extractores y el constructor son **los mismos** en entrenamiento y operación.

### 7.7 `ConstructorInstancias`
Genera las filas de entrenamiento. Para cada `Muestra`: leer la imagen, `Preprocesador.procesar`, rasterizar etiquetas con `MapaEtiquetas.rasterizar(..., escala=Preprocesador.escala(imagen))`.

- **Modo `poligonos`:** por cada componente conexa de cada clase en el mapa de etiquetas (área ≥ `area_minima_px`): una instancia `origen="poligono"` con esa máscara; además `parches_por_poligono` instancias `origen="parche"`: puntos aleatorios (generador con `semilla` y el `id_foto`) dentro del polígono erosionado 3 px, máscara = disco de `radio_parche` ∩ clase; descartar parches con menos del 50% del área del disco.
- **Modo `segmentacion`:** detectar plato, segmentar con el `Segmentador` configurado y, para cada región, calcular la fracción de sus píxeles que cae en cada clase del mapa. Si la fracción máxima ≥ `pureza_minima`, la región toma esa clase (`origen="region"`); en caso contrario se descarta (contar y registrar cuántas). Si falla la detección del plato en una foto, saltarla y registrar la advertencia.
- **Modo `ambos`:** concatenación de los dos.
- Cada instancia lleva `plato_id` e `id_foto` para poder agrupar.

### 7.8 `EntrenadorModelo`
1. Cargar muestras `split == "train"`, generar instancias según `entrenamiento.modo`.
2. Verificar que cada clase tenga al menos `min_instancias_por_clase` (advertir) y al menos 3 `plato_id` distintos (error si no).
3. Pipeline: `StandardScaler` + `SVC(kernel, C, gamma, class_weight, decision_function_shape="ovr", random_state=semilla)`.
4. Búsqueda `GridSearchCV` sobre `C` y `gamma` con `GroupKFold(n_splits=min(folds, n_platos))`, `scoring="f1_macro"`, `groups=plato_id`.
5. Reentrenar con los mejores parámetros sobre todo `train`. Guardar `.joblib` con: pipeline, lista de clases, `nombres()` del constructor, hash de configuración de visión, fecha y métricas de validación cruzada. Escribir el `.json` hermano y copiar a `modelo_actual.joblib`.

### 7.9 Clasificadores
- **`ClasificadorSVM.predecir_lote`:** `decision_function` (forma `(n, n_clases)`); clase = argmax; `score` = mejor − segundo mejor; `alternativa` = segunda clase; `requiere_revision = score < umbral`. La escala del score depende del número de clases y por eso el umbral se **calibra con datos** (sección 10.1). Manejar el caso de 2 clases (`decision_function` 1-D).
- `cargar` comprueba que `nombres()` del constructor actual coincida con el guardado; si no, `ModeloIncompatible`.
- **`ClasificadorFalso`:** asigna clases de forma determinista (semilla + media de H de la región), `score` fijo alto. Existe solo para el esqueleto, la interfaz y las pruebas.

### 7.10 `Posprocesador.consolidar`
Entrada: regiones ya clasificadas. Salida: regiones consolidadas con `estado` `aceptada` o `revisar`.

1. Construir un mapa de ids (`int32`, 0 = nada) a partir de las máscaras y la etiqueta de cada región.
2. **Fusión por etiqueta:** unir regiones vecinas (vecindad 4 en el mapa) con la misma `id_clase` en una región (máscara unión; `score` = promedio ponderado por área). Las regiones `fondo` también se fusionan entre sí.
3. **Absorción:** toda componente que no sea `fondo` con área menor que `componente_minimo_frac × área del plato` se reasigna a la clase vecina (no `fondo`) con la frontera compartida más larga y se vuelve a fusionar. Si solo tiene vecinos `fondo` o ninguno, se elimina. Registrar en la traza.
4. Eliminar todas las regiones `fondo` de la salida.
5. **Área mínima por clase:** descartar las clases cuya área total sea menor que `clase_minima_frac × área del plato`; registrar en la traza.
6. **Regla de proteínas:** si `max_proteinas` no es `null` y hay más proteínas distintas (según `categoria` del catálogo) que el límite, conservar las de mayor área total y marcar el resto `revisar` (no se eliminan).
7. `requiere_revision = score_promedio < umbral_confianza`; `estado = "revisar"` si es verdadero, `"aceptada"` si no.
8. Reasignar `id` consecutivos desde 1.

### 7.11 `CalculadoraCosto` y políticas
- Agrupar las regiones por `id_clase` (excluyendo `fondo` y regiones `eliminada`). Un `ItemRecibo` por clase, con `ids_region`, `area_frac` y `requiere_revision` (verdadero si alguna región del grupo lo es).
- Aplicar `ajustes` manuales: `agregar` crea un ítem `origen="manual"` sin regiones; `quitar` elimina el ítem de esa clase.
- Pasar los ítems a la `PoliticaCobro`.
- `CobroPorComponente`: `precio_cobrado_centavos = alimento.precio_centavos` para cada ítem.
- `CobroPorPlato`: si hay una proteína con precio en `por_plato.precios_por_proteina_centavos`, esa proteína cobra el precio del plato y el resto de ítems cobra 0; si no, se comporta como `CobroPorComponente`. `TODO(equipo)`: ajustar cuando la cafetería confirme su esquema.
- Clase sin entrada en el catálogo: `ClaseDesconocida`.

### 7.12 `Trazabilidad`
Cada etapa registra una `EntradaTraza` con un resumen de sus datos (conteos, parámetros, duración). Plantillas de texto, en español:
- `explicar_how(t, id_region)`: región, % del plato, método de segmentación, dimensión del vector, clase asignada con su score y umbral, y resultado (aceptada o revisar). Si fue fusionada, los ids de origen.
- `explicar_why(t, id_clase)`: precio del catálogo, política de cobro aplicada, regiones que lo respaldan y su área total.

### 7.13 `ControladorTransaccion`
Orden de `procesar(imagen)`, cada paso dentro de `with medidor.medir(<etapa>)`:
1. `preprocesamiento`, 2. `deteccion_plato`, 3. `segmentacion`, 4. `caracteristicas` (`ContextoImagen` único + `construir_lote`), 5. `clasificacion` (`predecir_lote`, asignar `resultado`), 6. `posprocesamiento`, 7. `cobro`.

- Guarda `regiones_crudas` (tras clasificar) y `regiones` (tras consolidar).
- `PlatoNoDetectado`: `estado="error"`, `mensaje_error` legible, recibo vacío.
- `estado` final: `requiere_revision` si `recibo.requiere_revision`, si no `procesada`.
- `corregir_item`: aplica la corrección a todas las regiones del ítem `id_clase_actual` y recalcula el recibo una sola vez (si el ítem es manual, lo reemplaza: quitar + agregar).
- `corregir_region`: cambia `resultado.id_clase` de la región a `id_clase`, `estado="corregida"`, `requiere_revision=False`; registra un `Ajuste`; recalcula el recibo. `quitar_item` y `agregar_item` registran `Ajuste` y recalculan el recibo con `CalculadoraCosto.calcular(regiones, ajustes)`.
- `confirmar`: `estado="confirmada"` y añade una fila a `salidas/transacciones.csv` (id, fecha y hora, ítems, total, tiempos, n de ajustes).

### 7.14 `FabricaPipeline`
Usa un **registro explícito** (diccionarios nombre → clase) para cámaras, segmentadores, extractores, clasificadores y políticas. Un nombre desconocido: `ConfiguracionInvalida`. Para `clasificador.tipo = "svm"` carga `rutas.modelo_actual`; si no existe, error claro que sugiere `python -m cobro_vision entrenar`.

---

## 8. CLI (`python -m cobro_vision <comando>`)

Implementada con `argparse`. Todos aceptan `--config` (por defecto `config.yaml`).

| Comando | Qué hace |
|---|---|
| `inicializar-catalogo` | Crea `catalogo.db` desde la semilla (idempotente) y escribe `data/clases.txt` (una clase por línea, para LabelMe) |
| `validar-etiquetas` | Corre `ValidadorEtiquetas`; imprime errores y advertencias; código ≠ 0 si hay errores |
| `construir-manifiesto` | Escanea `data/raw` y crea o actualiza `manifest.csv` |
| `dividir --prueba 0.2` | Asigna `split` por `plato_id` con la semilla |
| `resumen-dataset [--csv RUTA]` | Lee el manifiesto y los JSON; muestra cobertura por clase y opcionalmente guarda la misma tabla en CSV |
| `demo --imagen RUTA [--debug]` | Procesa una foto, imprime el recibo y guarda imágenes de depuración |
| `entrenar [--modo ...]` | Entrena y guarda el SVM |
| `evaluar clasificador\|segmentacion\|sistema` | Evalúa sobre `split == test` (sección 10) |
| `ablacion` | Corre las variantes de configuración y llena `experimentos/resultados.csv` |
| `app` | Abre la ventana del cajero (CustomTkinter) |

`resumen-dataset` requiere un manifiesto existente (si falta, indica ejecutar `construir-manifiesto`). No cambia sus splits ni los JSON. La tabla contiene `id_clase`, `platos_total`, `platos_train`, `platos_test`, `fotos`, `poligonos` y `estado`, con una fila por clase activa del catálogo en orden alfabético, incluido `fondo` y las clases con cero ejemplos:

- Los platos se cuentan por `plato_id` distinto, aunque tengan varias fotos o aparezcan en varias sesiones. Los splits vacíos aportan al total, pero no a train/test. Se rechaza cualquier `plato_id` presente tanto en train como en test.
- Cada foto cuenta una vez por clase presente; cada forma cuenta como un polígono, incluidos los rectángulos convertidos a cuatro puntos. Se aplican los alias configurados antes de contar, de modo que distintas etiquetas de origen pueden sumar a una misma clase.
- Se cuentan las anotaciones visibles de los JSON de las fotos del manifiesto; `contenido_esperado_override` no añade ejemplos. Las fotos sin JSON se omiten con una advertencia y las fotos fuera del manifiesto no se incluyen. Un JSON inválido o una etiqueta fuera del catálogo produce código de salida distinto de cero.
- `estado` contiene `FALTA` si hay menos de 3 platos en train y `POCO` si hay menos de 25 platos en total; las marcas son independientes y pueden aparecer juntas (`FALTA POCO`). Si no aplica ninguna, queda vacío. Son advertencias de cobertura: no causan un código de salida de error.
- `--csv RUTA` exporta las mismas columnas y filas en UTF-8, sin índice; crea la carpeta de destino si hace falta.

Ejemplo: `python -m cobro_vision resumen-dataset --config config.yaml --csv salidas/resumen_dataset.csv`.

---

## 9. Interfaz del cajero (CustomTkinter)

Aplicación de escritorio. `python -m cobro_vision app` crea el controlador con `FabricaPipeline`, abre `VistaCajero` y entra en `mainloop()`. Se configura con `ctk.set_appearance_mode(ui.tema)` y el tamaño `ui.tamano_ventana`.

**Tres archivos, para poder probar sin pantalla**
- `vista/render.py`: funciones puras sobre `numpy` (colores estables por clase, contornos y relleno translúcido, ajuste a una caja conservando la proporción).
- `vista/presentador.py`: funciones puras sin Tk: filas del recibo, total formateado, si se puede confirmar, opciones de corrección (la `alternativa` del clasificador primero), mensajes de alerta.
- `vista/vista_cajero.py`: `VistaCajero(ctk.CTk)`. Solo layout, widgets y eventos; llama al controlador y a las funciones anteriores. Convierte las imágenes a `PIL.Image` y las muestra con `CTkImage` en un `CTkLabel`.

**Layout**
- **Izquierda:** foto procesada con las regiones coloreadas por clase y la leyenda de clases debajo. Botón para alternar entre la foto sin marcas y con regiones.
- **Derecha:** `CTkScrollableFrame` con una fila por ítem del recibo: nombre, precio, marca "REVISAR" si corresponde, `CTkOptionMenu` "Cambiar a..." (llama `al_corregir`) y botón "Quitar". Debajo, un `CTkOptionMenu` "Agregar ítem" con su botón. Total en fuente grande con la moneda. Mensajes de alerta en un `CTkLabel` de color de advertencia.
- **Botones:** "Abrir foto..." (con `CamaraArchivo`, usa `tkinter.filedialog.askopenfilename`) o "Capturar" (con `CamaraUSB`); "Confirmar", deshabilitado según `puede_confirmar`; "Nueva transacción", que limpia la pantalla.
- **Paneles:** tiempos por etapa y total; botón "¿Por qué?" que abre un `CTkToplevel` con un `CTkTextbox` de solo lectura con `explicar_how` de cada región y `explicar_why` de cada ítem.
- **Vista previa en vivo (opcional, solo `CamaraUSB`):** `after(ui.vista_previa_ms)` lee un cuadro y llama a `mostrar_previsualizacion`; se detiene al capturar.

**Hilos (obligatorio)**
- Tkinter no es seguro entre hilos. `al_escanear` deshabilita los botones, muestra "Procesando..." y ejecuta `controlador.procesar` en un `threading.Thread`. El resultado (o la excepción) vuelve por una `queue.Queue` que `self.after(50, ...)` revisa **desde el hilo principal**. Ningún widget se toca desde el hilo de trabajo.
- Las excepciones del hilo se muestran con `mostrar_alerta`; la ventana nunca debe cerrarse por un error de procesamiento.

**Reglas**
- `VistaCajero` solo habla con `ControladorTransaccion`. La lógica de formato y de habilitación de botones vive en `presentador.py`, no en los widgets.
- La vista muestra la imagen **procesada** (640 px) escalada con `ajustar_a_caja`, no la foto original de celular.

---

## 10. Evaluación y métricas

Todas leen el `split == "test"` y escriben en `salidas/evaluacion/<fecha>/` y en `experimentos/resultados.csv` (vía `RegistroExperimentos.registrar(nombre, config, metricas)`, que guarda fecha, hash de configuración, hash de git si existe y las métricas).

### 10.1 `EvaluadorClasificador`
- Instancias de las fotos de prueba (mismo `modo` que el entrenamiento, y también `segmentacion` para ver el caso real).
- Por clase: precisión, recall, F1 y soporte. Global: exactitud. Matriz de confusión (CSV y PNG).
- **Tabla de umbral:** para umbrales de 0 a 2 en pasos de 0.1, la fracción de regiones aceptadas (cobertura) y la precisión de las aceptadas. Sirve para calibrar `umbral_confianza`.

### 10.2 `EvaluadorSegmentacion`
Con la segmentación automática sobre las fotos de prueba y el mapa de etiquetas:
- **Pureza ponderada:** suma de `área × fracción de la clase dominante` dividida por el área etiquetada total.
- **Techo de exactitud:** fracción de píxeles etiquetados cuya región tiene como clase dominante la clase real. Es la exactitud máxima del sistema si el clasificador fuera perfecto.
- Conteo de regiones por foto y fracción descartada por pureza baja.

### 10.3 `EvaluadorSistema`
Pipeline completo sobre cada foto de prueba; esperado = `GestorDataset.contenido_esperado`. Por plato y en resumen:
- `plato_exacto`: el conjunto de clases detectadas es igual al esperado.
- Precisión y recall de ítems (micro, sobre todos los platos).
- `total_exacto` y `error_total_medio` (B/.) comparando con el total calculado con el catálogo para el contenido esperado.
- Fracción de platos en `requiere_revision` (carga de trabajo del cajero).
- Tiempos: media y percentil 95 por etapa y total.

### 10.4 Ablación (`ablacion`)
Variantes mínimas: solo color vs. color + textura; HSV vs. Lab; con y sin canal V; `modo` de entrenamiento (`poligonos`, `segmentacion`, `ambos`); SVM vs. kNN vs. Random Forest (estas dos solo dentro del script de ablación con scikit-learn, no como clases de producción). Cada variante entrena sobre `train`, evalúa sobre `test` y registra una fila.

---

## 11. Pruebas

`tests/helpers.py` debe incluir generadores sintéticos, para que todo se pruebe sin fotos reales:
- `generar_plato_sintetico(semilla, sectores)`: fondo gris, plato blanco circular, sectores de colores conocidos (rojo, verde, amarillo, marrón) con ruido gaussiano leve y alguno con rayas (textura). Devuelve la imagen y el mapa de etiquetas verdadero.
- `escribir_dataset_sintetico(carpeta, n_platos)`: escribe imágenes y JSON de LabelMe coherentes en `data/raw/S01/...`.
- `ClasificadorPorColorDominante`: clasificador de prueba basado en reglas de color, para verificar el pipeline completo sin entrenar.

Cobertura mínima de pruebas:

| Módulo | Qué se verifica |
|---|---|
| Configuración | carga, valores inválidos, hash estable |
| Catálogo | inicializar idempotente, `obtener`/`ClaseDesconocida`, precios en centavos |
| Cobro | un ítem por clase, `CobroPorPlato`, totales exactos con `Decimal` |
| Lector/Mapa LabelMe | polígono y rectángulo, escala, orden de solapes, dimensiones que no coinciden |
| Validador | cada error y advertencia de la sección 6 |
| Gestor | manifiesto reproducible, **ningún `plato_id` en dos splits** |
| Preprocesador | tamaño de salida, gray-world sobre imagen con dominante de color |
| DetectorPlato | `hough`, `umbral`, `imagen_completa` y `PlatoNoDetectado` |
| Segmentadores | regiones disjuntas, áreas ≥ mínimo, orden determinista |
| Características | dimensión = `dimension()`, sin NaN, mismos vectores con la misma entrada |
| Posprocesador | fusión, absorción de componente pequeña, filtro por clase, regla de proteínas |
| Entrenador/SVM | exactitud ≥ 0.95 en el dataset sintético, guardar y cargar, `ModeloIncompatible` |
| Controlador | pipeline completo con `ClasificadorPorColorDominante` (total esperado exacto), correcciones, error de plato |
| Evaluadores | claves de métricas y forma de la matriz de confusión |
| Presentador y render | funciones puras: filas del recibo, total formateado (`B/. 4.25`), `puede_confirmar` (falso con ítems en revisión si `exigir_revision`), `opciones_correccion`, `dibujar_regiones` devuelve imagen del mismo tamaño |
| VistaCajero | humo: la ventana se crea y se destruye (la prueba se omite si no hay pantalla, por ejemplo en un servidor sin display) |

---

## 12. Requisitos no funcionales

- **Tiempo:** objetivo de procesar una foto en menos de 1.5 s en una laptop común con la imagen a 640 px. Medir y reportar; no optimizar prematuramente.
- **Reproducibilidad:** misma semilla, datos y configuración producen las mismas métricas (hay una prueba que lo comprueba).
- **Robustez:** ninguna excepción debe llegar sin capturar a la interfaz; los errores se muestran como mensajes legibles.
- **Fluidez de la interfaz:** la ventana no se congela mientras se procesa una foto (hilo de trabajo + cola, sección 9).
- **Registro:** `logging` con nivel INFO en la CLI; las advertencias del dataset (clases con pocos ejemplos, fotos sin JSON) siempre se informan.
- **Calidad:** `ruff check` sin errores; docstrings en clases y métodos públicos.

---

## 13. Pendientes que debe decidir el equipo (`TODO(equipo)`)

1. Menú real, `id_clase` definitivos y precios.
2. Política de cobro real de la cafetería (por componente, por plato o por proteína).
3. Plato redondo o envase con divisiones (si es envase rectangular, el detector de plato necesita un método por contorno).
4. Umbral de confianza final (se calibra con la tabla de umbral).
5. Si guardar las correcciones del cajero como muestras nuevas para reentrenar (etapa 8 opcional).
