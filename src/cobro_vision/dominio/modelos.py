"""Datos compartidos por los módulos del prototipo."""

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np

from cobro_vision.dominio.utilidades import validar_centavos

ID_FONDO = "fondo"


@dataclass(frozen=True)
class Alimento:
    """Clase del catálogo con precio entero en centavos."""

    id_clase: str
    nombre: str
    categoria: str
    precio_centavos: int
    activo: bool = True

    def __post_init__(self) -> None:
        validar_centavos(self.precio_centavos)

    @property
    def precio(self) -> Decimal:
        """Devuelve el precio exacto en balboas."""
        return Decimal(self.precio_centavos) / 100


@dataclass
class ResultadoClasificacion:
    """Clase predicha, margen y necesidad de revisión."""

    id_clase: str
    score: float
    requiere_revision: bool
    alternativa: str | None = None


@dataclass
class Region:
    """Máscara de una región y sus datos de clasificación."""

    id: int
    mascara: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    vector: np.ndarray | None = None
    resultado: ResultadoClasificacion | None = None
    estado: str = "pendiente"


@dataclass
class ItemRecibo:
    """Un alimento cobrado con sus regiones de origen."""

    alimento: Alimento
    origen: str
    ids_region: list[int]
    area_frac: float | None
    requiere_revision: bool
    precio_cobrado_centavos: int

    def __post_init__(self) -> None:
        validar_centavos(self.precio_cobrado_centavos)


@dataclass
class Recibo:
    """Ítems y totales exactos de una transacción."""

    items: list[ItemRecibo]

    @property
    def total_centavos(self) -> int:
        """Suma los importes asignados por la política de cobro."""
        for item in self.items:
            validar_centavos(item.precio_cobrado_centavos)
        return sum(item.precio_cobrado_centavos for item in self.items)

    @property
    def total(self) -> Decimal:
        """Devuelve el total exacto en balboas."""
        return Decimal(self.total_centavos) / 100

    @property
    def requiere_revision(self) -> bool:
        """Indica si algún ítem está pendiente de revisión."""
        return any(item.requiere_revision for item in self.items)


@dataclass
class EntradaTraza:
    """Descripción de un paso del procesamiento."""

    etapa: str
    descripcion: str
    datos: dict
    duracion_ms: float | None = None


@dataclass
class Ajuste:
    """Cambio solicitado por el cajero."""

    tipo: str
    id_clase: str
    id_region: int | None = None


@dataclass
class Transaccion:
    """Foto, resultados y registro de una operación del cajero."""

    id: str
    imagen_original: np.ndarray
    imagen_procesada: np.ndarray | None = None
    mascara_plato: np.ndarray | None = None
    regiones_crudas: list[Region] = field(default_factory=list)
    regiones: list[Region] = field(default_factory=list)
    recibo: Recibo | None = None
    ajustes: list[Ajuste] = field(default_factory=list)
    trazas: list[EntradaTraza] = field(default_factory=list)
    tiempos: dict[str, float] = field(default_factory=dict)
    estado: str = "nueva"
    mensaje_error: str | None = None


@dataclass
class Forma:
    """Polígono o rectángulo en coordenadas originales."""

    etiqueta: str
    tipo: str
    puntos: list[tuple[float, float]]


@dataclass
class Etiquetado:
    """Formas de LabelMe asociadas a una imagen."""

    ruta_imagen: Path
    ruta_json: Path
    ancho: int
    alto: int
    formas: list[Forma]


@dataclass
class Muestra:
    """Foto etiquetada con su plato y conjunto de pertenencia."""

    id_foto: str
    plato_id: str
    sesion: str
    split: str
    ruta_imagen: Path
    etiquetado: Etiquetado


@dataclass
class Instancia:
    """Vector de entrenamiento con procedencia identificable."""

    vector: np.ndarray
    id_clase: str
    plato_id: str
    id_foto: str
    origen: str
