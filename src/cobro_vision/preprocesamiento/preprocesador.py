"""Redimensionamiento, filtrado y corrección conservando BGR."""

from copy import deepcopy

import cv2
import numpy as np

from cobro_vision.dominio.utilidades import _validar_imagen
from cobro_vision.preprocesamiento.corrector_color import CorrectorColor


class Preprocesador:
    """Prepara la misma imagen para entrenamiento y operación."""

    def __init__(self, config_imagen: dict) -> None:
        self._config = deepcopy(config_imagen)
        self._corrector = CorrectorColor(self._config["correccion_color"])

    def escala(self, imagen_original: np.ndarray) -> float:
        """Calcula el factor de reducción, con tope de 1.0."""
        _validar_imagen(imagen_original)
        return min(1.0, self._config["lado_mayor"] / max(imagen_original.shape[:2]))

    def procesar(self, imagen: np.ndarray) -> np.ndarray:
        """Reduce, filtra y balancea el color; devuelve una imagen BGR uint8."""
        factor = self.escala(imagen)
        procesada = imagen.copy()
        if factor < 1:
            alto, ancho = imagen.shape[:2]
            procesada = cv2.resize(
                imagen,
                (max(1, round(ancho * factor)), max(1, round(alto * factor))),
                interpolation=cv2.INTER_AREA,
            )
        filtro = self._config["filtro"]
        if filtro["tipo"] == "mediana":
            procesada = cv2.medianBlur(procesada, filtro["tamano"])
        elif filtro["tipo"] == "gaussiano":
            tamano = filtro["tamano"]
            procesada = cv2.GaussianBlur(procesada, (tamano, tamano), 0)
        elif filtro["tipo"] != "ninguno":
            raise ValueError(f"Filtro desconocido: {filtro['tipo']}")
        return self._corrector.corregir(procesada)
