"""Balance de color clásico con ganancias acotadas por canal."""

from copy import deepcopy

import numpy as np

from cobro_vision.dominio.utilidades import _validar_imagen


class CorrectorColor:
    """Aplica gray-world, una tarjeta gris o una copia sin corrección."""

    def __init__(self, config_color: dict | None = None) -> None:
        self._config = (
            deepcopy(config_color)
            if config_color is not None
            else {"metodo": "gray_world"}
        )

    def corregir(self, imagen: np.ndarray) -> np.ndarray:
        """Devuelve BGR uint8 sin modificar la entrada ni dividir por cero."""
        _validar_imagen(imagen)
        metodo = self._config["metodo"]
        if metodo == "ninguna":
            return imagen.copy()
        referencia = imagen
        if metodo == "tarjeta_gris":
            # TODO(equipo): roi [x,y,w,h] usa coordenadas de la imagen redimensionada.
            x, y, ancho, alto = self._config["roi"]
            if (
                min(x, y) < 0
                or min(ancho, alto) <= 0
                or x + ancho > imagen.shape[1]
                or y + alto > imagen.shape[0]
            ):
                raise ValueError(
                    "El ROI de tarjeta gris debe quedar dentro de la imagen."
                )
            referencia = imagen[y : y + alto, x : x + ancho]
        elif metodo != "gray_world":
            raise ValueError(f"Método de corrección desconocido: {metodo}")
        medias = referencia.mean(axis=(0, 1))
        ganancias = np.divide(
            medias.mean(), medias, out=np.ones(3, dtype=np.float64), where=medias > 0
        )
        ganancias = np.clip(ganancias, 0.5, 2.0)
        return np.clip(imagen.astype(np.float64) * ganancias, 0, 255).astype(np.uint8)
