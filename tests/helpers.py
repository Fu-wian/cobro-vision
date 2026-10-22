"""Imágenes artificiales reproducibles; no representan fotografías reales."""

from pathlib import Path

import cv2
import numpy as np

from cobro_vision.clasificacion.base import Clasificador
from cobro_vision.dominio.modelos import ResultadoClasificacion


class ClasificadorPorColorDominante(Clasificador):
    """Reglas exclusivas de prueba para los cuatro sectores y el fondo neutro."""

    def __init__(self, nombres: list[str]) -> None:
        self._indices = [nombres.index(f"hsv_{canal}_media") for canal in ("s", "v")]
        self._indices_h = [
            i
            for i, nombre in enumerate(nombres)
            if nombre.startswith("hsv_h_") and nombre[6:].isdigit()
        ]

    def predecir_lote(self, vectores: np.ndarray) -> list[ResultadoClasificacion]:
        """Asocia rojo a pollo, verde a ensalada, amarillo a arroz y marrón a frijoles."""
        resultados = []
        for vector in vectores:
            saturacion, valor = vector[self._indices] * 256
            tono = (
                (np.argmax(vector[self._indices_h]) + 0.5) * 180 / len(self._indices_h)
            )
            # Estas reglas son únicamente etiquetas y precios de prueba de la semilla.
            if saturacion < 60:
                clase = "fondo"
            elif 35 < tono < 100:
                clase = "ensalada"
            elif 20 < tono < 35:
                clase = "arroz_amarillo"
            elif valor < 175:
                clase = "frijoles"
            else:
                clase = "pollo"
            resultados.append(ResultadoClasificacion(clase, 10.0, False))
        return resultados


def generar_plato_sintetico(
    semilla: int, sectores: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Genera BGR y etiquetas int16: 0=fondo, 1=rojo, 2=verde, 3=amarillo, 4=marrón."""
    # TODO(equipo): sectores es la cantidad de sectores iguales (0 a 4);
    # sus etiquetas son índices sintéticos, sin asignar clases ni precios reales.
    if type(sectores) is not int or not 0 <= sectores <= 4:
        raise ValueError("sectores debe ser un entero entre 0 y 4.")
    imagen = np.full((320, 320, 3), 85, dtype=np.uint8)
    etiquetas = np.zeros((320, 320), dtype=np.int16)
    cv2.circle(imagen, (160, 160), 140, (240, 240, 240), -1)
    filas, columnas = np.indices(etiquetas.shape)
    interior = (filas - 160) ** 2 + (columnas - 160) ** 2 <= 110**2
    angulos = (np.arctan2(filas - 160, columnas - 160) + 2 * np.pi) % (2 * np.pi)
    colores = ((35, 35, 215), (45, 175, 45), (30, 210, 220), (40, 80, 130))
    for indice in range(sectores):
        sector = interior & (angulos >= indice * 2 * np.pi / sectores)
        sector &= angulos < (indice + 1) * 2 * np.pi / sectores
        imagen[sector] = colores[indice]
        etiquetas[sector] = indice + 1
        if indice == sectores - 1:
            rayas = sector & ((columnas // 5) % 2 == 0)
            imagen[rayas] = np.maximum(imagen[rayas].astype(np.int16) - 30, 0)
    ruido = np.random.default_rng(semilla).normal(0, 3, imagen.shape)
    imagen = np.clip(np.rint(imagen.astype(np.float64) + ruido), 0, 255).astype(
        np.uint8
    )
    return imagen, etiquetas


if __name__ == "__main__":
    from cobro_vision.configuracion import Configuracion

    raiz = Path(__file__).resolve().parents[1]
    config = Configuracion.cargar(raiz / "config.yaml")
    imagen, _ = generar_plato_sintetico(config.semilla, 4)
    destino = raiz / "tests" / "fixtures" / "plato_sintetico.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destino), imagen):
        raise OSError(f"No se pudo escribir la imagen sintética: {destino}")
