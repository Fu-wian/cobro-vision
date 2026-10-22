"""Captura de fotografías BGR desde archivos o una cámara USB."""

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class Camara(ABC):
    """Origen intercambiable de una fotografía por transacción."""

    @abstractmethod
    def capturar(self) -> np.ndarray:
        """Obtiene una imagen BGR uint8."""

    def liberar(self) -> None:
        """Libera recursos si el origen los mantiene abiertos."""


class CamaraArchivo(Camara):
    """Recorre imágenes en orden alfabético y vuelve al principio."""

    def __init__(
        self, carpeta: Path, extensiones: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ) -> None:
        permitidas = {extension.lower() for extension in extensiones}
        # TODO(equipo): se incluyen subcarpetas para admitir data/raw/<SESION>.
        self._rutas = sorted(
            (
                ruta
                for ruta in carpeta.rglob("*")
                if ruta.is_file() and ruta.suffix.lower() in permitidas
            ),
            key=lambda ruta: (ruta.as_posix().casefold(), ruta.as_posix()),
        )
        self._carpeta = carpeta
        self._indice = 0

    @staticmethod
    def leer(ruta: Path) -> np.ndarray:
        """Lee BGR aplicando la orientación EXIF mediante OpenCV."""
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe la imagen: {ruta}")
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen is None:
            raise ValueError(f"No se pudo decodificar la imagen: {ruta}")
        return imagen

    def capturar(self) -> np.ndarray:
        """Devuelve la siguiente imagen, de forma cíclica."""
        if not self._rutas:
            raise FileNotFoundError(f"No hay imágenes en la carpeta: {self._carpeta}")
        imagen = self.leer(self._rutas[self._indice])
        self._indice = (self._indice + 1) % len(self._rutas)
        return imagen


class CamaraUSB(Camara):
    """Captura USB con descarte de cuadros de calentamiento."""

    def __init__(
        self,
        indice: int = 0,
        ancho: int = 1280,
        alto: int = 720,
        frames_calentamiento: int = 5,
    ) -> None:
        if frames_calentamiento < 0:
            raise ValueError("frames_calentamiento no puede ser negativo.")
        self._captura = cv2.VideoCapture(indice)
        self._frames_calentamiento = frames_calentamiento
        if not self._captura.isOpened():
            self.liberar()
            raise RuntimeError(f"No se pudo abrir la cámara USB de índice {indice}.")
        self._captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self._captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)

    def capturar(self) -> np.ndarray:
        """Descarta los cuadros de calentamiento y obtiene uno nuevo."""
        for _ in range(self._frames_calentamiento + 1):
            correcto, imagen = self._captura.read()
            if not correcto or imagen is None:
                raise RuntimeError("No se pudo obtener un cuadro de la cámara USB.")
        return imagen

    def liberar(self) -> None:
        """Cierra el dispositivo; admite llamadas repetidas."""
        self._captura.release()
