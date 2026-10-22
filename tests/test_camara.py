"""Capturas de archivos sintéticos y USB simulado, sin dispositivo físico."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from cobro_vision.adquisicion.camara import Camara, CamaraArchivo, CamaraUSB


def test_archivos_orden_ciclo_subcarpetas_y_bgr(tmp_path: Path) -> None:
    """El recorrido ordenado filtra extensiones y conserva BGR uint8."""
    (tmp_path / "sesion").mkdir()
    for nombre, color in (
        ("b.PNG", (10, 20, 30)),
        ("a.png", (40, 50, 60)),
        ("sesion/c.png", (70, 80, 90)),
    ):
        assert cv2.imwrite(str(tmp_path / nombre), np.full((8, 9, 3), color, np.uint8))
    (tmp_path / "ignorar.txt").write_text("sin imagen", encoding="utf-8")
    camara = CamaraArchivo(tmp_path)
    assert isinstance(camara, Camara)
    for color in ((40, 50, 60), (10, 20, 30), (70, 80, 90), (40, 50, 60)):
        imagen = camara.capturar()
        assert imagen.shape == (8, 9, 3)
        assert imagen.dtype == np.uint8
        np.testing.assert_array_equal(imagen[0, 0], color)
    camara.liberar()
    solo_png = CamaraArchivo(tmp_path, extensiones=(".png",))
    np.testing.assert_array_equal(solo_png.capturar()[0, 0], (40, 50, 60))


def test_archivo_vacio_inexistente_o_corrupto(tmp_path: Path) -> None:
    """Los archivos ausentes o ilegibles generan errores claros."""
    for carpeta in (tmp_path, tmp_path / "ausente"):
        with pytest.raises(FileNotFoundError, match="No hay imágenes"):
            CamaraArchivo(carpeta).capturar()
    with pytest.raises(FileNotFoundError, match="No existe"):
        CamaraArchivo.leer(tmp_path / "ausente.png")
    corrupta = tmp_path / "corrupta.png"
    corrupta.write_bytes(b"esto no es una imagen")
    with pytest.raises(ValueError, match="decodificar"):
        CamaraArchivo.leer(corrupta)


def test_orientacion_exif(tmp_path: Path) -> None:
    """Una marca roja sintética gira 90 grados según EXIF, igual que LabelMe."""
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:, :20, 0] = 230
    exif = Image.Exif()
    exif[274] = 6
    ruta = tmp_path / "orientada.jpg"
    Image.fromarray(rgb).save(ruta, exif=exif, quality=100, subsampling=0)
    imagen = CamaraArchivo.leer(ruta)
    assert imagen.shape == (40, 20, 3)
    assert imagen[:15, :, 2].mean() > 220
    assert imagen[-15:, :, 2].mean() < 5


class CapturaSimulada:
    """Sustituto mínimo de VideoCapture con errores controlables."""

    def __init__(self, abierta: bool = True, falla_en: int | None = None) -> None:
        self.abierta = abierta
        self.falla_en = falla_en
        self.lecturas = 0
        self.propiedades: dict[int, int] = {}
        self.liberada = False

    def isOpened(self) -> bool:
        """Imita la consulta de apertura del dispositivo."""
        return self.abierta

    def set(self, propiedad: int, valor: int) -> bool:
        """Registra las dimensiones solicitadas."""
        self.propiedades[propiedad] = valor
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Devuelve cuadros distintos o un fallo predefinido."""
        self.lecturas += 1
        if self.lecturas == self.falla_en or self.liberada:
            return False, None
        return True, np.full((8, 9, 3), self.lecturas, np.uint8)

    def release(self) -> None:
        """Simula la liberación idempotente."""
        self.liberada = True


def test_usb_calentamiento_y_liberacion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Descarta N cuadros por captura y permite cerrar el recurso dos veces."""
    captura = CapturaSimulada()
    monkeypatch.setattr(cv2, "VideoCapture", lambda indice: captura)
    camara = CamaraUSB(0, ancho=9, alto=8, frames_calentamiento=2)
    assert captura.propiedades[cv2.CAP_PROP_FRAME_WIDTH] == 9
    assert captura.propiedades[cv2.CAP_PROP_FRAME_HEIGHT] == 8
    assert np.all(camara.capturar() == 3)
    assert np.all(camara.capturar() == 6)
    camara.liberar()
    camara.liberar()
    with pytest.raises(RuntimeError, match="cuadro"):
        camara.capturar()


@pytest.mark.parametrize("abierta,falla_en", [(False, None), (True, 2)])
def test_usb_errores(
    monkeypatch: pytest.MonkeyPatch, abierta: bool, falla_en: int | None
) -> None:
    """Una cámara ausente o un cuadro fallido produce RuntimeError."""
    captura = CapturaSimulada(abierta, falla_en)
    monkeypatch.setattr(cv2, "VideoCapture", lambda indice: captura)
    if not abierta:
        with pytest.raises(RuntimeError, match="abrir"):
            CamaraUSB()
        assert captura.liberada
    else:
        camara = CamaraUSB(frames_calentamiento=2)
        with pytest.raises(RuntimeError, match="cuadro"):
            camara.capturar()
        camara.liberar()
