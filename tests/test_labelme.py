"""Lectura y rasterización de anotaciones artificiales en coordenadas originales."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from cobro_vision.adquisicion.lector_labelme import LectorLabelMe
from cobro_vision.adquisicion.mapa_etiquetas import MapaEtiquetas
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.preprocesamiento.preprocesador import Preprocesador


@pytest.fixture
def archivo_labelme(tmp_path: Path) -> Path:
    """Crea una imagen no cuadrada con un triángulo y un rectángulo superpuestos."""
    ruta = tmp_path / "P001_a.png"
    assert cv2.imwrite(str(ruta), np.full((80, 120, 3), 200, np.uint8))
    datos = {
        "imagePath": ruta.name,
        "imageWidth": 120,
        "imageHeight": 80,
        "imageData": "contenido ignorado, no es base64",
        "shapes": [
            {
                "label": "pollo",
                "shape_type": "polygon",
                "points": [[20.5, 20], [100, 20], [60, 70]],
            },
            {
                "label": "fondo",
                "shape_type": "rectangle",
                "points": [[70, 50], [40, 30]],
            },
        ],
    }
    ruta_json = ruta.with_suffix(".json")
    ruta_json.write_text(json.dumps(datos), encoding="utf-8")
    return ruta_json


def test_lector_poligono_rectangulo_e_imagedata(archivo_labelme: Path) -> None:
    """Conserva las coordenadas originales y transforma dos esquinas en cuatro."""
    etiquetado = LectorLabelMe().leer(archivo_labelme)
    assert (etiquetado.ancho, etiquetado.alto) == (120, 80)
    assert etiquetado.ruta_imagen == archivo_labelme.with_suffix(".png")
    assert etiquetado.formas[0].puntos[0] == (20.5, 20.0)
    assert etiquetado.formas[1].tipo == "rectangle"
    assert etiquetado.formas[1].puntos == [(70, 50), (40, 50), (40, 30), (70, 30)]


@pytest.mark.parametrize(
    "campo,valor,mensaje",
    [
        ("imageWidth", 121, "dimensiones distintas"),
        ("imageHeight", 81, "dimensiones distintas"),
        ("imageWidth", True, "enteros positivos"),
        ("imageHeight", 0, "enteros positivos"),
        ("imagePath", "ausente.png", "JSON sin imagen"),
        ("imagePath", None, "imagePath"),
        ("shapes", {}, "shapes"),
    ],
)
def test_lector_rechaza_metadatos(
    archivo_labelme: Path, campo: str, valor: object, mensaje: str
) -> None:
    """Comprueba ambos ejes y metadatos inválidos con mensajes identificables."""
    datos = json.loads(archivo_labelme.read_text(encoding="utf-8"))
    datos[campo] = valor
    archivo_labelme.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(EtiquetaInvalida, match=mensaje):
        LectorLabelMe().leer(archivo_labelme)


@pytest.mark.parametrize(
    "campo,valor,mensaje",
    [
        ("shape_type", "circle", "no soportado"),
        ("points", [[1, 1], [2, 2]], "menos de 3 puntos"),
        ("points", [[-1, 1], [2, 2], [3, 1]], "fuera de la imagen"),
        ("points", [[120, 1], [2, 2], [3, 1]], "fuera de la imagen"),
        ("points", [[1, 80], [2, 2], [3, 1]], "fuera de la imagen"),
        ("points", [[float("nan"), 1], [2, 2], [3, 1]], "coordenadas no válidas"),
        ("points", [[True, 1], [2, 2], [3, 1]], "coordenadas no válidas"),
        ("points", [[1], [2, 2], [3, 1]], "coordenadas no válidas"),
        ("label", "", "label"),
    ],
)
def test_lector_rechaza_formas(
    archivo_labelme: Path, campo: str, valor: object, mensaje: str
) -> None:
    """Rechaza tipos, polígonos incompletos y puntos inválidos sin excepciones crudas."""
    datos = json.loads(archivo_labelme.read_text(encoding="utf-8"))
    datos["shapes"][0][campo] = valor
    archivo_labelme.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(EtiquetaInvalida, match=mensaje):
        LectorLabelMe().leer(archivo_labelme)


def test_rectangulo_debe_tener_dos_puntos(archivo_labelme: Path) -> None:
    """El lector acepta exactamente dos puntos para un rectangle de LabelMe."""
    datos = json.loads(archivo_labelme.read_text(encoding="utf-8"))
    datos["shapes"][1]["points"].append([5, 5])
    archivo_labelme.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(EtiquetaInvalida, match="rectangle requiere 2 puntos"):
        LectorLabelMe().leer(archivo_labelme)


@pytest.mark.parametrize("contenido", ["{", "[]", "null"])
def test_json_invalido(archivo_labelme: Path, contenido: str) -> None:
    """Los JSON corruptos o de estructura distinta producen EtiquetaInvalida."""
    archivo_labelme.write_text(contenido, encoding="utf-8")
    with pytest.raises(EtiquetaInvalida):
        LectorLabelMe().leer(archivo_labelme)


def test_imagen_ilegible(archivo_labelme: Path) -> None:
    """La existencia del archivo no basta si OpenCV no puede decodificarlo."""
    archivo_labelme.with_suffix(".png").write_bytes(b"corrupta")
    with pytest.raises(EtiquetaInvalida, match="decodificar"):
        LectorLabelMe().leer(archivo_labelme)


def test_dimensiones_se_comparan_despues_de_exif(tmp_path: Path) -> None:
    """Acepta las dimensiones orientadas y rechaza las del JPEG sin rotar."""
    ruta = tmp_path / "P001_a.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8)).save(ruta, exif=exif)
    datos = {"imagePath": ruta.name, "imageWidth": 20, "imageHeight": 40, "shapes": []}
    ruta_json = ruta.with_suffix(".json")
    ruta_json.write_text(json.dumps(datos), encoding="utf-8")
    assert LectorLabelMe().leer(ruta_json).alto == 40
    datos.update(imageWidth=40, imageHeight=20)
    ruta_json.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(EtiquetaInvalida, match="EXIF"):
        LectorLabelMe().leer(ruta_json)


def test_mapa_escala_original_orden_y_no_etiquetado(
    archivo_labelme: Path, datos_config: dict
) -> None:
    """Aplica la misma escala del preprocesador a polígonos y rectángulos."""
    etiquetado = LectorLabelMe().leer(archivo_labelme)
    imagen = cv2.imread(str(etiquetado.ruta_imagen))
    datos_config["imagen"]["lado_mayor"] = 60
    preprocesador = Preprocesador(datos_config["imagen"])
    procesada = preprocesador.procesar(imagen)
    escala = preprocesador.escala(imagen)
    mapa = MapaEtiquetas(["pollo", "fondo"])
    salida = mapa.rasterizar(etiquetado, procesada.shape[:2], escala)
    assert salida.shape == (40, 60) and salida.dtype == np.int16
    assert salida[0, 0] == 0
    assert salida[12, 30] == 1
    assert salida[20, 25] == 2  # último rectángulo gana en el solape
    np.testing.assert_array_equal(salida, mapa.rasterizar(etiquetado, (40, 60), escala))
    assert mapa.clase_de(0) is None and mapa.clase_de(3) is None
    assert mapa.clase_de(1) == "pollo" and mapa.clase_de(2) == "fondo"
    etiquetado.formas.reverse()
    assert mapa.rasterizar(etiquetado, (40, 60), escala)[20, 25] == 1
    assert etiquetado.formas[1].puntos[0] == (20.5, 20)  # no se modifica el original


def test_mapa_sin_reduccion_y_dimension_redondeada(archivo_labelme: Path) -> None:
    """Admite escala 1 y redondea el borde a un píxel dentro del destino."""
    etiquetado = LectorLabelMe().leer(archivo_labelme)
    etiquetado.formas[0].puntos = [(110, 60), (119.9, 60), (119.9, 79.9)]
    mapa = MapaEtiquetas(["pollo", "fondo"])
    assert mapa.rasterizar(etiquetado, (80, 120), 1)[75, 119] == 1
    assert mapa.rasterizar(etiquetado, (40, 60), 0.5)[38, 59] == 1
    assert mapa.rasterizar(etiquetado, (27, 40), 1 / 3).shape == (27, 40)


def test_mapa_rechaza_clase_y_escala_incorrectas(archivo_labelme: Path) -> None:
    """No deja que etiquetas desconocidas se conviertan silenciosamente en fondo."""
    etiquetado = LectorLabelMe().leer(archivo_labelme)
    with pytest.raises(EtiquetaInvalida, match="desconocida"):
        MapaEtiquetas(["fondo"]).rasterizar(etiquetado, (80, 120), 1)
    with pytest.raises(ValueError, match="escala"):
        MapaEtiquetas([]).rasterizar(etiquetado, (40, 60), 1)
    with pytest.raises(ValueError, match="escala"):
        MapaEtiquetas([]).rasterizar(etiquetado, (80, 120), 0)
    with pytest.raises(ValueError, match="únicas"):
        MapaEtiquetas(["pollo", "pollo"])
