"""Integración de la parte A, sin clasificador, controlador ni fábrica."""

from pathlib import Path

import numpy as np
import pytest

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.caracteristicas.color import ExtractorHistogramaColor
from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.caracteristicas.textura import ExtractorTextura
from cobro_vision.preprocesamiento.detector_plato import DetectorPlato
from cobro_vision.preprocesamiento.preprocesador import Preprocesador
from cobro_vision.segmentacion.base import Segmentador
from cobro_vision.segmentacion.superpixeles import SegmentadorSuperpixeles
from cobro_vision.segmentacion.ventanas import SegmentadorVentanas


@pytest.mark.parametrize("clase", [SegmentadorSuperpixeles, SegmentadorVentanas])
def test_archivo_hasta_vectores(
    ruta_proyecto: Path, datos_config: dict, clase: type[Segmentador]
) -> None:
    """La imagen versionada atraviesa el núcleo y genera descriptores reproducibles."""
    imagen = CamaraArchivo.leer(ruta_proyecto / "tests/fixtures/plato_sintetico.png")
    original = imagen.copy()
    datos_config["imagen"]["lado_mayor"] = 256
    procesada = Preprocesador(datos_config["imagen"]).procesar(imagen)
    mascara = DetectorPlato(datos_config["plato"]).detectar(procesada)
    regiones = clase(datos_config["segmentacion"]).segmentar(procesada, mascara)
    color, textura = datos_config["caracteristicas"]
    constructor = ConstructorVector(
        [ExtractorHistogramaColor(color), ExtractorTextura(textura)]
    )
    vectores = constructor.construir_lote(procesada, regiones)
    assert procesada.shape == (256, 256, 3)
    assert regiones and vectores.shape == (len(regiones), constructor.dimension())
    assert np.isfinite(vectores).all() and vectores.dtype == np.float32
    np.testing.assert_array_equal(
        vectores, constructor.construir_lote(procesada, regiones)
    )
    np.testing.assert_array_equal(imagen, original)
