"""Geometría, exclusión del fondo y determinismo de los segmentadores."""

import numpy as np
import pytest

from cobro_vision.dominio.modelos import Region
from cobro_vision.segmentacion.base import Segmentador
from cobro_vision.segmentacion.superpixeles import SegmentadorSuperpixeles
from cobro_vision.segmentacion.ventanas import SegmentadorVentanas


def verificar_regiones(
    regiones: list[Region], mascara_plato: np.ndarray, minimo: int
) -> None:
    """Comprueba máscaras disjuntas, cajas exactas y metadatos coherentes."""
    ocupada = np.zeros_like(mascara_plato)
    assert [region.id for region in regiones] == list(range(1, len(regiones) + 1))
    for region in regiones:
        assert region.mascara.dtype == np.bool_
        assert region.mascara.shape == mascara_plato.shape
        assert not np.any(region.mascara & ~mascara_plato)
        assert not np.any(region.mascara & ocupada)
        assert region.area == np.count_nonzero(region.mascara) >= minimo
        filas, columnas = np.nonzero(region.mascara)
        assert region.bbox == (
            columnas.min(),
            filas.min(),
            columnas.max() + 1,
            filas.max() + 1,
        )
        assert region.estado == "pendiente"
        assert region.vector is None and region.resultado is None
        ocupada |= region.mascara


@pytest.mark.parametrize("clase", [SegmentadorSuperpixeles, SegmentadorVentanas])
def test_regiones_deterministas_disjuntas_y_area(
    datos_config: dict,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
    clase: type[Segmentador],
) -> None:
    """Dos ejecuciones conservan ids, cajas, áreas y todos los píxeles de máscara."""
    imagen, mapa = plato_sintetico
    original = imagen.copy()
    mascara = mapa > 0
    config = datos_config["segmentacion"]
    segmentador = clase(config)
    primera = segmentador.segmentar(imagen, mascara)
    segunda = segmentador.segmentar(imagen, mascara)
    assert primera
    verificar_regiones(primera, mascara, config["area_minima_px"])
    assert len(primera) == len(segunda)
    for region, repetida in zip(primera, segunda):
        assert (region.id, region.area, region.bbox) == (
            repetida.id,
            repetida.area,
            repetida.bbox,
        )
        np.testing.assert_array_equal(region.mascara, repetida.mascara)
    np.testing.assert_array_equal(imagen, original)


@pytest.mark.parametrize("clase", [SegmentadorSuperpixeles, SegmentadorVentanas])
def test_mascara_vacia_area_y_forma(
    datos_config: dict, clase: type[Segmentador]
) -> None:
    """Una máscara vacía o regiones menores al mínimo no producen regiones."""
    imagen = np.zeros((30, 40, 3), np.uint8)
    config = datos_config["segmentacion"]
    config["area_minima_px"] = 2000
    segmentador = clase(config)
    assert segmentador.segmentar(imagen, np.zeros((30, 40), bool)) == []
    assert segmentador.segmentar(imagen, np.ones((30, 40), bool)) == []
    with pytest.raises(ValueError, match="máscara"):
        segmentador.segmentar(imagen, np.ones((40, 30), bool))
    with pytest.raises(ValueError, match="bool"):
        segmentador.segmentar(imagen, np.ones((30, 40), np.uint8))


def test_slic_recibe_rgb(datos_config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """El límite con SLIC recibe rojo en RGB aunque la entrada esté en BGR."""
    imagen = np.full((20, 30, 3), (5, 15, 220), dtype=np.uint8)
    mascara = np.ones((20, 30), bool)
    mascara[:, :5] = False
    recibidos: dict = {}

    def slic_simulado(rgb: np.ndarray, **parametros: object) -> np.ndarray:
        recibidos["rgb"] = rgb
        recibidos.update(parametros)
        etiquetas = np.ones((20, 30), dtype=np.int32)
        etiquetas[:, 15:] = 2
        etiquetas[~mascara] = 0
        return etiquetas

    monkeypatch.setattr("cobro_vision.segmentacion.superpixeles.slic", slic_simulado)
    regiones = SegmentadorSuperpixeles(datos_config["segmentacion"]).segmentar(
        imagen, mascara
    )
    np.testing.assert_array_equal(recibidos["rgb"][0, 0], (220, 15, 5))
    assert recibidos["start_label"] == 1 and recibidos["enforce_connectivity"] is True
    assert recibidos["mask"] is mascara
    assert len(regiones) == 2  # No fusionar aunque ambas tengan el mismo color.
    assert regiones[0].bbox == (5, 0, 15, 20)
    assert regiones[1].bbox == (15, 0, 30, 20)


@pytest.mark.parametrize("paso", [10, 20, 25])
def test_ventanas_bordes_solapes_y_huecos(datos_config: dict, paso: int) -> None:
    """Recorta bordes, asigna solapes a la primera ventana y respeta pasos mayores."""
    imagen = np.zeros((45, 55, 3), dtype=np.uint8)
    mascara = np.ones(imagen.shape[:2], dtype=bool)
    mascara[:3, :5] = False
    config = datos_config["segmentacion"]
    config["ventanas"] = {"tamano": 20, "paso": paso}
    config["area_minima_px"] = 1
    regiones = SegmentadorVentanas(config).segmentar(imagen, mascara)
    verificar_regiones(regiones, mascara, 1)
    union = np.logical_or.reduce([region.mascara for region in regiones])
    if paso <= 20:
        np.testing.assert_array_equal(union, mascara)
    else:
        assert not union[:, 21].any()
    assert regiones[0].bbox == (0, 0, 20, 20)


def test_slic_un_pixel(datos_config: dict) -> None:
    """Una máscara mínima válida no se confunde con fondo ni provoca división por cero."""
    config = datos_config["segmentacion"]
    config["n_segmentos"] = 1
    config["area_minima_px"] = 1
    imagen = np.zeros((10, 10, 3), np.uint8)
    mascara = np.zeros((10, 10), bool)
    mascara[5, 5] = True
    regiones = SegmentadorSuperpixeles(config).segmentar(imagen, mascara)
    assert len(regiones) == 1
    assert regiones[0].bbox == (5, 5, 6, 6)
