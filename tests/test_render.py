"""Dibujo puro sobre imágenes sintéticas BGR con dimensiones conocidas."""

import numpy as np
import pytest

from cobro_vision.dominio.modelos import Transaccion
from cobro_vision.vista.render import ajustar_a_caja, dibujar_regiones, paleta_clases


def test_paleta_estable_por_clase_sin_depender_del_orden() -> None:
    """El color se conserva al reordenar o agregar clases, sin aleatoriedad."""
    paleta = paleta_clases(["pollo", "ensalada"])
    assert paleta == paleta_clases(["ensalada", "pollo"])
    assert paleta["pollo"] == paleta_clases(["pollo", "frijoles"])["pollo"]
    assert paleta["pollo"] != paleta["ensalada"]
    assert all(0 <= canal <= 255 for color in paleta.values() for canal in color)


def test_regiones_preservan_entrada_forma_y_bgr(
    transaccion_cajero: Transaccion,
) -> None:
    """El relleno cambia el interior y el contorno usa el color BGR exacto."""
    t = transaccion_cajero
    original = t.imagen_procesada.copy()
    region = t.regiones[0]
    paleta = {"pollo": (10, 80, 200)}
    salida = dibujar_regiones(original, [region], paleta)
    assert salida.shape == original.shape and salida.dtype == np.uint8
    np.testing.assert_array_equal(original, t.imagen_procesada)
    assert not np.shares_memory(salida, original)
    np.testing.assert_array_equal(salida[0, 0], [10, 80, 200])
    np.testing.assert_allclose(salida[10, 10], [143, 164, 200], atol=1)
    np.testing.assert_array_equal(salida[:, 20:], original[:, 20:])


def test_regiones_eliminadas_sin_resultado_o_color_no_se_dibujan(
    transaccion_cajero: Transaccion,
) -> None:
    """Se omiten regiones que no deben aparecer en la imagen."""
    t = transaccion_cajero
    t.regiones[0].estado = "eliminada"
    t.regiones[1].resultado = None
    salida = dibujar_regiones(t.imagen_procesada, t.regiones, {"pollo": (0, 0, 255)})
    np.testing.assert_array_equal(salida, t.imagen_procesada)


@pytest.mark.parametrize(
    ("ancho", "alto", "forma"),
    [
        (40, 100, (20, 40, 3)),
        (100, 10, (10, 20, 3)),
        (100, 100, (40, 80, 3)),
        (1, 1, (1, 1, 3)),
    ],
)
def test_ajustar_a_caja_respeta_proporcion_y_limites(
    transaccion_cajero: Transaccion, ancho: int, alto: int, forma: tuple[int, int, int]
) -> None:
    """La imagen cabe en ambas dimensiones sin ampliarse."""
    salida = ajustar_a_caja(transaccion_cajero.imagen_procesada, ancho, alto)
    assert salida.shape == forma and salida.dtype == np.uint8


def test_imagen_mascara_y_caja_invalidas(transaccion_cajero: Transaccion) -> None:
    """Entradas incompatibles generan errores claros sin modificar las matrices."""
    t = transaccion_cajero
    with pytest.raises(ValueError, match="positivas"):
        ajustar_a_caja(t.imagen_procesada, 0, 10)
    with pytest.raises(ValueError, match="BGR"):
        dibujar_regiones(np.zeros((2, 2), dtype=np.uint8), [], {})
    t.regiones[0].mascara = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="máscara"):
        dibujar_regiones(t.imagen_procesada, t.regiones, paleta_clases(["pollo"]))
