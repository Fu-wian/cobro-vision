"""Prueba de humo del paquete instalable."""

import cobro_vision


def test_importar_paquete() -> None:
    """El paquete se puede importar tras la instalación editable."""
    assert cobro_vision.__name__ == "cobro_vision"
