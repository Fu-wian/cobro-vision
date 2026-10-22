"""Operaciones comunes de dinero y mapas de regiones."""

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cobro_vision.dominio.modelos import Region


def validar_centavos(centavos: int) -> None:
    """Exige una cantidad entera no negativa, sin aceptar booleanos."""
    if type(centavos) is not int or centavos < 0:
        raise ValueError("El precio debe ser un entero no negativo en centavos.")


def formatear_dinero(centavos: int, moneda: str = "B/.") -> str:
    """Muestra centavos con dos decimales sin pasar por float."""
    validar_centavos(centavos)
    return f"{moneda} {Decimal(centavos) / 100:.2f}"


def mapa_desde_regiones(regiones: list["Region"], forma: tuple[int, int]) -> np.ndarray:
    """Crea un mapa int32; cero indica vacío y el último solape gana."""
    mapa = np.zeros(forma, dtype=np.int32)
    for region in regiones:
        if region.mascara.shape != forma or region.mascara.dtype != np.bool_:
            raise ValueError("La máscara debe ser bool y tener la forma del mapa.")
        if not 0 < region.id <= np.iinfo(np.int32).max:
            raise ValueError("El id de región debe ser positivo y caber en int32.")
        mapa[region.mascara] = region.id
    return mapa
