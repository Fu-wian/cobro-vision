"""Políticas de cobro que conservan el origen de cada ítem."""

from abc import ABC, abstractmethod
from dataclasses import replace

from cobro_vision.dominio.modelos import ID_FONDO, ItemRecibo, Recibo
from cobro_vision.dominio.utilidades import validar_centavos


class PoliticaCobro(ABC):
    """Contrato para asignar los precios cobrados."""

    @abstractmethod
    def calcular(self, items: list[ItemRecibo]) -> Recibo:
        """Asigna los importes a los ítems del recibo."""


class CobroPorComponente(PoliticaCobro):
    """Cobra una vez el precio de catálogo de cada componente."""

    def calcular(self, items: list[ItemRecibo]) -> Recibo:
        """Devuelve ítems con precio de catálogo, excluyendo fondo."""
        return Recibo(
            [
                replace(item, precio_cobrado_centavos=item.alimento.precio_centavos)
                for item in items
                if _es_cobrable(item)
            ]
        )


class CobroPorPlato(PoliticaCobro):
    """Asigna la tarifa del plato a su proteína o usa la suma habitual."""

    def __init__(
        self, precios_por_proteina_centavos: dict[str, int] | None = None
    ) -> None:
        self._precios = dict(precios_por_proteina_centavos or {})
        for precio in self._precios.values():
            validar_centavos(precio)

    def calcular(self, items: list[ItemRecibo]) -> Recibo:
        """Cobra una tarifa de plato y deja el resto de importes en cero."""
        cobrables = [item for item in items if _es_cobrable(item)]
        candidatos = [
            item
            for item in cobrables
            if item.alimento.categoria == "proteina"
            and item.alimento.id_clase in self._precios
        ]
        # TODO(equipo): Confirmar cobro por plato; con varias tarifas se usa la mayor
        # y se desempata por id_clase para obtener siempre el mismo resultado.
        elegido = min(
            candidatos,
            key=lambda item: (
                -self._precios[item.alimento.id_clase],
                item.alimento.id_clase,
            ),
            default=None,
        )
        return Recibo(
            [
                replace(
                    item,
                    precio_cobrado_centavos=(
                        item.alimento.precio_centavos
                        if elegido is None
                        else self._precios[item.alimento.id_clase]
                        if item is elegido
                        else 0
                    ),
                )
                for item in cobrables
            ]
        )


def _es_cobrable(item: ItemRecibo) -> bool:
    return item.alimento.id_clase != ID_FONDO and item.alimento.categoria != "fondo"
