"""Agrupación de regiones e ítems antes de aplicar una política."""

import numpy as np

from cobro_vision.dominio.modelos import ID_FONDO, Ajuste, ItemRecibo, Recibo, Region
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.negocio.politicas import PoliticaCobro


class CalculadoraCosto:
    """Construye un ítem por clase y delega la asignación de precios."""

    def __init__(self, catalogo: CatalogoAlimentos, politica: PoliticaCobro) -> None:
        self._catalogo = catalogo
        self._politica = politica

    def calcular(self, regiones: list[Region], ajustes: list[Ajuste] = ()) -> Recibo:
        """Agrupa regiones clasificadas, aplica altas/bajas y calcula el recibo."""
        grupos: dict[str, list[Region]] = {}
        for region in regiones:
            if region.estado == "eliminada" or region.resultado is None:
                continue
            if region.resultado.id_clase != ID_FONDO:
                grupos.setdefault(region.resultado.id_clase, []).append(region)

        # TODO(equipo): calcular recibe regiones, no mascara_plato; area_frac usa
        # su unión (incluye fondo y eliminadas). Tras filtrar, solo aproxima el plato.
        area_referencia = (
            int(np.count_nonzero(np.logical_or.reduce([r.mascara for r in regiones])))
            if regiones
            else 0
        )
        items: dict[str, ItemRecibo] = {}
        for id_clase, grupo in grupos.items():
            alimento = self._catalogo.obtener(id_clase)
            if alimento.categoria == "fondo":
                continue
            area = int(
                np.count_nonzero(np.logical_or.reduce([r.mascara for r in grupo]))
            )
            items[id_clase] = ItemRecibo(
                alimento=alimento,
                origen="auto",
                ids_region=[r.id for r in grupo],
                area_frac=area / area_referencia if area_referencia else 0.0,
                requiere_revision=any(
                    r.resultado.requiere_revision or r.estado == "revisar"
                    for r in grupo
                ),
                precio_cobrado_centavos=0,
            )
        for ajuste in ajustes:
            if ajuste.tipo == "quitar":
                self._catalogo.obtener(ajuste.id_clase)
                items.pop(ajuste.id_clase, None)
            elif ajuste.tipo == "agregar":
                alimento = self._catalogo.obtener(ajuste.id_clase)
                if alimento.id_clase != ID_FONDO and alimento.categoria != "fondo":
                    items.setdefault(
                        alimento.id_clase,
                        ItemRecibo(alimento, "manual", [], None, False, 0),
                    )
            elif ajuste.tipo != "corregir":
                raise ValueError(f"Tipo de ajuste desconocido: {ajuste.tipo}")
            # Las correcciones ya están aplicadas a las regiones por el controlador.
        return self._politica.calcular([items[clave] for clave in sorted(items)])
