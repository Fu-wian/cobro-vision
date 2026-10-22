"""Registro y explicaciones en español del procesamiento y el cobro."""

from copy import deepcopy

from cobro_vision.dominio.modelos import EntradaTraza, Transaccion
from cobro_vision.dominio.utilidades import formatear_dinero


class Trazabilidad:
    """Conserva resúmenes sin incluir matrices de imagen en la traza."""

    def registrar(
        self,
        transaccion: Transaccion,
        etapa: str,
        descripcion: str,
        datos: dict | None = None,
        duracion_ms: float | None = None,
    ) -> None:
        """Añade una entrada independiente de posteriores cambios en los datos."""
        transaccion.trazas.append(
            EntradaTraza(
                etapa,
                descripcion,
                deepcopy(datos) if datos is not None else {},
                duracion_ms,
            )
        )

    def explicar_how(self, transaccion: Transaccion, id_region: int) -> str:
        """Explica área, segmentación, características y decisión de una región."""
        region = next((r for r in transaccion.regiones if r.id == id_region), None)
        if region is None or region.resultado is None:
            raise ValueError(f"No existe una región clasificada con id {id_region}.")
        area_plato = (
            int(transaccion.mascara_plato.sum())
            if (transaccion.mascara_plato is not None)
            else 0
        )
        segmentacion = self._datos_etapa(transaccion, "segmentacion")
        caracteristicas = self._datos_etapa(transaccion, "caracteristicas")
        clasificacion = self._datos_etapa(transaccion, "clasificacion")
        posproceso = self._datos_etapa(transaccion, "posprocesamiento")
        detalle = next(
            (r for r in posproceso.get("regiones", []) if r["id"] == id_region), {}
        )
        porcentaje = 100 * region.area / area_plato if area_plato else 0.0
        return (
            f"Región {id_region}: {region.area} píxeles ({porcentaje:.2f}% del plato). "
            f"Segmentación: {segmentacion.get('metodo', 'sin registro')}. "
            f"Dimensión del vector: {caracteristicas.get('dimension', 'sin registro')}. "
            f"Clase asignada: {region.resultado.id_clase}; "
            f"score {region.resultado.score:.3f}; "
            f"umbral {clasificacion.get('umbral', 'sin registro')}. "
            f"Resultado: {region.estado}. "
            f"Ids de origen (antes de fusionar): {detalle.get('ids_origen', [])}."
        )

    def explicar_why(self, transaccion: Transaccion, id_clase: str) -> str:
        """Explica el precio, la política y las regiones que respaldan un ítem."""
        item = (
            next(
                (
                    i
                    for i in transaccion.recibo.items
                    if i.alimento.id_clase == id_clase
                ),
                None,
            )
            if transaccion.recibo
            else None
        )
        if item is None:
            raise ValueError(f"No existe un ítem de la clase {id_clase} en el recibo.")
        area = sum(r.area for r in transaccion.regiones if r.id in item.ids_region)
        cobro = self._datos_etapa(transaccion, "cobro")
        return (
            f"{item.alimento.nombre} ({id_clase}): precio del catálogo "
            f"{formatear_dinero(item.alimento.precio_centavos)}; "
            f"política {cobro.get('politica', 'sin registro')}; "
            f"importe cobrado {formatear_dinero(item.precio_cobrado_centavos)}. "
            f"Regiones de respaldo: {item.ids_region}; área total: {area} píxeles "
            f"({100 * (item.area_frac or 0):.2f}% del plato)."
        )

    @staticmethod
    def _datos_etapa(transaccion: Transaccion, etapa: str) -> dict:
        return next(
            (
                e.datos
                for e in reversed(transaccion.trazas)
                if e.etapa == etapa and "accion" not in e.datos
            ),
            {},
        )
