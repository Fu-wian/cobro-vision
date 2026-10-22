"""Consolidación espacial de clases y reglas de revisión del recibo."""

from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.ndimage import label

from cobro_vision.dominio.modelos import ID_FONDO, Region, ResultadoClasificacion
from cobro_vision.dominio.utilidades import _validar_mascara, mapa_desde_regiones
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.segmentacion.base import _crear_region


class Posprocesador:
    """Fusiona por vecindad 4, absorbe ruido y aplica reglas por clase."""

    def __init__(
        self,
        config_posproceso: dict,
        catalogo: CatalogoAlimentos,
        umbral_confianza: float,
    ) -> None:
        self._config = deepcopy(config_posproceso)
        self._catalogo = catalogo
        self._umbral = umbral_confianza
        # TODO(equipo): consolidar no recibe transacción; el controlador incorpora
        # estos eventos y orígenes de la última llamada en la traza de esa foto.
        self.eventos: list[dict] = []
        self.origenes: dict[int, list[int]] = {}

    def consolidar(
        self, regiones: list[Region], mascara_plato: np.ndarray
    ) -> list[Region]:
        """Devuelve regiones nuevas sin alterar las predicciones crudas."""
        _validar_mascara(mascara_plato, mascara_plato.shape)
        self.eventos = []
        self.origenes = {}
        if len({r.id for r in regiones}) != len(regiones):
            raise ValueError("Los ids de las regiones deben ser únicos.")
        mapa_origen = mapa_desde_regiones(regiones, mascara_plato.shape)
        mapa_origen[~mascara_plato] = 0
        etiquetas = np.zeros(mascara_plato.shape, dtype=np.int32)
        margenes = np.zeros(mascara_plato.shape, dtype=np.float64)
        clases = sorted({r.resultado.id_clase for r in regiones if r.resultado})
        indices = {clase: i + 1 for i, clase in enumerate(clases)}
        for region in regiones:
            if region.resultado is None:
                raise ValueError("Todas las regiones deben estar clasificadas.")
            self._catalogo.obtener(region.resultado.id_clase)
            mascara = mapa_origen == region.id
            etiquetas[mascara] = indices[region.resultado.id_clase]
            margenes[mascara] = region.resultado.score
        area_plato = int(mascara_plato.sum())
        minimo = self._config["componente_minimo_frac"] * area_plato
        while True:
            componentes = self._fusionar(etiquetas, margenes, clases)
            pequenas = [
                r
                for r in componentes
                if r.resultado.id_clase != ID_FONDO and r.area < minimo
            ]
            if not pequenas:
                break
            pequena = min(pequenas, key=lambda r: (r.area, r.id))
            mapa = mapa_desde_regiones(componentes, mascara_plato.shape)
            fronteras = self._fronteras(pequena.mascara, mapa)
            fronteras_clase: Counter[str] = Counter()
            for region in componentes:
                if region.id in fronteras and region.resultado.id_clase != ID_FONDO:
                    fronteras_clase[region.resultado.id_clase] += fronteras[region.id]
            # TODO(equipo): empates de frontera se resuelven por id_clase;
            # se absorbe primero el componente menor y se fusiona tras cada cambio.
            destino = min(
                fronteras_clase,
                key=lambda clase: (-fronteras_clase[clase], clase),
                default=None,
            )
            etiquetas[pequena.mascara] = indices[destino] if destino else 0
            self.eventos.append(
                {
                    "accion": "absorcion" if destino else "eliminar_componente",
                    "clase": pequena.resultado.id_clase,
                    "destino": destino,
                    "area": pequena.area,
                    "ids_origen": self._ids_origen(pequena, mapa_origen),
                    "frontera": fronteras_clase[destino] if destino else 0,
                }
            )
        componentes = [r for r in componentes if r.resultado.id_clase != ID_FONDO]
        areas: Counter[str] = Counter()
        for region in componentes:
            areas[region.resultado.id_clase] += region.area
        descartadas = {
            clase
            for clase, area in areas.items()
            if area < self._config["clase_minima_frac"] * area_plato
        }
        for clase in sorted(descartadas):
            self.eventos.append(
                {
                    "accion": "eliminar_clase",
                    "clase": clase,
                    "area": areas[clase],
                }
            )
        componentes = [
            r for r in componentes if r.resultado.id_clase not in descartadas
        ]
        proteinas = sorted(
            (
                clase
                for clase in areas
                if clase not in descartadas
                and self._catalogo.obtener(clase).categoria == "proteina"
            ),
            key=lambda clase: (-areas[clase], clase),
        )
        limite = self._config["max_proteinas"]
        # TODO(equipo): proteínas con igual área se priorizan por id_clase.
        revisar = set(proteinas[limite:]) if limite is not None else set()
        for clase in sorted(revisar):
            self.eventos.append({"accion": "revisar_proteina", "clase": clase})
        for identificador, region in enumerate(componentes, 1):
            region.id = identificador
            region.resultado.requiere_revision = (
                region.resultado.score < self._umbral
                or region.resultado.id_clase in revisar
            )
            region.estado = (
                "revisar" if region.resultado.requiere_revision else "aceptada"
            )
            self.origenes[region.id] = self._ids_origen(region, mapa_origen)
        return componentes

    @staticmethod
    def _fusionar(
        etiquetas: np.ndarray, margenes: np.ndarray, clases: list[str]
    ) -> list[Region]:
        regiones = []
        for indice, clase in enumerate(clases, 1):
            # La estructura por defecto de scipy es la cruz de vecindad 4.
            componentes, cantidad = label(etiquetas == indice)
            for componente in range(1, cantidad + 1):
                mascara = componentes == componente
                region = _crear_region(len(regiones) + 1, mascara)
                region.resultado = ResultadoClasificacion(
                    clase,
                    float(margenes[mascara].mean()),
                    False,
                )
                regiones.append(region)
        return regiones

    @staticmethod
    def _fronteras(mascara: np.ndarray, mapa: np.ndarray) -> Counter[int]:
        fronteras: Counter[int] = Counter()
        for interior, exterior, vecinos in (
            (mascara[:-1], mascara[1:], mapa[1:]),
            (mascara[1:], mascara[:-1], mapa[:-1]),
            (mascara[:, :-1], mascara[:, 1:], mapa[:, 1:]),
            (mascara[:, 1:], mascara[:, :-1], mapa[:, :-1]),
        ):
            ids, cantidades = np.unique(
                vecinos[interior & ~exterior], return_counts=True
            )
            fronteras.update({int(i): int(n) for i, n in zip(ids, cantidades) if i})
        return fronteras

    @staticmethod
    def _ids_origen(region: Region, mapa_origen: np.ndarray) -> list[int]:
        return [int(i) for i in np.unique(mapa_origen[region.mascara]) if i]
