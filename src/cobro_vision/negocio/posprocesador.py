"""Consolidación espacial de clases y reglas de revisión del recibo."""

from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.ndimage import find_objects, label

from cobro_vision.dominio.modelos import ID_FONDO, Region, ResultadoClasificacion
from cobro_vision.dominio.utilidades import _validar_mascara, mapa_desde_regiones
from cobro_vision.negocio.catalogo import CatalogoAlimentos


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
        alimentos = {}
        for region in regiones:
            if region.resultado is None:
                raise ValueError("Todas las regiones deben estar clasificadas.")
            clase = region.resultado.id_clase
            if clase not in alimentos:
                alimentos[clase] = self._catalogo.obtener(clase)
            mascara = mapa_origen == region.id
            etiquetas[mascara] = indices[region.resultado.id_clase]
            margenes[mascara] = region.resultado.score
        modo_filtro = self._config.get(
            "filtro_evidencia", "desactivado"
        )

        areas_originales = Counter()
        evidencia_original = Counter()
        area_segmentada = 0

        if modo_filtro != "desactivado":
            for clase, indice in indices.items():
                mascara_original = etiquetas == indice

                area_original = int(
                    mascara_original.sum()
                )

                areas_originales[clase] = area_original

                evidencia_original[clase] = float(
                    margenes[mascara_original].sum()
                )

                area_segmentada += area_original

        modo_confianza = self._config.get(
            "confianza_absorcion", "original"
        )

        etiquetas_originales = (
            etiquetas.copy()
            if modo_confianza != "original"
            else None
        )

        area_plato = int(mascara_plato.sum())
        minimo = self._config["componente_minimo_frac"] * area_plato
        fusiones: dict[int, list[Region]] = {}
        while True:
            componentes = self._fusionar(etiquetas, margenes, clases, fusiones)
            pequenas = [
                r
                for r in componentes
                if r.resultado.id_clase != ID_FONDO and r.area < minimo
            ]
            if not pequenas:
                break
            pequena = min(pequenas, key=lambda r: (r.area, r.id))
            x0, y0, x1, y1 = pequena.bbox
            # El halo conserva todos los bordes de vecindad 4, incluso en el borde.
            recorte = (
                slice(max(0, y0 - 1), min(etiquetas.shape[0], y1 + 1)),
                slice(max(0, x0 - 1), min(etiquetas.shape[1], x1 + 1)),
            )
            fronteras = self._fronteras(pequena.mascara[recorte], etiquetas[recorte])
            # Contar etiquetas suma directamente las fronteras de igual clase.
            fronteras_clase = Counter(
                {
                    clases[i - 1]: n
                    for i, n in fronteras.items()
                    if clases[i - 1] != ID_FONDO
                }
            )
            # TODO(equipo): empates de frontera se resuelven por id_clase;
            # se absorbe primero el componente menor y se fusiona tras cada cambio.
            destino = min(
                fronteras_clase,
                key=lambda clase: (-fronteras_clase[clase], clase),
                default=None,
            )
            etiquetas[pequena.mascara] = indices[destino] if destino else 0
            # Solo cambian las componentes de la clase de origen y la de destino.
            fusiones.pop(indices[pequena.resultado.id_clase], None)
            if destino is not None:
                fusiones.pop(indices[destino], None)
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
        if etiquetas_originales is not None:

            for region in componentes:

                clase = region.resultado.id_clase

                propios = (
                    region.mascara
                    & (etiquetas_originales == indices[clase])
                )

                cantidad_propia = int(propios.sum())

                score_anterior = region.resultado.score

                suma_propia = float(
                    margenes[propios].sum()
                )

                if modo_confianza == "evidencia_propia":

                    score = (
                        suma_propia / cantidad_propia
                        if cantidad_propia else 0.0
                    )

                elif modo_confianza == "penalizada":

                    score = (
                        suma_propia / region.area
                        if region.area else 0.0
                    )

                else:
                    raise ValueError(
                        "Modo de confianza desconocido."
                    )

                region.resultado.score = score

                if cantidad_propia != region.area:

                    self.eventos.append({
                        "accion": "ajustar_confianza_absorcion",
                        "clase": clase,
                        "modo": modo_confianza,
                        "area": region.area,
                        "pixeles_propios": cantidad_propia,
                        "pixeles_absorbidos": (
                            region.area - cantidad_propia
                        ),
                        "score_anterior": score_anterior,
                        "score_nuevo": score
                    })

        areas: Counter[str] = Counter()
        for region in componentes:
            areas[region.resultado.id_clase] += region.area
        clases_excluidas = set()

        if modo_filtro != "desactivado" and area_segmentada:

            for clase, area_final in areas.items():

                area_original = areas_originales[clase]

                if not area_original:
                    continue

                ao = area_original / area_segmentada
                af = area_final / area_segmentada

                score = (
                    evidencia_original[clase]
                    / area_original
                )

                d1 = af < 0.04 and score < 0.65

                d7 = (
                    ao < 0.05
                    and af < 0.06
                    and score < 0.70
                )

                excluir = (
                    d7
                    if modo_filtro == "d7"
                    else d1 or d7
                )

                if excluir:
                    clases_excluidas.add(clase)

                    self.eventos.append({
                        "accion": "excluir_evidencia",
                        "clase": clase,
                        "modo": modo_filtro,
                        "area_original": area_original,
                        "area_final": area_final,
                        "area_segmentada": area_segmentada,
                        "area_original_frac": ao,
                        "area_final_frac": af,
                        "score_original": score,
                        "regla_d1": d1,
                        "regla_d7": d7
                    })

            for clase in clases_excluidas:
                del areas[clase]

        revisar_area = {
            clase
            for clase, area in areas.items()
            if area < self._config["clase_minima_frac"] * area_plato
        }
        for clase in sorted(revisar_area):
            self.eventos.append(
                {
                    "accion": "revisar_clase_area",
                    "clase": clase,
                    "area": areas[clase],
                }
            )
        proteinas = sorted(
            (
                clase
                for clase in areas
                if alimentos[clase].categoria == "proteina"
            ),
            key=lambda clase: (-areas[clase], clase),
        )
        limite = self._config["max_proteinas"]
        # TODO(equipo): proteínas con igual área se priorizan por id_clase.
        revisar = set(proteinas[limite:]) if limite is not None else set()
        revisar.update(revisar_area)
        for clase in sorted(revisar - revisar_area):
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
        for region in componentes:
            if region.resultado.id_clase in clases_excluidas:
                region.estado = "eliminada"
                region.resultado.requiere_revision = True

        return componentes

    @staticmethod
    def _fusionar(
        etiquetas: np.ndarray,
        margenes: np.ndarray,
        clases: list[str],
        fusiones: dict[int, list[Region]],
    ) -> list[Region]:
        regiones = []
        for indice, clase in enumerate(clases, 1):
            if indice not in fusiones:
                # La estructura por defecto de scipy es la cruz de vecindad 4.
                componentes, cantidad = label(etiquetas == indice)
                areas = np.bincount(componentes.ravel(), minlength=cantidad + 1)
                fusiones[indice] = []
                # Dos recorridos por clase sustituyen nonzero/sum por componente.
                for componente, caja in enumerate(
                    find_objects(componentes, cantidad) if cantidad else [], 1
                ):
                    filas, columnas = caja
                    mascara = componentes == componente
                    fusiones[indice].append(
                        Region(
                            id=0,
                            mascara=mascara,
                            bbox=(
                                columnas.start,
                                filas.start,
                                columnas.stop,
                                filas.stop,
                            ),
                            area=int(areas[componente]),
                            resultado=ResultadoClasificacion(
                                clase,
                                # Mismo orden de píxeles y misma reducción float64.
                                float(margenes[mascara].mean()),
                                False,
                            ),
                        )
                    )
            for region in fusiones[indice]:
                region.id = len(regiones) + 1
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
