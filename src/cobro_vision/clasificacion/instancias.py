"""Instancias etiquetadas con el mismo procesamiento que usa el cajero."""

import hashlib
import logging

import cv2
import numpy as np

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.adquisicion.mapa_etiquetas import MapaEtiquetas
from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida, PlatoNoDetectado
from cobro_vision.dominio.modelos import Instancia, Muestra, Region
from cobro_vision.preprocesamiento.detector_plato import DetectorPlato
from cobro_vision.preprocesamiento.preprocesador import Preprocesador
from cobro_vision.segmentacion.base import Segmentador, _crear_region

_REGISTRO = logging.getLogger(__name__)


class ConstructorInstancias:
    """Genera polígonos, parches y regiones sin alterar las imágenes de entrada."""

    def __init__(
        self,
        config: Configuracion,
        preprocesador: Preprocesador,
        detector: DetectorPlato,
        segmentador: Segmentador,
        constructor: ConstructorVector,
        mapa: MapaEtiquetas,
    ) -> None:
        self._config = config
        self.preprocesador = preprocesador
        self.detector = detector
        self.segmentador = segmentador
        self.constructor = constructor
        self.mapa = mapa

    @property
    def clases(self) -> list[str]:
        """Devuelve todas las clases esperadas, incluidas las que no tengan filas."""
        clases = []
        while (clase := self.mapa.clase_de(len(clases) + 1)) is not None:
            clases.append(clase)
        return clases

    def generar(self, muestras: list[Muestra], modo: str) -> list[Instancia]:
        """Rasteriza a escala y conserva la identidad del plato en cada instancia."""
        if modo not in ("poligonos", "segmentacion", "ambos"):
            raise ValueError("modo debe ser poligonos, segmentacion o ambos.")
        instancias = []
        for muestra in sorted(muestras, key=lambda muestra: muestra.id_foto):
            original = CamaraArchivo.leer(muestra.ruta_imagen)
            if original.shape[:2] != (
                muestra.etiquetado.alto,
                muestra.etiquetado.ancho,
            ):
                raise EtiquetaInvalida(
                    f"{muestra.id_foto}: dimensiones de imagen y etiquetas distintas."
                )
            imagen = self.preprocesador.procesar(original)
            mapa = self.mapa.rasterizar(
                muestra.etiquetado,
                imagen.shape[:2],
                self.preprocesador.escala(original),
            )
            filas = []
            if modo in ("poligonos", "ambos"):
                filas.extend(self._poligonos(mapa, muestra.id_foto))
            if modo in ("segmentacion", "ambos"):
                try:
                    mascara = self.detector.detectar(imagen)
                except PlatoNoDetectado as error:
                    # TODO(equipo): en ambos se conservan los polígonos si falla
                    # la detección; solo se omiten las regiones automáticas.
                    _REGISTRO.warning(
                        "%s: se omite segmentacion: %s", muestra.id_foto, error
                    )
                else:
                    filas.extend(self._regiones(imagen, mascara, mapa, muestra.id_foto))
            vectores = self.constructor.construir_lote(
                imagen, [region for region, _, _ in filas]
            )
            for vector, (_, clase, origen) in zip(vectores, filas):
                instancias.append(
                    Instancia(vector, clase, muestra.plato_id, muestra.id_foto, origen)
                )
        return instancias

    def _poligonos(
        self, mapa: np.ndarray, id_foto: str
    ) -> list[tuple[Region, str, str]]:
        config = self._config.seccion("entrenamiento")
        minimo = self._config.seccion("segmentacion")["area_minima_px"]
        huella = hashlib.sha256(id_foto.encode("utf-8")).digest()
        generador = np.random.default_rng(
            np.random.SeedSequence(
                [self._config.semilla, int.from_bytes(huella, "big")]
            )
        )
        radio = config["radio_parche"]
        disco = np.zeros((2 * radio + 1, 2 * radio + 1), dtype=np.uint8)
        cv2.circle(disco, (radio, radio), radio, 1, -1)
        area_disco = int(disco.sum())
        filas = []
        for indice in np.unique(mapa):
            if indice == 0:
                continue
            clase = self.mapa.clase_de(int(indice))
            cantidad, componentes = cv2.connectedComponents(
                (mapa == indice).astype(np.uint8), connectivity=8
            )
            for componente in range(1, cantidad):
                mascara = componentes == componente
                if mascara.sum() < minimo:
                    continue
                filas.append(
                    (_crear_region(len(filas) + 1, mascara), clase, "poligono")
                )
                erosionada = cv2.erode(
                    mascara.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                puntos = np.argwhere(erosionada)
                if not puntos.size:
                    continue
                elegidos = generador.integers(
                    len(puntos), size=config["parches_por_poligono"]
                )
                for y, x in puntos[elegidos]:
                    parche = np.zeros(mapa.shape, dtype=np.uint8)
                    cv2.circle(parche, (int(x), int(y)), radio, 1, -1)
                    parche = parche.astype(bool) & mascara
                    if parche.sum() >= area_disco * 0.5:
                        filas.append(
                            (_crear_region(len(filas) + 1, parche), clase, "parche")
                        )
        return filas

    def _regiones(
        self, imagen: np.ndarray, mascara: np.ndarray, mapa: np.ndarray, id_foto: str
    ) -> list[tuple[Region, str, str]]:
        minimo = self._config.seccion("entrenamiento")["pureza_minima"]
        filas = []
        regiones = self.segmentador.segmentar(imagen, mascara)
        for region in regiones:
            conteos = np.bincount(mapa[region.mascara])
            # 0 significa sin etiqueta: no compite como clase ni reduce el denominador.
            conteos[0] = 0
            dominante = int(conteos.argmax())
            if dominante > 0 and conteos[dominante] / region.area >= minimo:
                filas.append((region, self.mapa.clase_de(dominante), "region"))
        _REGISTRO.info(
            "%s: %d regiones aceptadas; %d descartadas por pureza o falta de etiqueta.",
            id_foto,
            len(filas),
            len(regiones) - len(filas),
        )
        return filas
