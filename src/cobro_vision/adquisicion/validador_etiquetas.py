"""Auditoría de archivos, etiquetas y geometría antes del entrenamiento."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cobro_vision.adquisicion.gestor_dataset import GestorDataset, _escanear_archivos
from cobro_vision.adquisicion.lector_labelme import _cargar_alias, _leer_etiquetado
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.dominio.modelos import Forma, Muestra
from cobro_vision.negocio.catalogo import CatalogoAlimentos


@dataclass
class InformeValidacion:
    """Errores que bloquean el uso del dataset y advertencias informativas."""

    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Indica si no se encontraron errores."""
        return not self.errores


class ValidadorEtiquetas:
    """Revisa todas las parejas de archivos y las muestras recibidas sin detenerse."""

    # TODO(equipo): el contrato no define constructor; config y catalogo permiten
    # revisar huérfanos, clases activas y el mínimo configurado sin cambiar validar.
    def __init__(self, config: Configuracion, catalogo: CatalogoAlimentos) -> None:
        self._config = config
        self._catalogo = catalogo
        self._carpeta = Path(config.seccion("rutas")["imagenes"])
        self._minimo = config.seccion("entrenamiento")["min_instancias_por_clase"]
        self._alias = _cargar_alias(config, catalogo)

    def validar(self, muestras: list[Muestra]) -> InformeValidacion:
        """Acumula cada error y advertencia, incluidas fotos que el lector rechaza."""
        informe = InformeValidacion()
        clases = set(self._catalogo.clases_validas())
        conteos: Counter[str] = Counter()
        try:
            imagenes, archivos_json = _escanear_archivos(self._carpeta)
        except OSError as error:
            informe.errores.append(str(error))
            imagenes, archivos_json = [], []
        rutas_json = set(archivos_json) | {
            muestra.etiquetado.ruta_json.resolve() for muestra in muestras
        }
        por_nombre = {ruta.with_suffix("") for ruta in archivos_json}
        for ruta in imagenes:
            if ruta.with_suffix("") not in por_nombre:
                informe.advertencias.append(f"Imagen sin JSON (se ignora): {ruta}")
        for ruta_json in sorted(rutas_json):
            etiquetado, errores = _leer_etiquetado(ruta_json, self._alias)
            informe.errores.extend(errores)
            if etiquetado is None:
                continue
            if etiquetado.ruta_imagen.with_suffix("") != ruta_json.with_suffix(""):
                informe.errores.append(
                    f"{ruta_json}: imagePath debe referirse a la imagen del mismo nombre."
                )
            if not etiquetado.formas:
                # TODO(equipo): una foto sin formas se advierte y no aporta ejemplos;
                # no bloquea la validación, igual que una foto aún sin JSON.
                informe.advertencias.append(f"{ruta_json}: foto sin ninguna forma.")
            for numero, forma in enumerate(etiquetado.formas, start=1):
                if forma.etiqueta not in clases:
                    informe.errores.append(
                        f"{ruta_json}, forma {numero}: etiqueta fuera del catálogo: {forma.etiqueta!r}."
                    )
                elif not errores:
                    # TODO(equipo): en etapa 4, un ejemplo es una forma anotada;
                    # las instancias y los parches se generarán en la etapa 5.
                    conteos[forma.etiqueta] += 1
            if not errores:
                self._revisar_solapes(etiquetado.formas, ruta_json, informe)
        asignados: dict[str, str] = {}
        for muestra in muestras:
            if muestra.split and muestra.split not in ("train", "test"):
                informe.errores.append(
                    f"{muestra.id_foto}: split inválido {muestra.split!r}."
                )
            if muestra.split:
                anterior = asignados.setdefault(muestra.plato_id, muestra.split)
                if anterior != muestra.split:
                    informe.errores.append(
                        f"Fuga de datos: plato_id {muestra.plato_id} aparece en train y test."
                    )
        try:
            gestor = GestorDataset(self._config, self._catalogo)
            manifiesto = gestor._leer_manifiesto()
            for fila in manifiesto.to_dict("records"):
                override = {
                    valor.strip()
                    for valor in fila["contenido_esperado_override"].split(";")
                    if valor.strip()
                }
                for clase in sorted(override - clases):
                    informe.errores.append(
                        f"{fila['id_foto']}: override fuera del catálogo: {clase!r}."
                    )
        except (OSError, ValueError, EtiquetaInvalida) as error:
            informe.errores.append(str(error))
        for clase in sorted(clases):
            if conteos[clase] < self._minimo:
                informe.advertencias.append(
                    f"Clase {clase}: {conteos[clase]} ejemplos; menos de {self._minimo}."
                )
        return informe

    @staticmethod
    def _revisar_solapes(
        formas: list[Forma], ruta_json: Path, informe: InformeValidacion
    ) -> None:
        mascaras = []
        for forma in formas:
            puntos = np.rint(forma.puntos).astype(np.int32)
            x, y, ancho, alto = cv2.boundingRect(puntos)
            mascara = np.zeros((alto, ancho), dtype=np.uint8)
            cv2.fillPoly(mascara, [puntos - (x, y)], 1)
            mascaras.append((x, y, mascara, int(mascara.sum())))
        for indice, (x0, y0, primera, area_primera) in enumerate(mascaras):
            for otra in range(indice + 1, len(mascaras)):
                x1, y1, segunda, area_segunda = mascaras[otra]
                izquierda, arriba = max(x0, x1), max(y0, y1)
                derecha = min(x0 + primera.shape[1], x1 + segunda.shape[1])
                abajo = min(y0 + primera.shape[0], y1 + segunda.shape[0])
                if derecha <= izquierda or abajo <= arriba:
                    continue
                recorte_a = primera[
                    arriba - y0 : abajo - y0, izquierda - x0 : derecha - x0
                ]
                recorte_b = segunda[
                    arriba - y1 : abajo - y1, izquierda - x1 : derecha - x1
                ]
                area_menor = min(area_primera, area_segunda)
                fraccion = np.count_nonzero(recorte_a & recorte_b) / max(1, area_menor)
                if fraccion > 0.05:
                    informe.advertencias.append(
                        f"{ruta_json}: solape entre formas {indice + 1} y {otra + 1} "
                        f"de {fraccion:.1%} del polígono menor (> 5%)."
                    )
