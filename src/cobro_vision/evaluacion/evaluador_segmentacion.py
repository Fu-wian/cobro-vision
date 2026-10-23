"""Pureza y techo de exactitud de las regiones automáticas."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import EtiquetaInvalida, PlatoNoDetectado
from cobro_vision.evaluacion.comun import cargar_prueba, guardar_evaluacion
from cobro_vision.evaluacion.registro import RegistroExperimentos


@dataclass
class EvaluadorSegmentacion:
    """Mide todas las regiones, incluidas las que no pasarían el filtro de pureza."""

    config: Configuracion
    registro: RegistroExperimentos = field(default_factory=RegistroExperimentos)
    carpeta_salida: Path | None = field(default=None, init=False)

    def evaluar(self) -> dict:
        """Rasteriza a escala e incluye los píxeles etiquetados sin región como fallos."""
        _, _, muestras = cargar_prueba(self.config)
        constructor = FabricaPipeline.crear_constructor_instancias(self.config)
        minimo = self.config.seccion("entrenamiento")["pureza_minima"]
        fotos, detalles = [], []
        for muestra in muestras:
            original = CamaraArchivo.leer(muestra.ruta_imagen)
            imagen = constructor.preprocesador.procesar(original)
            mapa = constructor.mapa.rasterizar(
                muestra.etiquetado,
                imagen.shape[:2],
                constructor.preprocesador.escala(original),
            )
            error = ""
            try:
                mascara = constructor.detector.detectar(imagen)
                regiones = constructor.segmentador.segmentar(imagen, mascara)
            except PlatoNoDetectado as excepcion:
                regiones = []
                error = str(excepcion)
            etiquetados = int(np.count_nonzero(mapa))
            aciertos, descartadas = 0, 0
            for region in regiones:
                conteos = np.bincount(mapa[region.mascara])
                conteos[0] = (
                    0  # Sin etiqueta no compite con las clases, incluido fondo.
                )
                dominante = int(conteos.argmax())
                cantidad = int(conteos[dominante])
                pureza = cantidad / region.area
                descartada = pureza < minimo
                aciertos += cantidad
                descartadas += int(descartada)
                detalles.append(
                    {
                        "id_foto": muestra.id_foto,
                        "id_region": region.id,
                        "area": region.area,
                        "clase_dominante": constructor.mapa.clase_de(dominante),
                        "pixeles_dominantes": cantidad,
                        "pureza": pureza,
                        "descartada": descartada,
                    }
                )
            # TODO(equipo): con regiones disjuntas ambas fórmulas de la sección
            # 10.2 coinciden: sum(area * pureza) / total de píxeles etiquetados.
            # El denominador incluye etiquetas fuera del plato y áreas sin región.
            valor = aciertos / etiquetados if etiquetados else None
            fotos.append(
                {
                    "id_foto": muestra.id_foto,
                    "plato_id": muestra.plato_id,
                    "pixeles_etiquetados": etiquetados,
                    "pixeles_dominantes": aciertos,
                    "pureza_ponderada": valor,
                    "techo_exactitud": valor,
                    "n_regiones": len(regiones),
                    "regiones_descartadas": descartadas,
                    "fraccion_descartada": descartadas / len(regiones)
                    if regiones
                    else 0.0,
                    "error": error,
                }
            )
        etiquetados = sum(foto["pixeles_etiquetados"] for foto in fotos)
        if not etiquetados:
            raise EtiquetaInvalida(
                "No hay píxeles etiquetados en test para evaluar segmentación."
            )
        aciertos = sum(foto["pixeles_dominantes"] for foto in fotos)
        regiones = sum(foto["n_regiones"] for foto in fotos)
        descartadas = sum(foto["regiones_descartadas"] for foto in fotos)
        metricas = {
            "split": "test",
            "n_fotos": len(fotos),
            "pureza_ponderada": aciertos / etiquetados,
            "techo_exactitud": aciertos / etiquetados,
            "pixeles_etiquetados": etiquetados,
            "n_regiones": regiones,
            "regiones_por_foto_media": regiones / len(fotos),
            "regiones_descartadas": descartadas,
            "fraccion_descartada": descartadas / regiones if regiones else 0.0,
            "n_errores": sum(bool(foto["error"]) for foto in fotos),
        }
        self.carpeta_salida = guardar_evaluacion(
            "segmentacion",
            self.config,
            metricas,
            {
                "por_foto": pd.DataFrame(fotos),
                "por_region": pd.DataFrame(
                    detalles,
                    columns=[
                        "id_foto",
                        "id_region",
                        "area",
                        "clase_dominante",
                        "pixeles_dominantes",
                        "pureza",
                        "descartada",
                    ],
                ),
            },
            self.registro,
        )
        return metricas
