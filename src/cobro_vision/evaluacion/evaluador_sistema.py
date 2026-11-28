"""Evaluación del recibo automático, revisión y tiempos por fotografía."""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.modelos import Ajuste
from cobro_vision.evaluacion.comun import cargar_prueba, guardar_evaluacion
from cobro_vision.evaluacion.registro import RegistroExperimentos

_ETAPAS = (
    "preprocesamiento",
    "deteccion_plato",
    "segmentacion",
    "caracteristicas",
    "clasificacion",
    "posprocesamiento",
    "cobro",
    "total",
)


@dataclass
class EvaluadorSistema:
    """Compara las clases y los centavos del pipeline completo con lo esperado."""

    config: Configuracion
    registro: RegistroExperimentos = field(default_factory=RegistroExperimentos)
    carpeta_salida: Path | None = field(default=None, init=False)

    def evaluar(self) -> dict:
        """Procesa test sin corregir ni confirmar transacciones ni ajustar parámetros."""
        _, gestor, muestras = cargar_prueba(self.config)
        controlador = FabricaPipeline.crear_evaluacion(self.config)
        constructor_etiquetas = FabricaPipeline.crear_constructor_instancias(
            self.config
        )
        minimo_pureza = self.config.seccion("entrenamiento")["pureza_minima"]        
        fotos, mediciones, eventos_posprocesamiento = [], [], []
        regiones_predicciones = []
        try:
            for muestra in muestras:
                esperado = gestor.contenido_esperado(muestra)
                recibo_esperado = controlador.calculadora.calcular(
                    [], [Ajuste("agregar", clase) for clase in sorted(esperado)]
                )
                transaccion = controlador.procesar(
                    CamaraArchivo.leer(muestra.ruta_imagen)
                )
                mapa_real = None

                if transaccion.imagen_procesada is not None:
                    mapa_real = constructor_etiquetas.mapa.rasterizar(
                        muestra.etiquetado,
                        transaccion.imagen_procesada.shape[:2],
                        constructor_etiquetas.preprocesador.escala(
                            transaccion.imagen_original
                        ),
                    )
                for region in transaccion.regiones_crudas:
                    resultado = region.resultado

                    if resultado is None:
                        continue

                    clase_real = ""
                    pureza = None
                    pureza_etiquetada = None
                    fraccion_etiquetada = None
                    pixeles_etiquetados = 0
                    pixeles_dominantes = 0
                    evaluable = False
                    correcta = None

                    if mapa_real is not None:
                        conteos = np.bincount(mapa_real[region.mascara])

                        if len(conteos):
                            conteos[0] = 0

                            dominante = int(conteos.argmax())
                            pixeles_dominantes = int(conteos[dominante])
                            pixeles_etiquetados = int(conteos[1:].sum())

                            clase_real = (
                                constructor_etiquetas.mapa.clase_de(dominante)
                                if pixeles_dominantes > 0
                                else ""
                            )

                            pureza = pixeles_dominantes / region.area

                            fraccion_etiquetada = (
                                pixeles_etiquetados / region.area
                            )

                            pureza_etiquetada = (
                                pixeles_dominantes / pixeles_etiquetados
                                if pixeles_etiquetados
                                else None
                            )

                            evaluable = (
                                pixeles_dominantes > 0
                                and pureza >= minimo_pureza
                            )

                            correcta = (
                                resultado.id_clase == clase_real
                                if evaluable
                                else None
                            )

                    regiones_predicciones.append(
                        {
                            "id_foto": muestra.id_foto,
                            "plato_id": muestra.plato_id,
                            "id_region": region.id,
                            "clase_predicha": resultado.id_clase,
                            "score_svm": resultado.score,
                            "requiere_revision_clasificador": (
                                resultado.requiere_revision
                            ),
                            "clase_alternativa": resultado.alternativa or "",
                            "area": region.area,
                            "estado_inicial": region.estado,
                            "bbox": json.dumps(region.bbox),
                            "clase_real": clase_real,
                            "pixeles_etiquetados": pixeles_etiquetados,
                            "pixeles_dominantes": pixeles_dominantes,
                            "pureza": pureza,
                            "pureza_etiquetada": pureza_etiquetada,
                            "fraccion_etiquetada": fraccion_etiquetada,
                            "evaluable": evaluable,
                            "correcta": correcta,
                        }
                    )
                for indice, traza in enumerate(transaccion.trazas):
                    if traza.etapa != "posprocesamiento":
                        continue

                    eventos_posprocesamiento.append(
                        {
                            "id_foto": muestra.id_foto,
                            "plato_id": muestra.plato_id,
                            "indice_evento": indice,
                            "etapa": traza.etapa,
                            "descripcion": traza.descripcion,
                            "duracion_ms": traza.duracion_ms,
                            "datos": json.dumps(
                                traza.datos,
                                ensure_ascii=False,
                                default=str,
                            ),
                        }
                    )
                recibo = transaccion.recibo
                detectado = (
                    {item.alimento.id_clase for item in recibo.items}
                    if recibo
                    else set()
                )
                total = recibo.total_centavos if recibo else 0
                total_esperado = recibo_esperado.total_centavos
                correcta = transaccion.estado != "error" and recibo is not None
                # TODO(equipo): se evalúa cada foto como una transacción; se conserva
                # plato_id para agrupar después, sin unir predicciones entre tomas.
                fotos.append(
                    {
                        "id_foto": muestra.id_foto,
                        "plato_id": muestra.plato_id,
                        "esperado": json.dumps(sorted(esperado), ensure_ascii=False),
                        "detectado": json.dumps(sorted(detectado), ensure_ascii=False),
                        "plato_exacto": correcta and detectado == esperado,
                        "verdaderos_positivos": len(detectado & esperado),
                        "falsos_positivos": len(detectado - esperado),
                        "falsos_negativos": len(esperado - detectado),
                        "total_centavos": total,
                        "total_esperado_centavos": total_esperado,
                        "total_exacto": correcta and total == total_esperado,
                        "error_absoluto_centavos": abs(total - total_esperado),
                        "requiere_revision": transaccion.estado == "requiere_revision",
                        "estado": transaccion.estado,
                        "error": transaccion.mensaje_error or "",
                        **{
                            f"{etapa}_ms": transaccion.tiempos.get(etapa)
                            for etapa in _ETAPAS
                        },
                    }
                )
                mediciones.extend(
                    {"id_foto": muestra.id_foto, "etapa": etapa, "ms": valor}
                    for etapa, valor in transaccion.tiempos.items()
                )
        finally:
            controlador.camara.liberar()
        cantidad = len(fotos)
        verdaderos = sum(foto["verdaderos_positivos"] for foto in fotos)
        falsos_positivos = sum(foto["falsos_positivos"] for foto in fotos)
        falsos_negativos = sum(foto["falsos_negativos"] for foto in fotos)
        tiempos = {}
        for etapa in _ETAPAS:
            valores = [
                medicion["ms"] for medicion in mediciones if medicion["etapa"] == etapa
            ]
            tiempos[etapa] = {
                "n": len(valores),
                "media": float(np.mean(valores)) if valores else None,
                "p95": float(np.percentile(valores, 95)) if valores else None,
            }
        # TODO(equipo): precisión/recall sin denominador se informan como 0;
        # las etapas no ejecutadas no aportan tiempos ficticios de cero.
        metricas = {
            "split": "test",
            "n_fotos": cantidad,
            "plato_exacto": sum(foto["plato_exacto"] for foto in fotos) / cantidad,
            "precision_items": verdaderos / (verdaderos + falsos_positivos)
            if verdaderos + falsos_positivos
            else 0.0,
            "recall_items": verdaderos / (verdaderos + falsos_negativos)
            if verdaderos + falsos_negativos
            else 0.0,
            "total_exacto": sum(foto["total_exacto"] for foto in fotos) / cantidad,
            # TODO(equipo): el error medio en B/. se serializa como texto decimal
            # para conservar dinero exacto en JSON/CSV, sin convertirlo a float.
            "error_total_medio": str(
                Decimal(sum(foto["error_absoluto_centavos"] for foto in fotos))
                / cantidad
                / 100
            ),
            "fraccion_revision": sum(foto["requiere_revision"] for foto in fotos)
            / cantidad,
            "fraccion_error": sum(foto["estado"] == "error" for foto in fotos)
            / cantidad,
            "tiempos_ms": tiempos,
        }
        self.carpeta_salida = guardar_evaluacion(
            "sistema",
            self.config,
            metricas,
            {
                "por_foto": pd.DataFrame(fotos),
                "tiempos": pd.DataFrame(
                    [{"etapa": etapa, **valores} for etapa, valores in tiempos.items()]
                ),
                "eventos_posprocesamiento": pd.DataFrame(eventos_posprocesamiento),
                "regiones_predicciones": pd.DataFrame(regiones_predicciones),
            },
            self.registro,
        )
        return metricas
