"""Métricas del modelo cargado y tabla informativa de umbrales."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.evaluacion.comun import cargar_prueba, guardar_evaluacion
from cobro_vision.evaluacion.registro import RegistroExperimentos


@dataclass
class EvaluadorClasificador:
    """Evalúa el modo de entrenamiento y segmentación sin modificar el modelo."""

    config: Configuracion
    registro: RegistroExperimentos = field(default_factory=RegistroExperimentos)
    carpeta_salida: Path | None = field(default=None, init=False)

    def evaluar(self) -> dict:
        """Predice una vez por modo; los umbrales solo filtran los scores obtenidos."""
        catalogo, _, muestras = cargar_prueba(self.config)
        constructor = FabricaPipeline.crear_constructor_instancias(self.config)
        clasificador = FabricaPipeline.crear_clasificador(
            self.config, constructor.constructor, catalogo
        )
        # El modo persistido tiene prioridad si se entrenó usando --modo.
        modo = getattr(clasificador, "meta", {}).get(
            "modo", self.config.seccion("entrenamiento")["modo"]
        )
        modos = list(dict.fromkeys([modo, "segmentacion"]))
        metricas = {"split": "test", "modo_entrenamiento": modo, "modos": {}}
        tablas, matrices = {}, {}
        for modo in modos:
            instancias = constructor.generar(muestras, modo)
            if not instancias:
                raise EtiquetaInvalida(
                    f"No hay instancias test evaluables en modo {modo}."
                )
            resultados = clasificador.predecir_lote(
                np.asarray(
                    [instancia.vector for instancia in instancias], dtype=np.float32
                )
            )
            if len(resultados) != len(instancias):
                raise ValueError(
                    "El clasificador debe devolver un resultado por instancia."
                )
            reales = [instancia.id_clase for instancia in instancias]
            predichas = [resultado.id_clase for resultado in resultados]
            clases = sorted(
                set(catalogo.clases_validas()) | set(reales) | set(predichas)
            )
            precision, recall, f1, soporte = precision_recall_fscore_support(
                reales, predichas, labels=clases, zero_division=0
            )
            por_clase = [
                {
                    "id_clase": clase,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "soporte": int(s),
                }
                for clase, p, r, f, s in zip(
                    clases, precision, recall, f1, soporte, strict=True
                )
            ]
            matriz = confusion_matrix(reales, predichas, labels=clases)
            puntajes = np.asarray([resultado.score for resultado in resultados])
            if not np.isfinite(puntajes).all():
                raise ValueError("Los scores deben ser finitos para evaluar umbrales.")
            aciertos = np.asarray(reales) == np.asarray(predichas)
            umbrales = []
            for paso in range(21):
                umbral = paso / 10
                aceptadas = puntajes >= umbral
                cantidad = int(aceptadas.sum())
                # TODO(equipo): sin aceptadas la precisión es null (no definida);
                # no se selecciona ni se aplica ningún umbral con los datos test.
                umbrales.append(
                    {
                        "umbral": umbral,
                        "aceptadas": cantidad,
                        "cobertura": cantidad / len(resultados),
                        "precision_aceptadas": float(aciertos[aceptadas].mean())
                        if cantidad
                        else None,
                    }
                )
            metricas["modos"][modo] = {
                "n_instancias": len(instancias),
                "n_fotos": len(muestras),
                "exactitud": float(accuracy_score(reales, predichas)),
                "f1_macro": float(f1.mean()),
                "clases": clases,
                "por_clase": por_clase,
                "matriz_confusion": matriz.tolist(),
                "tabla_umbral": umbrales,
            }
            tablas[f"{modo}_por_clase"] = pd.DataFrame(por_clase)
            tablas[f"{modo}_umbrales"] = pd.DataFrame(umbrales)
            tablas[f"{modo}_predicciones"] = pd.DataFrame(
                [
                    {
                        "id_foto": instancia.id_foto,
                        "plato_id": instancia.plato_id,
                        "origen": instancia.origen,
                        "real": instancia.id_clase,
                        "predicha": resultado.id_clase,
                        "score": resultado.score,
                        "requiere_revision": resultado.requiere_revision,
                    }
                    for instancia, resultado in zip(instancias, resultados, strict=True)
                ]
            )
            matrices[f"{modo}_matriz_confusion"] = pd.DataFrame(
                matriz, index=clases, columns=clases
            )
        self.carpeta_salida = guardar_evaluacion(
            "clasificador", self.config, metricas, tablas, self.registro, matrices
        )
        return metricas
