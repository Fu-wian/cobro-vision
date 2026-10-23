"""SVM persistente y predicciones con margen de revisión."""

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.clasificacion.base import Clasificador
from cobro_vision.dominio.errores import ModeloIncompatible
from cobro_vision.dominio.modelos import ResultadoClasificacion

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


class ClasificadorSVM(Clasificador):
    """Envuelve el pipeline ajustado y los metadatos de sus características."""

    def __init__(
        self, pipeline: "Pipeline", clases: list[str], umbral: float, meta: dict
    ) -> None:
        if len(clases) < 2 or len(set(clases)) != len(clases):
            raise ModeloIncompatible("El SVM necesita al menos dos clases distintas.")
        if list(pipeline.classes_) != clases:
            raise ModeloIncompatible(
                "El orden de las clases no coincide con el pipeline."
            )
        if not np.isfinite(umbral) or umbral < 0:
            raise ValueError("El umbral debe ser finito y no negativo.")
        self.pipeline = pipeline
        self.clases = list(clases)
        self.umbral = float(umbral)
        self.meta = deepcopy(meta)

    def predecir_lote(self, vectores: np.ndarray) -> list[ResultadoClasificacion]:
        """Compara las dos mejores decisiones, también en el caso binario."""
        if (
            vectores.ndim != 2
            or vectores.shape[1] != self.pipeline.n_features_in_
            or not np.isfinite(vectores).all()
        ):
            raise ValueError(
                "Se esperaba una matriz finita de dimensión (n, d) del modelo."
            )
        if not len(vectores):
            return []
        decisiones = self.pipeline.decision_function(vectores)
        if decisiones.ndim == 1:
            # TODO(equipo): el caso binario usa decisiones simétricas [-d, d];
            # el margen es 2*abs(d), cuyo umbral requiere calibración con datos.
            decisiones = np.column_stack((-decisiones, decisiones))
        if decisiones.shape != (len(vectores), len(self.clases)):
            raise ModeloIncompatible(
                "Forma de decision_function incompatible con las clases."
            )
        orden = np.argsort(-decisiones, axis=1, kind="stable")
        resultados = []
        for fila, (mejor, segunda) in enumerate(orden[:, :2]):
            score = float(decisiones[fila, mejor] - decisiones[fila, segunda])
            resultados.append(
                ResultadoClasificacion(
                    self.clases[mejor], score, score < self.umbral, self.clases[segunda]
                )
            )
        return resultados

    def guardar(self, ruta: Path) -> None:
        """Escribe el modelo local y un JSON legible con sus metadatos."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        metadatos = {**self.meta, "clases": self.clases, "umbral": self.umbral}
        contenido = json.dumps(metadatos, ensure_ascii=False, indent=2, allow_nan=False)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "clases": self.clases,
                "umbral": self.umbral,
                "meta": self.meta,
            },
            ruta,
        )
        ruta.with_suffix(".json").write_text(contenido + "\n", encoding="utf-8")

    @classmethod
    def cargar(
        cls, ruta: Path, constructor: ConstructorVector, umbral: float | None = None
    ) -> "ClasificadorSVM":
        """Rechaza modelos cuyas características no coincidan en nombre y orden."""
        datos = joblib.load(ruta)
        try:
            meta = datos["meta"]
            if meta["nombres_caracteristicas"] != constructor.nombres():
                raise ModeloIncompatible(
                    "Los nombres u orden de los extractores difieren del modelo guardado."
                )
            if datos["pipeline"].n_features_in_ != constructor.dimension():
                raise ModeloIncompatible(
                    "La dimensión del modelo difiere del constructor."
                )
            return cls(
                datos["pipeline"],
                datos["clases"],
                datos["umbral"] if umbral is None else umbral,
                meta,
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise ModeloIncompatible(
                "El archivo no contiene un modelo SVM compatible."
            ) from error
