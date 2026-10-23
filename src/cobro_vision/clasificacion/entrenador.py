"""Entrenamiento del SVM con selección de parámetros agrupada por plato."""

import logging
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.clasificacion.instancias import ConstructorInstancias
from cobro_vision.clasificacion.svm import ClasificadorSVM
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.dominio.modelos import Instancia, Muestra

_REGISTRO = logging.getLogger(__name__)


class EntrenadorModelo:
    """Ajusta solo train y persiste el mejor pipeline con su validación cruzada."""

    def __init__(
        self,
        config: Configuracion,
        constructor_instancias: ConstructorInstancias,
        constructor: ConstructorVector,
    ) -> None:
        self._config = config
        self.constructor_instancias = constructor_instancias
        self.constructor = constructor
        self._busqueda: GridSearchCV | None = None

    def entrenar(self, muestras: list[Muestra]) -> ClasificadorSVM:
        """Valida los splits, ajusta train y guarda una copia del modelo actual."""
        splits: dict[str, set[str]] = defaultdict(set)
        for muestra in muestras:
            if muestra.split not in ("train", "test"):
                raise EtiquetaInvalida(
                    f"{muestra.id_foto}: split sin asignar; ejecute dividir antes de entrenar."
                )
            splits[muestra.plato_id].add(muestra.split)
        mezclados = sorted(
            plato for plato, conjuntos in splits.items() if len(conjuntos) > 1
        )
        if mezclados:
            raise EtiquetaInvalida(
                f"Fuga de datos: plato_id en train y test: {mezclados}"
            )
        entrenamiento = [muestra for muestra in muestras if muestra.split == "train"]
        if not entrenamiento:
            raise EtiquetaInvalida(
                "No hay muestras train; construya el manifiesto y ejecute dividir."
            )
        modo = self._config.seccion("entrenamiento")["modo"]
        instancias = self.constructor_instancias.generar(entrenamiento, modo)
        metricas = self.validacion_cruzada(instancias)
        fecha = datetime.now().astimezone()
        meta = {
            "nombres_caracteristicas": self.constructor.nombres(),
            "hash_configuracion": self._config.hash(
                ["imagen", "plato", "segmentacion", "caracteristicas"]
            ),
            "fecha": fecha.isoformat(),
            "modo": modo,
            "semilla": self._config.semilla,
            "metricas_cv": metricas,
        }
        # La fábrica sigue siendo el único lugar que instancia variantes de Clasificador.
        from cobro_vision.controlador.fabrica import FabricaPipeline

        modelo = FabricaPipeline._crear_svm_entrenado(
            self._busqueda.best_estimator_,
            list(self._busqueda.classes_),
            self._config.seccion("clasificador")["umbral_confianza"],
            meta,
        )
        rutas = self._config.seccion("rutas")
        ruta = Path(rutas["modelos"]) / f"svm_{fecha:%Y%m%d_%H%M%S}.joblib"
        # TODO(equipo): si dos entrenamientos terminan en el mismo segundo,
        # se añade un contador al nombre para conservar ambos artefactos.
        contador = 1
        while ruta.exists():
            ruta = (
                Path(rutas["modelos"]) / f"svm_{fecha:%Y%m%d_%H%M%S}_{contador}.joblib"
            )
            contador += 1
        modelo.guardar(ruta)
        actual = Path(rutas["modelo_actual"])
        actual.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ruta, actual)
        _REGISTRO.info("Modelo guardado en %s y copiado a %s.", ruta, actual)
        return modelo

    def validacion_cruzada(self, instancias: list[Instancia]) -> dict:
        """Busca C y gamma con escalado dentro de cada fold y grupos por plato."""
        conteos = Counter(instancia.id_clase for instancia in instancias)
        platos: dict[str, set[str]] = defaultdict(set)
        for instancia in instancias:
            platos[instancia.id_clase].add(instancia.plato_id)
        # TODO(equipo): se exige cobertura de todas las clases activas del mapa
        # (incluido fondo); una clase sin instancias útiles cuenta como 0 platos.
        clases = sorted(set(self.constructor_instancias.clases) | set(conteos))
        insuficientes = [
            f"{clase}: {len(platos[clase])}"
            for clase in clases
            if len(platos[clase]) < 3
        ]
        if insuficientes:
            raise EtiquetaInvalida(
                "Se requieren al menos 3 platos distintos (plato_id) por clase en train. "
                "Clases insuficientes (platos con instancias útiles): "
                + "; ".join(insuficientes)
            )
        if len(conteos) < 2:
            raise EtiquetaInvalida(
                "El SVM necesita al menos 2 clases con instancias en train."
            )
        minimo = self._config.seccion("entrenamiento")["min_instancias_por_clase"]
        for clase in clases:
            if conteos[clase] < minimo:
                _REGISTRO.warning(
                    "Clase %s: %d instancias, menos de min_instancias_por_clase=%d.",
                    clase,
                    conteos[clase],
                    minimo,
                )
        vectores = np.asarray(
            [instancia.vector for instancia in instancias], dtype=np.float32
        )
        if (
            vectores.shape != (len(instancias), self.constructor.dimension())
            or not np.isfinite(vectores).all()
        ):
            raise ValueError(
                "Las instancias deben contener vectores finitos de dimensión uniforme."
            )
        etiquetas = np.array([instancia.id_clase for instancia in instancias])
        grupos = np.array([instancia.plato_id for instancia in instancias])
        config = self._config.seccion("clasificador")
        pipeline = Pipeline(
            [
                ("escalador", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=config["kernel"],
                        C=config["C"],
                        gamma=config["gamma"],
                        class_weight=config["class_weight"],
                        decision_function_shape="ovr",
                        random_state=self._config.semilla,
                    ),
                ),
            ]
        )
        folds = min(config["busqueda"]["folds"], len(set(grupos)))
        self._busqueda = GridSearchCV(
            pipeline,
            {
                "svm__C": config["busqueda"]["C"],
                "svm__gamma": config["busqueda"]["gamma"],
            },
            cv=GroupKFold(n_splits=folds),
            scoring="f1_macro",
            refit=True,
            error_score="raise",
        )
        self._busqueda.fit(vectores, etiquetas, groups=grupos)
        resultados = self._busqueda.cv_results_
        mejor = self._busqueda.best_index_
        return {
            "f1_macro_media": float(self._busqueda.best_score_),
            "f1_macro_desviacion": float(resultados["std_test_score"][mejor]),
            "f1_macro_por_fold": [
                float(resultados[f"split{i}_test_score"][mejor]) for i in range(folds)
            ],
            "mejores_parametros": {
                clave.removeprefix("svm__"): valor
                for clave, valor in self._busqueda.best_params_.items()
            },
            "folds": folds,
            "n_instancias": len(instancias),
            "n_platos": len(set(grupos)),
            "instancias_por_clase": {clase: conteos[clase] for clase in clases},
            "platos_por_clase": {clase: len(platos[clase]) for clase in clases},
        }
