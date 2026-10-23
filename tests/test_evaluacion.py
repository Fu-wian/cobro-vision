"""Evaluación exclusivamente con fotos y etiquetas sintéticas, sin ajustes."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.clasificacion.entrenador import EntrenadorModelo
from cobro_vision.cli import principal
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import EtiquetaInvalida, PlatoNoDetectado
from cobro_vision.dominio.modelos import Ajuste, ResultadoClasificacion, Transaccion
from cobro_vision.evaluacion.comun import cargar_prueba
from cobro_vision.evaluacion.evaluador_clasificador import EvaluadorClasificador
from cobro_vision.evaluacion.evaluador_segmentacion import EvaluadorSegmentacion
from cobro_vision.evaluacion.evaluador_sistema import EvaluadorSistema
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.segmentacion.base import _crear_region


@pytest.fixture
def config_prueba(
    config_entrenamiento: Configuracion, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Configuracion:
    """Aísla las salidas y el registro; no entrena ni carga ningún dato real."""
    datos = deepcopy(config_entrenamiento._datos)
    datos["rutas"]["salidas"] = str(tmp_path / "salidas")
    datos["camara"]["carpeta"] = datos["rutas"]["imagenes"]
    monkeypatch.chdir(tmp_path)
    return Configuracion(datos)


def test_clasificador_modelo_persistido_y_determinismo(
    config_prueba: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """El evaluador usa el modo guardado y solo test; conserva modelo, YAML y splits."""
    entrenador = FabricaPipeline.crear_entrenador(config_prueba)
    entrenador.entrenar(GestorDataset(config_prueba, catalogo).cargar_muestras())
    datos = deepcopy(config_prueba._datos)
    datos["clasificador"]["tipo"] = "svm"
    datos["entrenamiento"]["modo"] = "ambos"  # No altera el modo del SVM ya guardado.
    config = Configuracion(datos)
    rutas = config.seccion("rutas")
    modelo = Path(rutas["modelo_actual"])
    manifiesto = Path(rutas["manifiesto"])
    originales = modelo.read_bytes(), manifiesto.read_bytes(), config.hash()
    # Un JSON train inválido demuestra que no se leen sus etiquetas al evaluar.
    for muestra in GestorDataset(config, catalogo).cargar_muestras("train"):
        muestra.etiquetado.ruta_json.write_text("no es JSON", encoding="utf-8")
    crear = FabricaPipeline.crear_constructor_instancias
    llamadas = []

    def crear_observado(config: Configuracion):
        constructor = crear(config)
        generar = constructor.generar

        def observar(muestras, modo):
            assert muestras and all(muestra.split == "test" for muestra in muestras)
            llamadas.append(modo)
            return generar(muestras, modo)

        constructor.generar = observar
        return constructor

    with (
        patch.object(FabricaPipeline, "crear_constructor_instancias", crear_observado),
        patch.object(
            EntrenadorModelo, "entrenar", side_effect=AssertionError("reentrenó")
        ),
        patch("sklearn.svm.SVC.fit", side_effect=AssertionError("ajustó SVM")),
        patch(
            "sklearn.model_selection.GridSearchCV.fit",
            side_effect=AssertionError("buscó parámetros"),
        ),
        patch(
            "sklearn.preprocessing.StandardScaler.fit",
            side_effect=AssertionError("ajustó escalador"),
        ),
    ):
        evaluador = EvaluadorClasificador(config)
        primero = evaluador.evaluar()
        segundo = evaluador.evaluar()
    assert primero == segundo
    assert llamadas == ["poligonos", "segmentacion"] * 2
    assert primero["modo_entrenamiento"] == "poligonos"
    for modo, metricas in primero["modos"].items():
        assert metricas["exactitud"] >= 0.95
        assert {"precision", "recall", "f1", "soporte"} <= metricas["por_clase"][
            0
        ].keys()
        assert np.asarray(metricas["matriz_confusion"]).shape == (5, 5)
        assert (
            np.asarray(metricas["matriz_confusion"]).sum() == metricas["n_instancias"]
        )
        matriz = pd.read_csv(
            evaluador.carpeta_salida / f"{modo}_matriz_confusion.csv", index_col=0
        )
        assert matriz.shape == (5, 5)
        assert list(matriz.columns) == metricas["clases"] == list(matriz.index)
        with Image.open(
            evaluador.carpeta_salida / f"{modo}_matriz_confusion.png"
        ) as imagen:
            assert imagen.width > 0 and imagen.height > 0
    assert originales == (modelo.read_bytes(), manifiesto.read_bytes(), config.hash())
    assert len(pd.read_csv("experimentos/resultados.csv")) == 2


def test_tabla_umbral_incluye_igualdad_y_no_selecciona_umbral(
    config_prueba: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Se calculan 21 puntos sobre los mismos scores; null si no hay aceptadas."""
    constructor = FabricaPipeline.crear_constructor_instancias(config_prueba)
    clasificador = Mock(meta={"modo": "poligonos"}, umbral=0.5)
    resultados_por_modo = []

    def generar(muestras, modo):
        instancias = constructor.generar(muestras, modo)
        resultados = [
            ResultadoClasificacion(
                instancia.id_clase if indice % 2 else "pollo",
                (indice % 3) / 10,
                indice % 3 < 2,
            )
            for indice, instancia in enumerate(instancias)
        ]
        resultados_por_modo.append((instancias, resultados))
        clasificador.predecir_lote.return_value = resultados
        return instancias

    with (
        patch.object(FabricaPipeline, "crear_clasificador", return_value=clasificador),
        patch.object(
            FabricaPipeline,
            "crear_constructor_instancias",
            return_value=SimpleNamespace(
                generar=generar, constructor=constructor.constructor
            ),
        ),
    ):
        metricas = EvaluadorClasificador(config_prueba).evaluar()
    assert clasificador.predecir_lote.call_count == 2
    assert clasificador.umbral == 0.5
    for modo, (instancias, resultados) in zip(
        metricas["modos"].values(), resultados_por_modo, strict=True
    ):
        tabla = modo["tabla_umbral"]
        assert [fila["umbral"] for fila in tabla] == [paso / 10 for paso in range(21)]
        assert tabla[0]["cobertura"] == 1
        aceptadas = [
            i for i, resultado in enumerate(resultados) if resultado.score >= 0.1
        ]
        assert tabla[1]["aceptadas"] == len(aceptadas)
        assert tabla[1]["precision_aceptadas"] == pytest.approx(
            sum(instancias[i].id_clase == resultados[i].id_clase for i in aceptadas)
            / len(aceptadas)
        )
        assert tabla[-1]["cobertura"] == 0
        assert tabla[-1]["precision_aceptadas"] is None
    assert catalogo.clases_validas()  # El catálogo de prueba permanece disponible.


def test_segmentacion_pureza_pesa_pixeles_y_penaliza_areas_sin_region(
    config_prueba: Configuracion,
) -> None:
    """El 0 no es una clase; las zonas sin región permanecen en el denominador."""
    mapa = np.array([[1, 1, 1, 2], [1, 1, 0, 2]], dtype=np.int16)
    regiones = [
        _crear_region(1, np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)),
        _crear_region(2, np.array([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=bool)),
    ]
    constructor = FabricaPipeline.crear_constructor_instancias(config_prueba)
    with (
        patch.object(
            FabricaPipeline, "crear_constructor_instancias", return_value=constructor
        ),
        patch.object(FabricaPipeline, "crear_clasificador") as clasificar,
        patch.object(
            constructor.preprocesador,
            "procesar",
            return_value=np.zeros((2, 4, 3), dtype=np.uint8),
        ),
        patch.object(constructor.mapa, "rasterizar", return_value=mapa),
        patch.object(
            constructor.detector, "detectar", return_value=np.ones((2, 4), dtype=bool)
        ),
        patch.object(constructor.segmentador, "segmentar", return_value=regiones),
    ):
        evaluador = EvaluadorSegmentacion(config_prueba)
        metricas = evaluador.evaluar()
    clasificar.assert_not_called()
    assert (
        metricas["pureza_ponderada"]
        == metricas["techo_exactitud"]
        == pytest.approx(4 / 7)
    )
    assert metricas["regiones_por_foto_media"] == 2
    assert metricas["fraccion_descartada"] == 0.5
    fotos = pd.read_csv(evaluador.carpeta_salida / "por_foto.csv")
    assert set(fotos["pixeles_etiquetados"]) == {7}
    assert set(fotos["n_regiones"]) == {2}


def test_segmentacion_fallo_detector_no_desaparece_del_informe(
    config_prueba: Configuracion,
) -> None:
    """Un fallo de plato tiene cero regiones y techo cero, sin división por cero."""
    constructor = FabricaPipeline.crear_constructor_instancias(config_prueba)
    with (
        patch.object(
            FabricaPipeline, "crear_constructor_instancias", return_value=constructor
        ),
        patch.object(
            constructor.detector, "detectar", side_effect=PlatoNoDetectado("Sin plato")
        ),
    ):
        evaluador = EvaluadorSegmentacion(config_prueba)
        metricas = evaluador.evaluar()
    assert metricas["n_errores"] == metricas["n_fotos"]
    assert metricas["techo_exactitud"] == metricas["n_regiones"] == 0
    assert metricas["fraccion_descartada"] == 0
    assert pd.read_csv(evaluador.carpeta_salida / "por_region.csv").empty


def test_sistema_micro_dinero_override_errores_y_percentiles(
    config_prueba: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Cuatro transacciones conocidas distinguen exactitud, micro, revisión y error."""
    catalogo.agregar(
        replace(
            catalogo.obtener("pollo"),
            id_clase="pescado_frito",
            nombre="Pescado sintetico",
            precio_centavos=275,
        )
    )
    manifiesto = Path(config_prueba.seccion("rutas")["manifiesto"])
    tabla = pd.read_csv(manifiesto, keep_default_na=False)
    indices = tabla.index[tabla["split"] == "test"]
    tabla.loc[indices[0], "contenido_esperado_override"] = (
        "pollo;arroz_amarillo;ensalada"
    )
    tabla.to_csv(manifiesto, index=False)
    controlador = FabricaPipeline.crear(config_prueba)
    clases = [
        ["pollo", "arroz_amarillo", "ensalada"],
        ["pollo", "arroz_amarillo", "frijoles"],
        ["pollo", "arroz_amarillo", "ensalada", "frijoles", "pescado_frito"],
        [],
    ]
    transacciones = []
    for indice, detectadas in enumerate(clases, start=1):
        recibo = controlador.calculadora.calcular(
            [], [Ajuste("agregar", clase) for clase in detectadas]
        )
        transacciones.append(
            Transaccion(
                str(indice),
                np.zeros((2, 2, 3), dtype=np.uint8),
                recibo=recibo,
                estado="error"
                if indice == 4
                else "requiere_revision"
                if indice == 2
                else "procesada",
                mensaje_error="Sin plato" if indice == 4 else None,
                tiempos={
                    "preprocesamiento": indice * 10,
                    "deteccion_plato": indice * 10,
                    "total": indice * 10,
                    **({"segmentacion": indice * 10} if indice < 4 else {}),
                },
            )
        )
    with (
        patch.object(FabricaPipeline, "crear_evaluacion", return_value=controlador),
        patch.object(controlador, "procesar", side_effect=transacciones),
        patch.object(controlador.camara, "liberar") as liberar,
    ):
        evaluador = EvaluadorSistema(config_prueba)
        metricas = evaluador.evaluar()
    liberar.assert_called_once()
    assert metricas["plato_exacto"] == metricas["total_exacto"] == 0.25
    assert metricas["precision_items"] == pytest.approx(10 / 11)
    assert metricas["recall_items"] == pytest.approx(10 / 15)
    assert metricas["error_total_medio"] == "2.125"
    assert metricas["fraccion_revision"] == metricas["fraccion_error"] == 0.25
    assert metricas["tiempos_ms"]["total"] == {"n": 4, "media": 25, "p95": 38.5}
    assert metricas["tiempos_ms"]["segmentacion"]["n"] == 3
    assert metricas["tiempos_ms"]["cobro"] == {"n": 0, "media": None, "p95": None}
    fotos = pd.read_csv(evaluador.carpeta_salida / "por_foto.csv")
    assert fotos["total_esperado_centavos"].tolist() == [425, 500, 500, 500]
    assert fotos["error_absoluto_centavos"].tolist() == [0, 75, 275, 500]


def test_sistema_total_esperado_respeta_cobro_por_plato(
    config_prueba: Configuracion,
) -> None:
    """La tarifa configurada se aplica también al contenido esperado del catálogo."""
    datos = deepcopy(config_prueba._datos)
    datos["cobro"] = {
        "politica": "CobroPorPlato",
        "por_plato": {"precios_por_proteina_centavos": {"pollo": 450}},
    }
    config = Configuracion(datos)
    controlador = FabricaPipeline.crear(config)
    recibo = controlador.calculadora.calcular([], [Ajuste("agregar", "pollo")])
    with (
        patch.object(FabricaPipeline, "crear_evaluacion", return_value=controlador),
        patch.object(
            controlador,
            "procesar",
            return_value=Transaccion(
                "prueba",
                np.zeros((2, 2, 3), dtype=np.uint8),
                recibo=recibo,
                estado="procesada",
            ),
        ),
    ):
        metricas = EvaluadorSistema(config).evaluar()
    assert metricas["total_exacto"] == 1
    assert metricas["error_total_medio"] == "0"
    assert metricas["plato_exacto"] == 0


def test_cli_tres_evaluaciones_con_svm_sintetico(
    config_prueba: Configuracion,
    catalogo: CatalogoAlimentos,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Los comandos reales generan informes y tres filas sin entrenar al evaluar."""
    FabricaPipeline.crear_entrenador(config_prueba).entrenar(
        GestorDataset(config_prueba, catalogo).cargar_muestras()
    )
    for muestra in GestorDataset(config_prueba, catalogo).cargar_muestras("train"):
        muestra.etiquetado.ruta_json.write_text("no es JSON", encoding="utf-8")
    datos = deepcopy(config_prueba._datos)
    datos["clasificador"]["tipo"] = "svm"
    ruta = tmp_path / "evaluacion.yaml"
    ruta.write_text(yaml.safe_dump(datos), encoding="utf-8")
    originales = {
        archivo: archivo.read_bytes()
        for archivo in [
            ruta,
            Path(datos["rutas"]["manifiesto"]),
            *Path(datos["rutas"]["modelos"]).iterdir(),
        ]
    }
    with (
        patch("sklearn.svm.SVC.fit", side_effect=AssertionError("ajustó SVM")),
        patch(
            "sklearn.model_selection.GridSearchCV.fit",
            side_effect=AssertionError("buscó parámetros"),
        ),
        patch(
            "sklearn.preprocessing.StandardScaler.fit",
            side_effect=AssertionError("ajustó escalador"),
        ),
        patch.object(
            EntrenadorModelo, "entrenar", side_effect=AssertionError("reentrenó")
        ),
    ):
        for objetivo in ("clasificador", "segmentacion", "sistema"):
            assert principal(["evaluar", objetivo, "--config", str(ruta)]) == 0
            salida = capsys.readouterr().out
            assert "test" in salida
            if objetivo == "sistema":
                for clave in (
                    "plato_exacto",
                    "precision_items",
                    "recall_items",
                    "total_exacto",
                    "error_total_medio",
                    "fraccion_revision",
                    "tiempos_ms",
                    "p95",
                ):
                    assert clave in salida
    assert all(
        archivo.read_bytes() == contenido for archivo, contenido in originales.items()
    )
    carpetas = list((tmp_path / "salidas" / "evaluacion").iterdir())
    assert len(carpetas) == 3
    assert all(
        (carpeta / "metricas.json").is_file() and list(carpeta.glob("*.csv"))
        for carpeta in carpetas
    )
    assert list((tmp_path / "salidas" / "evaluacion").rglob("*.png"))
    assert len(pd.read_csv("experimentos/resultados.csv")) == 3
    assert not (tmp_path / "salidas" / "transacciones.csv").exists()


@pytest.mark.parametrize(
    "evaluador", [EvaluadorClasificador, EvaluadorSegmentacion, EvaluadorSistema]
)
def test_rechaza_test_vacio_y_fuga_sin_reescribir_splits(
    config_prueba: Configuracion,
    evaluador: type,
) -> None:
    """El manifiesto no se divide ni se reconstruye al fallar una evaluación."""
    ruta = Path(config_prueba.seccion("rutas")["manifiesto"])
    tabla = pd.read_csv(ruta)
    tabla["split"] = "train"
    tabla.to_csv(ruta, index=False)
    original = ruta.read_bytes()
    with pytest.raises(EtiquetaInvalida, match="No hay muestras test"):
        evaluador(config_prueba).evaluar()
    assert ruta.read_bytes() == original
    tabla.loc[0, "split"] = "test"
    tabla.to_csv(ruta, index=False)
    with pytest.raises(EtiquetaInvalida, match="Fuga"):
        evaluador(config_prueba).evaluar()


def test_exige_manifiesto_existente(config_prueba: Configuracion) -> None:
    """La evaluación no escanea ni asigna splits de manera implícita."""
    Path(config_prueba.seccion("rutas")["manifiesto"]).unlink()
    with pytest.raises(FileNotFoundError, match="construir-manifiesto"):
        cargar_prueba(config_prueba)


@pytest.mark.parametrize("objetivo", ["clasificador", "segmentacion", "sistema"])
def test_cli_error_legible_sin_test(
    config_prueba: Configuracion,
    objetivo: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Un test vacío retorna uno y deja una explicación sin traceback."""
    ruta = tmp_path / "evaluacion.yaml"
    ruta.write_text(yaml.safe_dump(config_prueba._datos), encoding="utf-8")
    manifiesto = Path(config_prueba.seccion("rutas")["manifiesto"])
    tabla = pd.read_csv(manifiesto)
    tabla["split"] = ""
    tabla.to_csv(manifiesto, index=False)
    assert principal(["evaluar", objetivo, "--config", str(ruta)]) == 1
    assert "No hay muestras test" in caplog.text
    assert not Path("experimentos/resultados.csv").exists()


def test_sistema_no_abre_camara_usb(config_prueba: Configuracion) -> None:
    """La cámara del cajero no se abre para evaluar fotos del manifiesto."""
    datos = deepcopy(config_prueba._datos)
    datos["camara"]["clase"] = "CamaraUSB"
    with patch("cv2.VideoCapture", side_effect=AssertionError("Abrió USB")):
        assert EvaluadorSistema(Configuracion(datos)).evaluar()["n_fotos"] == 4


@pytest.mark.parametrize("objetivo", ["clasificador", "sistema"])
def test_cli_modelo_ausente_no_entrena_ni_usa_clasificador_falso(
    config_prueba: Configuracion,
    objetivo: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """La ausencia del SVM se explica sin cambiar el tipo de clasificador."""
    datos = deepcopy(config_prueba._datos)
    datos["clasificador"]["tipo"] = "svm"
    ruta = tmp_path / "evaluacion.yaml"
    ruta.write_text(yaml.safe_dump(datos), encoding="utf-8")
    with patch.object(
        EntrenadorModelo, "entrenar", side_effect=AssertionError("reentrenó")
    ):
        assert principal(["evaluar", objetivo, "--config", str(ruta)]) == 1
    assert "No existe el modelo SVM" in caplog.text
    assert not Path(datos["rutas"]["modelo_actual"]).exists()
    assert not Path("experimentos/resultados.csv").exists()
