"""Entrenamiento solo con train sintético, grupos y archivos reproducibles."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from sklearn.model_selection import GroupKFold

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.clasificacion.svm import ClasificadorSVM
from cobro_vision.cli import principal
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import EtiquetaInvalida, ModeloIncompatible
from cobro_vision.dominio.modelos import Instancia
from cobro_vision.negocio.catalogo import CatalogoAlimentos


@pytest.mark.parametrize("modo", ["poligonos", "segmentacion"])
def test_exactitud_persistencia_y_operacion(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
    modo: str,
) -> None:
    """Cada modo supera 0.95 en platos de prueba y se carga en el pipeline del cajero."""
    datos = config_entrenamiento._datos
    datos["entrenamiento"]["modo"] = modo
    config = Configuracion(datos)
    gestor = GestorDataset(config, catalogo)
    muestras = gestor.cargar_muestras()
    entrenador = FabricaPipeline.crear_entrenador(config)
    with patch.object(
        entrenador.constructor_instancias,
        "generar",
        wraps=entrenador.constructor_instancias.generar,
    ) as generar:
        modelo = entrenador.entrenar(muestras)
    assert all(m.split == "train" for m in generar.call_args.args[0])
    prueba = entrenador.constructor_instancias.generar(
        gestor.cargar_muestras("test"), modo
    )
    vectores = np.array([i.vector for i in prueba])
    predicciones = modelo.predecir_lote(vectores)
    exactitud = np.mean(
        [r.id_clase == i.id_clase for r, i in zip(predicciones, prueba, strict=True)]
    )
    assert exactitud >= 0.95, f"{modo}: exactitud={exactitud}"
    carpeta = Path(config.seccion("rutas")["modelos"])
    archivos = list(carpeta.glob("svm_*.joblib"))
    assert len(archivos) == 1
    ruta = archivos[0]
    assert (
        ruta.read_bytes() == Path(config.seccion("rutas")["modelo_actual"]).read_bytes()
    )
    metadatos = json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadatos["clases"] == catalogo.clases_validas()
    assert metadatos["nombres_caracteristicas"] == entrenador.constructor.nombres()
    assert metadatos["hash_configuracion"] == config.hash(
        ["imagen", "plato", "segmentacion", "caracteristicas"]
    )
    assert metadatos["metricas_cv"]["n_platos"] == 4
    assert metadatos["fecha"] and metadatos["modo"] == modo
    cargado = ClasificadorSVM.cargar(ruta, entrenador.constructor)
    assert cargado.predecir_lote(vectores) == predicciones
    assert cargado.predecir(vectores[0]) == predicciones[0]
    revisado = ClasificadorSVM.cargar(ruta, entrenador.constructor, umbral=100)
    assert all(r.requiere_revision for r in revisado.predecir_lote(vectores))
    datos["clasificador"]["tipo"] = "svm"
    controlador = FabricaPipeline.crear(Configuracion(datos))
    assert controlador.clasificador.predecir_lote(vectores) == predicciones
    foto = gestor.cargar_muestras("test")[0]
    transaccion = controlador.procesar(CamaraArchivo.leer(foto.ruta_imagen))
    assert transaccion.estado != "error" and transaccion.recibo is not None
    assert isinstance(
        controlador.preprocesador, type(entrenador.constructor_instancias.preprocesador)
    )
    assert type(controlador.constructor) is type(entrenador.constructor)
    assert controlador.constructor.nombres() == entrenador.constructor.nombres()
    datos["caracteristicas"][0]["espacio"] = "lab"  # Misma dimensión, otros nombres.
    incompatible = FabricaPipeline.crear_entrenador(Configuracion(datos)).constructor
    with pytest.raises(ModeloIncompatible, match="extractores"):
        ClasificadorSVM.cargar(ruta, incompatible)


def test_determinismo_y_folds_sin_fuga(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
) -> None:
    """Misma configuración produce métricas idénticas; cada fold agrupa los platos."""
    config = config_entrenamiento
    muestras = GestorDataset(config, catalogo).cargar_muestras()
    primero = FabricaPipeline.crear_entrenador(config)
    segundo = FabricaPipeline.crear_entrenador(config)
    particiones = []
    dividir = GroupKFold.split

    def registrar(self, X, y=None, groups=None):
        for train, test in dividir(self, X, y, groups):
            assert set(groups[train]).isdisjoint(groups[test])
            particiones.append((train.copy(), test.copy()))
            yield train, test

    with patch.object(GroupKFold, "split", registrar):
        modelo_a = primero.entrenar(muestras)
        modelo_b = segundo.entrenar(list(reversed(muestras)))
    assert len(particiones) == 6
    assert modelo_a.meta["metricas_cv"] == modelo_b.meta["metricas_cv"]
    assert modelo_a.meta["hash_configuracion"] == modelo_b.meta["hash_configuracion"]
    assert modelo_a.meta["metricas_cv"]["mejores_parametros"]
    assert len(list(Path(config.seccion("rutas")["modelos"]).glob("svm_*.joblib"))) == 2


def test_clases_con_menos_de_tres_platos_incluye_ausentes_y_no_cuenta_fotos(
    config_entrenamiento: Configuracion,
) -> None:
    """Muchas filas y tomas del mismo plato no permiten evadir el mínimo fijo de 3."""
    entrenador = FabricaPipeline.crear_entrenador(config_entrenamiento)
    instancias = [
        Instancia(
            np.zeros(entrenador.constructor.dimension()),
            clase,
            f"P{plato}",
            f"P{plato}_{toma}",
            "region",
        )
        for clase, cantidad in [
            ("pollo", 2),
            ("fondo", 1),
            ("ensalada", 3),
            ("frijoles", 3),
        ]
        for plato in range(cantidad)
        for toma in range(30)
    ]
    with pytest.raises(EtiquetaInvalida) as error:
        entrenador.validacion_cruzada(instancias)
    assert "al menos 3 platos" in str(error.value)
    for texto in ("pollo: 2", "fondo: 1", "arroz_amarillo: 0"):
        assert texto in str(error.value)
    assert not Path(config_entrenamiento.seccion("rutas")["modelos"]).exists()


def test_avisar_pocas_instancias_sin_bajar_minimo_platos(
    config_entrenamiento: Configuracion,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tres platos por clase se permiten aun si hay menos de 20 filas, con advertencia."""
    datos = config_entrenamiento._datos
    datos["clasificador"]["busqueda"]["folds"] = 5
    entrenador = FabricaPipeline.crear_entrenador(Configuracion(datos))
    instancias = [
        Instancia(
            np.full(entrenador.constructor.dimension(), indice),
            clase,
            f"P{plato}",
            f"P{plato}_a",
            "region",
        )
        for indice, clase in enumerate(entrenador.constructor_instancias.clases)
        for plato in range(3)
    ]
    metricas = entrenador.validacion_cruzada(instancias)
    assert metricas["folds"] == 3 and metricas["n_platos"] == 3
    assert "min_instancias_por_clase=20" in caplog.text


def test_rechaza_fuga_y_splits_sin_asignar(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
) -> None:
    """La API de entrenamiento verifica los grupos aunque se salte el gestor."""
    muestras = GestorDataset(config_entrenamiento, catalogo).cargar_muestras("train")
    entrenador = FabricaPipeline.crear_entrenador(config_entrenamiento)
    with pytest.raises(EtiquetaInvalida, match="Fuga"):
        entrenador.entrenar([muestras[0], replace(muestras[1], split="test")])
    with pytest.raises(EtiquetaInvalida, match="dividir"):
        entrenador.entrenar([replace(muestras[0], split="")])
    with pytest.raises(EtiquetaInvalida, match="No hay muestras train"):
        entrenador.entrenar([])


def test_cli_entrena_ambos_y_falla_con_clases_insuficientes(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """El comando aplica --modo, escribe los tres artefactos y lista faltantes al fallar."""
    ruta = tmp_path / "entrenamiento.yaml"
    ruta.write_text(yaml.safe_dump(config_entrenamiento._datos), encoding="utf-8")
    assert principal(["entrenar", "--config", str(ruta), "--modo", "ambos"]) == 0
    assert "SVM entrenado (ambos)" in capsys.readouterr().out
    archivos = list(
        Path(config_entrenamiento.seccion("rutas")["modelos"]).glob("svm_*.json")
    )
    assert len(archivos) == 1
    assert json.loads(archivos[0].read_text(encoding="utf-8"))["modo"] == "ambos"
    assert (
        yaml.safe_load(ruta.read_text(encoding="utf-8"))["entrenamiento"]["modo"]
        == "poligonos"
    )
    manifiesto = Path(config_entrenamiento.seccion("rutas")["manifiesto"])
    import pandas as pd

    tabla = pd.read_csv(manifiesto)
    tabla["split"] = np.where(tabla["plato_id"].isin(["P001", "P002"]), "train", "test")
    tabla.to_csv(manifiesto, index=False)
    assert principal(["entrenar", "--config", str(ruta)]) == 1
    for clase in catalogo.clases_validas():
        assert f"{clase}: 2" in caplog.text
    assert len(list(archivos[0].parent.glob("svm_*.joblib"))) == 1
