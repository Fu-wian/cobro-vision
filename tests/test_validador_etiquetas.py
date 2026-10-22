"""Errores acumulados, advertencias y códigos de salida de las herramientas de dataset."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.adquisicion.validador_etiquetas import ValidadorEtiquetas
from cobro_vision.cli import principal
from cobro_vision.configuracion import Configuracion
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def test_dataset_valido_advierte_clases_escasas(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Las pocas anotaciones son advertencias y no invalidan el dataset sintético."""
    muestras = GestorDataset(config_dataset, catalogo).cargar_muestras()
    informe = ValidadorEtiquetas(config_dataset, catalogo).validar(muestras)
    assert informe.ok and informe.errores == []
    assert (
        len(informe.advertencias) == 6
    )  # una por clase del catálogo, sin duplicar muestras
    assert any(
        "Clase pollo: 12 ejemplos; menos de 20" in aviso
        for aviso in informe.advertencias
    )
    assert any(
        "Clase pescado_frito: 0 ejemplos" in aviso for aviso in informe.advertencias
    )


def test_acumula_errores_independientes_en_la_misma_foto(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Dimensiones, etiqueta desconocida y dos puntos se informan por separado."""
    ruta = Path(config_dataset.seccion("rutas")["imagenes"]) / "S01/P001_a.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["imageWidth"] = 321
    datos["shapes"][0].update(label="desconocida", points=[[1, 1], [2, 2]])
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    informe = ValidadorEtiquetas(config_dataset, catalogo).validar([])
    assert not informe.ok and len(informe.errores) == 3
    for mensaje in ("dimensiones distintas", "menos de 3 puntos", "fuera del catálogo"):
        assert any(
            mensaje in error and "P001_a.json" in error for error in informe.errores
        )


def test_parejas_huerfanas_sin_formas_y_json_corrupto(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Un huérfano o JSON inválido no oculta los problemas de los otros archivos."""
    carpeta = Path(config_dataset.seccion("rutas")["imagenes"]) / "S01"
    (carpeta / "P001_a.json").unlink()
    (carpeta / "P002_a.png").unlink()
    (carpeta / "P003_a.json").write_text("{", encoding="utf-8")
    ruta = carpeta / "P004_a.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["shapes"] = []
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    informe = ValidadorEtiquetas(config_dataset, catalogo).validar([])
    assert not informe.ok
    assert any(
        "JSON sin imagen" in error and "P002_a" in error for error in informe.errores
    )
    assert any(
        "no se pudo leer el JSON" in error and "P003_a" in error
        for error in informe.errores
    )
    assert any(
        "Imagen sin JSON" in aviso and "P001_a" in aviso
        for aviso in informe.advertencias
    )
    assert any(
        "sin ninguna forma" in aviso and "P004_a" in aviso
        for aviso in informe.advertencias
    )


@pytest.mark.parametrize(
    "cambio,mensaje",
    [
        ({"shape_type": "line"}, "tipo de forma no soportado"),
        ({"points": [[-1, 1], [10, 10], [20, 1]]}, "fuera de la imagen"),
        ({"points": [[1, 1], [10, float("inf")], [20, 1]]}, "coordenadas no válidas"),
    ],
)
def test_validador_reporta_tipo_y_puntos_invalidos(
    config_dataset: Configuracion,
    catalogo: CatalogoAlimentos,
    cambio: dict,
    mensaje: str,
) -> None:
    """Los problemas estructurales del lector aparecen también en el informe."""
    ruta = Path(config_dataset.seccion("rutas")["imagenes"]) / "S01/P001_a.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["shapes"][0].update(cambio)
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    informe = ValidadorEtiquetas(config_dataset, catalogo).validar([])
    assert not informe.ok and any(mensaje in error for error in informe.errores)


def test_solapes_usan_area_del_poligono_menor_y_umbral_estricto(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """El 5% exacto no se avisa; el 10% y una forma contenida sí, sin dar error."""
    ruta = Path(config_dataset.seccion("rutas")["imagenes"]) / "S01/P001_a.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["shapes"] = [
        {"label": "pollo", "shape_type": "rectangle", "points": [[0, 0], [19, 19]]},
        {"label": "pollo", "shape_type": "rectangle", "points": [[19, 0], [38, 19]]},
    ]
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    validador = ValidadorEtiquetas(config_dataset, catalogo)
    assert not any("solape" in aviso for aviso in validador.validar([]).advertencias)
    datos["shapes"][1]["points"] = [[18, 0], [37, 19]]
    datos["shapes"].append(
        {
            "label": "ensalada",
            "shape_type": "rectangle",
            "points": [[5, 5], [6, 6]],
        }
    )
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    informe = validador.validar([])
    assert informe.ok
    solapes = [aviso for aviso in informe.advertencias if "solape" in aviso]
    assert len(solapes) == 2
    assert any("10.0%" in aviso for aviso in solapes)
    assert any("100.0%" in aviso for aviso in solapes)


def test_validador_detecta_split_conflictivo_y_override_desconocido(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """La auditoría revisa también inconsistencias del manifiesto y de las muestras."""
    gestor = GestorDataset(config_dataset, catalogo)
    muestras = gestor.cargar_muestras()
    muestras[0].split, muestras[1].split = "train", "test"
    validador = ValidadorEtiquetas(config_dataset, catalogo)
    assert any(
        "Fuga de datos" in error for error in validador.validar(muestras).errores
    )
    ruta = Path(config_dataset.seccion("rutas")["manifiesto"])
    datos = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    datos.loc[0, "contenido_esperado_override"] = "clase_inventada"
    datos.to_csv(ruta, index=False)
    assert any(
        "override fuera del catálogo" in error
        for error in validador.validar([]).errores
    )
    datos.loc[0, "split"], datos.loc[1, "split"] = "train", "test"
    datos.to_csv(ruta, index=False)
    assert any("Fuga de datos" in error for error in validador.validar([]).errores)


def test_cli_dataset_completo_y_errores_por_modulo(
    config_dataset: Configuracion,
) -> None:
    """El punto de entrada real valida, construye y divide; devuelve 1 con tres errores."""
    rutas = config_dataset.seccion("rutas")
    ruta_config = Path(rutas["imagenes"]).parents[1] / "dataset.yaml"
    for argumentos, mensaje in (
        (["validar-etiquetas"], "0 errores"),
        (["construir-manifiesto"], "12 fotos"),
        (["dividir", "--prueba", "0.2"], "por plato_id"),
    ):
        resultado = subprocess.run(
            [
                sys.executable,
                "-m",
                "cobro_vision",
                *argumentos,
                "--config",
                str(ruta_config),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert resultado.returncode == 0, resultado.stderr + resultado.stdout
        assert mensaje in resultado.stdout
    datos_csv = pd.read_csv(rutas["manifiesto"], dtype=str)
    assert datos_csv.groupby("plato_id")["split"].nunique().eq(1).all()
    ruta = Path(rutas["imagenes"]) / "S01/P001_a.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["imageHeight"] = 321
    datos["shapes"][0].update(label="clase_inventada", points=[[1, 1], [2, 2]])
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    resultado = subprocess.run(
        [
            sys.executable,
            "-m",
            "cobro_vision",
            "validar-etiquetas",
            "--config",
            str(ruta_config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 1
    assert resultado.stdout.count("ERROR:") == 3
    for mensaje in ("dimensiones distintas", "menos de 3 puntos", "fuera del catálogo"):
        assert mensaje in resultado.stdout
    assert "Traceback" not in resultado.stderr


@pytest.mark.parametrize(
    "comando", ["validar-etiquetas", "construir-manifiesto", "dividir"]
)
def test_cli_configuracion_ausente(
    tmp_path: Path, comando: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Los comandos nuevos convierten errores de configuración en un código útil."""
    argumentos = [comando, "--config", str(tmp_path / "ausente.yaml")]
    if comando == "dividir":
        argumentos += ["--prueba", "0.2"]
    assert principal(argumentos) == 1
    assert "No se pudo ejecutar" in caplog.text
