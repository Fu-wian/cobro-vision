"""Una fila por evaluación, aun con métricas de distinta estructura."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cobro_vision.configuracion import Configuracion
from cobro_vision.evaluacion.registro import RegistroExperimentos


@pytest.mark.parametrize("con_git", [True, False])
def test_registro_agrega_filas_y_conserva_metricas(
    datos_config: dict,
    tmp_path: Path,
    con_git: bool,
) -> None:
    """Git es opcional y no se pierden columnas entre distintos evaluadores."""
    config = Configuracion(datos_config)
    ruta = tmp_path / "experimentos" / "resultados.csv"
    registro = RegistroExperimentos(ruta, tmp_path)
    completado = subprocess.CompletedProcess([], 0 if con_git else 128, "abc123\n", "")
    with patch(
        "cobro_vision.evaluacion.registro.subprocess.run", return_value=completado
    ):
        for indice, metricas in enumerate(
            [
                {"exactitud": 1.0},
                {"tiempos_ms": {"total": {"media": 12}}},
                {"error_total_medio": "0.25"},
            ]
        ):
            registro.registrar("prueba", config, metricas)
            tabla = pd.read_csv(ruta, keep_default_na=False)
            assert len(tabla) == indice + 1
            assert json.loads(tabla.iloc[-1]["metricas"]) == metricas
    assert set(tabla["hash_configuracion"]) == {config.hash()}
    assert set(tabla["hash_git"]) == {"abc123" if con_git else ""}
    assert tabla["fecha"].str.contains("T").all()


def test_registro_sin_ejecutable_git(datos_config: dict, tmp_path: Path) -> None:
    """El registro sigue funcionando si git no está instalado."""
    ruta = tmp_path / "resultados.csv"
    with patch(
        "cobro_vision.evaluacion.registro.subprocess.run", side_effect=FileNotFoundError
    ):
        RegistroExperimentos(ruta).registrar(
            "segmentacion", Configuracion(datos_config), {}
        )
    assert len(pd.read_csv(ruta)) == 1
