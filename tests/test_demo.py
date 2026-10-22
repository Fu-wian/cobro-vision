"""CLI real y exportación verificable de imágenes, recibo y traza."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cobro_vision.cli import principal
from cobro_vision.dominio.utilidades import formatear_dinero


@pytest.mark.parametrize("debug", [False, True])
def test_demo_desde_modulo_con_y_sin_debug(
    ruta_config_temporal: Path,
    ruta_proyecto: Path,
    tmp_path: Path,
    debug: bool,
) -> None:
    """El ejecutable imprime el recibo y exporta exactamente los artefactos previstos."""
    datos = yaml.safe_load(ruta_config_temporal.read_text(encoding="utf-8"))
    datos["rutas"]["salidas"] = str(tmp_path / "salidas")
    ruta_config_temporal.write_text(yaml.safe_dump(datos), encoding="utf-8")
    argumentos = [
        sys.executable,
        "-m",
        "cobro_vision",
        "demo",
        "--imagen",
        str(ruta_proyecto / "tests/fixtures/plato_sintetico.png"),
        "--config",
        str(ruta_config_temporal),
    ]
    if debug:
        argumentos.append("--debug")
    resultado = subprocess.run(argumentos, capture_output=True, text=True, check=False)
    assert resultado.returncode == 0, resultado.stderr
    assert "Recibo T-" in resultado.stdout and "Total: B/." in resultado.stdout
    carpeta = tmp_path / "salidas/debug/plato_sintetico"
    if not debug:
        assert not carpeta.exists()
        return
    nombres = {
        "01_procesada.png",
        "02_mascara_plato.png",
        "03_regiones_crudas.png",
        "04_clasificadas.png",
        "05_consolidadas.png",
        "recibo.txt",
        "traza.json",
    }
    assert {p.name for p in carpeta.iterdir()} == nombres
    for nombre in sorted(nombres):
        if nombre.endswith(".png"):
            imagen = cv2.imread(str(carpeta / nombre), cv2.IMREAD_UNCHANGED)
            assert imagen is not None and imagen.shape[:2] == (320, 320)
            if nombre.startswith("02"):
                assert set(np.unique(imagen)) == {0, 255}
    recibo = (carpeta / "recibo.txt").read_text(encoding="utf-8")
    assert recibo == resultado.stdout
    traza = json.loads((carpeta / "traza.json").read_text(encoding="utf-8"))
    assert len(traza["tiempos"]) == 8
    assert len([e for e in traza["trazas"] if e["duracion_ms"] is not None]) == 7
    assert traza["trazas"][4]["datos"]["clasificador"] == "ClasificadorFalso"
    cobro = traza["trazas"][-1]
    assert cobro["etapa"] == "cobro"
    assert formatear_dinero(cobro["datos"]["total_centavos"]) in recibo


def test_demo_imagen_ausente_o_plato_no_detectado(
    ruta_config_temporal: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Los errores habituales se informan con código uno y mensaje legible."""
    assert (
        principal(
            [
                "demo",
                "--imagen",
                str(tmp_path / "ausente.png"),
                "--config",
                str(ruta_config_temporal),
            ]
        )
        == 1
    )
    assert "No existe la imagen" in caplog.text
    negra = tmp_path / "negra.png"
    assert cv2.imwrite(str(negra), np.zeros((100, 100, 3), dtype=np.uint8))
    assert (
        principal(
            ["demo", "--imagen", str(negra), "--config", str(ruta_config_temporal)]
        )
        == 1
    )
    assert "plato" in caplog.text.lower()
