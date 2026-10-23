"""Pruebas de la ayuda, los argumentos y los comandos pendientes."""

import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cobro_vision.cli import construir_analizador, principal
from cobro_vision.controlador.controlador import ControladorTransaccion


@pytest.mark.parametrize(
    ("argumentos", "etapa"),
    [
        (["ablacion"], 7),
    ],
)
def test_comandos_pendientes(
    argumentos: list[str], etapa: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cada comando válido informa su etapa y devuelve el código acordado."""
    assert principal(argumentos) == 2
    salida = capsys.readouterr()
    assert salida.out == f"No implementado aún (etapa {etapa})\n"
    assert salida.err == ""


@pytest.mark.parametrize(
    "argumentos",
    [
        ["--config", "otra.yaml", "app"],
        ["app", "--config", "otra.yaml"],
    ],
)
def test_config_antes_o_despues_del_comando(argumentos: list[str]) -> None:
    """La ruta indicada se conserva a ambos lados del subcomando."""
    opciones = construir_analizador().parse_args(argumentos)
    assert opciones.config == Path("otra.yaml")


@pytest.mark.parametrize(
    "argumentos",
    [
        ["demo"],
        ["dividir"],
        ["evaluar", "desconocido"],
        ["entrenar", "--modo", "desconocido"],
    ],
)
def test_rechazar_argumentos_invalidos(argumentos: list[str]) -> None:
    """La CLI exige los argumentos y las opciones declarados."""
    with pytest.raises(SystemExit) as error:
        principal(argumentos)
    assert error.value.code == 2


def test_ayuda_desde_modulo(ruta_proyecto: Path) -> None:
    """El punto de entrada real muestra todos los subcomandos sin errores."""
    resultado = subprocess.run(
        [sys.executable, "-m", "cobro_vision", "--help"],
        cwd=ruta_proyecto,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stderr == ""
    for comando in (
        "inicializar-catalogo",
        "validar-etiquetas",
        "construir-manifiesto",
        "dividir",
        "resumen-dataset",
        "demo",
        "entrenar",
        "evaluar",
        "ablacion",
        "app",
    ):
        assert comando in resultado.stdout


def test_inicializar_catalogo_dos_veces(ruta_config_temporal: Path) -> None:
    """El comando real no duplica filas y exporta etiquetas aptas para LabelMe."""
    import yaml

    datos = yaml.safe_load(ruta_config_temporal.read_text(encoding="utf-8"))
    ruta_catalogo = Path(datos["rutas"]["catalogo"])
    for _ in range(2):
        resultado = subprocess.run(
            [
                sys.executable,
                "-m",
                "cobro_vision",
                "inicializar-catalogo",
                "--config",
                str(ruta_config_temporal),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert resultado.returncode == 0, resultado.stderr
        assert "Catálogo inicializado" in resultado.stdout
    with sqlite3.connect(ruta_catalogo) as conexion:
        filas = conexion.execute(
            "SELECT id_clase FROM alimentos ORDER BY id_clase"
        ).fetchall()
    assert [fila[0] for fila in filas] == [
        "arroz_amarillo",
        "ensalada",
        "fondo",
        "frijoles",
        "pescado_frito",
        "pollo",
    ]
    clases = (
        ruta_catalogo.with_name("clases.txt").read_text(encoding="utf-8").splitlines()
    )
    assert clases == [fila[0] for fila in filas]
    assert "fondo" in clases


def test_inicializar_con_config_invalida(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Los errores se reportan sin traceback y devuelven código distinto de cero."""
    assert (
        principal(["inicializar-catalogo", "--config", str(tmp_path / "ausente.yaml")])
        == 1
    )
    assert "No se pudo inicializar" in caplog.text


def test_app_crea_controlador_ventana_y_mainloop(
    ruta_config_temporal: Path, controlador_cajero: ControladorTransaccion
) -> None:
    """El comando app ensambla el pipeline y entra al bucle de CustomTkinter."""
    with (
        patch(
            "cobro_vision.cli.FabricaPipeline.crear", return_value=controlador_cajero
        ) as crear,
        patch("cobro_vision.vista.vista_cajero.VistaCajero") as ventana,
    ):
        assert principal(["app", "--config", str(ruta_config_temporal)]) == 0
    crear.assert_called_once()
    assert ventana.call_args.args[0] is controlador_cajero
    ventana.return_value.mainloop.assert_called_once()
    ventana.return_value.destroy.assert_called_once()


def test_app_error_de_arranque_libera_camara(
    ruta_config_temporal: Path,
    controlador_cajero: ControladorTransaccion,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Un error al crear la ventana devuelve uno y libera el dispositivo."""
    with (
        patch(
            "cobro_vision.cli.FabricaPipeline.crear", return_value=controlador_cajero
        ),
        patch(
            "cobro_vision.vista.vista_cajero.VistaCajero",
            side_effect=RuntimeError("Sin pantalla"),
        ),
        patch.object(controlador_cajero.camara, "liberar") as liberar,
    ):
        assert principal(["app", "--config", str(ruta_config_temporal)]) == 1
    liberar.assert_called_once()
    assert "Sin pantalla" in caplog.text
