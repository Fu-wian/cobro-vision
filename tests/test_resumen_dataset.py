"""Cobertura por clase con fotos, platos y anotaciones generados por código."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.adquisicion.resumen_dataset import resumir_dataset
from cobro_vision.cli import principal
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def _escribir_foto(ruta: Path, etiquetas: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(ruta), np.zeros((16, 16, 3), dtype=np.uint8))
    formas = [
        {"label": etiqueta, "shape_type": "polygon", "points": [[1, 1], [5, 1], [1, 5]]}
        for etiqueta in etiquetas
    ]
    if "fondo" in etiquetas:
        formas[-1].update(shape_type="rectangle", points=[[8, 8], [14, 14]])
    ruta.with_suffix(".json").write_text(
        json.dumps(
            {
                "imagePath": ruta.name,
                "imageWidth": 16,
                "imageHeight": 16,
                "shapes": formas,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def dataset_resumen(
    tmp_path: Path, datos_config: dict, catalogo: CatalogoAlimentos
) -> tuple[Configuracion, Path]:
    """Crea 25 platos; dos tomas por plato y otra sesión del mismo P001."""
    carpeta = tmp_path / "raw"
    for plato in range(1, 26):
        etiquetas = ["pollo", "ave"]
        if plato < 25:
            etiquetas.append("arroz_amarillo")
        if plato <= 3:
            etiquetas.append("ensalada")
        if plato <= 2:
            etiquetas.append("frijoles")
        if plato == 1:
            etiquetas.append("fondo")
        for toma in ("a", "b"):
            _escribir_foto(carpeta / "S01" / f"P{plato:03d}_{toma}.png", etiquetas)
    _escribir_foto(carpeta / "S02/P001_c.png", ["pollo", "ave"])
    datos_config["rutas"].update(
        imagenes=str(carpeta),
        manifiesto=str(tmp_path / "manifest.csv"),
        catalogo=str(catalogo.ruta_db),
    )
    datos_config["etiquetas"]["alias"] = {"ave": "pollo"}
    ruta_config = tmp_path / "config.yaml"
    ruta_config.write_text(yaml.safe_dump(datos_config), encoding="utf-8")
    config = Configuracion.cargar(ruta_config)
    manifiesto = GestorDataset(config, catalogo).construir_manifiesto()
    manifiesto["split"] = manifiesto["plato_id"].map(
        lambda plato: "train" if plato <= "P003" else "test" if plato < "P025" else ""
    )
    manifiesto.loc[0, "contenido_esperado_override"] = "pescado_frito"
    manifiesto.to_csv(datos_config["rutas"]["manifiesto"], index=False)
    # Esta anotación no pertenece al manifiesto y no debe afectar la cobertura.
    _escribir_foto(carpeta / "S01/P026_a.png", ["pescado_frito"])
    # P025_b sí está en el CSV, pero se ignora con advertencia al faltarle el JSON.
    (carpeta / "S01/P025_b.json").unlink()
    return config, ruta_config


def test_resumen_cuenta_platos_fotos_poligonos_y_limites(
    dataset_resumen: tuple[Configuracion, Path],
    catalogo: CatalogoAlimentos,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """25 platos y 3 en train no se marcan; repetición, alias y overrides no inflan."""
    config, _ = dataset_resumen
    rutas = config.seccion("rutas")
    archivos = [Path(rutas["manifiesto"]), *Path(rutas["imagenes"]).rglob("*.json")]
    originales = {ruta: ruta.read_bytes() for ruta in archivos}
    tabla = resumir_dataset(config, catalogo)
    assert tabla["id_clase"].tolist() == catalogo.clases_validas()
    assert tabla.set_index("id_clase").to_dict("index") == {
        "arroz_amarillo": {
            "platos_total": 24,
            "platos_train": 3,
            "platos_test": 21,
            "fotos": 48,
            "poligonos": 48,
            "estado": "POCO",
        },
        "ensalada": {
            "platos_total": 3,
            "platos_train": 3,
            "platos_test": 0,
            "fotos": 6,
            "poligonos": 6,
            "estado": "POCO",
        },
        "fondo": {
            "platos_total": 1,
            "platos_train": 1,
            "platos_test": 0,
            "fotos": 2,
            "poligonos": 2,
            "estado": "FALTA POCO",
        },
        "frijoles": {
            "platos_total": 2,
            "platos_train": 2,
            "platos_test": 0,
            "fotos": 4,
            "poligonos": 4,
            "estado": "FALTA POCO",
        },
        "pescado_frito": {
            "platos_total": 0,
            "platos_train": 0,
            "platos_test": 0,
            "fotos": 0,
            "poligonos": 0,
            "estado": "FALTA POCO",
        },
        "pollo": {
            "platos_total": 25,
            "platos_train": 3,
            "platos_test": 21,
            "fotos": 50,
            "poligonos": 100,
            "estado": "",
        },
    }
    assert "Imagen sin JSON" in caplog.text and "P025_b" in caplog.text
    assert all(ruta.read_bytes() == contenido for ruta, contenido in originales.items())


def test_resumen_sin_manifiesto_y_con_manifiesto_vacio(
    dataset_resumen: tuple[Configuracion, Path], catalogo: CatalogoAlimentos
) -> None:
    """No crea el manifiesto implícitamente; uno vacío mantiene clases con cero."""
    config, _ = dataset_resumen
    ruta = Path(config.seccion("rutas")["manifiesto"])
    columnas = pd.read_csv(ruta).columns
    ruta.unlink()
    with pytest.raises(FileNotFoundError, match="construir-manifiesto"):
        resumir_dataset(config, catalogo)
    assert not ruta.exists()
    pd.DataFrame(columns=columnas).to_csv(ruta, index=False)
    tabla = resumir_dataset(config, catalogo)
    assert len(tabla) == len(catalogo.clases_validas())
    assert tabla["platos_total"].eq(0).all()
    assert tabla["estado"].eq("FALTA POCO").all()


@pytest.mark.parametrize("problema", ["fuga", "desconocida", "json"])
def test_resumen_rechaza_dataset_invalido(
    dataset_resumen: tuple[Configuracion, Path],
    catalogo: CatalogoAlimentos,
    problema: str,
) -> None:
    """No presenta conteos aparentemente válidos si hay fuga o etiquetas ilegibles."""
    config, _ = dataset_resumen
    rutas = config.seccion("rutas")
    if problema == "fuga":
        manifiesto = pd.read_csv(rutas["manifiesto"], keep_default_na=False)
        manifiesto.loc[0, "split"] = "test"
        manifiesto.to_csv(rutas["manifiesto"], index=False)
        mensaje = "Fuga de datos"
    else:
        ruta = Path(rutas["imagenes"]) / "S01/P001_a.json"
        if problema == "json":
            ruta.write_text("{", encoding="utf-8")
            mensaje = "no se pudo leer el JSON"
        else:
            etiquetas = json.loads(ruta.read_text(encoding="utf-8"))
            etiquetas["shapes"][0]["label"] = "desconocida"
            ruta.write_text(json.dumps(etiquetas), encoding="utf-8")
            mensaje = "fuera del catálogo"
    with pytest.raises(EtiquetaInvalida, match=mensaje):
        resumir_dataset(config, catalogo)


def test_cli_resumen_desde_modulo_y_csv(
    dataset_resumen: tuple[Configuracion, Path],
    catalogo: CatalogoAlimentos,
    tmp_path: Path,
) -> None:
    """El comando real muestra la misma tabla que exporta, con rutas con espacios."""
    config, ruta_config = dataset_resumen
    destino = tmp_path / "salida con espacios" / "resumen.csv"
    resultado = subprocess.run(
        [
            sys.executable,
            "-m",
            "cobro_vision",
            "resumen-dataset",
            "--config",
            str(ruta_config),
            "--csv",
            str(destino),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr
    tabla = resumir_dataset(config, catalogo)
    assert resultado.stdout.strip() == tabla.to_string(index=False).strip()
    pd.testing.assert_frame_equal(pd.read_csv(destino, keep_default_na=False), tabla)


def test_cli_resumen_solo_pantalla_y_errores(
    dataset_resumen: tuple[Configuracion, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Acepta --config antes del comando y reporta fallos con código uno."""
    config, ruta_config = dataset_resumen
    assert principal(["--config", str(ruta_config), "resumen-dataset"]) == 0
    assert "FALTA POCO" in capsys.readouterr().out
    assert not list(tmp_path.rglob("resumen.csv"))
    datos = config._datos.copy()
    datos["etiquetas"] = {"alias": {"ave": "inexistente"}}
    ruta_config.write_text(yaml.safe_dump(datos), encoding="utf-8")
    assert principal(["resumen-dataset", "--config", str(ruta_config)]) == 1
    assert "etiquetas.alias" in caplog.text
    ruta_config.write_text(yaml.safe_dump(config._datos), encoding="utf-8")
    Path(config.seccion("rutas")["manifiesto"]).unlink()
    assert principal(["resumen-dataset", "--config", str(ruta_config)]) == 1
    assert "construir-manifiesto" in caplog.text
