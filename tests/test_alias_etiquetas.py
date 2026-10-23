"""Normalización de etiquetas sintéticas sin reescribir los JSON originales."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.adquisicion.lector_labelme import LectorLabelMe
from cobro_vision.adquisicion.validador_etiquetas import ValidadorEtiquetas
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import ConfiguracionInvalida
from cobro_vision.negocio.catalogo import CatalogoAlimentos


@pytest.mark.parametrize("catalogo_explicito", [False, True])
def test_lector_aplica_alias_una_vez_y_conserva_original(
    config_dataset: Configuracion,
    catalogo: CatalogoAlimentos,
    catalogo_explicito: bool,
) -> None:
    """Un alias puede contener espacios; el destino no se vuelve a remapear."""
    datos = deepcopy(config_dataset._datos)
    datos["etiquetas"]["alias"] = {"Pollo asado": "pollo", "pollo": "frijoles"}
    ruta = Path(datos["rutas"]["imagenes"]) / "S01/P001_a.json"
    etiquetas = json.loads(ruta.read_text(encoding="utf-8"))
    etiquetas["shapes"][0]["label"] = "Pollo asado"
    etiquetas["shapes"].append({**etiquetas["shapes"][0], "label": "pollo"})
    ruta.write_text(json.dumps(etiquetas), encoding="utf-8")
    original = ruta.read_bytes()
    lector = LectorLabelMe(
        Configuracion(datos), catalogo if catalogo_explicito else None
    )
    etiquetado = lector.leer(ruta)
    assert etiquetado.formas[0].etiqueta == "pollo"
    assert etiquetado.formas[-1].etiqueta == "frijoles"
    assert etiquetado.formas[1].etiqueta == etiquetas["shapes"][1]["label"]
    assert etiquetado.formas[0].puntos == [
        tuple(punto) for punto in etiquetas["shapes"][0]["points"]
    ]
    assert LectorLabelMe().leer(ruta).formas[0].etiqueta == "Pollo asado"
    assert ruta.read_bytes() == original


def test_validador_y_gestor_comparten_alias_y_conteos(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Los alias suman ejemplos canónicos y llegan a las muestras del entrenamiento."""
    datos = deepcopy(config_dataset._datos)
    datos["etiquetas"]["alias"] = {"ave": "pollo"}
    datos["entrenamiento"]["min_instancias_por_clase"] = 12
    carpeta = Path(datos["rutas"]["imagenes"])
    for ruta in carpeta.rglob("*_a.json"):
        etiquetas = json.loads(ruta.read_text(encoding="utf-8"))
        etiquetas["shapes"][0]["label"] = "ave"
        ruta.write_text(json.dumps(etiquetas), encoding="utf-8")
    originales = {ruta: ruta.read_bytes() for ruta in carpeta.rglob("*.json")}
    config = Configuracion(datos)
    gestor = GestorDataset(config, catalogo)
    muestras = gestor.cargar_muestras()
    assert len(muestras) == 12
    assert all(muestra.etiquetado.formas[0].etiqueta == "pollo" for muestra in muestras)
    assert "pollo" in gestor.contenido_esperado(muestras[0])
    validador = ValidadorEtiquetas(config, catalogo)
    informe = validador.validar(muestras)
    assert informe.ok
    assert not any("Clase pollo:" in aviso for aviso in informe.advertencias)
    assert all(ruta.read_bytes() == original for ruta, original in originales.items())
    ruta = carpeta / "S01/P001_a.json"
    etiquetas = json.loads(ruta.read_text(encoding="utf-8"))
    etiquetas["shapes"][0]["points"] = [[1, 1], [2, 2]]
    etiquetas["shapes"][1]["label"] = "desconocida"
    ruta.write_text(json.dumps(etiquetas), encoding="utf-8")
    errores = "\n".join(validador.validar([]).errores)
    assert "menos de 3 puntos" in errores
    assert "fuera del catálogo: 'desconocida'" in errores
    assert "fuera del catálogo: 'ave'" not in errores


@pytest.mark.parametrize(
    "componente", [LectorLabelMe, ValidadorEtiquetas, GestorDataset]
)
@pytest.mark.parametrize("destino", ["inexistente", "pollo"])
def test_destino_desconocido_o_inactivo_es_configuracion_invalida(
    config_dataset: Configuracion,
    catalogo: CatalogoAlimentos,
    componente: type,
    destino: str,
) -> None:
    """Se validan también alias no usados en ningún JSON y clases desactivadas."""
    if destino == "pollo":
        catalogo.agregar(replace(catalogo.obtener(destino), activo=False))
    datos = deepcopy(config_dataset._datos)
    datos["etiquetas"]["alias"] = {"alias_no_usado": destino}
    with pytest.raises(ConfiguracionInvalida, match=f"etiquetas.alias.*{destino}"):
        componente(Configuracion(datos), catalogo)
