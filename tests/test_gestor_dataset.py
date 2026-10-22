"""Manifiestos estables y separación por plato con varias fotos y sesiones."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def test_manifiesto_reproducible(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Cada foto tiene identidad, ruta relativa y fecha; repetir no cambia el CSV."""
    gestor = GestorDataset(config_dataset, catalogo)
    primero = gestor.construir_manifiesto()
    ruta = Path(config_dataset.seccion("rutas")["manifiesto"])
    contenido = ruta.read_bytes()
    segundo = gestor.construir_manifiesto()
    pd.testing.assert_frame_equal(primero, segundo)
    assert ruta.read_bytes() == contenido
    assert len(primero) == 12
    assert primero["split"].eq("").all()
    fila = primero.iloc[0]
    assert fila["id_foto"] == "S01/P001_a"
    assert fila["ruta_imagen"] == "S01/P001_a.png"
    assert fila["plato_id"] == "P001" and fila["sesion"] == "S01"
    assert len(fila["fecha"]) == 10


def test_division_por_plato_determinista(
    config_dataset: Configuracion,
    catalogo: CatalogoAlimentos,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Las dos tomas de cada plato comparten split y la semilla reproduce el CSV."""
    gestor = GestorDataset(config_dataset, catalogo)
    gestor.dividir(0.2, config_dataset.semilla)
    ruta = Path(config_dataset.seccion("rutas")["manifiesto"])
    contenido = ruta.read_bytes()
    train, test = gestor.cargar_muestras("train"), gestor.cargar_muestras("test")
    platos_train = {muestra.plato_id for muestra in train}
    platos_test = {muestra.plato_id for muestra in test}
    assert platos_train.isdisjoint(platos_test)
    assert len(platos_train) == 4 and len(platos_test) == 2
    assert len(train) + len(test) == 12
    assert {muestra.split for muestra in train} == {"train"}
    assert {muestra.split for muestra in test} == {"test"}
    gestor.dividir(0.2, config_dataset.semilla)
    assert ruta.read_bytes() == contenido
    assert "Clase pescado_frito sin ejemplos en train" in caplog.text
    assert "Clase pescado_frito sin ejemplos en test" in caplog.text


def test_agregar_foto_en_otra_sesion_conserva_split_y_override(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Un plato conserva su split incluso cuando aparece de nuevo en otra sesión."""
    gestor = GestorDataset(config_dataset, catalogo)
    gestor.dividir(0.2, config_dataset.semilla)
    rutas = config_dataset.seccion("rutas")
    manifiesto = Path(rutas["manifiesto"])
    anterior = pd.read_csv(manifiesto, dtype=str, keep_default_na=False)
    anterior.loc[0, "contenido_esperado_override"] = "pollo;pescado_frito"
    anterior.to_csv(manifiesto, index=False)
    carpeta = Path(rutas["imagenes"])
    nueva = carpeta / "S02" / "P001_c.png"
    nueva.parent.mkdir()
    nueva.write_bytes((carpeta / "S01/P001_a.png").read_bytes())
    datos = json.loads((carpeta / "S01/P001_a.json").read_text(encoding="utf-8"))
    datos["imagePath"] = nueva.name
    nueva.with_suffix(".json").write_text(json.dumps(datos), encoding="utf-8")
    actualizado = gestor.construir_manifiesto()
    assert len(actualizado) == 13
    assert actualizado.groupby("plato_id")["split"].nunique().eq(1).all()
    pd.testing.assert_series_equal(
        actualizado.set_index("id_foto").loc[anterior["id_foto"], "split"],
        anterior.set_index("id_foto")["split"],
    )
    assert actualizado.iloc[0]["contenido_esperado_override"] == "pollo;pescado_frito"
    assert len(gestor.cargar_muestras()) == 13
    gestor.dividir(0.2, config_dataset.semilla)
    train = {muestra.plato_id for muestra in gestor.cargar_muestras("train")}
    test = {muestra.plato_id for muestra in gestor.cargar_muestras("test")}
    assert train.isdisjoint(test)


def test_rechaza_fuga_antes_de_filtrar_o_escribir(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Un CSV contradictorio nunca se acepta como entrada, ni filtrando solo train."""
    gestor = GestorDataset(config_dataset, catalogo)
    datos = gestor.construir_manifiesto()
    datos.loc[0, "split"] = "train"
    datos.loc[1, "split"] = "test"
    ruta = Path(config_dataset.seccion("rutas")["manifiesto"])
    datos.to_csv(ruta, index=False)
    contenido = ruta.read_bytes()
    for operacion in (
        gestor.construir_manifiesto,
        lambda: gestor.cargar_muestras("train"),
        lambda: gestor.cargar_muestras("test"),
        lambda: gestor.dividir(0.2, config_dataset.semilla),
    ):
        with pytest.raises(EtiquetaInvalida, match="Fuga de datos.*P001"):
            operacion()
        assert ruta.read_bytes() == contenido


def test_contenido_esperado_visible_y_override(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """El override sustituye el conjunto visible y fondo nunca es contenido cobrable."""
    gestor = GestorDataset(config_dataset, catalogo)
    muestra = gestor.cargar_muestras()[0]
    assert gestor.contenido_esperado(muestra) == {
        "pollo",
        "ensalada",
        "arroz_amarillo",
        "frijoles",
    }
    ruta = Path(config_dataset.seccion("rutas")["manifiesto"])
    datos = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    datos.loc[0, "contenido_esperado_override"] = "pollo; pescado_frito ;fondo;pollo"
    datos.to_csv(ruta, index=False)
    assert gestor.contenido_esperado(muestra) == {"pollo", "pescado_frito"}
    datos.loc[0, "contenido_esperado_override"] = "inexistente"
    datos.to_csv(ruta, index=False)
    with pytest.raises(EtiquetaInvalida, match="fuera del catálogo"):
        gestor.contenido_esperado(muestra)


def test_imagen_sin_json_se_registra_pero_no_se_carga(
    config_dataset: Configuracion,
    catalogo: CatalogoAlimentos,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Conserva la foto en el manifiesto e informa por qué no aporta una muestra."""
    carpeta = Path(config_dataset.seccion("rutas")["imagenes"])
    (carpeta / "S01/P001_b.json").unlink()
    gestor = GestorDataset(config_dataset, catalogo)
    assert len(gestor.construir_manifiesto()) == 12
    assert len(gestor.cargar_muestras()) == 11
    assert "Imagen sin JSON" in caplog.text and "P001_b.png" in caplog.text


@pytest.mark.parametrize("fraccion", [0, 1, -0.1, 1.1, float("nan"), float("inf")])
def test_rechaza_fraccion_invalida(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos, fraccion: float
) -> None:
    """No permite divisiones vacías ni fracciones no finitas."""
    with pytest.raises(ValueError, match="fracción"):
        GestorDataset(config_dataset, catalogo).dividir(
            fraccion, config_dataset.semilla
        )


def test_rechaza_identidad_manipulada_y_split_invalido(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Los ids se derivan del archivo y no pueden alterarse para eludir la agrupación."""
    gestor = GestorDataset(config_dataset, catalogo)
    datos = gestor.construir_manifiesto()
    ruta = Path(config_dataset.seccion("rutas")["manifiesto"])
    datos.loc[0, "plato_id"] = "otro"
    datos.to_csv(ruta, index=False)
    with pytest.raises(EtiquetaInvalida, match="identidad inconsistente"):
        gestor.cargar_muestras()
    datos.loc[0, "plato_id"] = "P001"
    datos.loc[0, "split"] = "NA"
    datos.to_csv(ruta, index=False)
    with pytest.raises(EtiquetaInvalida, match="split inválido"):
        gestor.cargar_muestras()


def test_rechaza_dos_formatos_de_la_misma_foto(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """Dos extensiones con igual sesión y nombre no pueden compartir id_foto."""
    carpeta = Path(config_dataset.seccion("rutas")["imagenes"])
    (carpeta / "S01/P001_a.jpg").write_bytes((carpeta / "S01/P001_a.png").read_bytes())
    with pytest.raises(EtiquetaInvalida, match="id_foto duplicado"):
        GestorDataset(config_dataset, catalogo).construir_manifiesto()


def test_rechaza_imagepath_de_otra_foto(
    config_dataset: Configuracion, catalogo: CatalogoAlimentos
) -> None:
    """El JSON adyacente debe describir su propia foto."""
    ruta = Path(config_dataset.seccion("rutas")["imagenes"]) / "S01/P001_a.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["imagePath"] = "P001_b.png"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(EtiquetaInvalida, match="no corresponde"):
        GestorDataset(config_dataset, catalogo).cargar_muestras()
