"""Reglas espaciales, área y revisión del posprocesador."""

from copy import deepcopy

import numpy as np
import pytest

from cobro_vision.dominio.modelos import Region, ResultadoClasificacion
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.negocio.posprocesador import Posprocesador
from cobro_vision.segmentacion.base import _crear_region


def regiones_de_mapa(
    mapa: np.ndarray, clases: dict[int, str], margenes: dict[int, float] | None = None
) -> list[Region]:
    """Convierte etiquetas artificiales en predicciones sin entrenar."""
    regiones = []
    for indice, clase in clases.items():
        region = _crear_region(indice, mapa == indice)
        region.resultado = ResultadoClasificacion(
            clase, (margenes or {}).get(indice, 1.0), False
        )
        regiones.append(region)
    return regiones


def crear_posprocesador(
    catalogo: CatalogoAlimentos, **parametros: object
) -> Posprocesador:
    """Desactiva filtros salvo los que la prueba requiere explícitamente."""
    config = {
        "componente_minimo_frac": 0,
        "clase_minima_frac": 0,
        "max_proteinas": None,
    }
    config.update(parametros)
    return Posprocesador(config, catalogo, 0.5)


def test_fusion_vecindad_cuatro_score_ponderado_y_sin_mutar(
    catalogo: CatalogoAlimentos,
) -> None:
    """Vecinos de igual clase se unen; contacto diagonal no une componentes."""
    mapa = np.array([[1, 2, 2, 0], [0, 0, 0, 3]])
    regiones = regiones_de_mapa(
        mapa, {1: "pollo", 2: "pollo", 3: "pollo"}, {1: 0.1, 2: 0.9, 3: 0.2}
    )
    originales = deepcopy(regiones)
    posprocesador = crear_posprocesador(catalogo)
    salida = posprocesador.consolidar(regiones, np.ones_like(mapa, dtype=bool))
    assert [r.id for r in salida] == [1, 2]
    assert [r.area for r in salida] == [3, 1]
    assert salida[0].resultado.score == pytest.approx((0.1 + 2 * 0.9) / 3)
    assert salida[0].estado == "aceptada" and salida[1].estado == "revisar"
    assert salida[1].resultado.requiere_revision
    assert posprocesador.origenes == {1: [1, 2], 2: [3]}
    assert salida[0].bbox == (0, 0, 3, 1)
    for actual, original in zip(regiones, originales):
        np.testing.assert_array_equal(actual.mascara, original.mascara)
        assert actual.resultado == original.resultado and actual.estado == "pendiente"
    salida[0].mascara[:] = False
    assert regiones[0].mascara.any()


def test_absorcion_por_frontera_mas_larga_y_nueva_fusion(
    catalogo: CatalogoAlimentos,
) -> None:
    """Dos píxeles se absorben en ensalada (cuatro bordes frente a dos de arroz)."""
    mapa = np.ones((10, 10), dtype=int)
    mapa[:, 5:] = 2
    mapa[2:4, 5] = 3
    regiones = regiones_de_mapa(
        mapa, {1: "arroz_amarillo", 2: "ensalada", 3: "pollo"}, {2: 0.8, 3: 0.1}
    )
    posprocesador = crear_posprocesador(catalogo, componente_minimo_frac=0.03)
    salida = posprocesador.consolidar(regiones, np.ones_like(mapa, dtype=bool))
    assert {r.resultado.id_clase: r.area for r in salida} == {
        "arroz_amarillo": 50,
        "ensalada": 50,
    }
    ensalada = next(r for r in salida if r.resultado.id_clase == "ensalada")
    assert ensalada.resultado.score == pytest.approx((48 * 0.8 + 2 * 0.1) / 50)
    assert posprocesador.origenes[ensalada.id] == [2, 3]
    assert posprocesador.eventos[0]["accion"] == "absorcion"
    assert posprocesador.eventos[0]["destino"] == "ensalada"
    assert posprocesador.eventos[0]["frontera"] == 4


@pytest.mark.parametrize("con_fondo", [False, True])
def test_componente_pequena_aislada_o_solo_junto_a_fondo_se_elimina(
    catalogo: CatalogoAlimentos, con_fondo: bool
) -> None:
    """El fondo no absorbe alimentos y nunca aparece como región cobrable."""
    mapa = np.full((5, 5), 2 if con_fondo else 0)
    mapa[2, 2] = 1
    clases = {1: "pollo", **({2: "fondo"} if con_fondo else {})}
    posprocesador = crear_posprocesador(catalogo, componente_minimo_frac=0.1)
    assert (
        posprocesador.consolidar(
            regiones_de_mapa(mapa, clases), np.ones_like(mapa, dtype=bool)
        )
        == []
    )
    assert posprocesador.eventos[0]["accion"] == "eliminar_componente"


@pytest.mark.parametrize(("minimo", "requiere_revision"), [(0.2, False), (0.21, True)])
def test_area_minima_suma_componentes_de_clase(
    catalogo: CatalogoAlimentos, minimo: float, requiere_revision: bool
) -> None:
    """Conserva las regiones y revisa la clase si no alcanza el area minima."""
    mapa = np.zeros((10, 10), dtype=int)
    mapa[0] = 1
    mapa[9] = 2
    posprocesador = crear_posprocesador(catalogo, clase_minima_frac=minimo)
    salida = posprocesador.consolidar(
        regiones_de_mapa(mapa, {1: "pollo", 2: "pollo"}), np.ones_like(mapa, dtype=bool)
    )

    assert len(salida) == 2
    assert sum(region.area for region in salida) == 20
    assert all(
        region.estado == ("revisar" if requiere_revision else "aceptada")
        for region in salida
    )
    assert all(
        region.resultado.requiere_revision == requiere_revision
        for region in salida
    )

    eventos_area = [
        evento for evento in posprocesador.eventos
        if evento["accion"] == "revisar_clase_area"
    ]
    assert len(eventos_area) == int(requiere_revision)
    if requiere_revision:
        assert eventos_area[0]["clase"] == "pollo"
        assert eventos_area[0]["area"] == 20


@pytest.mark.parametrize(
    ("limite", "revisadas"),
    [(None, set()), (1, {"pescado_frito"}), (0, {"pollo", "pescado_frito"})],
)
def test_regla_proteinas_conserva_todas_y_revisa_exceso(
    catalogo: CatalogoAlimentos, limite: int | None, revisadas: set[str]
) -> None:
    """La proteína con mayor área total tiene prioridad; no desaparecen ítems."""
    mapa = np.array([[1, 1, 0, 3], [2, 2, 0, 3], [0, 0, 0, 3]])
    posprocesador = crear_posprocesador(catalogo, max_proteinas=limite)
    salida = posprocesador.consolidar(
        regiones_de_mapa(mapa, {1: "pollo", 2: "pollo", 3: "pescado_frito"}),
        np.ones_like(mapa, dtype=bool),
    )
    assert {r.resultado.id_clase for r in salida} == {"pollo", "pescado_frito"}
    assert {r.resultado.id_clase for r in salida if r.estado == "revisar"} == revisadas
    assert all(r.resultado.requiere_revision == (r.estado == "revisar") for r in salida)


def test_recorte_al_plato_y_vacio_reinician_origenes(
    catalogo: CatalogoAlimentos,
) -> None:
    """La consolidación ignora fuera del plato y no conserva eventos de otra foto."""
    mapa = np.ones((4, 4), dtype=int)
    plato = np.zeros((4, 4), dtype=bool)
    plato[1:3, 1:3] = True
    posprocesador = crear_posprocesador(catalogo)
    salida = posprocesador.consolidar(regiones_de_mapa(mapa, {1: "pollo"}), plato)
    assert salida[0].area == 4 and salida[0].bbox == (1, 1, 3, 3)
    assert posprocesador.consolidar([], plato) == []
    assert posprocesador.eventos == [] and posprocesador.origenes == {}


def test_absorcion_suma_fronteras_de_componentes_de_la_misma_clase(
    catalogo: CatalogoAlimentos,
) -> None:
    """Dos vecinos de arroz suman más frontera que el único vecino de ensalada."""
    mapa = np.ones((6, 7), dtype=int)
    mapa[1:3, 1:3] = 2
    mapa[2:4, 4:6] = 2
    mapa[1:4, 3] = 3
    mapa[1, 4] = 0
    # Dos componentes de arroz: 2 + 2 bordes; ensalada: 3 bordes.
    posprocesador = crear_posprocesador(catalogo, componente_minimo_frac=0.095)
    salida = posprocesador.consolidar(
        regiones_de_mapa(mapa, {1: "ensalada", 2: "arroz_amarillo", 3: "pollo"}),
        np.ones_like(mapa, dtype=bool),
    )
    assert {r.resultado.id_clase: r.area for r in salida} == {
        "ensalada": 30,
        "arroz_amarillo": 11,
    }
    assert posprocesador.eventos[0]["destino"] == "arroz_amarillo"
    assert posprocesador.eventos[0]["frontera"] == 4


def test_regla_proteinas_tiene_prioridad_sobre_score_alto(
    catalogo: CatalogoAlimentos,
) -> None:
    """La proteína principal de confianza baja y la excedente alta se revisan ambas."""
    mapa = np.array([[1, 1, 1, 2, 2]])
    posprocesador = crear_posprocesador(catalogo, max_proteinas=1)
    salida = posprocesador.consolidar(
        regiones_de_mapa(mapa, {1: "pollo", 2: "pescado_frito"}, {1: 0.2, 2: 10.0}),
        np.ones_like(mapa, dtype=bool),
    )
    assert len(salida) == 2 and all(r.estado == "revisar" for r in salida)


@pytest.mark.parametrize(
    ("modo", "esperado"),
    [
        ("original", (48 * 0.8 + 2 * 0.1) / 50),
        ("evidencia_propia", 0.8),
        ("penalizada", 48 * 0.8 / 50),
    ],
)
def test_e14_confianza_absorcion(
    catalogo: CatalogoAlimentos,
    modo: str,
    esperado: float,
) -> None:

    mapa = np.ones((10, 10), dtype=int)

    mapa[:, 5:] = 2
    mapa[2:4, 5] = 3

    regiones = regiones_de_mapa(
        mapa,
        {
            1: "arroz_amarillo",
            2: "ensalada",
            3: "pollo",
        },
        {2: 0.8, 3: 0.1},
    )

    posprocesador = crear_posprocesador(
        catalogo,
        componente_minimo_frac=0.03,
        confianza_absorcion=modo,
    )

    salida = posprocesador.consolidar(
        regiones,
        np.ones_like(mapa, dtype=bool),
    )

    ensalada = next(
        r for r in salida
        if r.resultado.id_clase == "ensalada"
    )

    assert ensalada.area == 50

    assert ensalada.resultado.score == pytest.approx(
        esperado
    )

    assert len(salida) == 2

    assert regiones[1].resultado.score == 0.8
    assert regiones[2].resultado.score == 0.1

    eventos = [
        e for e in posprocesador.eventos
        if e["accion"] == "ajustar_confianza_absorcion"
    ]

    if modo == "original":
        assert not eventos
    else:
        assert len(eventos) == 1
        assert eventos[0]["pixeles_propios"] == 48
        assert eventos[0]["pixeles_absorbidos"] == 2


@pytest.mark.parametrize(
    "modo",
    ["original", "evidencia_propia", "penalizada"],
)
def test_e14_confianza_sin_absorcion(
    catalogo: CatalogoAlimentos,
    modo: str,
) -> None:

    mapa = np.ones((5, 5), dtype=int)

    regiones = regiones_de_mapa(
        mapa,
        {1: "pollo"},
        {1: 0.75},
    )

    posprocesador = crear_posprocesador(
        catalogo,
        confianza_absorcion=modo,
    )

    salida = posprocesador.consolidar(
        regiones,
        np.ones_like(mapa, dtype=bool),
    )

    assert len(salida) == 1
    assert salida[0].area == 25

    assert salida[0].resultado.score == pytest.approx(
        0.75
    )

    assert not posprocesador.eventos


@pytest.mark.parametrize(
    ("modo", "excluidas"),
    [
        ("desactivado", set()),
        ("d7", {"carne_guisada", "chuleta"}),
        ("d1_d7", {"carne_guisada", "chuleta"}),
    ],
)
def test_e14_filtro_evidencia(
    catalogo: CatalogoAlimentos,
    modo: str,
    excluidas: set,
    monkeypatch,
) -> None:

    from types import SimpleNamespace

    categorias = {
        "arroz_blanco": "acompanamiento",
        "pollo": "proteina",
        "carne_guisada": "proteina",
        "chuleta": "proteina",
        "ensalada": "acompanamiento",
        "fondo": "fondo",
    }

    monkeypatch.setattr(
        type(catalogo),
        "obtener",
        lambda self, clase: SimpleNamespace(
            categoria=categorias[clase]
        )
    )


    mapa = np.ones((100, 100), dtype=int)

    mapa[60:80, :] = 2
    mapa[80:83, :] = 3
    mapa[83:87, :] = 4
    mapa[87:90, :] = 5
    mapa[90:100, :] = 6

    regiones = regiones_de_mapa(
        mapa,
        {
            1: "arroz_blanco",
            2: "pollo",
            3: "carne_guisada",
            4: "chuleta",
            5: "ensalada",
            6: "fondo",
        },
        {
            1: 0.90,
            2: 0.85,
            3: 0.60,
            4: 0.68,
            5: 0.85,
            6: 0.95,
        }
    )

    posprocesador = crear_posprocesador(
        catalogo,
        filtro_evidencia=modo,
        confianza_absorcion="evidencia_propia",
        componente_minimo_frac=0.0,
        clase_minima_frac=0.0,
        max_proteinas=None,
    )

    salida = posprocesador.consolidar(
        regiones,
        np.ones_like(mapa, dtype=bool),
    )

    eliminadas = {
        r.resultado.id_clase
        for r in salida
        if r.estado == "eliminada"
    }

    assert eliminadas == excluidas

    activas = {
        r.resultado.id_clase
        for r in salida
        if r.estado != "eliminada"
    }

    assert activas == (
        {
            "arroz_blanco",
            "pollo",
            "carne_guisada",
            "chuleta",
            "ensalada",
        } - excluidas
    )

    eventos = {
        e["clase"]
        for e in posprocesador.eventos
        if e["accion"] == "excluir_evidencia"
    }

    assert eventos == excluidas

    assert all(
        r.estado != "eliminada"
        for r in regiones
    )
