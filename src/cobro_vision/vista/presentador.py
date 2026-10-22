"""Transformaciones de presentación comprobables sin pantalla ni Tk."""

from dataclasses import dataclass

from cobro_vision.dominio.modelos import Recibo, Transaccion
from cobro_vision.dominio.utilidades import formatear_dinero
from cobro_vision.explicacion.trazabilidad import Trazabilidad
from cobro_vision.negocio.catalogo import CatalogoAlimentos


@dataclass
class FilaRecibo:
    """Contenido de una fila del recibo mostrado al cajero."""

    id_clase: str
    nombre: str
    precio_texto: str
    estado: str
    origen: str


def filas_recibo(recibo: Recibo, moneda: str) -> list[FilaRecibo]:
    """Presenta el importe cobrado por la política, con su marca de revisión."""
    return [
        FilaRecibo(
            item.alimento.id_clase,
            item.alimento.nombre,
            formatear_dinero(item.precio_cobrado_centavos, moneda),
            "revisar" if item.requiere_revision else "ok",
            item.origen,
        )
        for item in recibo.items
    ]


def texto_total(recibo: Recibo | None, moneda: str) -> str:
    """Formatea el total exacto o cero cuando aún no hay recibo."""
    return formatear_dinero(recibo.total_centavos if recibo else 0, moneda)


def puede_editar(t: Transaccion | None) -> bool:
    """Permite cambios solamente en transacciones procesadas sin confirmar."""
    return (
        t is not None
        and t.estado in {"procesada", "requiere_revision"}
        and (t.recibo is not None)
    )


def puede_confirmar(t: Transaccion, exigir_revision: bool) -> bool:
    """Bloquea errores, recibos vacíos, confirmaciones repetidas y revisiones."""
    return bool(
        puede_editar(t)
        and t.recibo.items
        and not (exigir_revision and t.recibo.requiere_revision)
    )


def opciones_correccion(
    catalogo: CatalogoAlimentos, t: Transaccion, id_clase: str
) -> list[str]:
    """Prioriza alternativas activas cobrables y conserva el orden del catálogo."""
    clases = [a.id_clase for a in catalogo.listar(solo_cobrables=True)]
    item = (
        next((i for i in t.recibo.items if i.alimento.id_clase == id_clase), None)
        if t.recibo
        else None
    )
    sugerencias = []
    for region in t.regiones:
        if item and region.id in item.ids_region and region.resultado:
            alternativa = region.resultado.alternativa
            if alternativa in clases and alternativa not in sugerencias:
                sugerencias.append(alternativa)
    # TODO(equipo): se ofrece también la misma clase para aceptar un ítem en revisión.
    return sugerencias + [clase for clase in clases if clase not in sugerencias]


def mensajes_alerta(t: Transaccion) -> list[str]:
    """Resume errores y acciones pendientes en lenguaje del cajero."""
    if t.estado == "error":
        return [t.mensaje_error or "No se pudo procesar la foto."]
    if t.estado == "confirmada":
        return ["Transacción confirmada y guardada."]
    if t.recibo is None:
        return ["Abra una foto o capture un plato."]
    if not t.recibo.items:
        return ["No hay alimentos en el recibo. Agregue los que correspondan."]
    if t.recibo.requiere_revision:
        return [
            "Revise los ítems marcados; puede elegir su misma clase para aceptarlos."
        ]
    return []


def texto_tiempos(t: Transaccion) -> str:
    """Muestra las duraciones disponibles en milisegundos."""
    return "\n".join(
        f"{etapa}: {duracion:.2f} ms" for etapa, duracion in t.tiempos.items()
    )


def clases_visibles(t: Transaccion) -> list[str]:
    """Enumera las clases de las regiones que se dibujan en la foto."""
    return sorted(
        {
            r.resultado.id_clase
            for r in t.regiones
            if r.resultado is not None and r.estado != "eliminada"
        }
    )


def etiquetas_alimentos(catalogo: CatalogoAlimentos) -> dict[str, str]:
    """Identifica cada opción por nombre e id, incluso si hay nombres repetidos."""
    return {
        a.id_clase: f"{a.nombre} ({a.id_clase})"
        for a in catalogo.listar(solo_cobrables=True)
    }


def texto_explicacion(t: Transaccion, trazabilidad: Trazabilidad) -> str:
    """Reúne las explicaciones de cada región y de cada ítem del recibo."""
    explicaciones = [
        trazabilidad.explicar_how(t, r.id)
        for r in t.regiones
        if r.resultado is not None and r.estado != "eliminada"
    ]
    if t.recibo:
        explicaciones.extend(
            trazabilidad.explicar_why(t, i.alimento.id_clase) for i in t.recibo.items
        )
    return "\n\n".join(explicaciones) or "Todavía no hay resultados para explicar."
