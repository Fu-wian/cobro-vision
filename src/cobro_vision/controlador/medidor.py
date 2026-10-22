"""Medición acumulativa de las etapas del pipeline."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


class MedidorTiempos:
    """Acumula milisegundos por etapa, incluso cuando una etapa falla."""

    def __init__(self) -> None:
        self._tiempos: dict[str, float] = {}

    @contextmanager
    def medir(self, etapa: str) -> Iterator[None]:
        """Mide un bloque y añade su duración a la etapa indicada."""
        inicio = perf_counter()
        try:
            yield
        finally:
            self._tiempos[etapa] = (
                self._tiempos.get(etapa, 0.0) + (perf_counter() - inicio) * 1000
            )

    def resumen(self) -> dict[str, float]:
        """Devuelve una copia y el total de las siete etapas secuenciales."""
        return {**self._tiempos, "total": sum(self._tiempos.values())}

    def reiniciar(self) -> None:
        """Borra las mediciones antes de iniciar otra transacción."""
        self._tiempos.clear()
