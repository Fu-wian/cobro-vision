"""Catálogo SQLite inicializado desde una semilla CSV."""

import csv
import sqlite3
from contextlib import closing
from pathlib import Path

from cobro_vision.dominio.errores import ClaseDesconocida
from cobro_vision.dominio.modelos import ID_FONDO, Alimento
from cobro_vision.dominio.utilidades import validar_centavos

_CREAR_TABLA = """
CREATE TABLE IF NOT EXISTS alimentos (
    id_clase TEXT PRIMARY KEY,
    nombre TEXT NOT NULL,
    categoria TEXT NOT NULL CHECK (
        categoria IN ('base','proteina','acompanante','ensalada','fondo')
    ),
    precio_centavos INTEGER NOT NULL DEFAULT 0 CHECK (precio_centavos >= 0),
    activo INTEGER NOT NULL DEFAULT 1
)
"""
_GUARDAR = """
INSERT INTO alimentos (id_clase, nombre, categoria, precio_centavos, activo)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(id_clase) DO UPDATE SET
    nombre=excluded.nombre, categoria=excluded.categoria,
    precio_centavos=excluded.precio_centavos, activo=excluded.activo
"""


class CatalogoAlimentos:
    """Consulta y mantiene el catálogo sin conservar conexiones abiertas."""

    def __init__(self, ruta_db: Path) -> None:
        self.ruta_db = ruta_db

    def inicializar(self, ruta_semilla: Path) -> None:
        """Crea la tabla y actualiza la semilla de forma atómica e idempotente."""
        alimentos = []
        with ruta_semilla.open(encoding="utf-8-sig", newline="") as archivo:
            lector = csv.DictReader(archivo)
            columnas = {"id_clase", "nombre", "categoria", "precio_centavos", "activo"}
            if set(lector.fieldnames or ()) != columnas or len(
                lector.fieldnames or ()
            ) != len(columnas):
                raise ValueError("La semilla no contiene las columnas del catálogo.")
            for numero, fila in enumerate(lector, start=2):
                try:
                    if None in fila or any(valor is None for valor in fila.values()):
                        raise ValueError("Cantidad de columnas incorrecta.")
                    if fila["activo"] not in {"0", "1"}:
                        raise ValueError("activo debe ser 0 o 1.")
                    alimento = Alimento(
                        fila["id_clase"],
                        fila["nombre"],
                        fila["categoria"],
                        int(fila["precio_centavos"]),
                        fila["activo"] == "1",
                    )
                    self._validar(alimento)
                    alimentos.append(self._valores(alimento))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Semilla, fila {numero}: {error}") from error
        self.ruta_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.ruta_db)) as conexion, conexion:
            conexion.execute(_CREAR_TABLA)
            conexion.executemany(_GUARDAR, alimentos)

    def obtener(self, id_clase: str) -> Alimento:
        """Obtiene una clase activa o lanza ClaseDesconocida."""
        with closing(sqlite3.connect(self.ruta_db)) as conexion:
            fila = conexion.execute(
                "SELECT * FROM alimentos WHERE id_clase = ? AND activo = 1", (id_clase,)
            ).fetchone()
        if fila is None:
            raise ClaseDesconocida(f"Clase desconocida o inactiva: {id_clase}")
        return self._alimento(fila)

    def listar(self, solo_cobrables: bool = False) -> list[Alimento]:
        """Lista clases activas por id; las cobrables excluyen fondo."""
        with closing(sqlite3.connect(self.ruta_db)) as conexion:
            filas = conexion.execute(
                "SELECT * FROM alimentos WHERE activo = 1 ORDER BY id_clase"
            ).fetchall()
        return [
            self._alimento(fila)
            for fila in filas
            if not solo_cobrables or (fila[0] != ID_FONDO and fila[2] != "fondo")
        ]

    def clases_validas(self) -> list[str]:
        """Devuelve ids activos en orden alfabético, incluido fondo."""
        return [alimento.id_clase for alimento in self.listar()]

    def agregar(self, alimento: Alimento) -> None:
        """Inserta o actualiza un alimento validado."""
        self._validar(alimento)
        with closing(sqlite3.connect(self.ruta_db)) as conexion, conexion:
            conexion.execute(_GUARDAR, self._valores(alimento))

    def actualizar_precio(self, id_clase: str, precio_centavos: int) -> None:
        """Cambia los centavos de una clase activa."""
        validar_centavos(precio_centavos)
        with closing(sqlite3.connect(self.ruta_db)) as conexion, conexion:
            cursor = conexion.execute(
                "UPDATE alimentos SET precio_centavos = ? WHERE id_clase = ? AND activo = 1",
                (precio_centavos, id_clase),
            )
            if cursor.rowcount == 0:
                raise ClaseDesconocida(f"Clase desconocida o inactiva: {id_clase}")

    @staticmethod
    def _validar(alimento: Alimento) -> None:
        validar_centavos(alimento.precio_centavos)
        if not alimento.id_clase.strip() or not alimento.nombre.strip():
            raise ValueError("id_clase y nombre no pueden estar vacíos.")
        if alimento.categoria not in {
            "base",
            "proteina",
            "acompanante",
            "ensalada",
            "fondo",
        }:
            raise ValueError(f"Categoría inválida: {alimento.categoria}")
        if type(alimento.activo) is not bool:
            raise ValueError("activo debe ser bool.")

    @staticmethod
    def _valores(alimento: Alimento) -> tuple:
        return (
            alimento.id_clase,
            alimento.nombre,
            alimento.categoria,
            alimento.precio_centavos,
            int(alimento.activo),
        )

    @staticmethod
    def _alimento(fila: tuple) -> Alimento:
        return Alimento(*fila[:4], activo=bool(fila[4]))
