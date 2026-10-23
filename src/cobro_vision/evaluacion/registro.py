"""Registro acumulativo de evaluaciones y su configuración."""

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cobro_vision.configuracion import Configuracion


@dataclass
class RegistroExperimentos:
    """Añade una fila por llamada, con las métricas anidadas en una celda JSON."""

    ruta: Path = Path("experimentos/resultados.csv")
    raiz_git: Path | None = None

    def registrar(self, nombre: str, config: Configuracion, metricas: dict) -> None:
        """Conserva fecha, hashes y métricas sin cambiar el esquema entre objetivos."""
        contenido = json.dumps(metricas, ensure_ascii=False, allow_nan=False)
        hash_git = ""
        try:
            resultado = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.raiz_git,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if resultado.returncode == 0:
                hash_git = resultado.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass  # Un proyecto sin git también puede evaluar.
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        escribir_cabecera = not self.ruta.exists() or self.ruta.stat().st_size == 0
        with self.ruta.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=[
                    "fecha",
                    "nombre",
                    "hash_configuracion",
                    "hash_git",
                    "metricas",
                ],
            )
            if escribir_cabecera:
                escritor.writeheader()
            escritor.writerow(
                {
                    "fecha": datetime.now().astimezone().isoformat(),
                    "nombre": nombre,
                    "hash_configuracion": config.hash(),
                    "hash_git": hash_git,
                    "metricas": contenido,
                }
            )
