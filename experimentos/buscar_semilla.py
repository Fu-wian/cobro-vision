import re
import subprocess
import sys
from pathlib import Path

cfg = Path("config.yaml")
original = cfg.read_text(encoding="utf-8")


def dividir():
    r = subprocess.run(
        [sys.executable, "-m", "cobro_vision", "dividir", "--prueba", "0.2"],
        capture_output=True, text=True,
    )
    avisos = [x for x in (r.stderr + r.stdout).splitlines() if "sin ejemplos" in x]
    return r.returncode == 0 and not avisos, len(avisos)


for s in range(43, 103):
    cfg.write_text(re.sub(r"(?m)^semilla:\s*\d+", f"semilla: {s}", original), encoding="utf-8")
    ok, n = dividir()
    print(s, "OK" if ok else f"{n} aviso(s)")
    if ok:
        print(f"-> config.yaml quedó con semilla: {s}")
        break
else:
    cfg.write_text(original, encoding="utf-8")
    dividir()
    print("Ninguna semilla cumplió; config.yaml restaurado.")