from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

from fgi_engine import MotorFGI
from regime_detector import RegimeDetector


# =========================
# App
# =========================

app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="0.2.0",
    description="Backend do MotorFGI + RegimeDetector (DNA + regime).",
)

# =========================
# Paths / Config
# =========================

BASE_DIR = Path(__file__).resolve().parent

LAB_CONFIG_PATH = BASE_DIR / "lab_config.yaml"
DNA_LAST25_PATH = BASE_DIR / "dna_last25.yaml"
HISTORICO_LAST25_CSV = BASE_DIR / "lotofacil_ultimos_25_concursos.csv"
HISTORICO_TOTAL_CSV = BASE_DIR / "lotofacil_historico_completo.csv"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


LAB_CONFIG: Dict[str, Any] = _load_yaml(LAB_CONFIG_PATH)

# =========================
# Motor
# =========================

motor_fgi = MotorFGI(
    historico_csv=str(HISTORICO_LAST25_CSV),
    universo_max=25,
)

# =========================
# Request Models
# =========================

class PrototipoRequest(BaseModel):
    k: int
    regime_id: Optional[str] = "estavel"
    max_candidatos: Optional[int] = 2000
    incluir_contexto_dna: bool = True


# =========================
# Endpoints básicos
# =========================

@app.get("/lab/status")
def lab_status():
    return {
        "status": "online",
        "versao": app.version,
    }


@app.get("/lab/config")
def lab_config():
    return LAB_CONFIG


# =========================
# DNA / Regime
# =========================

@app.get("/lab/dna_last25")
def dna_last25():
    try:
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV if HISTORICO_TOTAL_CSV.exists() else None,
        )
        return detector.extrair_dna()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/lab/regime_atual")
def regime_atual():
    try:
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV if HISTORICO_TOTAL_CSV.exists() else None,
        )
        return detector.detectar_regime()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================
# Protótipos (endpoint principal)
# =========================

@app.post("/prototipos")
def gerar_prototipos(req: PrototipoRequest):
    try:
        resultado = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
            incluir_contexto_dna=False,
        )

        if not req.incluir_contexto_dna:
            return resultado

        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV if HISTORICO_TOTAL_CSV.exists() else None,
        )

        resultado["contexto_lab"] = {
            "dna_last25": detector.extrair_dna(),
            "regime_atual": detector.detectar_regime(),
        }

        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
