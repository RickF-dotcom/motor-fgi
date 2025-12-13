from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

from fgi_engine import MotorFGI
from regime_detector import RegimeDetector


app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="0.2.0",
    description="Backend do MotorFGI + PONTO C + RegimeDetector (DNA + regime).",
)

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

# IMPORTANTÍSSIMO: carrega o histórico (últimos 25) no MotorFGI
motor_fgi = MotorFGI(historico_csv=str(HISTORICO_LAST25_CSV), universo_max=25)


class PrototipoRequest(BaseModel):
    k: int = 15
    regime_id: Optional[str] = "estavel"
    max_candidatos: Optional[int] = 2000
    incluir_contexto_dna: bool = True

    # overrides do request (Swagger)
    pesos_override: Optional[Dict[str, float]] = None
    constraints_override: Optional[Dict[str, Any]] = None


@app.get("/lab/config")
def get_lab_config():
    return LAB_CONFIG


@app.get("/lab/status")
def status():
    return {
        "status": "online",
        "versao_laboratorio": LAB_CONFIG.get("versao_laboratorio", "desconhecida"),
    }


@app.get("/lab/dna_last25")
def lab_dna_last25():
    try:
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV if HISTORICO_TOTAL_CSV.exists() else None,
        )
        dna = detector.extrair_dna()
        return {"origem": "ultimos_25_concursos", "dna": dna}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/lab/regime_atual")
def lab_regime_atual():
    try:
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV if HISTORICO_TOTAL_CSV.exists() else None,
        )
        regime = detector.detectar_regime()
        return {"regime_atual": regime}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prototipos")
def gerar_prototipos(req: PrototipoRequest):
    try:
        out = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
            incluir_contexto_dna=False,
            pesos_override=req.pesos_override,
            constraints_override=req.constraints_override,
        )

        if not req.incluir_contexto_dna:
            return out

        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV if HISTORICO_TOTAL_CSV.exists() else None,
        )
        dna = detector.extrair_dna()
        regime = detector.detectar_regime()

        out["contexto_lab"] = {
            "dna_last25": dna,
            "regime_atual": regime,
        }
        return out

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
