from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

from fgi_engine import MotorFGI
from regime_detector import RegimeDetector


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="0.3.1",
    description=(
        "Backend do MotorFGI + PONTO C + RegimeDetector "
        "(DNA + regime) com âncora fractal desacoplada."
    ),
)

BASE_DIR = Path(__file__).resolve().parent

DNA_LAST25_PATH = BASE_DIR / "dna_last25.yaml"
HISTORICO_LAST25_CSV = BASE_DIR / "lotofacil_ultimos_25_concursos.csv"
HISTORICO_TOTAL_CSV = BASE_DIR / "lotofacil_historico_completo.csv"


# ============================================================
# Utils
# ============================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido (não é dict): {path}")
    return data


# ============================================================
# Motor
# ============================================================

motor_fgi = MotorFGI(
    historico_csv=str(HISTORICO_LAST25_CSV),
)


# ============================================================
# Models
# ============================================================

class PrototipoRequest(BaseModel):
    k: int = 15
    regime_id: Optional[str] = "estavel"
    max_candidatos: Optional[int] = 2000
    incluir_contexto_dna: bool = True

    # âncora fractal (janela do DNA)
    dna_anchor_window: int = 12

    # overrides experimentais
    pesos_override: Optional[Dict[str, float]] = None
    constraints_override: Optional[Dict[str, Any]] = None


# ============================================================
# Endpoints básicos
# ============================================================

@app.get("/lab/status")
def lab_status():
    return {
        "status": "online",
        "versao": app.version,
    }


@app.get("/lab/dna_last25")
def lab_dna_last25():
    try:
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV
            if HISTORICO_TOTAL_CSV.exists()
            else None,
        )
        dna = detector.extrair_dna()
        return {
            "origem": "ultimos_25_concursos",
            "dna": dna,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/lab/regime_atual")
def lab_regime_atual():
    try:
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV
            if HISTORICO_TOTAL_CSV.exists()
            else None,
        )
        regime = detector.detectar_regime()
        return {
            "regime_atual": regime,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# Prototipos
# ============================================================

@app.post("/prototipos")
def gerar_prototipos(req: PrototipoRequest):
    try:
        # --- Detector ---
        detector = RegimeDetector(
            dna_path=DNA_LAST25_PATH,
            historico_csv=HISTORICO_LAST25_CSV,
            historico_total_csv=HISTORICO_TOTAL_CSV
            if HISTORICO_TOTAL_CSV.exists()
            else None,
        )

        dna = detector.extrair_dna()
        regime = detector.detectar_regime()

        # ====================================================
        # ÂNCORA DNA (forma universal, não quebra engine)
        # ====================================================
        motor_fgi.set_dna_anchor(
            dna_anchor={
                "dna_last25": dna,
                "window": req.dna_anchor_window,
            }
        )

        # --- Geração ---
        out = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id or regime.get("regime", "estavel"),
            max_candidatos=req.max_candidatos or 2000,
            incluir_contexto_dna=False,  # controlado aqui
            pesos_override=req.pesos_override,
            constraints_override=req.constraints_override,
        )

        # --- Contexto explícito ---
        if req.incluir_contexto_dna:
            out["contexto_lab"] = {
                "dna_last25": dna,
                "regime_atual": regime,
                "dna_anchor_window": req.dna_anchor_window,
            }

        return out

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno: {e}",
        )
