from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from pathlib import Path
import yaml

from fgi_engine import MotorFGI
from regime_detector import RegimeDetector


app = FastAPI(
    title="ATHENAH LABORATORIO PMFC",
    version="0.1.0",
    description="Backend do MotorFGI + PONTO C + Config YAML. Laboratório de engenharia de padrões.",
)


# ==============================
# Paths / Config
# ==============================

BASE_DIR = Path(__file__).resolve().parent
LAB_CONFIG_PATH = BASE_DIR / "lab_config.yaml"

DNA_LAST25_PATH = BASE_DIR / "dna_last25.yaml"
HISTORICO_LAST25_CSV = BASE_DIR / "lotofacil_ultimos_25_concursos.csv"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido (não é dict): {path}")
    return data


LAB_CONFIG: Dict[str, Any] = {}
try:
    LAB_CONFIG = _load_yaml(LAB_CONFIG_PATH)
except Exception:
    LAB_CONFIG = {}


# ==============================
# Request models
# ==============================

class PrototipoRequest(BaseModel):
    k: int = 5
    regime_id: Optional[str] = "R2"
    max_candidatos: Optional[int] = 2000


# ==============================
# Health / Config endpoints
# ==============================

@app.get("/lab/config")
def get_lab_config():
    return LAB_CONFIG


@app.get("/lab/status")
def status():
    return {
        "status": "online",
        "versao_laboratorio": LAB_CONFIG.get("versao_laboratorio", "desconhecida"),
    }


# ==============================
# FGI endpoint - protótipos
# ==============================

@app.post("/prototipos")
def gerar_prototipos(req: PrototipoRequest):
    """
    Gera protótipos estruturais (LHEs/FGIs) usando:
    - MotorFGI (decoder)
    - PontoCEngine (constraints + score)
    - Grupo de Milhões (combinações não sorteadas)
    """
    try:
        motor = MotorFGI()
        prototipos = motor.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
            incluir_contexto_dna=True,
        )
        return prototipos
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")


# ==============================
# LAB - DNA e Regime (Passo C)
# ==============================

def _assert_lab_files():
    if not DNA_LAST25_PATH.exists():
        raise HTTPException(status_code=500, detail="Arquivo dna_last25.yaml não encontrado")
    if not HISTORICO_LAST25_CSV.exists():
        raise HTTPException(status_code=500, detail="Arquivo lotofacil_ultimos_25_concursos.csv não encontrado")


@app.get("/lab/dna_last25")
def lab_dna_last25():
    """
    Retorna o DNA estrutural calculado a partir das últimas 25 sequências reais.
    """
    _assert_lab_files()

    detector = RegimeDetector(
        dna_path=DNA_LAST25_PATH,
        historico_csv=HISTORICO_LAST25_CSV,
    )

    dna = detector.extrair_dna()
    return {"origem": "ultimos_25_concursos", "dna": dna}


@app.get("/lab/regime_atual")
def lab_regime_atual():
    """
    Diagnóstico do regime atual baseado no DNA + histórico das últimas 25.
    """
    _assert_lab_files()

    detector = RegimeDetector(
        dna_path=DNA_LAST25_PATH,
        historico_csv=HISTORICO_LAST25_CSV,
    )

    regime = detector.detectar_regime()
    return {
        "regime_atual": {
            "regime_id": regime.regime_id,
            "score_regime": regime.score_regime,
            "diagnostico": regime.diagnostico,
        }
    }
