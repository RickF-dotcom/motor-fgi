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
    description="Backend do MotorFGI + PONTO C + RegimeDetector (DNA + regime).",
)

# =========================
# Paths / Config
# =========================

BASE_DIR = Path(__file__).resolve().parent
LAB_CONFIG_PATH = BASE_DIR / "lab_config.yaml"

DNA_LAST25_PATH = BASE_DIR / "dna_last25.yaml"
HISTORICO_LAST25_CSV = BASE_DIR / "lotofacil_ultimos_25_concursos.csv"

# opcional: se você subir um CSV do histórico completo no repo, ele vira baseline REAL
HISTORICO_TOTAL_CSV = BASE_DIR / "lotofacil_historico_completo.csv"


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

# =========================
# Motor
# =========================

motor_fgi = MotorFGI()


# =========================
# Request Models
# =========================

class PrototipoRequest(BaseModel):
    k: int = 5
    regime_id: Optional[str] = "R2"
    max_candidatos: Optional[int] = 2000
    incluir_contexto_dna: bool = True


# =========================
# Endpoints básicos
# =========================

@app.get("/lab/config")
def get_lab_config():
    return LAB_CONFIG


@app.get("/lab/status")
def status():
    return {
        "status": "online",
        "versao_laboratorio": LAB_CONFIG.get("versao_laboratorio", "desconhecida"),
    }


# =========================
# Laboratório: DNA e Regime
# =========================

@app.get("/lab/dna_last25")
def lab_dna_last25():
    """
    Retorna o DNA estrutural calculado a partir das últimas 25 sequências reais.
    """
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
    """
    Diagnóstico do regime atual baseado em:
      - DNA(25) das últimas 25
      - baseline macro (histórico total se disponível; senão fallback)
    """
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


# =========================
# Endpoint principal: protótipos
# =========================

@app.post("/prototipos")
def gerar_prototipos(req: PrototipoRequest):
    """
    Gera protótipos estruturais (LHE/LHS) usando:
      - MotorFGI (decoder + score)
      - Grupo de Milhões (combinações não sorteadas)
    E injeta contexto do laboratório (DNA + regime) se solicitado.
    """
    try:
        # gera protótipos do motor
        out = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
            incluir_contexto_dna=False,  # vamos controlar aqui fora
        )

        if not req.incluir_contexto_dna:
            return out

        # injeta contexto do laboratório
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
