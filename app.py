from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml

from fgi_engine import MotorFGI
from regime_detector import RegimeDetector

# NOVO: fractal temporal (janela 25/13/8 etc.)
from temporal_fractal_engine import rankear_prototipos_por_fractal_temporal


# =========================
# App
# =========================

app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="0.3.0",
    description="Backend do MotorFGI + PONTO C + RegimeDetector + Fractal Temporal (DNA por janelas).",
)

# =========================
# Paths / Config
# =========================

BASE_DIR = Path(__file__).resolve().parent
LAB_CONFIG_PATH = BASE_DIR / "lab_config.yaml"

DNA_LAST25_PATH = BASE_DIR / "dna_last25.yaml"
HISTORICO_LAST25_CSV = BASE_DIR / "lotofacil_ultimos_25_concursos.csv"

# opcional: se você subir um CSV do histórico completo no repo, vira baseline REAL
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

# importante: inicializa o motor já com o CSV que existe no repo
motor_fgi = MotorFGI(
    historico_csv=str(HISTORICO_LAST25_CSV) if HISTORICO_LAST25_CSV.exists() else None,
    universo_max=25,
)


# =========================
# Request Models
# =========================

class PrototipoRequest(BaseModel):
    k: int = 15
    regime_id: Optional[str] = "estavel"
    max_candidatos: Optional[int] = 2000
    incluir_contexto_dna: bool = True

    # compat com seus testes no Swagger
    pesos_override: Optional[Dict[str, float]] = None
    constraints_override: Optional[Dict[str, Any]] = None


class FractalRankRequest(BaseModel):
    # gera protótipos e já ranqueia pelo fractal temporal
    k: int = 15
    regime_id: Optional[str] = "estavel"
    max_candidatos: Optional[int] = 2000
    incluir_contexto_dna: bool = True

    pesos_override: Optional[Dict[str, float]] = None
    constraints_override: Optional[Dict[str, Any]] = None

    # fractal temporal
    windows: Optional[List[int]] = None              # default: [25, 13, 8]
    pesos_windows: Optional[Dict[int, float]] = None # default: {25:0.25, 13:0.50, 8:0.25}
    pesos_metricas: Optional[Dict[str, float]] = None # default: {"soma":1,"pares":1,"adj":1}
    top_n: int = 50


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
        "app_version": app.version,
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
        out = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
            incluir_contexto_dna=False,  # controlamos aqui fora
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
    except TypeError as e:
        # pega exatamente o erro que você já viu (keyword inesperada etc.)
        raise HTTPException(status_code=500, detail=f"Erro de assinatura (engine/app desalinhados): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")


# =========================
# NOVO Endpoint: Fractal Temporal (ranking por “parecido com o agora”)
# =========================

@app.post("/lab/fractal_rank")
def lab_fractal_rank(req: FractalRankRequest):
    """
    1) Gera protótipos (MotorFGI)
    2) Rankeia os protótipos pela distância estrutural para o DNA temporal
       em janelas (ex.: 25/13/8)
    """
    try:
        base = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
            incluir_contexto_dna=req.incluir_contexto_dna,
            pesos_override=req.pesos_override,
            constraints_override=req.constraints_override,
        )

        protos = base.get("prototipos", [])

        ranking = rankear_prototipos_por_fractal_temporal(
            prototipos=protos,
            historico_csv=HISTORICO_LAST25_CSV,
            windows=req.windows,
            pesos_windows=req.pesos_windows,
            pesos_metricas=req.pesos_metricas,
            top_n=req.top_n,
        )

        return {
            "base": base,
            "ranking_temporal": ranking,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
