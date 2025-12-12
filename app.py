from __future__ import annotations

import os

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fgi_engine import MotorFGI
from ponto_c_engine import PontoCEngine


# ============================================================
#  LOAD DO ARQUIVO lab_config.yaml
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_CONFIG_PATH = os.path.join(BASE_DIR, "lab_config.yaml")

if not os.path.exists(LAB_CONFIG_PATH):
    raise FileNotFoundError(
        f"Arquivo de configuração não encontrado: {LAB_CONFIG_PATH}"
    )

with open(LAB_CONFIG_PATH, "r", encoding="utf-8") as f:
    LAB_CONFIG = yaml.safe_load(f)


# ============================================================
#  INSTÂNCIAS GLOBAIS: PontoCEngine + MotorFGI
# ============================================================

ponto_c_engine = PontoCEngine(LAB_CONFIG)
motor_fgi = MotorFGI(ponto_c=ponto_c_engine)


# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="ATHENAH LABORATORIO PMFC",
    description=(
        "Backend do MotorFGI + PONTO C + Config YAML. "
        "Laboratório de engenharia de padrões em janelas 25−."
    ),
    version=str(LAB_CONFIG.get("versao_laboratorio", "0.1.0")),
)


# ============================================================
#  MODELOS Pydantic
# ============================================================

class PrototipoRequest(BaseModel):
    k: Optional[int] = None
    regime_id: Optional[str] = None
    max_candidatos: Optional[int] = None


# ============================================================
#  ENDPOINTS BÁSICOS
# ============================================================

@app.get("/lab/config")
def get_lab_config():
    """
    Retorna o conteúdo completo do arquivo lab_config.yaml como JSON.
    """
    return JSONResponse(content=LAB_CONFIG)


@app.get("/lab/status")
def status():
    """
    Status simples do laboratório.
    """
    return {
        "status": "online",
        "versao_laboratorio": LAB_CONFIG.get("versao_laboratorio", "desconhecida"),
    }


# ============================================================
#  ENDPOINT: GERAR PROTÓTIPOS (FGI ESTRUTURAL)
# ============================================================

@app.post("/prototipos")
def gerar_prototipos(req: PrototipoRequest):
    """
    Gera protótipos estruturais (FGIs) usando:

      - MotorFGI (decoder)
      - PontoCEngine (constraints + score)
      - Grupo de Milhões (combinações não sorteadas)

    Corpo da requisição (JSON):

      {
        "k": 20,                # opcional, quantidade de protótipos
        "regime_id": "R2",      # opcional, regime (default vem do lab_config)
        "max_candidatos": 5000  # opcional, limite de candidatos avaliados
      }

    Retorno: lista de objetos com:
      - sequencia
      - score_total
      - coerencias
      - violacoes
      - detalhes (por tipo de verificação)
    """
    try:
        prototipos = motor_fgi.gerar_prototipos_json(
            k=req.k,
            regime_id=req.regime_id,
            max_candidatos=req.max_candidatos,
        )
        return prototipos
    except ValueError as e:
        # Erros de domínio (grupo de milhões vazio, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Erros inesperados
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
