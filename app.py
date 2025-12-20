
# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from fgi_engine import MotorFGI
from fgi_engine_v2 import MotorFGI_V2
from fgi_engine_v3 import MotorFGI_V3

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# =========================================================
# Build identity (Render)
# =========================================================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")

app = FastAPI(title="ATHENA LABORATORIO PMF", version="1.2.1")


# =========================================================
# REQUEST MODEL
# =========================================================
class PrototiposRequest(BaseModel):
    engine: str = "v1"
    k: int = 15
    top_n: int = 10
    max_candidatos: int = 800

    # V2
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75


# =========================================================
# ROOT / STATUS
# =========================================================
@app.get("/")
def root():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
        "endpoints": [
            "/lab/status",
            "/lab/dna_last25",
            "/lab/regime_atual",
            "/prototipos",
        ],
    }


@app.get("/lab/status")
def lab_status():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "motores": {
            "v1": "Filtro",
            "v2": "Direcional / SCF",
            "v3": "Contraste / DCR",
        },
    }


# =========================================================
# LAB CONTEXTO
# =========================================================
@app.get("/lab/dna_last25")
def dna_last25():
    return RegimeDetector().get_dna_last25()


@app.get("/lab/regime_atual")
def regime_atual():
    return RegimeDetector().detectar_regime()


# =========================================================
# HELPERS
# =========================================================
def _extract_seq_list(items: List[Any]) -> List[List[int]]:
    out = []
    for it in items:
        if isinstance(it, dict) and "sequencia" in it:
            out.append(it["sequencia"])
        elif isinstance(it, list):
            out.append(it)
    return out


def _extract_candidates_for_v3(v2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = v2_result.get("top")
    if not isinstance(top, list):
        raise HTTPException(status_code=500, detail="V2 não retornou top válido")

    candidatos = []
    for it in top:
        candidatos.append(
            {
                "sequencia": it["sequencia"],
                "score": it["score"],
                "detail": it["detail"],
            }
        )
    return candidatos


# =========================================================
# PROTÓTIPOS
# =========================================================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):

    engine = req.engine.lower()

    detector = RegimeDetector()
    contexto_lab = {
        "dna_last25": detector.get_dna_last25(),
        "regime": detector.detectar_regime(),
    }

    grupo = GrupoMilhoes()
    candidatos = grupo.get_candidatos(
        k=req.k,
        max_candidatos=req.max_candidatos,
        shuffle=True,
        seed=1337,
    )

    # -------------------------
    # V1
    # -------------------------
    motor_v1 = MotorFGI()
    filtrados = motor_v1.gerar_prototipos(
        candidatos=candidatos,
        top_n=req.max_candidatos,
    )

    if engine == "v1":
        return {"engine_used": "v1", "prototipos": filtrados[: req.top_n]}

    seqs = _extract_seq_list(filtrados)

    # -------------------------
    # V2 (defaults garantidos)
    # -------------------------
    motor_v2 = MotorFGI_V2()
    resultado_v2 = motor_v2.rerank(
        candidatos=seqs,
        contexto_lab=contexto_lab,
        overrides={
            "top_n": max(req.top_n * 5, 50),
        },
    )

    if engine == "v2":
        return resultado_v2

    # -------------------------
    # V3
    # -------------------------
    candidatos_v3 = _extract_candidates_for_v3(resultado_v2)

    motor_v3 = MotorFGI_V3()
    resultado_v3 = motor_v3.rank(
        candidatos=candidatos_v3,
        top_n=req.top_n,
        alpha_contraste=req.alpha_contraste,
        beta_diversidade=req.beta_diversidade,
        gamma_base=req.gamma_base,
        jaccard_penalty_threshold=req.jaccard_penalty_threshold,
    )

    return JSONResponse(content=resultado_v3)
