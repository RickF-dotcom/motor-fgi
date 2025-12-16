
# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from fgi_engine_v2 import MotorFGI_V2
from fgi_engine_v3 import MotorFGI_V3

from grupo_de_milhoes import GrupoDeMilhoes
from regime_detector import RegimeDetector


# ==============================
#   Build identity (Render)
# ==============================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")

app = FastAPI(title="ATHENA LABORATORIO PMF")


# ==============================
#   ROOT / HEALTH
# ==============================

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "ok": True,
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID
    }


# ==============================
#   REQUEST
# ==============================

class PrototiposRequest(BaseModel):
    engine: str = "v1"  # v1 | v2 | v3

    top_n: int = 30
    max_candidatos: int = 3000

    # V2
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75


# ==============================
#   STATUS
# ==============================

@app.get("/lab/status")
def lab_status():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
        "motores": {
            "v1": "Pass-through (congelado)",
            "v2": "Direcional / SCF",
            "v3": "Contraste / DCR"
        }
    }


# ==============================
#   DNA / REGIME
# ==============================

@app.get("/lab/dna_last25")
def dna_last25():
    return RegimeDetector().get_dna_last25()


@app.get("/lab/regime_atual")
def regime_atual():
    return RegimeDetector().detectar_regime()


# ==============================
#   HELPERS
# ==============================

def _extract_candidates_for_v3(v2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = v2_result.get("top")
    if not isinstance(top, list):
        raise HTTPException(500, "V2 não retornou lista válida")

    out = []
    for it in top:
        detail = it.get("detail", {})
        metricas = detail.get("metricas")
        if not isinstance(metricas, dict):
            raise HTTPException(500, "V3 exige metricas do V2")

        out.append({
            "sequencia": it["sequencia"],
            "score": float(it.get("score", 0)),
            "detail": {
                "scf_total": float(detail.get("scf_total", 0)),
                "metricas": metricas
            }
        })
    return out


# ==============================
#   PROTÓTIPOS
# ==============================

@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):

    engine = req.engine.lower()

    detector = RegimeDetector()
    contexto_lab = {
        "dna_last25": detector.get_dna_last25(),
        "regime": detector.detectar_regime(),
        "build_commit": BUILD_COMMIT
    }

    # =========================
    # Grupo de Milhões (15 dezenas)
    # =========================
    grupo = GrupoDeMilhoes()
    candidatos = grupo.get_candidatos(
        k=15,
        max_candidatos=req.max_candidatos,
        shuffle=True,
        seed=1337
    )

    if not candidatos:
        raise HTTPException(500, "Grupo de Milhões vazio")

    # =========================
    # v1 — PASS THROUGH
    # =========================
    if engine == "v1":
        return {
            "engine_used": "v1",
            "prototipos": candidatos[:req.top_n],
            "contexto_lab": contexto_lab
        }

    # =========================
    # v2 — SCF
    # =========================
    motor_v2 = MotorFGI_V2()
    resultado_v2 = motor_v2.rerank(
        candidatos=candidatos,
        contexto_lab=contexto_lab,
        overrides=req.dict()
    )

    if engine == "v2":
        resultado_v2["contexto_lab"] = contexto_lab
        return resultado_v2

    # =========================
    # v3 — CONTRASTE
    # =========================
    motor_v3 = MotorFGI_V3()
    candidatos_v3 = _extract_candidates_for_v3(resultado_v2)

    resultado_v3 = motor_v3.rank(
        candidatos=candidatos_v3,
        top_n=req.top_n,
        alpha_contraste=req.alpha_contraste,
        beta_diversidade=req.beta_diversidade,
        gamma_base=req.gamma_base,
        jaccard_penalty_threshold=req.jaccard_penalty_threshold
    )

    resultado_v3["contexto_lab"] = contexto_lab
    return resultado_v3
