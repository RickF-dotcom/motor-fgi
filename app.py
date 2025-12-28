from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import traceback

# ==============================
# IMPORTS DOS MOTORES
# ==============================
from fgi_engine import MotorFGI
from fgi_engine_v2 import MotorFGI_V2
from fgi_engine_v3 import MotorFGI_V3

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import detectar_regime_atual

# ==============================
# APP
# ==============================
app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="0.1.0"
)

# ==============================
# MODELOS
# ==============================
class PrototiposRequest(BaseModel):
    engine: str
    k: int
    top_n: int
    max_candidatos: int

# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    return {
        "ok": True,
        "build_commit": os.getenv("RENDER_GIT_COMMIT", "local"),
        "service_id": os.getenv("RENDER_SERVICE_ID", "local")
    }

# ==============================
# HEALTH
# ==============================
@app.get("/health")
def health():
    return {"status": "ok"}

# ==============================
# LAB STATUS
# ==============================
@app.get("/lab/status")
def lab_status():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "motores": {
            "v1": "Pass-through (congelado)",
            "v2": "Direcional / SCF",
            "v3": "Contraste / DCR"
        }
    }

# ==============================
# DNA LAST 25
# ==============================
@app.get("/lab/dna_last25")
def dna_last25():
    try:
        motor = MotorFGI()
        return motor.dna_last25()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# REGIME ATUAL
# ==============================
@app.get("/lab/regime_atual")
def regime_atual():
    try:
        return detectar_regime_atual()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# GERAR PROTÓTIPOS
# ==============================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    try:
        grupo = GrupoMilhoes()

        if req.engine == "v1":
            motor = MotorFGI()
        elif req.engine == "v2":
            motor = MotorFGI_V2()
        elif req.engine == "v3":
            motor = MotorFGI_V3()
        else:
            raise HTTPException(status_code=400, detail="Engine inválido")

        prototipos = motor.gerar_prototipos(
            grupo=grupo,
            k=req.k,
            top_n=req.top_n,
            max_candidatos=req.max_candidatos
        )

        return {
            "engine_used": req.engine,
            "prototipos": prototipos
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
