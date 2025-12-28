# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from grupo_de_milhoes import GrupoDeMilhoes
from fgi_engine import MotorFGI
from fgi_engine_v2 import MotorFGI_V2
from fgi_engine_v3 import MotorFGI_V3

# =========================
# App
# =========================

app = FastAPI(
    title="Motor FGI",
    version="1.0.0",
    description="API do laboratório FGI – protótipos estruturais",
)

# =========================
# Models
# =========================

class PrototiposRequest(BaseModel):
    engine: str = "v2"
    k: int = 15
    top_n: int = 5
    max_candidatos: int = 50


# =========================
# Helpers
# =========================

def get_motor(engine: str):
    if engine == "v1":
        return MotorFGI()
    if engine == "v2":
        return MotorFGI_V2()
    if engine == "v3":
        return MotorFGI_V3()
    raise HTTPException(status_code=400, detail="Engine inválido")


# =========================
# Rotas básicas
# =========================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "motor-fgi",
        "env": os.environ.get("RENDER_SERVICE_ID", "local"),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================
# Prototipos (FGI)
# =========================

@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    motor = get_motor(req.engine)

    resultado = motor.gerar_prototipos(
        k=req.k,
        top_n=req.top_n,
        max_candidatos=req.max_candidatos,
    )

    return resultado


# =========================
# Grupo de Milhões
# =========================

@app.post("/grupo-de-milhoes/sample")
def sample_grupo_de_milhoes(
    jogos_drawn: List[List[int]],
    n: int = 10,
):
    grupo = GrupoDeMilhoes(
        jogos_drawn=[tuple(j) for j in jogos_drawn],
        k=len(jogos_drawn[0]) if jogos_drawn else 15,
    )

    return grupo.sample_not_drawn(n=n)


@app.get("/grupo-de-milhoes/status")
def status_grupo_de_milhoes(
    jogos_drawn: List[List[int]],
):
    grupo = GrupoDeMilhoes(
        jogos_drawn=[tuple(j) for j in jogos_drawn],
        k=len(jogos_drawn[0]) if jogos_drawn else 15,
    )

    return grupo.status()
