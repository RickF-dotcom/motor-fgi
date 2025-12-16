
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from grupo_de_milhoes import GrupoMilhoes
from fgi_engine_v2 import MotorFGI_V2

app = FastAPI(title="ATHENA LABORATORIO PMF")

class PrototiposRequest(BaseModel):
    engine: str = "v2"
    top_n: int = 30
    redundancy_jaccard_threshold: float = 0.75
    redundancy_penalty: float = 0.5
    align_temperature: float = 2.0

@app.get("/lab/status")
def status():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "engine_disponivel": ["v2"],
        "schema_version": "v2.1",
        "scoring_mode": "scf_governing"
    }

@app.post("/prototipos")
def prototipos(req: PrototiposRequest):
    if req.engine != "v2":
        return {"erro": "Somente v2 habilitado neste modo de teste"}

    grupo = GrupoMilhoes()
    candidatos = grupo.gerar_combinacoes()[:200]  # TESTE CONTROLADO

    motor = MotorFGI_V2()

    return motor.rerank(
        candidatos=candidatos,
        contexto_lab={},
        overrides=req.dict()
    )
