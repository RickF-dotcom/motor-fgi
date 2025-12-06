from fastapi import FastAPI
from pydantic import BaseModel
from fgi_engine import FGIMotor

app = FastAPI(title="Motor FGI - V1")

class DrawData(BaseModel):
    draws: list[list[int]]
    quantity: int = 30
    min_absents: int = 4

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Motor FGI rodando. Abra /docs para testar o endpoint."
    }

@app.post("/fgi")
def create_fgi(data: DrawData):
    motor = FGIMotor(total_dezenas=25, dezenas_por_concurso=15)
    motor.carregar_concursos(data.draws)

    fgis = motor.gerar_fgis(
        n_fgis=data.quantity,
        min_ausentes=data.min_absents
    )

    return {
        "resumo": motor.resumo_basico(),
        "parametros": {
            "n_fgis": data.quantity,
            "min_absentes": data.min_absents,
        },
        "fgis": motor.fgis_para_dicts(fgis),
    }
