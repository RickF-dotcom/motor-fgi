from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from fgi_engine import FGIMotor


class CarregarRequest(BaseModel):
    concursos: List[List[int]]  # cada concurso é uma lista de 15 dezenas


app = FastAPI()
motor = FGIMotor()


@app.post("/carregar")
def carregar(req: CarregarRequest):
    """
    Exemplo de body (JSON):

    {
      "concursos": [
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        [ ... 15 dezenas ... ],
        ...
      ]
    }
    """
    motor.carregar_concursos(req.concursos)
    return motor.resumo_basico()


@app.get("/gerar_fino")
def gerar_fino(
    n: int = 32,
    min_frias: int = 5,
    min_quentes: int = 4,
):
    """
    Gera N jogos já peneirados.

    Exemplo de uso no Hoppscotch:

    GET https://SEU-SERVICO.onrender.com/gerar_fino?n=32&min_frias=5&min_quentes=4
    """

    fgis = motor.gerar_fgi_fino(
        n_resultados=n,
        min_frias=min_frias,
        min_quentes=min_quentes,
    )
    return motor.fgis_para_dicts(fgis)
