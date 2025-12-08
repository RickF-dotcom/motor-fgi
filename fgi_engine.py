from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fgi_engine import FGIMotor


class CarregarRequest(BaseModel):
    concursos: List[List[int]]


app = FastAPI(
    title="Motor FGI",
    version="1.0.0",
    description="API do motor de FGIs com score."
)

motor = FGIMotor()


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "motor-fgi online"}


@app.post("/carregar")
def carregar(req: CarregarRequest):
    try:
        estado = motor.carregar(req.concursos)
        return estado
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"erro interno: {e}")


@app.get("/gerar_fino")
def gerar_fino(
    n: int = 32,
    min_frias: int = 5,
    min_quentes: int = 4,
):
    try:
        jogos = motor.gerar_fino(
            n=n,
            min_frias=min_frias,
            min_quentes=min_quentes,
        )
        return jogos
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"erro interno: {e}")
