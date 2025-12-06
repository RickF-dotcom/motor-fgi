from fastapi import FastAPI
from fgi_engine import FGIMotor

app = FastAPI()

motor = FGIMotor()

@app.post("/carregar")
def carregar(data: dict):
    concursos = data.get("concursos", [])
    motor.carregar_concursos(concursos)
    return motor.resumo_basico()

@app.get("/gerar")
def gerar(n: int = 30, min_aus: int = 4):
    fgis = motor.gerar_fgis(n_fgis=n, min_ausentes=min_aus)
    return motor.fgis_para_dicts(fgis)
