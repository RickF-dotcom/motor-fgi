from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.responses import PlainTextResponse
from io import StringIO
import csv

from fgi_engine import FGIMotor


# -------------------------
# Modelo de entrada
# -------------------------

class CarregarRequest(BaseModel):
    # cada concurso é uma lista de dezenas (idealmente 15, de 1 a 25)
    concursos: List[List[int]]


# -------------------------
# Instância principal
# -------------------------

app = FastAPI(
    title="Motor FGI",
    version="1.0.0",
    description="API do motor de FGIs com score."
)

motor = FGIMotor()


# -------------------------
# Ping raiz
# -------------------------

@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "motor-fgi online"}


# -------------------------
# /carregar
# -------------------------

@app.post("/carregar")
def carregar(req: CarregarRequest):
    """
    Carrega uma janela de concursos no motor e devolve
    as estatísticas básicas (freq, frias, quentes).

    Exemplo de body (JSON):

    {
      "concursos": [
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        [1,3,5,7,9,11,13,15,17,19,21,23,25,2,4]
      ]
    }
    """
    try:
        # Atualiza o motor com os concursos
        # motor.carregar retorna um AnaliseEstado (objeto)
        estado = motor.carregar(req.concursos)

        # Usa ATRIBUTOS do objeto, não índice
        return {
            "status": "ok",
            "total_concursos": estado.total_concursos,
            "freq": estado.freq,
            "frias": estado.frias,
            "quentes": estado.quentes,
        }

    except ValueError as e:
        # Erros de validação / entrada ruim
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Erro interno qualquer → 500
        raise HTTPException(status_code=500, detail=f"erro interno: {e}")


# -------------------------
# /gerar_fino
# -------------------------

@app.get("/gerar_fino")
def gerar_fino(
    n: int = 32,
    min_frias: int = 5,
    min_quentes: int = 4,
):
    """
    Gera N jogos (FGIs) usando o motor com score.

    Parâmetros:
      - n: quantidade de jogos (default 32)
      - min_frias: mínimo de dezenas frias por jogo
      - min_quentes: mínimo de dezenas quentes por jogo

    Exemplo de uso:

      GET /gerar_fino?n=32&min_frias=5&min_quentes=4
    """
    try:
        jogos = motor.gerar_fino(
            n=n,
            min_frias=min_frias,
            min_quentes=min_quentes,
        )
        return {
            "status": "ok",
            "quantidade": len(jogos),
            "jogos": jogos,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"erro interno: {e}")


# -------------------------
# /ultimos25.csv
# -------------------------

@app.get("/ultimos25.csv", response_class=PlainTextResponse)
def ultimos_25_csv():
    """
    Exporta os últimos 25 concursos carregados em formato CSV.

    - Baseado em motor.concursos (ordem em que foram carregados).
    - Se existirem menos de 25 concursos, exporta só o que tiver.
    - Separador: ponto e vírgula (;)

    Cabeçalho:
      concurso;d1;d2;...;d15
    """
    if not getattr(motor, "concursos", None):
        raise HTTPException(
            status_code=400,
            detail="Nenhum concurso carregado. Chame /carregar antes."
        )

    concursos = motor.concursos
    qtd = min(25, len(concursos))
    ultimos = concursos[-qtd:]

    buffer = StringIO()
    writer = csv.writer(buffer, delimiter=';')

    # Cabeçalho
    header = ["concurso"] + [f"d{i}" for i in range(1, 16)]
    writer.writerow(header)

    primeiro_idx = len(concursos) - qtd + 1

    # Linhas
    for idx, jogo in enumerate(ultimos):
        row = [primeiro_idx + idx] + list(jogo)
        writer.writerow(row)

    csv_str = buffer.getvalue()
    buffer.close()

    return csv_str
