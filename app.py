from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from io import StringIO
import csv

from fgi_engine import FGIMotor


# -------------------------
# Modelos de entrada
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
    description="API do motor de FGIs com score + grupo de milhões.",
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
def carregar(req: CarregarRequest) -> Dict[str, Any]:
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
        # Usa o método novo que já integra com o GrupoDeMilhoes
        estado = motor.carregar_concursos(req.concursos)

        # estado pode ser um objeto (dataclass) ou um dict,
        # então trato os dois casos de forma segura.
        if isinstance(estado, dict):
            total_concursos = estado.get("total_concursos")
            freq = estado.get("freq")
            frias = estado.get("frias")
            quentes = estado.get("quentes")
        else:
            # AnaliseEstado
            total_concursos = getattr(estado, "total_concursos", None)
            freq = getattr(estado, "freq", None)
            frias = getattr(estado, "frias", None)
            quentes = getattr(estado, "quentes", None)

        return {
            "status": "ok",
            "total_concursos": total_concursos,
            "freq": freq,
            "frias": frias,
            "quentes": quentes,
        }

    except HTTPException:
        # Se eu mesmo lancei HTTPException, só repasso
        raise
    except ValueError as e:
        # Entrada inválida (ex: dezenas fora de 1..25, tamanho errado etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Erro interno inesperado
        raise HTTPException(status_code=500, detail=f"erro interno: {e}")


# -------------------------
# /gerar_fino
# -------------------------

@app.get("/gerar_fino")
def gerar_fino(
    n: int = 32,
    min_frias: int = 5,
    min_quentes: int = 4,
) -> Dict[str, Any]:
    """
    Gera N jogos (FGIs) usando o motor com score,
    respeitando mínimos de frias/quentes.

    Parâmetros:
      - n: quantidade de jogos (default 32)
      - min_frias: mínimo de dezenas frias por jogo
      - min_quentes: mínimo de dezenas quentes por jogo

    Exemplo:
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
def ultimos_25_csv() -> str:
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
