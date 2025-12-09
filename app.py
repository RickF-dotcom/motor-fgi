from typing import List, Dict, Any, Optional

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

# motor é recriado sempre que /carregar é chamado
motor: Optional[FGIMotor] = None


def novo_motor() -> FGIMotor:
    """
    Cria um motor novo já integrado ao grupo de milhões.

    O FGIMotor, por dentro, carrega o grupo de milhões
    e remove do universo as combinações já sorteadas
    com base no CSV histórico.
    """
    return FGIMotor()


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
    Carrega uma JANELA de concursos no motor e devolve
    as estatísticas básicas (freq, frias, quentes).

    Você manda APENAS os concursos da janela (ex.: últimos 25),
    o motor é recriado do zero para analisar só essa janela.

    Exemplo de body (JSON):

    {
      "concursos": [
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        [1,3,5,7,9,11,13,15,17,19,21,23,25,2,4]
      ]
    }
    """
    global motor

    try:
        # sempre começa com um motor novo para esta janela
        motor = novo_motor()

        # fgi_engine.FGIMotor.carregar devolve um objeto de estado
        estado = motor.carregar(req.concursos)

        # estado pode ser dict (versão antiga) ou dataclass AnaliseEstado
        if isinstance(estado, dict):
            total_concursos = estado.get("total_concursos")
            freq = estado.get("freq")
            frias = estado.get("frias")
            quentes = estado.get("quentes")
        else:
            # dataclass AnaliseEstado
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

    except ValueError as e:
        # erro de validação de entrada
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # se já for HTTPException, só repassa
        raise
    except Exception as e:
        # qualquer outro erro é 500
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
    Gera N jogos (FGIs) usando o motor com score, baseado
    na ÚLTIMA janela carregada em /carregar.

    Parâmetros:
      - n: quantidade de jogos (default 32)
      - min_frias: mínimo de dezenas frias por jogo
      - min_quentes: mínimo de dezenas quentes por jogo

    Exemplo de uso:

      GET /gerar_fino?n=32&min_frias=5&min_quentes=4
    """
    if motor is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum concurso carregado. Chame /carregar antes.",
        )

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
    Exporta os últimos 25 concursos carregados EM MEMÓRIA
    (ou menos, se tiver menos carregados) em formato CSV.

    - Baseado em motor.concursos (ordem em que foram carregados).
    - Separador: ponto e vírgula (;)

    Cabeçalho:
      concurso;d1;d2;...;d15
    """
    if motor is None or not getattr(motor, "concursos", None):
        raise HTTPException(
            status_code=400,
            detail="Nenhum concurso carregado. Chame /carregar antes.",
        )

    concursos = motor.concursos
    qtd = min(25, len(concursos))
    ultimos = concursos[-qtd:]

    buffer = StringIO()
    writer = csv.writer(buffer, delimiter=";")

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
