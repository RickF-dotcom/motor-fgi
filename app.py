from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List

from fgi_engine import MotorFGI
from grupo_de_milhoes import GrupoMilhoes

# IMPORTA O MÓDULO DE MATURAÇÃO
from maturacao import calcular_maturacao, score_maturacao_jogo


# ==============================
#   MODELO DO REQUEST /carregar
# ==============================
class CarregarRequest(BaseModel):
    concursos: List[List[int]]


# ======================
#   APLICAÇÃO FASTAPI
# ======================
app = FastAPI(
    title="Motor FGI",
    version="1.0.0",
    description="API do motor de FGIs com score + grupo de milhões + maturação."
)

motor = None
HISTORICO = []   # Guarda a sequência carregada p/ maturação
grupo = GrupoMilhoes()


# ============
#   ROOT
# ============
@app.get("/")
async def root():
    return {"status": "online", "msg": "Motor FGI ativo"}


# =======================
#   POST /carregar
# =======================
@app.post("/carregar")
async def carregar(req: CarregarRequest):
    global motor, HISTORICO

    if len(req.concursos) == 0:
        raise HTTPException(status_code=400, detail="Nenhum concurso enviado.")

    motor = MotorFGI(req.concursos)
    HISTORICO = req.concursos[:]  # para maturação

    freq = motor.freq
    frias = motor.frias
    quentes = motor.quentes

    return {
        "status": "ok",
        "total_concursos": len(req.concursos),
        "freq": freq,
        "frias": frias,
        "quentes": quentes
    }


# ======================================
#   GET /gerar_fino
# ======================================
@app.get("/gerar_fino")
async def gerar_fino(
    n: int = 32,
    min_frias: int = 3,
    min_quentes: int = 4,
):
    """
    Gera N FGIs usando o score tradicional...
    + ORDENA tudo pela MATURAÇÃO.
    """
    global motor, HISTORICO

    if motor is None:
        raise HTTPException(status_code=400, detail="Use /carregar antes.")

    # 1) FGIs base do motor
    jogos = motor.gerar_fino(
        n=n,
        min_frias=min_frias,
        min_quentes=min_quentes
    )

    # ===========================
    # 2) MATRIZ DE MATURAÇÃO
    # ===========================
    try:
        mapa_maturacao = calcular_maturacao(
            HISTORICO,
            janela_longa=25,
            janela_curta=10
        )
    except Exception as e:
        print("Erro na maturação:", e)
        mapa_maturacao = {}

    # ===========================
    # 3) SCORE DE MATURAÇÃO
    # ===========================
    jogos_com_score = []
    for j in jogos:
        s = score_maturacao_jogo(j, mapa_maturacao)
        jogos_com_score.append((s, j))

    # Ordena por maturação (maior score primeiro)
    jogos_ordenados = [j for (s, j) in sorted(jogos_com_score, key=lambda x: x[0], reverse=True)]

    return {
        "status": "ok",
        "quantidade": len(jogos_ordenados),
        "jogos": jogos_ordenados
    }


# ======================================
#   GET /ultimos25.csv  (para Excel)
# ======================================
@app.get("/ultimos25.csv")
async def baixar_csv():
    if motor is None:
        return PlainTextResponse("ERRO: carregue concursos primeiro.")

    linhas = []
    for conc in HISTORICO[-25:]:
        linha = ",".join(str(x) for x in conc)
        linhas.append(linha)

    conteudo = "\n".join(linhas)
    return PlainTextResponse(conteudo, media_type="text/csv")
