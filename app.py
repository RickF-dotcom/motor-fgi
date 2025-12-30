from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Optional

# Baseline do Grupo de Milhões (acabou de ser commitado)
from grupo_de_milhoes import GrupoMilhoes

app = FastAPI(title="motor-fgi-baseline", version="0.1.0")

# Instância global (segura)
GM = GrupoMilhoes(k=15, min_n=1, max_n=25, seed=123)


@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "service": "motor-fgi-baseline"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/grupo/status")
def grupo_status() -> Dict[str, Any]:
    return {"status": "ok", "grupo": GM.status()}


@app.post("/grupo/add-drawn")
def grupo_add_drawn(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload esperado:
      {"jogo": [1,2,3,...,15]}
    """
    jogo = payload.get("jogo")
    if not isinstance(jogo, list):
        return JSONResponse(status_code=400, content={"status": "error", "detail": "payload precisa ter 'jogo' como lista"})
    try:
        GM.add_drawn(jogo)
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(e)})
    return {"status": "ok", "grupo": GM.status()}


@app.get("/grupo/sample")
def grupo_sample(n: int = 10) -> Dict[str, Any]:
    """
    Gera jogos NAO sorteados (aleatorio por rejeicao).
    """
    try:
        data = GM.generate_not_drawn(n=n)
        return {"status": "ok", **data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
