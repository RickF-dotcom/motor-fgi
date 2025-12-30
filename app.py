# -*- coding: utf-8 -*-
import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Motores
from fgi_engine import MotorFGI
from fgi_engine_v2 import MotorFGI_V2
from fgi_engine_v3 import MotorFGI_V3

# Núcleo
from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector

# Maturação
from maturacao import calcular_maturacao, score_maturacao_jogo


APP_NAME = "motor-fgi"
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =========================
# Helpers
# =========================
def _trace_short() -> str:
    tb = traceback.format_exc()
    lines = tb.splitlines()[-25:]
    return "\n".join(lines)


def _as_jogo(nums: List[int]) -> List[int]:
    if not isinstance(nums, list) or not nums:
        raise ValueError("jogo deve ser uma lista de inteiros")
    out = []
    for x in nums:
        if not isinstance(x, int):
            raise ValueError("jogo contém item não-inteiro")
        out.append(int(x))
    out.sort()
    return out


# =========================
# Singletons (cria 1x)
# =========================
try:
    GM = GrupoMilhoes()
    MOTOR_V1 = MotorFGI()
    MOTOR_V2 = MotorFGI_V2()
    MOTOR_V3 = MotorFGI_V3()
    REGIME = RegimeDetector()
except Exception as e:
    # Se falhar aqui, o Render cai. Então vamos forçar erro explícito.
    raise RuntimeError(f"Falha ao inicializar singletons: {e}\n{_trace_short()}")


# =========================
# Endpoints base
# =========================
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "render_service_id": SERVICE_ID,
        "build_commit": BUILD_COMMIT,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


# =========================
# Grupo de Milhões
# =========================
@app.get("/gm/status")
def gm_status() -> Dict[str, Any]:
    try:
        return GM.status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gm_status: {e}")


@app.post("/gm/is_drawn")
def gm_is_drawn(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        jogo = _as_jogo(payload.get("jogo", []))
        return {"jogo": jogo, "is_drawn": GM.is_drawn(jogo)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gm_is_drawn: {e}")


@app.post("/gm/sample_not_drawn")
def gm_sample_not_drawn(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload:
      n: int (qtd)
      max_candidates: int (opcional)
      timeout_sec: float (opcional)
    """
    try:
        n = int(payload.get("n", 10))
        max_candidates = payload.get("max_candidates", None)
        timeout_sec = float(payload.get("timeout_sec", 2.0))
        if max_candidates is not None:
            max_candidates = int(max_candidates)

        return GM.sample_not_drawn(
            n=n,
            max_candidates=max_candidates,
            timeout_sec=timeout_sec,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gm_sample_not_drawn: {e}")


# =========================
# Maturação
# =========================
@app.post("/maturacao")
def api_maturacao(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        jogo = _as_jogo(payload.get("jogo", []))
        mat = calcular_maturacao(jogo)
        score = score_maturacao_jogo(jogo)
        return {"jogo": jogo, "maturacao": mat, "score": score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"maturacao: {e}")


# =========================
# Motores FGI (exemplo mínimo)
# =========================
@app.post("/fgi/v1")
def fgi_v1(payload: Dict[str, Any]) -> Any:
    try:
        # Ajuste aqui conforme a assinatura real do seu MotorFGI
        return MOTOR_V1.process(payload)
    except AttributeError:
        raise HTTPException(status_code=500, detail="MotorFGI não tem método process(payload). Ajuste a chamada.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": _trace_short()})


@app.post("/fgi/v2")
def fgi_v2(payload: Dict[str, Any]) -> Any:
    try:
        return MOTOR_V2.process(payload)
    except AttributeError:
        raise HTTPException(status_code=500, detail="MotorFGI_V2 não tem método process(payload). Ajuste a chamada.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": _trace_short()})


@app.post("/fgi/v3")
def fgi_v3(payload: Dict[str, Any]) -> Any:
    try:
        return MOTOR_V3.process(payload)
    except AttributeError:
        raise HTTPException(status_code=500, detail="MotorFGI_V3 não tem método process(payload). Ajuste a chamada.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": _trace_short()})
