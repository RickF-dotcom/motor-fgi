# app.py
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fgi_engine import MotorFGI                 # v1
from fgi_engine_v2 import MotorFGI_V2           # v2
from fgi_engine_v3 import MotorFGI_V3           # v3

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# =========================
# Build identity (Render)
# =========================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")

# Liga debug de erro detalhado (sem precisar mexer em código depois)
DEBUG_ERRORS = os.environ.get("DEBUG_ERRORS", "1") == "1"

BASE_DIR = Path(__file__).resolve().parent
DNA_FILE = BASE_DIR / "dna_last25.yaml"
LAB_CONFIG_FILE = BASE_DIR / "lab_config.yaml"


# =========================
# Helpers
# =========================
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.name} (path={path})")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido: {path.name} não retornou dict.")
    return data


def _trace_short() -> str:
    tb = traceback.format_exc()
    lines = tb.strip().splitlines()
    # pega só o final (o que importa)
    return "\n".join(lines[-25:])


def _fail(where: str, e: Exception) -> None:
    # Loga no Render (para ver nos logs)
    print(f"[ERROR] {where}: {repr(e)}")
    if DEBUG_ERRORS:
        print(_trace_short())
    # Sobe como HTTPException com detalhe útil
    detail = {
        "where": where,
        "error": str(e),
    }
    if DEBUG_ERRORS:
        detail["trace"] = _trace_short()
    raise HTTPException(status_code=500, detail=detail)


# =========================
# FastAPI
# =========================
app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="1.3.0",
)


# =========================
# Request Model (contrato)
# =========================
class PrototiposRequest(BaseModel):
    # engine obrigatório
    engine: str = Field(..., description="v1 | v2 | v3")

    # tamanho da combinação (default 15)
    k: int = 15

    # quantos retornos finais
    top_n: int = 10

    # limita candidatos gerados/amostrados
    max_candidatos: int = 800

    # V2 parâmetros SCF (opcionais)
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancia_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3 parâmetros DCR (defaults no request)
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75


# =========================
# Health / Root / Status
# =========================
@app.get("/")
def root():
    return {
        "ok": True,
        "laboratorio": "ATHENA LABORATORIO PMF",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": [
            "/lab/status",
            "/lab/dna_last25",
            "/lab/regime_atual",
            "/prototipos",
        ],
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/lab/status")
def lab_status():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
        "motores": {
            "v1": "Pass-through (congelado)",
            "v2": "Direcional / SCF",
            "v3": "Contraste / DCR",
        },
        "debug_errors": DEBUG_ERRORS,
    }


@app.get("/lab/dna_last25")
def dna_last25():
    try:
        dna = _read_yaml(DNA_FILE)
        return {"ok": True, "dna_last25": dna}
    except Exception as e:
        _fail("GET /lab/dna_last25", e)


@app.get("/lab/regime_atual")
def regime_atual():
    try:
        dna = _read_yaml(DNA_FILE)
        detector = RegimeDetector()
        regime = detector.detectar_regime(dna)
        return {"ok": True, "regime": regime}
    except Exception as e:
        _fail("GET /lab/regime_atual", e)


# =========================
# Core: gerar protótipos
# =========================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    try:
        # contexto do laboratório (sempre carregado do dna_last25.yaml)
        dna = _read_yaml(DNA_FILE)
        contexto_lab: Dict[str, Any] = {"dna_last25": dna}

        # carrega config opcional do lab (se existir)
        if LAB_CONFIG_FILE.exists():
            contexto_lab["lab_config"] = _read_yaml(LAB_CONFIG_FILE)

        # gera candidatos (Grupo de Milhões)
        grupo = GrupoMilhoes()
        candidatos: List[List[int]] = grupo.gerar_candidatos(
            k=req.k,
            max_candidatos=req.max_candidatos,
        )

        engine = (req.engine or "").strip().lower()

        # v1
        if engine == "v1":
            motor = MotorFGI()
            protos = motor.gerar_prototipos(
                candidatos=candidatos,
                top_n=req.top_n,
                contexto_lab=contexto_lab,
            )
            return {"engine_used": "v1", "prototipos": protos, "contexto_lab": contexto_lab}

        # v2
        if engine == "v2":
            motor = MotorFGI_V2()
            overrides: Dict[str, Any] = {
                "top_n": req.top_n,
                "windows": req.windows,
                "dna_anchor_window": req.dna_anchor_window,
                "pesos_windows": req.pesos_windows,
                "pesos_metricas": req.pesos_metricas,
                "redundancia_jaccard_threshold": req.redundancia_jaccard_threshold,
                "redundancy_penalty": req.redundancy_penalty,
                "z_cap": req.z_cap,
                "align_temperature": req.align_temperature,
            }
            # remove Nones
            overrides = {k: v for k, v in overrides.items() if v is not None}

            out = motor.rerank(
                candidatos=candidatos,
                contexto_lab=contexto_lab,
                overrides=overrides,
            )
            return {"engine_used": "v2", **out, "contexto_lab": contexto_lab}

        # v3
        if engine == "v3":
            motor = MotorFGI_V3()
            overrides_v3: Dict[str, Any] = {
                "top_n": req.top_n,
                "alpha_contraste": req.alpha_contraste,
                "beta_diversidade": req.beta_diversidade,
                "gamma_base": req.gamma_base,
                "jaccard_penalty_threshold": req.jaccard_penalty_threshold,
            }

            out = motor.rerank(
                candidatos=candidatos,
                contexto_lab=contexto_lab,
                overrides=overrides_v3,
            )
            return {"engine_used": "v3", **out, "contexto_lab": contexto_lab}

        raise HTTPException(status_code=422, detail=f"engine inválido: {req.engine}. Use v1|v2|v3")

    except HTTPException:
        raise
    except Exception as e:
        _fail("POST /prototipos", e)


# =========================
# Global exception handler (backup)
# =========================
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    # evita 500 “mudo”
    print(f"[UNHANDLED] {request.method} {request.url}: {repr(exc)}")
    if DEBUG_ERRORS:
        print(_trace_short())
    payload = {"detail": "Internal Server Error"}
    if DEBUG_ERRORS:
        payload = {
            "detail": "Internal Server Error",
            "error": str(exc),
            "trace": _trace_short(),
        }
    return JSONResponse(status_code=500, content=payload)
