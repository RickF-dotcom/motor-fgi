# app.py
import os
import yaml
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from fgi_engine import MotorFGI              # v1 (congelado)
from fgi_engine_v2 import MotorFGI_V2        # v2 (SCF)
from fgi_engine_v3 import MotorFGI_V3        # v3 (DCR/Contraste)

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# =========================================================
# Build identity (Render)
# =========================================================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")


# =========================================================
# App
# =========================================================
app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="1.3.1",
)


# =========================================================
# Request Model (contrato real)
# - aceita campos extras sem quebrar (para Swagger / evoluções)
# =========================================================
class PrototiposRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # engine: v1 | v2 | v3
    engine: str = Field("v1", description="v1 | v2 | v3")

    # tamanho da combinação
    k: int = 15

    # resposta final
    top_n: int = 10

    # triagem inicial no grupo (performance)
    max_candidatos: int = 800

    # V2/V3 (podem vir ou não)
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None

    # redundância / diversidade
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None

    # V2: cap e alinhamento
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3 knobs (aceitos, mas hoje não são obrigatórios)
    alpha_contraste: Optional[float] = None
    beta_diversidade: Optional[float] = None
    gamma_base: Optional[float] = None
    jaccard_penalty_threshold: Optional[float] = None


# =========================================================
# Helpers: carregar YAML
# =========================================================
def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _contexto_lab() -> Dict[str, Any]:
    dna = _load_yaml("dna_last25.yaml")
    lab_cfg = _load_yaml("lab_config.yaml")
    return {
        "dna_last25": dna or {},
        "lab_config": lab_cfg or {},
    }


# =========================================================
# Helpers: normalizar payload para V2
# (V3 depende do contrato do V2; então V3 sempre executa V2 primeiro)
# =========================================================
def _normalize_v2_overrides(req: PrototiposRequest) -> Dict[str, Any]:
    # defaults estáveis
    overrides: Dict[str, Any] = {
        "top_n": int(req.top_n),
        "windows": req.windows if req.windows else [7, 10, 12, 15, 25],
        "dna_anchor_window": int(req.dna_anchor_window) if req.dna_anchor_window else 25,
        "redundancy_jaccard_threshold": float(req.redundancy_jaccard_threshold) if req.redundancy_jaccard_threshold is not None else 0.75,
        "redundancy_penalty": float(req.redundancy_penalty) if req.redundancy_penalty is not None else 0.25,
        "z_cap": float(req.z_cap) if req.z_cap is not None else 3.0,
        "align_temperature": float(req.align_temperature) if req.align_temperature is not None else 1.0,
    }

    # pesos opcionais
    if isinstance(req.pesos_windows, dict) and req.pesos_windows:
        overrides["pesos_windows"] = {str(k): float(v) for k, v in req.pesos_windows.items()}

    if isinstance(req.pesos_metricas, dict) and req.pesos_metricas:
        overrides["pesos_metricas"] = {str(k): float(v) for k, v in req.pesos_metricas.items()}

    return overrides


# =========================================================
# Core: extrair candidatos do Grupo de Milhões
# =========================================================
def _extract_candidates(k: int, max_candidatos: int) -> List[List[int]]:
    gm = GrupoMilhoes()
    # contrato esperado: gm deve fornecer candidatos possíveis (não sorteados)
    # tentativas de compatibilidade:
    if hasattr(gm, "sample"):
        return gm.sample(k=k, n=max_candidatos)
    if hasattr(gm, "amostrar"):
        return gm.amostrar(k=k, n=max_candidatos)
    if hasattr(gm, "candidatos"):
        return gm.candidatos(k=k, n=max_candidatos)
    if hasattr(gm, "gerar"):
        return gm.gerar(k=k, n=max_candidatos)
    raise RuntimeError("GrupoMilhoes nao expõe método de amostragem (sample/amostrar/candidatos/gerar)")


# =========================================================
# Endpoints básicos
# =========================================================
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
            "/health",
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
    }


@app.get("/lab/dna_last25")
def dna_last25():
    ctx = _contexto_lab()
    return ctx.get("dna_last25", {})


@app.get("/lab/regime_atual")
def regime_atual():
    ctx = _contexto_lab()
    det = RegimeDetector()
    # compat: detect / detectar
    if hasattr(det, "detect"):
        return det.detect(ctx)
    if hasattr(det, "detectar"):
        return det.detectar(ctx)
    return {"status": "regime_detector_sem_metodo"}


# =========================================================
# Engine runners
# =========================================================
def _v1_run(candidatos: List[List[int]], req: PrototiposRequest) -> Dict[str, Any]:
    motor = MotorFGI()
    # compat: gerar_prototipos / prototipos / run
    if hasattr(motor, "gerar_prototipos"):
        protos = motor.gerar_prototipos(candidatos, top_n=req.top_n)
    elif hasattr(motor, "prototipos"):
        protos = motor.prototipos(candidatos, top_n=req.top_n)
    elif hasattr(motor, "run"):
        protos = motor.run(candidatos, top_n=req.top_n)
    else:
        # fallback: só devolve as primeiras
        protos = candidatos[: req.top_n]

    return {"engine_used": "v1", "prototipos": protos}


def _v2_run(candidatos: List[List[int]], req: PrototiposRequest) -> Dict[str, Any]:
    motor = MotorFGI_V2()
    ctx = _contexto_lab()
    overrides = _normalize_v2_overrides(req)
    return motor.rerank(candidatos=candidatos, contexto_lab=ctx, overrides=overrides)


def _v3_run(candidatos: List[List[int]], req: PrototiposRequest) -> Dict[str, Any]:
    # V3 SEMPRE roda V2 primeiro (contrato)
    v2_out = _v2_run(candidatos, req)

    # v2_out pode ter "top" ou outra chave — blindagem
    candidatos_v2 = v2_out.get("top")
    if not isinstance(candidatos_v2, list) or not candidatos_v2:
        # tenta caminhos alternativos
        candidatos_v2 = v2_out.get("candidatos_v2") or v2_out.get("resultados") or []
    if not isinstance(candidatos_v2, list):
        candidatos_v2 = []

    motor = MotorFGI_V3()
    v3_out = motor.rerank(candidatos_v2=candidatos_v2, overrides={"top_n": int(req.top_n)})

    # anexa rastreabilidade mínima
    v3_out["trace"] = {
        "v2_engine_used": v2_out.get("engine_used", "v2"),
        "v2_schema": v2_out.get("schema_version", None),
        "input_candidates": len(candidatos),
        "v2_items": len(candidatos_v2),
    }
    return v3_out


# =========================================================
# Endpoint principal
# =========================================================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    try:
        candidatos = _extract_candidates(k=int(req.k), max_candidatos=int(req.max_candidatos))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"falha_extract_candidates: {e}")

    engine = (req.engine or "v1").lower().strip()

    try:
        if engine == "v1":
            return JSONResponse(content=_v1_run(candidatos, req))

        if engine == "v2":
            return JSONResponse(content=_v2_run(candidatos, req))

        if engine == "v3":
            return JSONResponse(content=_v3_run(candidatos, req))

        raise HTTPException(status_code=422, detail="engine inválido (use v1 | v2 | v3)")

    except HTTPException:
        raise
    except Exception as e:
        # aqui tem que virar 500 com detalhe (pra não ficar “Internal Server Error” cego)
        raise HTTPException(status_code=500, detail=f"erro_no_engine_{engine}: {e}")
