
# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from fgi_engine import MotorFGI              # v1 (Filtro) – congelado
from fgi_engine_v2 import MotorFGI_V2        # v2 (SCF)
from fgi_engine_v3 import MotorFGI_V3        # v3 (DCR)

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# ==============================
# Build identity (Render)
# ==============================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")


app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="1.3.0",
)


# ==============================
# REQUEST MODEL (CONTRATO)
# ==============================
class PrototiposRequest(BaseModel):
    engine: str = Field(..., description="v1 | v2 | v3")

    k: int = 15
    top_n: int = 30
    max_candidatos: int = 3000

    # V2 – SCF
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3 – DCR
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75


# ==============================
# ROOT / HEALTH / STATUS
# ==============================
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
            "v2": "SCF direcional",
            "v3": "DCR contraste",
        },
    }


# ==============================
# DNA / REGIME
# ==============================
@app.get("/lab/dna_last25")
def dna_last25():
    return RegimeDetector().get_dna_last25()


@app.get("/lab/regime_atual")
def regime_atual():
    return RegimeDetector().detectar_regime()


# ==============================
# HELPERS
# ==============================
def _extract_seq_list(items: List[Any]) -> List[List[int]]:
    seqs: List[List[int]] = []
    for it in items:
        if isinstance(it, dict) and "sequencia" in it:
            seqs.append([int(x) for x in it["sequencia"]])
        elif isinstance(it, (list, tuple)):
            seqs.append([int(x) for x in it])
    return seqs


def _normalize_v2_payload(resultado_v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = resultado_v2.get("top")
    if not isinstance(top, list) or not top:
        raise HTTPException(
            status_code=422,
            detail="V3 exige retorno do V2 com lista não vazia em 'top'.",
        )

    out: List[Dict[str, Any]] = []
    for it in top:
        if not isinstance(it, dict):
            continue

        seq = it.get("sequencia")
        detail = it.get("detail", {})
        metricas = detail.get("metricas")
        scf_total = detail.get("scf_total", it.get("score"))

        if not isinstance(seq, list) or not isinstance(metricas, dict):
            raise HTTPException(
                status_code=422,
                detail="Contrato V3 violado: item do V2 sem sequencia ou metricas.",
            )

        out.append(
            {
                "sequencia": [int(x) for x in seq],
                "score": float(it.get("score", scf_total or 0.0)),
                "detail": {
                    "metricas": metricas,
                    "scf_total": float(scf_total or 0.0),
                },
            }
        )

    if not out:
        raise HTTPException(
            status_code=422,
            detail="Normalização V2→V3 resultou vazia.",
        )

    return out


# ==============================
# PROTÓTIPOS
# ==============================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = req.engine.lower().strip()
    if engine not in ("v1", "v2", "v3"):
        raise HTTPException(status_code=400, detail="engine inválido")

    if req.k <= 0 or req.top_n <= 0 or req.max_candidatos <= 0:
        raise HTTPException(status_code=400, detail="Parâmetros numéricos inválidos")

    detector = RegimeDetector()
    contexto_lab = {
        "dna_last25": detector.get_dna_last25(),
        "regime": detector.detectar_regime(),
    }

    grupo = GrupoMilhoes()
    candidatos = grupo.get_candidatos(
        k=req.k,
        max_candidatos=req.max_candidatos,
        shuffle=True,
        seed=1337,
    )

    if not candidatos:
        raise HTTPException(status_code=500, detail="Grupo de Milhões vazio")

    # -------- v1 --------
    v1 = MotorFGI()
    filtrados = v1.gerar_prototipos(
        candidatos=candidatos,
        top_n=req.max_candidatos,
    )

    if engine == "v1":
        return {
            "engine_used": "v1",
            "prototipos": filtrados[: req.top_n],
            "contexto_lab": contexto_lab,
        }

    seqs = _extract_seq_list(filtrados)
    if not seqs:
        raise HTTPException(status_code=500, detail="Falha ao extrair sequências do v1")

    # -------- v2 --------
    overrides_v2: Dict[str, Any] = {
        "top_n": req.top_n,
        "max_candidatos": req.max_candidatos,
    }

    for campo in (
        "windows",
        "dna_anchor_window",
        "pesos_windows",
        "pesos_metricas",
        "redundancy_jaccard_threshold",
        "redundancy_penalty",
        "z_cap",
        "align_temperature",
    ):
        val = getattr(req, campo)
        if val is not None:
            overrides_v2[campo] = val

    v2 = MotorFGI_V2()
    resultado_v2 = v2.rerank(
        candidatos=seqs,
        contexto_lab=contexto_lab,
        overrides=overrides_v2,
    )

    if engine == "v2":
        resultado_v2["contexto_lab"] = contexto_lab
        return resultado_v2

    # -------- v3 --------
    candidatos_v3 = _normalize_v2_payload(resultado_v2)

    v3 = MotorFGI_V3()
    resultado_v3 = v3.rank(
        candidatos=candidatos_v3,
        top_n=req.top_n,
        alpha_contraste=req.alpha_contraste,
        beta_diversidade=req.beta_diversidade,
        gamma_base=req.gamma_base,
        jaccard_penalty_threshold=req.jaccard_penalty_threshold,
    )

    resultado_v3["contexto_lab"] = contexto_lab
    return JSONResponse(content=resultado_v3)
