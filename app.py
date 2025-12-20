# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from fgi_engine import MotorFGI              # v1 (Filtro) - congelado
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional / SCF)
from fgi_engine_v3 import MotorFGI_V3        # v3 (Contraste / DCR)

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# =========================================================
# Build identity (Render)
# =========================================================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")


app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="1.2.0",
)


# =========================================================
# REQUEST MODEL (CONTRATO REAL)
# =========================================================
class PrototiposRequest(BaseModel):
    # engine é obrigatório — sem fallback silencioso
    engine: str = Field(..., description="v1 | v2 | v3")

    # tamanho da combinação
    k: int = 15

    top_n: int = 30
    max_candidatos: int = 3000

    # -------------------------
    # V2 — parâmetros SCF
    # -------------------------
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # -------------------------
    # V3 — parâmetros DCR
    # -------------------------
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75


# =========================================================
# ROOT / HEALTH / STATUS
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
            "v1": "Filtro (congelado)",
            "v2": "Direcional / SCF",
            "v3": "Contraste / DCR (depende do v2)",
        },
    }


# =========================================================
# DNA / REGIME
# =========================================================
@app.get("/lab/dna_last25")
def dna_last25():
    detector = RegimeDetector()
    return detector.get_dna_last25()


@app.get("/lab/regime_atual")
def regime_atual():
    detector = RegimeDetector()
    return detector.detectar_regime()


# =========================================================
# HELPERS
# =========================================================
def _extract_seq_list(items: List[Any]) -> List[List[int]]:
    out: List[List[int]] = []
    for it in items:
        if isinstance(it, dict) and "sequencia" in it:
            out.append([int(x) for x in it["sequencia"]])
        elif isinstance(it, (list, tuple)):
            out.append([int(x) for x in it])
    return out


def _extract_candidates_for_v3(v2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = None

    if isinstance(v2_result, dict):
        if isinstance(v2_result.get("top"), list):
            top = v2_result["top"]
        elif isinstance(v2_result.get("prototipos"), list):
            top = v2_result["prototipos"]

    if not isinstance(top, list) or not top:
        raise HTTPException(
            status_code=422,
            detail="V3 exige saída válida do V2 (campo 'top' ou 'prototipos')."
        )

    normalized: List[Dict[str, Any]] = []

    for it in top:
        if not isinstance(it, dict) or "sequencia" not in it:
            continue

        detail = it.get("detail", {})
        metricas = detail.get("metricas")
        scf_total = detail.get("scf_total", it.get("score"))

        if not isinstance(metricas, dict):
            raise HTTPException(
                status_code=422,
                detail="V3 exige metricas vindas do V2 (detail.metricas)."
            )

        normalized.append(
            {
                "sequencia": it["sequencia"],
                "score": float(it.get("score", 0.0)),
                "detail": {
                    "scf_total": float(scf_total),
                    "metricas": metricas,
                },
            }
        )

    if not normalized:
        raise HTTPException(
            status_code=422,
            detail="Não foi possível normalizar candidatos do V2 para o V3."
        )

    return normalized


# =========================================================
# PROTÓTIPOS — PIPELINE OFICIAL
# =========================================================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = req.engine.lower().strip()

    if engine not in ("v1", "v2", "v3"):
        raise HTTPException(
            status_code=400,
            detail="engine deve ser 'v1', 'v2' ou 'v3'",
        )

    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k inválido")

    # -----------------------------------------------------
    # Contexto do laboratório
    # -----------------------------------------------------
    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()

    contexto_lab = {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso"),
        "build_commit": BUILD_COMMIT,
    }

    # -----------------------------------------------------
    # Grupo de Milhões
    # -----------------------------------------------------
    grupo = GrupoMilhoes()
    candidatos = grupo.get_candidatos(
        k=req.k,
        max_candidatos=req.max_candidatos,
        shuffle=True,
        seed=1337,
    )

    if not candidatos:
        raise HTTPException(
            status_code=500,
            detail="Grupo de Milhões vazio",
        )

    # -----------------------------------------------------
    # v1 — FILTRO
    # -----------------------------------------------------
    motor_v1 = MotorFGI()
    filtrados = motor_v1.gerar_prototipos(
        candidatos=candidatos,
        top_n=req.max_candidatos,
    )

    if engine == "v1":
        return {
            "engine_used": "v1",
            "prototipos": filtrados[: req.top_n],
            "contexto_lab": contexto_lab,
        }

    # -----------------------------------------------------
    # v2 — SCF
    # -----------------------------------------------------
    seqs_filtradas = _extract_seq_list(filtrados)

    overrides_v2: Dict[str, Any] = {
        "top_n": req.top_n,
        "max_candidatos": req.max_candidatos,
    }

    for campo in [
        "windows",
        "dna_anchor_window",
        "pesos_windows",
        "pesos_metricas",
        "redundancy_jaccard_threshold",
        "redundancy_penalty",
        "z_cap",
        "align_temperature",
    ]:
        valor = getattr(req, campo)
        if valor is not None:
            overrides_v2[campo] = valor

    motor_v2 = MotorFGI_V2()
    resultado_v2 = motor_v2.rerank(
        candidatos=seqs_filtradas,
        contexto_lab=contexto_lab,
        overrides=overrides_v2,
    )

    if engine == "v2":
        resultado_v2["contexto_lab"] = contexto_lab
        return resultado_v2

    # -----------------------------------------------------
    # v3 — CONTRASTE (DCR)
    # -----------------------------------------------------
    candidatos_v3 = _extract_candidates_for_v3(resultado_v2)

    motor_v3 = MotorFGI_V3()
    resultado_v3 = motor_v3.rank(
        candidatos=candidatos_v3,
        top_n=req.top_n,
        alpha_contraste=req.alpha_contraste,
        beta_diversidade=req.beta_diversidade,
        gamma_base=req.gamma_base,
        jaccard_penalty_threshold=req.jaccard_penalty_threshold,
    )

    resultado_v3["contexto_lab"] = contexto_lab
    return JSONResponse(content=resultado_v3)
