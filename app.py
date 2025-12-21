# app.py
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fgi_engine import MotorFGI              # v1 (Filtro) - congelado
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional/SCF) - contrato: detail.metricas + detail.scf_total
from fgi_engine_v3 import MotorFGI_V3        # v3 (Contraste/DCR)

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
# REQUEST MODEL (CONTRATO REAL)
# ==============================
class PrototiposRequest(BaseModel):
    # obrigatório: sem fallback silencioso
    engine: str = Field(..., description="v1 | v2 | v3")

    # tamanho da combinação (ex.: 15)
    k: int = 15

    top_n: int = 30
    max_candidatos: int = 3000

    # V2 — parâmetros SCF
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3 — parâmetros DCR
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
        "endpoints": ["/lab/status", "/lab/dna_last25", "/lab/regime_atual", "/prototipos"],
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


# ==============================
# DNA / REGIME
# ==============================
@app.get("/lab/dna_last25")
def dna_last25():
    detector = RegimeDetector()
    return detector.get_dna_last25()


@app.get("/lab/regime_atual")
def regime_atual():
    detector = RegimeDetector()
    return detector.detectar_regime()


# ==============================
# HELPERS
# ==============================
def _extract_seq_list(items: List[Any]) -> List[List[int]]:
    """
    Aceita:
      - lista de listas (sequências)
      - lista de dicts contendo "sequencia"
    Retorna lista de listas de int.
    """
    out: List[List[int]] = []
    for it in items:
        if isinstance(it, dict) and "sequencia" in it:
            out.append([int(x) for x in it["sequencia"]])
        elif isinstance(it, (list, tuple)):
            out.append([int(x) for x in it])
    return out


def _normalize_v2_payload(req: PrototiposRequest) -> Dict[str, Any]:
    """
    Monta overrides V2 apenas com o que veio no request.
    """
    overrides: Dict[str, Any] = {
        "top_n": int(req.top_n),
        "max_candidatos": int(req.max_candidatos),
    }
    if req.windows is not None:
        overrides["windows"] = req.windows
    if req.dna_anchor_window is not None:
        overrides["dna_anchor_window"] = int(req.dna_anchor_window)
    if req.pesos_windows is not None:
        overrides["pesos_windows"] = req.pesos_windows
    if req.pesos_metricas is not None:
        overrides["pesos_metricas"] = req.pesos_metricas
    if req.redundancy_jaccard_threshold is not None:
        overrides["redundancy_jaccard_threshold"] = float(req.redundancy_jaccard_threshold)
    if req.redundancy_penalty is not None:
        overrides["redundancy_penalty"] = float(req.redundancy_penalty)
    if req.z_cap is not None:
        overrides["z_cap"] = float(req.z_cap)
    if req.align_temperature is not None:
        overrides["align_temperature"] = float(req.align_temperature)
    return overrides


def _extract_candidates_for_v3(v2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    V3 precisa de candidatos com:
      - sequencia
      - detail.metricas (dict numérico)
      - detail.scf_total (float)
    """
    top = None
    if isinstance(v2_result, dict):
        if isinstance(v2_result.get("top"), list):
            top = v2_result["top"]
        elif isinstance(v2_result.get("prototipos"), list):
            top = v2_result["prototipos"]

    if not isinstance(top, list) or not top:
        raise HTTPException(status_code=500, detail="V2 não retornou lista de candidatos (top/prototipos).")

    normalized: List[Dict[str, Any]] = []
    for it in top:
        if not isinstance(it, dict) or "sequencia" not in it:
            continue

        detail = it.get("detail") if isinstance(it.get("detail"), dict) else {}
        metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else None
        scf_total = detail.get("scf_total", it.get("score", 0.0))

        # fallback adicional (se alguém mudou contrato do v2)
        if metricas is None:
            metricas = it.get("metricas") if isinstance(it.get("metricas"), dict) else None

        if metricas is None:
            raise HTTPException(
                status_code=500,
                detail="V3 exige metricas do V2 (detail.metricas). V2 não está expondo metricas no retorno.",
            )

        normalized.append(
            {
                "sequencia": [int(x) for x in it["sequencia"]],
                "score": float(it.get("score", 0.0)),
                "detail": {"scf_total": float(scf_total), "metricas": metricas},
            }
        )

    if not normalized:
        raise HTTPException(status_code=500, detail="Não consegui normalizar candidatos do V2 para o V3.")
    return normalized


def _v1_run(candidatos: List[List[int]], req: PrototiposRequest) -> List[Any]:
    motor_v1 = MotorFGI()
    filtrados = motor_v1.gerar_prototipos(
        candidatos=candidatos,
        top_n=int(req.max_candidatos),
    )
    if not isinstance(filtrados, list):
        raise HTTPException(status_code=500, detail="V1 retornou formato inválido.")
    return filtrados


def _v2_run(seqs_filtradas: List[List[int]], contexto_lab: Dict[str, Any], req: PrototiposRequest) -> Dict[str, Any]:
    motor_v2 = MotorFGI_V2()
    overrides_v2 = _normalize_v2_payload(req)
    resultado_v2 = motor_v2.rerank(
        candidatos=seqs_filtradas,
        contexto_lab=contexto_lab,
        overrides=overrides_v2,
    )
    if not isinstance(resultado_v2, dict):
        raise HTTPException(status_code=500, detail="V2 retornou formato inválido (esperado dict).")
    return resultado_v2


def _v3_run(v2_result: Dict[str, Any], req: PrototiposRequest) -> Dict[str, Any]:
    candidatos_v3 = _extract_candidates_for_v3(v2_result)

    motor_v3 = MotorFGI_V3()
    resultado_v3 = motor_v3.rank(
        candidatos=candidatos_v3,
        top_n=int(req.top_n),
        alpha_contraste=float(req.alpha_contraste),
        beta_diversidade=float(req.beta_diversidade),
        gamma_base=float(req.gamma_base),
        jaccard_penalty_threshold=float(req.jaccard_penalty_threshold),
    )
    if not isinstance(resultado_v3, dict):
        raise HTTPException(status_code=500, detail="V3 retornou formato inválido (esperado dict).")
    return resultado_v3


# ==============================
# PROTÓTIPOS (PIPELINE FORÇADO)
# ==============================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = (req.engine or "").lower().strip()
    if engine not in ("v1", "v2", "v3"):
        raise HTTPException(status_code=400, detail="engine deve ser 'v1', 'v2' ou 'v3'")

    if int(req.k) <= 0:
        raise HTTPException(status_code=400, detail="k inválido")
    if int(req.top_n) <= 0:
        raise HTTPException(status_code=400, detail="top_n inválido")
    if int(req.max_candidatos) <= 0:
        raise HTTPException(status_code=400, detail="max_candidatos inválido")

    # Contexto do laboratório
    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()

    contexto_lab = {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso") if isinstance(regime, dict) else None,
    }

    # Grupo de Milhões (amostragem)
    grupo = GrupoMilhoes()
    try:
        candidatos = grupo.get_candidatos(
            k=int(req.k),
            max_candidatos=int(req.max_candidatos),
            shuffle=True,
            seed=1337,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha no Grupo de Milhões: {e}")

    if not candidatos:
        raise HTTPException(status_code=500, detail="Grupo de Milhões vazio")

    # =========================
    # PIPELINE SEMPRE:
    #   v1 -> (se precisar) v2 -> (se precisar) v3
    # =========================
    filtrados = _v1_run(candidatos=candidatos, req=req)

    if engine == "v1":
        return {
            "engine_used": "v1",
            "prototipos": filtrados[: int(req.top_n)],
            "contexto_lab": contexto_lab,
        }

    seqs_filtradas = _extract_seq_list(filtrados)
    if not seqs_filtradas:
        raise HTTPException(status_code=500, detail="V1 não produziu sequências válidas para alimentar o V2.")

    resultado_v2 = _v2_run(seqs_filtradas=seqs_filtradas, contexto_lab=contexto_lab, req=req)

    if engine == "v2":
        resultado_v2["contexto_lab"] = contexto_lab
        return JSONResponse(content=resultado_v2)

    # engine == v3
    # REGRA DURA: V3 SEMPRE roda em cima do resultado do V2 (não aceita execução direta sem métricas)
    resultado_v3 = _v3_run(v2_result=resultado_v2, req=req)
    resultado_v3["contexto_lab"] = contexto_lab
    return JSONResponse(content=resultado_v3)
```0
