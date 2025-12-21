# app.py
import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fgi_engine import MotorFGI              # v1 (Filtro) - congelado
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional/SCF) - contrato + métricas
from fgi_engine_v3 import MotorFGI_V3        # v3 (Contraste/DCR) - depende de métricas do v2

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# ==============================
# Build identity (Render)
# ==============================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")


app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="1.3.1",
)


# ==============================
# REQUEST MODEL (contrato real)
# ==============================
class PrototiposRequest(BaseModel):
    # engine é obrigatório e SEM fallback silencioso
    engine: str = Field(..., description="v1 | v2 | v3")

    # tamanho da combinação (ex.: 15)
    k: int = 15

    top_n: int = 30
    max_candidatos: int = 3000

    # V2 (SCF)
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None

    # V3 (Contraste / DCR)
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
            "v2": "Direcional / SCF (metricas obrigatórias)",
            "v3": "Contraste / DCR (depende de metricas do v2)",
        },
        "contrato": {
            "v2_output_required": {
                "top": [
                    {
                        "sequencia": "List[int]",
                        "score": "float",
                        "detail": {"metricas": "Dict[str,float]", "scf_total": "float"},
                    }
                ]
            }
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
            try:
                out.append([int(x) for x in it["sequencia"]])
            except Exception:
                continue
        elif isinstance(it, (list, tuple)):
            try:
                out.append([int(x) for x in it])
            except Exception:
                continue
    return out


def _normalize_v2_result(resultado_v2: Any) -> Dict[str, Any]:
    """
    Garante que o V2 devolveu um dict com lista em `top`.
    Se vier como `prototipos`, converte.
    """
    if not isinstance(resultado_v2, dict):
        raise HTTPException(status_code=500, detail="V2 retornou payload inválido (não é dict).")

    if isinstance(resultado_v2.get("top"), list):
        return resultado_v2

    if isinstance(resultado_v2.get("prototipos"), list):
        resultado_v2["top"] = resultado_v2["prototipos"]
        return resultado_v2

    raise HTTPException(status_code=500, detail="V2 não retornou lista de candidatos em `top`/`prototipos`.")


def _extract_candidates_for_v3(resultado_v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    V3 precisa de candidatos com:
      - sequencia
      - detail.metricas (dict numérico)
      - detail.scf_total (float)
    """
    top = resultado_v2.get("top")
    if not isinstance(top, list) or not top:
        raise HTTPException(status_code=500, detail="V2 retornou `top` vazio; V3 não pode rodar.")

    normalized: List[Dict[str, Any]] = []

    for it in top:
        if not isinstance(it, dict) or "sequencia" not in it:
            continue

        seq = it.get("sequencia")
        if not isinstance(seq, list) or not seq:
            continue

        # detail
        detail = it.get("detail") if isinstance(it.get("detail"), dict) else {}

        metricas = detail.get("metricas")
        if metricas is None and isinstance(it.get("metricas"), dict):
            metricas = it.get("metricas")

        if not isinstance(metricas, dict) or not metricas:
            raise HTTPException(
                status_code=500,
                detail="V3 exige metricas do V2 (detail.metricas). Seu V2 não está expondo metricas.",
            )

        # scf_total obrigatório (cai no score se não existir, mas força float)
        scf_total = detail.get("scf_total", it.get("score", 0.0))

        normalized.append(
            {
                "sequencia": [int(x) for x in seq],
                "score": float(it.get("score", 0.0)),
                "detail": {
                    "metricas": metricas,
                    "scf_total": float(scf_total),
                },
            }
        )

    if not normalized:
        raise HTTPException(status_code=500, detail="Não consegui normalizar candidatos do V2 para o V3.")

    return normalized


def _build_contexto_lab() -> Dict[str, Any]:
    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()
    return {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso") if isinstance(regime, dict) else None,
    }


def _v1_run(k: int, max_candidatos: int) -> List[Any]:
    grupo = GrupoMilhoes()
    candidatos = grupo.get_candidatos(
        k=k,
        max_candidatos=max_candidatos,
        shuffle=True,
        seed=1337,
    )
    if not candidatos:
        raise HTTPException(status_code=500, detail="Grupo de Milhões vazio.")
    motor_v1 = MotorFGI()
    filtrados = motor_v1.gerar_prototipos(
        candidatos=candidatos,
        top_n=max_candidatos,
    )
    if not filtrados:
        raise HTTPException(status_code=500, detail="V1 retornou vazio.")
    return filtrados


def _v2_run(
    seqs_filtradas: List[List[int]],
    contexto_lab: Dict[str, Any],
    req: PrototiposRequest,
) -> Dict[str, Any]:
    overrides_v2: Dict[str, Any] = {
        "top_n": req.top_n,
        "max_candidatos": req.max_candidatos,
    }
    if req.windows is not None:
        overrides_v2["windows"] = req.windows
    if req.dna_anchor_window is not None:
        overrides_v2["dna_anchor_window"] = req.dna_anchor_window
    if req.pesos_windows is not None:
        overrides_v2["pesos_windows"] = req.pesos_windows
    if req.pesos_metricas is not None:
        overrides_v2["pesos_metricas"] = req.pesos_metricas
    if req.redundancy_jaccard_threshold is not None:
        overrides_v2["redundancy_jaccard_threshold"] = req.redundancy_jaccard_threshold
    if req.redundancy_penalty is not None:
        overrides_v2["redundancy_penalty"] = req.redundancy_penalty
    if req.z_cap is not None:
        overrides_v2["z_cap"] = req.z_cap
    if req.align_temperature is not None:
        overrides_v2["align_temperature"] = req.align_temperature

    motor_v2 = MotorFGI_V2()
    resultado_v2 = motor_v2.rerank(
        candidatos=seqs_filtradas,
        contexto_lab=contexto_lab,
        overrides=overrides_v2,
    )

    resultado_v2 = _normalize_v2_result(resultado_v2)
    resultado_v2["contexto_lab"] = contexto_lab
    return resultado_v2


def _v3_run(
    candidatos_v3: List[Dict[str, Any]],
    contexto_lab: Dict[str, Any],
    req: PrototiposRequest,
) -> Dict[str, Any]:
    motor_v3 = MotorFGI_V3()
    resultado_v3 = motor_v3.rank(
        candidatos=candidatos_v3,
        top_n=req.top_n,
        alpha_contraste=req.alpha_contraste,
        beta_diversidade=req.beta_diversidade,
        gamma_base=req.gamma_base,
        jaccard_penalty_threshold=req.jaccard_penalty_threshold,
    )
    if isinstance(resultado_v3, dict):
        resultado_v3["contexto_lab"] = contexto_lab
    return resultado_v3


# ==============================
# PROTÓTIPOS (pipeline obrigatório)
# ==============================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = (req.engine or "").lower().strip()
    if engine not in ("v1", "v2", "v3"):
        raise HTTPException(status_code=400, detail="engine deve ser 'v1', 'v2' ou 'v3'.")

    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k inválido.")

    if req.top_n <= 0:
        raise HTTPException(status_code=400, detail="top_n inválido.")

    if req.max_candidatos <= 0:
        raise HTTPException(status_code=400, detail="max_candidatos inválido.")

    try:
        # contexto
        contexto_lab = _build_contexto_lab()

        # v1 sempre roda (fonte das sequências)
        filtrados = _v1_run(k=req.k, max_candidatos=req.max_candidatos)

        if engine == "v1":
            return {
                "engine_used": "v1",
                "prototipos": filtrados[: req.top_n],
                "contexto_lab": contexto_lab,
            }

        # prepara input do v2
        seqs_filtradas = _extract_seq_list(filtrados)
        if not seqs_filtradas:
            raise HTTPException(status_code=500, detail="Falha ao extrair sequências após V1.")

        # v2 sempre roda quando engine != v1
        resultado_v2 = _v2_run(seqs_filtradas=seqs_filtradas, contexto_lab=contexto_lab, req=req)

        if engine == "v2":
            return resultado_v2

        # v3 depende de métricas do v2 (contrato)
        candidatos_v3 = _extract_candidates_for_v3(resultado_v2)
        resultado_v3 = _v3_run(candidatos_v3=candidatos_v3, contexto_lab=contexto_lab, req=req)

        return JSONResponse(content=resultado_v3)

    except HTTPException:
        raise
    except Exception as e:
        # 500 com causa explícita (sem “Internal Server Error” cego)
        raise HTTPException(status_code=500, detail=f"Falha interna em /prototipos: {type(e).__name__}: {e}")
```0
