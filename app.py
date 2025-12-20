# app.py
import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from fgi_engine import MotorFGI              # v1 (Filtro) - mantém
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional/SCF)
from fgi_engine_v3 import MotorFGI_V3        # v3 (Contraste/DCR)

from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector


# ==============================
# Build identity (Render)
# ==============================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")

app = FastAPI(title="ATHENA LABORATORIO PMF")


# ==============================
# REQUESTS
# ==============================
class PrototiposRequest(BaseModel):
    engine: str = "v1"  # v1 | v2  (v3 sai daqui)

    # tamanho da combinação (ex.: 15)
    k: int = 15

    top_n: int = 30
    max_candidatos: int = 3000

    # V2
    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None
    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None


class V3ContrasteRequest(BaseModel):
    # knobs do V3
    top_n: int = 10
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75

    # ENTRADA OBRIGATÓRIA: JSON inteiro retornado pelo V2
    input_metrics: Dict[str, Any]


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
            "/v3/contraste",
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
            "v1": "Filtro / baseline (congelado)",
            "v2": "Direcional / SCF",
            "v3": "Contraste / DCR (endpoint dedicado /v3/contraste)",
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


def _extract_candidates_for_v3(v2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    V3 precisa de candidatos com:
      - sequencia
      - detail.metricas (dict numérico)
      - detail.scf_total (ou score)
    Aceita v2_result contendo 'top' ou 'prototipos'.
    """
    top = None
    if isinstance(v2_result, dict):
        if isinstance(v2_result.get("top"), list):
            top = v2_result["top"]
        elif isinstance(v2_result.get("prototipos"), list):
            top = v2_result["prototipos"]

    if not isinstance(top, list) or not top:
        raise HTTPException(status_code=422, detail="input_metrics não contém lista válida (esperado: top).")

    normalized: List[Dict[str, Any]] = []
    for it in top:
        if not isinstance(it, dict) or "sequencia" not in it:
            continue

        detail = it.get("detail") if isinstance(it.get("detail"), dict) else {}
        metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else None
        scf_total = detail.get("scf_total", it.get("score", 0.0))

        if metricas is None:
            metricas = it.get("metricas") if isinstance(it.get("metricas"), dict) else None

        if metricas is None:
            raise HTTPException(
                status_code=422,
                detail="V3 exige metricas do V2: cada item precisa ter detail.metricas (dict).",
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
        raise HTTPException(status_code=422, detail="Não consegui normalizar input_metrics.top para o V3.")
    return normalized


def _contexto_lab() -> Dict[str, Any]:
    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()
    return {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso"),
        "build_commit": BUILD_COMMIT,
    }


# ==============================
# PROTÓTIPOS (v1/v2 APENAS)
# ==============================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = (req.engine or "v1").lower().strip()

    # trava v3 aqui (agora é endpoint dedicado)
    if engine == "v3":
        raise HTTPException(status_code=422, detail="V3 não roda em /prototipos. Use /v3/contraste.")

    if engine not in ("v1", "v2"):
        raise HTTPException(status_code=400, detail="engine deve ser 'v1' ou 'v2'")

    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k inválido")

    contexto_lab = _contexto_lab()

    # Grupo de Milhões (amostragem)
    grupo = GrupoMilhoes()
    try:
        candidatos = grupo.get_candidatos(
            k=req.k,
            max_candidatos=req.max_candidatos,
            shuffle=True,
            seed=1337,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha no Grupo de Milhões: {e}")

    if not candidatos:
        raise HTTPException(status_code=500, detail="Grupo de Milhões vazio")

    # =========================
    # v1 — FILTRO (mantém)
    # =========================
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

    # =========================
    # v2 — SCF (re-ranking)
    # =========================
    seqs_filtradas = _extract_seq_list(filtrados)

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

    if isinstance(resultado_v2, dict):
        resultado_v2["contexto_lab"] = contexto_lab

    return JSONResponse(content=resultado_v2)


# ==============================
# V3 (endpoint dedicado)
# ==============================
@app.post("/v3/contraste")
def v3_contraste(req: V3ContrasteRequest = Body(...)):
    """
    V3 roda APENAS a partir do output do V2 (input_metrics).
    Isso elimina o looping "V3 exige métricas do V2".
    """
    engine_used = str(req.input_metrics.get("engine_used", "")).lower().strip()
    if engine_used != "v2":
        raise HTTPException(status_code=422, detail="V3 exige métricas do V2 (input_metrics.engine_used='v2').")

    # normaliza input do V2 para o formato do V3
    candidatos_v3 = _extract_candidates_for_v3(req.input_metrics)

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
        resultado_v3["contexto_lab"] = _contexto_lab()

    return JSONResponse(content=resultado_v3)
