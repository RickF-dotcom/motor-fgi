
# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from fgi_engine import MotorFGI              # v1 (Filtro) - congelado
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional/SCF) - garante contrato (metricas + scf_total)
from fgi_engine_v3 import MotorFGI_V3        # v3 (Contraste/DCR) - rank final

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
    # engine é obrigatório e SEM fallback silencioso (contrato)
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


def _normalize_v2_payload(resultado_v2: Any) -> List[Dict[str, Any]]:
    """
    Normaliza o retorno do V2 para lista de candidatos.
    Aceita chaves: "top" ou "prototipos".
    """
    top = None
    if isinstance(resultado_v2, dict):
        if isinstance(resultado_v2.get("top"), list):
            top = resultado_v2["top"]
        elif isinstance(resultado_v2.get("prototipos"), list):
            top = resultado_v2["prototipos"]

    if not isinstance(top, list) or not top:
        return []
    # garante formato dict com "sequencia"
    out = [x for x in top if isinstance(x, dict) and "sequencia" in x]
    return out


def _extract_candidates_for_v3(resultado_v2: Any) -> List[Dict[str, Any]]:
    """
    V3 precisa de candidatos com:
      - sequencia
      - detail.metricas (dict numérico)
      - detail.scf_total (float)
    """
    top = _normalize_v2_payload(resultado_v2)
    if not top:
        raise ValueError("V2 não retornou lista de candidatos (top/prototipos).")

    normalized: List[Dict[str, Any]] = []
    for it in top:
        detail = it.get("detail") if isinstance(it.get("detail"), dict) else {}
        metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else None
        scf_total = detail.get("scf_total", it.get("score", 0.0))

        # fallback (antigo): alguns V2 podem expor "metricas" no topo
        if metricas is None and isinstance(it.get("metricas"), dict):
            metricas = it["metricas"]

        if metricas is None:
            raise ValueError("V3 exige metricas do V2 (detail.metricas).")

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
        raise ValueError("Não consegui normalizar candidatos do V2 para o V3.")
    return normalized


def _v1_run(candidatos: List[List[int]], top_n: int) -> List[Any]:
    motor_v1 = MotorFGI()
    return motor_v1.gerar_prototipos(candidatos=candidatos, top_n=top_n)


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
    return motor_v2.rerank(
        candidatos=seqs_filtradas,
        contexto_lab=contexto_lab,
        overrides=overrides_v2,
    )


# ==============================
# PROTÓTIPOS (PIPELINE COM FALLBACK)
# ==============================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = (req.engine or "").lower().strip()
    if engine not in ("v1", "v2", "v3"):
        raise HTTPException(status_code=400, detail="engine deve ser 'v1', 'v2' ou 'v3'")

    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k inválido")

    # Contexto do laboratório
    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()

    contexto_lab = {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso"),
    }

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

    # -------------------------
    # V1 sempre existe (base)
    # -------------------------
    filtrados = _v1_run(candidatos=candidatos, top_n=req.max_candidatos)

    if engine == "v1":
        return {
            "engine_used": "v1",
            "schema_version": "v1.0",
            "prototipos": filtrados[: req.top_n],
            "contexto_lab": contexto_lab,
        }

    seqs_filtradas = _extract_seq_list(filtrados)

    # -------------------------
    # V2 (com fallback -> V1)
    # -------------------------
    try:
        resultado_v2 = _v2_run(
            seqs_filtradas=seqs_filtradas,
            contexto_lab=contexto_lab,
            req=req,
        )
    except Exception as e:
        # fallback duro: V2 falhou, volta V1
        return {
            "engine_used": "v1",
            "schema_version": "v1.0",
            "fallback": {
                "requested_engine": engine,
                "fell_back_to": "v1",
                "reason": f"V2 falhou: {str(e)}",
            },
            "prototipos": filtrados[: req.top_n],
            "contexto_lab": contexto_lab,
        }

    if engine == "v2":
        if isinstance(resultado_v2, dict):
            resultado_v2["contexto_lab"] = contexto_lab
        return resultado_v2

    # -------------------------
    # V3 (com fallback -> V2)
    # -------------------------
    try:
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

        if isinstance(resultado_v3, dict):
            resultado_v3["contexto_lab"] = contexto_lab

        return JSONResponse(content=resultado_v3)

    except Exception as e:
        # fallback: V3 não conseguiu operar (normalmente por métricas ausentes)
        if isinstance(resultado_v2, dict):
            resultado_v2["contexto_lab"] = contexto_lab
            resultado_v2["fallback"] = {
                "requested_engine": "v3",
                "fell_back_to": "v2",
                "reason": str(e),
            }
        return resultado_v2
```0
