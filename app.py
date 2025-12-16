
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ===== Motores =====
from fgi_engine import MotorFGI              # v1 (Filtro) — CONGELADO
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional / SCF)

# ===== Infra =====
from grupo_de_milhoes import GrupoMilhoes
from regime_detector import RegimeDetector

# ===== App =====
app = FastAPI(title="ATHENA LABORATORIO PMF")


# ==============================
#   MODELO DO REQUEST
# ==============================

class PrototiposRequest(BaseModel):
    engine: str = "v1"  # v1 | v2

    top_n: int = 30
    max_candidatos: int = 3000

    windows: Optional[List[int]] = None
    dna_anchor_window: Optional[int] = None

    pesos_windows: Optional[Dict[str, float]] = None
    pesos_metricas: Optional[Dict[str, float]] = None

    # parâmetros opcionais v2
    redundancy_jaccard_threshold: Optional[float] = None
    redundancy_penalty: Optional[float] = None
    z_cap: Optional[float] = None
    align_temperature: Optional[float] = None


# ==============================
#   STATUS
# ==============================

@app.get("/lab/status")
def lab_status():
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "motores": {
            "v1": "Filtro (congelado)",
            "v2": "Direcional / SCF"
        }
    }


# ==============================
#   DNA / REGIME
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
#   PROTÓTIPOS
# ==============================

@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = req.engine.lower().strip()

    if engine not in ("v1", "v2"):
        raise HTTPException(status_code=400, detail="engine deve ser 'v1' ou 'v2'")

    # --------------------------------
    # 1. CONTEXTO DO LABORATÓRIO
    # --------------------------------
    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()

    contexto_lab = {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso")
    }

    # --------------------------------
    # 2. GRUPO DE MILHÕES (BASE ÚNICA)
    # --------------------------------
    grupo = GrupoMilhoes()
    candidatos = grupo.gerar_combinacoes()

    if not candidatos:
        raise HTTPException(status_code=500, detail="Grupo de Milhões vazio")

    # --------------------------------
    # 3. MOTOR v1 — FILTRO (SEMPRE)
    # --------------------------------
    motor_v1 = MotorFGI()
    filtrados = motor_v1.gerar_prototipos(
        candidatos=candidatos,
        top_n=req.max_candidatos
    )

    # Se o usuário pediu v1, acabou aqui
    if engine == "v1":
        return {
            "engine": "v1",
            "top": filtrados[:req.top_n],
            "contexto_lab": contexto_lab
        }

    # --------------------------------
    # 4. MOTOR v2 — SCF (RE-RANKING)
    # --------------------------------
    motor_v2 = MotorFGI_V2()

    overrides = {
        "top_n": req.top_n,
        "max_candidatos": req.max_candidatos,
    }

    if req.windows is not None:
        overrides["windows"] = req.windows
    if req.dna_anchor_window is not None:
        overrides["dna_anchor_window"] = req.dna_anchor_window
    if req.pesos_windows is not None:
        overrides["pesos_windows"] = req.pesos_windows
    if req.pesos_metricas is not None:
        overrides["pesos_metricas"] = req.pesos_metricas
    if req.redundancy_jaccard_threshold is not None:
        overrides["redundancy_jaccard_threshold"] = req.redundancy_jaccard_threshold
    if req.redundancy_penalty is not None:
        overrides["redundancy_penalty"] = req.redundancy_penalty
    if req.z_cap is not None:
        overrides["z_cap"] = req.z_cap
    if req.align_temperature is not None:
        overrides["align_temperature"] = req.align_temperature

    resultado_v2 = motor_v2.rerank(
        candidatos=[x["sequencia"] if isinstance(x, dict) else x for x in filtrados],
        contexto_lab=contexto_lab,
        overrides=overrides
    )

    return resultado_v2
