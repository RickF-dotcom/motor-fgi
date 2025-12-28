# app.py
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================================
# Identidade do build (Render)
# ==========================================================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")
DEBUG_ERRORS = os.environ.get("DEBUG_ERRORS", "1") == "1"

BASE_DIR = Path(__file__).resolve().parent
DNA_FILE = BASE_DIR / "dna_last25.yaml"
LAB_CONFIG_FILE = BASE_DIR / "lab_config.yaml"
ULTIMOS_25_CSV = BASE_DIR / "lotofacil_ultimos_25_concursos.csv"

# ==========================================================
# Imports robustos (evita quebrar deploy por detalhe de nome)
# ==========================================================
# Motores
try:
    from fgi_engine import MotorFGI as MotorV1
except Exception:
    MotorV1 = None  # type: ignore

try:
    from fgi_engine_v2 import MotorFGI_V2 as MotorV2
except Exception:
    try:
        from fgi_engine_v2 import MotorFGI as MotorV2  # fallback
    except Exception:
        MotorV2 = None  # type: ignore

try:
    from fgi_engine_v3 import MotorFGI_V3 as MotorV3
except Exception:
    try:
        from fgi_engine_v3 import MotorFGI as MotorV3  # fallback
    except Exception:
        MotorV3 = None  # type: ignore

# Grupo de milhões (seu log mostrou NameError por nome errado)
GrupoMilhoes = None
try:
    from grupo_de_milhoes import GrupoMilhoes as _GrupoMilhoes

    GrupoMilhoes = _GrupoMilhoes
except Exception:
    try:
        # fallback: se alguém nomeou diferente
        from grupo_de_milhoes import GrupoDeMilhoes as _GrupoMilhoes  # type: ignore

        GrupoMilhoes = _GrupoMilhoes
    except Exception:
        GrupoMilhoes = None  # type: ignore

# Regime detector
RegimeDetector = None
try:
    from regime_detector import RegimeDetector as _RegimeDetector

    RegimeDetector = _RegimeDetector
except Exception:
    RegimeDetector = None  # type: ignore

# Maturação (opcional)
calcular_maturacao = None
score_maturacao_jogo = None
try:
    from maturacao import calcular_maturacao as _calcular_maturacao, score_maturacao_jogo as _score_maturacao_jogo

    calcular_maturacao = _calcular_maturacao
    score_maturacao_jogo = _score_maturacao_jogo
except Exception:
    pass


# ==========================================================
# Helpers
# ==========================================================
def _trace_short() -> str:
    tb = traceback.format_exc()
    lines = tb.strip().splitlines()
    return "\n".join(lines[-25:])


def _read_yaml_if_possible(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"arquivo não encontrado: {path.name}", "path": str(path)}
    try:
        import yaml  # depende do requirements.txt ter PyYAML

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {"ok": False, "error": f"YAML inválido: {path.name} não retornou dict", "path": str(path)}
        return {"ok": True, "data": data, "path": str(path)}
    except Exception as e:
        payload = {"ok": False, "error": str(e), "path": str(path)}
        if DEBUG_ERRORS:
            payload["trace"] = _trace_short()
        return payload


def _make_grupo(k: int) -> Any:
    """
    Cria GrupoMilhoes de forma tolerante a assinaturas diferentes.
    """
    if GrupoMilhoes is None:
        raise RuntimeError("GrupoMilhoes não importou (grupo_de_milhoes.py com erro ou nome diferente).")

    # tenta assinaturas comuns
    for ctor in (
        lambda: GrupoMilhoes(k=k, csv_path=str(ULTIMOS_25_CSV)),  # type: ignore
        lambda: GrupoMilhoes(k=k, data_path=str(ULTIMOS_25_CSV)),  # type: ignore
        lambda: GrupoMilhoes(k=k, historico_csv=str(ULTIMOS_25_CSV)),  # type: ignore
        lambda: GrupoMilhoes(k=k),  # type: ignore
        lambda: GrupoMilhoes(k),  # type: ignore
    ):
        try:
            return ctor()
        except TypeError:
            continue

    raise RuntimeError("Não consegui instanciar GrupoMilhoes (assinatura inesperada).")


def _call_motor_generate(motor: Any, k: int, top_n: int, max_candidatos: int, grupo: Any) -> List[Any]:
    """
    Chama o método correto do motor, independente do nome interno.
    """
    candidates = [
        ("gerar_prototipos", dict(k=k, top_n=top_n, max_candidatos=max_candidatos, grupo=grupo)),
        ("gerar_prototipos", dict(k=k, top_n=top_n, max_candidatos=max_candidatos)),
        ("gerar", dict(k=k, top_n=top_n, max_candidatos=max_candidatos, grupo=grupo)),
        ("gerar", dict(k=k, top_n=top_n, max_candidatos=max_candidatos)),
        ("run", dict(k=k, top_n=top_n, max_candidatos=max_candidatos, grupo=grupo)),
        ("run", dict(k=k, top_n=top_n, max_candidatos=max_candidatos)),
    ]

    last_err: Optional[Exception] = None
    for name, kwargs in candidates:
        fn = getattr(motor, name, None)
        if callable(fn):
            try:
                out = fn(**kwargs)
                if out is None:
                    return []
                if isinstance(out, list):
                    return out
                # se o motor retorna dict com "prototipos"
                if isinstance(out, dict) and "prototipos" in out:
                    return out["prototipos"]  # type: ignore
                return [out]
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break

    raise RuntimeError(f"Motor não respondeu (método incompatível). Último erro: {last_err}")


# ==========================================================
# API
# ==========================================================
app = FastAPI(
    title="ATHENA LABORATORIO PMF",
    version="0.1.0",
)


class PrototiposRequest(BaseModel):
    engine: str = Field(default="v1", description="v1 | v2 | v3")
    k: int = Field(default=15, ge=1, le=25)
    top_n: int = Field(default=5, ge=1, le=500)
    max_candidatos: int = Field(default=50, ge=1, le=200000)


@app.get("/")
def root() -> Dict[str, Any]:
    # ESSENCIAL pro Render não ficar travado no “Application loading”
    return {
        "ok": True,
        "service": "motor-fgi",
        "status": "online",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
        "docs": "/docs",
        "endpoints": ["/health", "/lab/status", "/lab/dna_last25", "/lab/regime_atual", "/prototipos"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "healthy",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
    }


@app.get("/lab/status")
def lab_status() -> Dict[str, Any]:
    motores = {
        "v1": "Pass-through (congelado)" if MotorV1 else "indisponível (import falhou)",
        "v2": "Direcional / SCF" if MotorV2 else "indisponível (import falhou)",
        "v3": "Contraste / DCR" if MotorV3 else "indisponível (import falhou)",
    }
    return {
        "laboratorio": "ATHENA LABORATORIO PMF",
        "status": "online",
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
        "motores": motores,
        "arquivos": {
            "dna_last25.yaml": DNA_FILE.exists(),
            "lab_config.yaml": LAB_CONFIG_FILE.exists(),
            "lotofacil_ultimos_25_concursos.csv": ULTIMOS_25_CSV.exists(),
        },
    }


@app.get("/lab/dna_last25")
def dna_last25() -> Dict[str, Any]:
    return _read_yaml_if_possible(DNA_FILE)


@app.get("/lab/regime_atual")
def regime_atual() -> Dict[str, Any]:
    if RegimeDetector is None:
        raise HTTPException(status_code=500, detail="RegimeDetector não importou (regime_detector.py com erro).")

    try:
        detector = RegimeDetector()  # type: ignore
        # tenta métodos comuns
        for m in ("regime_atual", "detectar", "run", "get_regime"):
            fn = getattr(detector, m, None)
            if callable(fn):
                out = fn()
                return {"ok": True, "regime": out, "build_commit": BUILD_COMMIT, "service_id": SERVICE_ID}
        return {"ok": False, "error": "RegimeDetector não tem método reconhecido."}
    except Exception as e:
        payload = {"ok": False, "error": str(e)}
        if DEBUG_ERRORS:
            payload["trace"] = _trace_short()
        raise HTTPException(status_code=500, detail=payload)


@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest) -> Dict[str, Any]:
    engine = (req.engine or "v1").strip().lower()

    motor_cls = {"v1": MotorV1, "v2": MotorV2, "v3": MotorV3}.get(engine)
    if motor_cls is None:
        raise HTTPException(status_code=400, detail=f"engine inválido/indisponível: {engine} (use v1|v2|v3)")

    try:
        grupo = _make_grupo(req.k)
        motor = motor_cls()  # type: ignore
        prototipos = _call_motor_generate(
            motor=motor,
            k=req.k,
            top_n=req.top_n,
            max_candidatos=req.max_candidatos,
            grupo=grupo,
        )

        return {
            "ok": True,
            "engine_used": engine,
            "k": req.k,
            "top_n": req.top_n,
            "max_candidatos": req.max_candidatos,
            "count": len(prototipos),
            "prototipos": prototipos,
            "build_commit": BUILD_COMMIT,
            "service_id": SERVICE_ID,
        }

    except HTTPException:
        raise
    except Exception as e:
        payload = {
            "ok": False,
            "engine_used": engine,
            "error": str(e),
        }
        if DEBUG_ERRORS:
            payload["trace"] = _trace_short()
        raise HTTPException(status_code=500, detail=payload)
```0
