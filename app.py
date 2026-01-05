# app.py
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fgi_engine import MotorFGI              # v1 (Filtro) - mantém
from fgi_engine_v2 import MotorFGI_V2        # v2 (Direcional/SCF)
from fgi_engine_v3 import MotorFGI_V3        # v3 (Contraste/DCR)

from regime_detector import RegimeDetector

# ==========================================================
# Build identity (Render)
# ==========================================================
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "unknown")
SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "unknown")

# ==========================================================
# Grupo de Milhões: compatibilidade de import (sem quebrar)
# - seu arquivo postado tem class GrupoDeMilhoes
# - seu baseline antigo importava GrupoMilhoes
# ==========================================================
GrupoClass = None
try:
    from grupo_de_milhoes import GrupoMilhoes as _GrupoMilhoes  # type: ignore
    GrupoClass = _GrupoMilhoes
except Exception:
    try:
        from grupo_de_milhoes import GrupoDeMilhoes as _GrupoDeMilhoes  # type: ignore
        GrupoClass = _GrupoDeMilhoes
    except Exception:
        GrupoClass = None


# ==========================================================
# Helpers: localizar e carregar histórico automaticamente
# ==========================================================
def _repo_root() -> Path:
    # Render roda em /app normalmente
    # mas isso mantém compatível localmente também
    here = Path(__file__).resolve()
    return here.parent


def _candidate_history_paths() -> List[Path]:
    # 1) ENV explícita (prioridade máxima)
    env = os.environ.get("HISTORICO_CSV", "").strip()
    paths: List[Path] = []
    if env:
        paths.append(Path(env))

    # 2) nomes prováveis no repo (fallback automático)
    root = _repo_root()
    paths += [
        root / "lotofacil_ultimos_25_concursos.csv",
        root / "lotofacil_ultimos_25_concursos.CSV",
        root / "historico_lotofacil.csv",
        root / "historico.csv",
    ]
    return paths


def _find_history_csv() -> Optional[Path]:
    for p in _candidate_history_paths():
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def _parse_history_csv(path: Path, universo_max: int = 25) -> List[List[int]]:
    """
    Parser robusto:
    - aceita CSV com colunas diversas
    - extrai todos os inteiros 1..universo_max por linha
    - retorna lista de jogos (listas ordenadas)
    """
    jogos: List[List[int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            nums: List[int] = []
            for cell in row:
                cell = (cell or "").strip()
                if not cell:
                    continue
                # quebra por separadores comuns
                parts = (
                    cell.replace(";", " ")
                        .replace("-", " ")
                        .replace(",", " ")
                        .replace("\t", " ")
                        .split()
                )
                for part in parts:
                    try:
                        n = int(part)
                    except Exception:
                        continue
                    if 1 <= n <= universo_max:
                        nums.append(n)
            if nums:
                uniq = sorted(set(nums))
                # só aceita jogos com tamanho mínimo plausível
                if len(uniq) >= 10:  # lotofácil geralmente >= 15; aqui é tolerante
                    jogos.append(uniq)
    return jogos


def _inject_history_into_grupo(grupo: Any, universo_max: int = 25) -> Dict[str, Any]:
    """
    Carrega histórico no Grupo SEM depender da interface da classe.
    Preferência:
    - se existir grupo.load_from_csv(path) -> usa
    - senão: parseia CSV e injeta em grupo.drawn (set de tuplas)
    """
    info: Dict[str, Any] = {
        "history_loaded": False,
        "history_path": None,
        "history_games": 0,
        "history_mode": None,
        "drawn_size": None,
        "error": None,
    }

    path = _find_history_csv()
    if not path:
        # sem histórico, não quebra nada
        try:
            if hasattr(grupo, "drawn"):
                info["drawn_size"] = len(getattr(grupo, "drawn"))
        except Exception:
            info["drawn_size"] = None
        return info

    info["history_path"] = str(path)

    try:
        # 1) se a classe tiver loader próprio
        if hasattr(grupo, "load_from_csv") and callable(getattr(grupo, "load_from_csv")):
            grupo.load_from_csv(path)  # type: ignore
            info["history_loaded"] = True
            info["history_mode"] = "grupo.load_from_csv"
        else:
            # 2) fallback universal: injeta em grupo.drawn
            jogos = _parse_history_csv(path, universo_max=universo_max)
            if not hasattr(grupo, "drawn"):
                raise RuntimeError("Classe do Grupo não expõe atributo 'drawn' e não tem load_from_csv().")

            drawn = getattr(grupo, "drawn")
            if not isinstance(drawn, set):
                raise RuntimeError("Atributo 'drawn' existe, mas não é set().")

            before = len(drawn)
            for j in jogos:
                drawn.add(tuple(sorted(j)))
            after = len(drawn)

            info["history_loaded"] = True
            info["history_mode"] = "inject_into_grupo.drawn"
            info["history_games"] = len(jogos)
            info["drawn_size"] = after
            # se nada entrou, ainda assim marcamos o caminho
            if after == before:
                info["history_mode"] += "_no_change"
        # pós
        try:
            if hasattr(grupo, "drawn"):
                info["drawn_size"] = len(getattr(grupo, "drawn"))
        except Exception:
            pass

        return info

    except Exception as e:
        info["error"] = str(e)
        try:
            if hasattr(grupo, "drawn"):
                info["drawn_size"] = len(getattr(grupo, "drawn"))
        except Exception:
            pass
        return info


def _make_grupo() -> Any:
    """
    Cria instância do Grupo de Milhões SEM exigir histórico.
    Se existir CSV no repo (ou configurado), carrega automaticamente.
    """
    if GrupoClass is None:
        raise HTTPException(
            status_code=500,
            detail="Não consegui importar GrupoMilhoes/GrupoDeMilhoes de grupo_de_milhoes.py",
        )

    universo_max = int(os.environ.get("UNIVERSO_MAX", "25"))
    kwargs: Dict[str, Any] = {}

    # compatibilidade: alguns construtores usam universo_max
    kwargs["universo_max"] = universo_max

    # tenta instanciar com kwargs; se falhar, tenta vazio
    try:
        grupo = GrupoClass(**kwargs)
    except TypeError:
        try:
            grupo = GrupoClass()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Falha ao instanciar Grupo: {e}")

    # injeta histórico se achar CSV (não quebra se não achar)
    _inject_history_into_grupo(grupo, universo_max=universo_max)

    return grupo


# ==========================================================
# App
# ==========================================================
app = FastAPI(title="ATHENA LABORATORIO PMF")


# ==========================================================
# Request schema
# ==========================================================
class PrototiposRequest(BaseModel):
    engine: str = "v1"  # v1 | v2 | v3

    # tamanho da combinação (Lotofácil: 15)
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

    # V3 (Contraste / DCR)
    alpha_contraste: float = 0.55
    beta_diversidade: float = 0.30
    gamma_base: float = 0.15
    jaccard_penalty_threshold: float = 0.75


# ==========================================================
# ROOT / HEALTH / STATUS
# ==========================================================
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
            "/grupo/status",
            "/grupo/sample",
            "/prototipos",
        ],
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
    }


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
        "grupo": {
            "enabled": True,
            "class": getattr(GrupoClass, "__name__", "unknown"),
            "universo_max": int(os.environ.get("UNIVERSO_MAX", "25")),
            "historico_csv_env": os.environ.get("HISTORICO_CSV", "").strip() or None,
            "historico_csv_auto": str(_find_history_csv()) if _find_history_csv() else None,
        },
    }


# ==========================================================
# DNA / REGIME
# ==========================================================
@app.get("/lab/dna_last25")
def dna_last25():
    detector = RegimeDetector()
    return detector.get_dna_last25()


@app.get("/lab/regime_atual")
def regime_atual():
    detector = RegimeDetector()
    return detector.detectar_regime()


# ==========================================================
# GRUPO: STATUS / SAMPLE
# ==========================================================
@app.get("/grupo/status")
def grupo_status():
    """
    Prova de integração:
    - importa
    - instancia
    - carrega histórico (se achar CSV)
    - responde com informações mínimas + drawn_size
    """
    universo_max = int(os.environ.get("UNIVERSO_MAX", "25"))
    grupo = _make_grupo()
    hist_info = _inject_history_into_grupo(grupo, universo_max=universo_max)

    payload: Dict[str, Any] = {
        "ok": True,
        "class": getattr(grupo, "__class__", type("x", (), {})).__name__,
        "universo_max": getattr(grupo, "universo_max", universo_max),
        "historico_csv_env": os.environ.get("HISTORICO_CSV", "").strip() or None,
        "historico_csv_auto": str(_find_history_csv()) if _find_history_csv() else None,
        "drawn_size": hist_info.get("drawn_size"),
        "history_loaded": hist_info.get("history_loaded"),
        "history_mode": hist_info.get("history_mode"),
        "history_games": hist_info.get("history_games"),
        "history_error": hist_info.get("error"),
        "build_commit": BUILD_COMMIT,
        "service_id": SERVICE_ID,
    }

    # se existir método total_sorteadas (em algumas versões)
    try:
        if hasattr(grupo, "total_sorteadas"):
            payload["total_sorteadas_k15"] = grupo.total_sorteadas(15)  # type: ignore
            payload["total_sorteadas_total"] = grupo.total_sorteadas()  # type: ignore
    except Exception:
        payload["total_sorteadas_k15"] = None
        payload["total_sorteadas_total"] = None

    return payload


@app.get("/grupo/sample")
def grupo_sample(
    k: int = Query(15, ge=1, le=25),
    n: int = Query(30, ge=1, le=5000),
    shuffle: bool = Query(True),
    seed: int = Query(1337),
):
    """
    Retorna uma amostra de candidatos do Grupo de Milhões.
    - k: tamanho da combinação (ex.: 15)
    - n: quantidade solicitada
    """
    grupo = _make_grupo()

    # compatibilidade: seu GrupoDeMilhoes usa get_candidatos
    if not hasattr(grupo, "get_candidatos"):
        raise HTTPException(status_code=500, detail="Classe do Grupo não possui get_candidatos()")

    try:
        candidatos = grupo.get_candidatos(  # type: ignore
            k=k,
            max_candidatos=n,
            shuffle=shuffle,
            seed=seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar sample: {e}")

    return {
        "ok": True,
        "k": k,
        "n": len(candidatos),
        "shuffle": shuffle,
        "seed": seed,
        "sample": candidatos,
    }


# ==========================================================
# HELPERS
# ==========================================================
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
        raise HTTPException(status_code=500, detail="V2 não retornou lista de candidatos (top/prototipos).")

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
                status_code=500,
                detail="V3 exige metricas do V2 (detail.metricas). Seu V2 não está expondo metricas.",
            )

        normalized.append(
            {
                "sequencia": it["sequencia"],
                "score": float(it.get("score", 0.0)),
                "detail": {"scf_total": float(scf_total), "metricas": metricas},
            }
        )

    if not normalized:
        raise HTTPException(status_code=500, detail="Não consegui normalizar candidatos do V2 para o V3.")
    return normalized


# ==========================================================
# PROTÓTIPOS
# ==========================================================
@app.post("/prototipos")
def gerar_prototipos(req: PrototiposRequest):
    engine = (req.engine or "v1").lower().strip()
    if engine not in ("v1", "v2", "v3"):
        raise HTTPException(status_code=400, detail="engine deve ser 'v1', 'v2' ou 'v3'")

    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k inválido")

    detector = RegimeDetector()
    dna = detector.get_dna_last25()
    regime = detector.detectar_regime()

    contexto_lab = {
        "dna_last25": dna,
        "regime": regime,
        "ultimo_concurso": regime.get("ultimo_concurso"),
    }

    # Grupo de Milhões (amostragem)
    grupo = _make_grupo()
    try:
        candidatos = grupo.get_candidatos(  # type: ignore
            k=req.k,
            max_candidatos=req.max_candidatos,
            shuffle=True,
            seed=1337,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha no Grupo de Milhões: {e}")

    if not candidatos:
        raise HTTPException(status_code=500, detail="Grupo de Milhões vazio")

    # v1 — FILTRO
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

    seqs_filtradas = _extract_seq_list(filtrados)

    # v2 — SCF (re-ranking)
    overrides_v2: Dict[str, Any] = {"top_n": req.top_n, "max_candidatos": req.max_candidatos}
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

    if engine == "v2":
        if isinstance(resultado_v2, dict):
            resultado_v2["contexto_lab"] = contexto_lab
        return resultado_v2

    # v3 — CONTRASTE (rank final)
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
