# fgi_engine_v3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import math


def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


def _vec(metricas: Dict[str, float], keys: List[str]) -> List[float]:
    return [float(metricas.get(k, 0.0)) for k in keys]


def _cosine(u: List[float], v: List[float]) -> float:
    dot = sum(ui * vi for ui, vi in zip(u, v))
    nu = math.sqrt(sum(ui * ui for ui in u))
    nv = math.sqrt(sum(vi * vi for vi in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


@dataclass
class MotorFGI_V3:
    """
    V3 (Contraste / DCR)

    Entrada: lista de candidatos (dict) contendo:
      - sequencia: List[int]
      - detail.metricas: Dict[str,float]
      - detail.scf_total: float

    Saída: top final com score_v3 e breakdown.
    """

    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int = 10,
        alpha_contraste: float = 0.55,
        beta_diversidade: float = 0.30,
        gamma_base: float = 0.15,
        jaccard_penalty_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        if not candidatos:
            return {"engine_used": "v3", "top": []}

        # garante chaves métricas comuns
        metric_keys = None
        for c in candidatos:
            detail = c.get("detail", {}) if isinstance(c.get("detail"), dict) else {}
            metricas = detail.get("metricas", {}) if isinstance(detail.get("metricas"), dict) else {}
            if metricas:
                metric_keys = sorted(list(metricas.keys()))
                break
        if not metric_keys:
            raise ValueError("V3 exige metricas no payload (detail.metricas).")

        # vetoriza e calcula base
        enriched: List[Dict[str, Any]] = []
        for c in candidatos:
            detail = c.get("detail", {}) if isinstance(c.get("detail"), dict) else {}
            metricas = detail.get("metricas", {}) if isinstance(detail.get("metricas"), dict) else {}
            scf_total = float(detail.get("scf_total", c.get("score", 0.0)))
            v = _vec(metricas, metric_keys)

            enriched.append(
                {
                    "sequencia": c["sequencia"],
                    "base": scf_total,
                    "metricas": metricas,
                    "vec": v,
                }
            )

        # score = alpha*contraste + beta*diversidade + gamma*base
        # contraste aqui é "distância" do centro (média vetorial) => mais diferente do miolo, maior contraste
        mean = [0.0] * len(metric_keys)
        for e in enriched:
            for i, val in enumerate(e["vec"]):
                mean[i] += val
        mean = [m / max(1, len(enriched)) for m in mean]

        def contrast_score(vec: List[float]) -> float:
            # 1 - cosine(similarity) com o centro
            sim = _cosine(vec, mean)
            return 1.0 - sim

        # pré-score
        for e in enriched:
            e["contraste"] = float(contrast_score(e["vec"]))

        # seleção gulosa com diversidade (anti-jaccard)
        enriched.sort(key=lambda x: (x["contraste"], x["base"]), reverse=True)

        selected: List[Dict[str, Any]] = []
        for cand in enriched:
            if len(selected) >= top_n:
                break

            # diversidade: penaliza Jaccard alto com selecionados
            pen = 0.0
            for prev in selected:
                j = _jaccard(cand["sequencia"], prev["sequencia"])
                if j >= jaccard_penalty_threshold:
                    pen += (j - jaccard_penalty_threshold) / max(1e-9, (1.0 - jaccard_penalty_threshold))

            # diversidade score = 1 - penalidade normalizada (clamp)
            diversidade = 1.0 - min(1.0, pen)

            score_v3 = (
                alpha_contraste * float(cand["contraste"])
                + beta_diversidade * float(diversidade)
                + gamma_base * float(cand["base"])
            )

            selected.append(
                {
                    "sequencia": cand["sequencia"],
                    "score": float(score_v3),
                    "detail": {
                        "score_v3": float(score_v3),
                        "contraste": float(cand["contraste"]),
                        "diversidade": float(diversidade),
                        "base_scf": float(cand["base"]),
                        "metricas": cand["metricas"],
                        "metric_keys": metric_keys,
                    },
                }
            )

        selected.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        return {
            "engine_used": "v3",
            "top": selected,
            "config_used": {
                "top_n": top_n,
                "alpha_contraste": alpha_contraste,
                "beta_diversidade": beta_diversidade,
                "gamma_base": gamma_base,
                "jaccard_penalty_threshold": jaccard_penalty_threshold,
            },
        }
