# fgi_engine_v3.py
# Motor v3 — Contrast Ranker (DCR)
# Quebra empate crônico por contraste relativo + diversidade estrutural

from __future__ import annotations
from typing import List, Dict, Any, Sequence
import math
import statistics


# =========================
# Utilidades
# =========================

def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# Motor V3
# =========================

class MotorFGI_V3:
    SCHEMA_VERSION = "v3.0"
    ENGINE_MODE = "contrast_ranker"

    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int = 30,
        alpha_contraste: float = 0.55,
        beta_diversidade: float = 0.30,
        gamma_base: float = 0.15,
        jaccard_penalty_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """
        candidatos: lista já validada (saída do V2)
        cada item DEVE conter:
          - sequencia
          - detail.scf_total (ou score)
          - detail.metricas (numéricas)
        """

        # -------------------------
        # 1. Vetorização
        # -------------------------
        vetores = []
        base_scores = []

        for c in candidatos:
            metrics = c.get("detail", {}).get("metricas", {})
            vec = [float(metrics[k]) for k in sorted(metrics.keys())]
            vetores.append(vec)
            base_scores.append(float(c.get("detail", {}).get("scf_total", c.get("score", 0.0))))

        if not vetores:
            return {"engine": "v3", "top": []}

        centroide = [
            statistics.mean(col) for col in zip(*vetores)
        ]

        # -------------------------
        # 2. Distância ao centroide (contraste)
        # -------------------------
        distancias = [
            _euclidean(v, centroide) for v in vetores
        ]

        max_dist = max(distancias) if max(distancias) > 0 else 1.0
        contraste_norm = [d / max_dist for d in distancias]

        # -------------------------
        # 3. Ranking com diversidade
        # -------------------------
        ranked = []
        selecionados = []

        for idx, c in sorted(
            enumerate(candidatos),
            key=lambda x: contraste_norm[x[0]],
            reverse=True
        ):
            seq = c["sequencia"]
            base = base_scores[idx]
            contraste = contraste_norm[idx]

            # diversidade vs já selecionados
            diversidade = 1.0
            if selecionados:
                max_sim = max(_jaccard(seq, s) for s in selecionados)
                if max_sim >= jaccard_penalty_threshold:
                    diversidade = 1.0 - max_sim
                diversidade = _clamp(diversidade, 0.0, 1.0)

            score_final = (
                alpha_contraste * contraste +
                beta_diversidade * diversidade +
                gamma_base * base
            )

            ranked.append({
                "sequencia": seq,
                "score": score_final,
                "detail": {
                    "contraste": contraste,
                    "diversidade": diversidade,
                    "base_scf": base
                }
            })

            selecionados.append(seq)
            if len(ranked) >= top_n:
                break

        ranked.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": self.SCHEMA_VERSION,
            "mode": self.ENGINE_MODE,
            "top": ranked
        }
