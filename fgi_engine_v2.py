
# fgi_engine_v2.py
# Motor v2 (Direcional / SCF) — INSTRUMENTADO

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import math
import statistics

# =========================
# Utilidades
# =========================

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

# =========================
# Config
# =========================

@dataclass
class MotorV2Config:
    top_n: int
    redundancy_jaccard_threshold: float
    redundancy_penalty: float
    align_temperature: float

# =========================
# Motor
# =========================

class MotorFGI_V2:
    SCHEMA_VERSION = "v2.1"
    SCORING_MODE = "scf_governing"

    def rerank(
        self,
        candidatos: List[Sequence[int]],
        contexto_lab: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:

        cfg = MotorV2Config(
            top_n=overrides.get("top_n", 30),
            redundancy_jaccard_threshold=overrides.get("redundancy_jaccard_threshold", 0.75),
            redundancy_penalty=overrides.get("redundancy_penalty", 0.50),
            align_temperature=overrides.get("align_temperature", 2.0),
        )

        scored = []

        for seq in candidatos:
            seq = sorted(seq)

            # ---------- componentes base ----------
            soma = sum(seq)
            pares = sum(1 for x in seq if x % 2 == 0)
            adj = sum(1 for i in range(1, len(seq)) if seq[i] - seq[i-1] == 1)

            # ---------- SCF (sem clamp intermediário) ----------
            score_direcional = _sigmoid(cfg.align_temperature * (soma - 195) / 15)
            score_consistencia = 1.0 - abs(adj - 3) / 5
            score_ancora = _sigmoid((pares - 7) / 2)

            raw_components = {
                "direcional": score_direcional,
                "consistencia": score_consistencia,
                "ancora": score_ancora,
            }

            weighted_components = {
                "direcional": 0.4 * score_direcional,
                "consistencia": 0.35 * score_consistencia,
                "ancora": 0.25 * score_ancora,
            }

            score_before_redundancy = sum(weighted_components.values())

            scored.append({
                "sequencia": seq,
                "score_raw_components": raw_components,
                "score_weighted_components": weighted_components,
                "score_before_redundancy": score_before_redundancy,
                "score_final": score_before_redundancy,
                "redundancy_penalty": 0.0
            })

        # ---------- ordena ----------
        scored.sort(key=lambda x: x["score_final"], reverse=True)

        # ---------- anti-clone ----------
        final = []
        selected = []

        for item in scored:
            if len(final) >= cfg.top_n:
                break

            sim = max((_jaccard(item["sequencia"], s) for s in selected), default=0.0)
            if sim >= cfg.redundancy_jaccard_threshold:
                penalty = cfg.redundancy_penalty * (
                    (sim - cfg.redundancy_jaccard_threshold) /
                    (1.0 - cfg.redundancy_jaccard_threshold)
                )
                penalty = _clamp(penalty, 0.0, 0.9)
                item["redundancy_penalty"] = penalty
                item["score_final"] *= (1.0 - penalty)

            final.append(item)
            selected.append(item["sequencia"])

        return {
            "engine_used": "v2",
            "schema_version": self.SCHEMA_VERSION,
            "scoring_mode": self.SCORING_MODE,
            "top": final
                }
