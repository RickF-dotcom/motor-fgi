
# fgi_engine_v3.py
from typing import List, Dict, Any
import math


class MotorFGI_V3:
    """
    V3 — Contraste / DCR

    ENTRADA (contrato obrigatório):
    candidatos = [
        {
            "sequencia": [...],
            "score": float,
            "detail": {
                "scf_total": float,
                "metricas": {str: float}
            }
        }
    ]
    """

    def __init__(self):
        pass

    # =========================================================
    # Similaridade Jaccard
    # =========================================================
    def _jaccard(self, a: List[int], b: List[int]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # =========================================================
    # RANK FINAL
    # =========================================================
    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int,
        alpha_contraste: float,
        beta_diversidade: float,
        gamma_base: float,
        jaccard_penalty_threshold: float,
    ) -> Dict[str, Any]:

        ranqueados: List[Dict[str, Any]] = []

        for cand in candidatos:
            seq = cand.get("sequencia")
            detail = cand.get("detail", {})
            metricas = detail.get("metricas", {})
            scf_total = float(detail.get("scf_total", 0.0))

            # blindagem absoluta
            if not isinstance(seq, list) or not isinstance(metricas, dict):
                continue

            # -----------------------------
            # CONTRASTE
            # -----------------------------
            contraste = scf_total

            # -----------------------------
            # BASE (média simples das métricas)
            # -----------------------------
            if metricas:
                base = sum(metricas.values()) / max(len(metricas), 1)
            else:
                base = 0.0

            score_final = (
                alpha_contraste * contraste
                + gamma_base * base
            )

            ranqueados.append(
                {
                    "sequencia": seq,
                    "score": score_final,
                    "detail": {
                        "scf_total": scf_total,
                        "metricas": metricas,
                        "contraste": contraste,
                        "base": base,
                    },
                }
            )

        # -----------------------------
        # Penalização por redundância
        # -----------------------------
        ranqueados.sort(key=lambda x: x["score"], reverse=True)

        finais: List[Dict[str, Any]] = []
        for cand in ranqueados:
            penalidade = 0.0
            for prev in finais:
                j = self._jaccard(cand["sequencia"], prev["sequencia"])
                if j >= jaccard_penalty_threshold:
                    penalidade += beta_diversidade * j

            cand["score"] = cand["score"] - penalidade
            finais.append(cand)

        finais.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": "v3.0",
            "score_mode": "contrast_dcr",
            "top": finais[:top_n],
        }
