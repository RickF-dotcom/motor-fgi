
# fgi_engine_v3.py
from typing import List, Dict, Any
from itertools import combinations


class MotorFGI_V3:
    """
    V3 — Contraste / DCR (Diversity-Contrast Rank)

    CONTRATO DE ENTRADA (OBRIGATÓRIO):
      candidatos: List[Dict] com:
        - sequencia: List[int]
        - score: float
        - detail.metricas: Dict[str, float]
        - detail.scf_total: float

    CONTRATO DE SAÍDA:
      - engine_used
      - schema_version
      - rank_mode
      - top
    """

    def __init__(self) -> None:
        pass

    # =====================================================
    # JACCARD SIMILARITY
    # =====================================================
    def _jaccard(self, a: List[int], b: List[int]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # =====================================================
    # DIVERSIDADE GLOBAL (penalização por similaridade)
    # =====================================================
    def _diversidade_penalty(
        self,
        seq: List[int],
        selecionados: List[List[int]],
        threshold: float,
    ) -> float:

        if not selecionados:
            return 0.0

        penalidade: float = 0.0
        for s in selecionados:
            j = self._jaccard(seq, s)
            if j >= threshold:
                penalidade += j

        return penalidade

    # =====================================================
    # RANK FINAL
    # =====================================================
    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int,
        alpha_contraste: float,
        beta_diversidade: float,
        gamma_base: float,
        jaccard_penalty_threshold: float,
    ) -> Dict[str, Any]:

        # blindagem
        validos: List[Dict[str, Any]] = []
        for c in candidatos:
            if (
                isinstance(c, dict)
                and "sequencia" in c
                and "detail" in c
                and isinstance(c["detail"], dict)
                and "metricas" in c["detail"]
                and "scf_total" in c["detail"]
            ):
                validos.append(c)

        if not validos:
            raise ValueError("V3 recebeu candidatos inválidos ou sem métricas.")

        # score base
        for c in validos:
            base = float(c["detail"]["scf_total"])
            c["_base_score"] = base

        # ordenação inicial
        validos.sort(key=lambda x: x["_base_score"], reverse=True)

        selecionados: List[Dict[str, Any]] = []
        selecionados_seq: List[List[int]] = []

        for c in validos:
            if len(selecionados) >= top_n:
                break

            seq = c["sequencia"]

            penalty_div = self._diversidade_penalty(
                seq,
                selecionados_seq,
                jaccard_penalty_threshold,
            )

            score_final = (
                alpha_contraste * c["_base_score"]
                - beta_diversidade * penalty_div
                + gamma_base * c["_base_score"]
            )

            selecionados.append(
                {
                    "sequencia": seq,
                    "score": float(score_final),
                    "detail": {
                        "scf_total": float(c["detail"]["scf_total"]),
                        "metricas": c["detail"]["metricas"],
                        "penalty_diversidade": float(penalty_div),
                    },
                }
            )

            selecionados_seq.append(seq)

        # ordenação final
        selecionados.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": "v3.0",
            "rank_mode": "contrast_dcr",
            "top": selecionados[:top_n],
        }
