
# fgi_engine_v3.py
from typing import List, Dict, Any
import itertools


class MotorFGI_V3:
    """
    V3 — Contraste / DCR (Diversity-Contrast Ranking)

    CONTRATO EXIGIDO (VINDO DO V2):
    Cada candidato deve conter:
      {
        "sequencia": List[int],
        "score": float,                      # score legado (opcional)
        "detail": {
            "scf_total": float,              # score SCF consolidado do V2
            "metricas": Dict[str, float]     # métricas numéricas do V2
        }
      }

    SAÍDA:
      - ranking final com penalização por redundância (Jaccard)
      - contraste controlado por alpha / beta / gamma
    """

    def __init__(self):
        pass

    # ==========================
    # Métricas auxiliares
    # ==========================
    @staticmethod
    def _jaccard(a: List[int], b: List[int]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _base_score(candidate: Dict[str, Any]) -> float:
        """
        Score base vindo do V2.
        Prioridade:
          1) detail.scf_total
          2) score
        """
        detail = candidate.get("detail", {})
        if isinstance(detail, dict) and "scf_total" in detail:
            return float(detail.get("scf_total", 0.0))
        return float(candidate.get("score", 0.0))

    @staticmethod
    def _metric_contrast(metricas: Dict[str, float]) -> float:
        """
        Contraste simples entre métricas:
        desvio médio absoluto em relação à média.
        """
        if not metricas:
            return 0.0
        values = list(metricas.values())
        mean = sum(values) / max(len(values), 1)
        return sum(abs(v - mean) for v in values) / max(len(values), 1)

    # ==========================
    # Rank principal
    # ==========================
    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int = 10,
        alpha_contraste: float = 0.55,
        beta_diversidade: float = 0.30,
        gamma_base: float = 0.15,
        jaccard_penalty_threshold: float = 0.75,
    ) -> Dict[str, Any]:

        if not candidatos or not isinstance(candidatos, list):
            raise ValueError("V3 recebeu candidatos inválidos ou vazios.")

        # ==========================
        # Validação do contrato V2
        # ==========================
        for c in candidatos:
            if not isinstance(c, dict):
                raise ValueError("Candidato inválido (não é dict).")

            if "sequencia" not in c or not isinstance(c["sequencia"], list):
                raise ValueError("Candidato sem 'sequencia' válida.")

            detail = c.get("detail")
            if not isinstance(detail, dict):
                raise ValueError("V3 exige 'detail' vindo do V2.")

            if "metricas" not in detail or not isinstance(detail["metricas"], dict):
                raise ValueError("V3 exige detail.metricas (contrato V2).")

            if "scf_total" not in detail:
                raise ValueError("V3 exige detail.scf_total (contrato V2).")

        # ==========================
        # Cálculo de scores
        # ==========================
        enriched: List[Dict[str, Any]] = []

        for c in candidatos:
            base = self._base_score(c)
            contraste = self._metric_contrast(c["detail"]["metricas"])

            enriched.append(
                {
                    "sequencia": c["sequencia"],
                    "base_score": base,
                    "contraste": contraste,
                    "detail": c["detail"],
                }
            )

        # ==========================
        # Penalização por redundância
        # ==========================
        for a, b in itertools.combinations(enriched, 2):
            jac = self._jaccard(a["sequencia"], b["sequencia"])
            if jac >= jaccard_penalty_threshold:
                # penaliza ambos de forma simétrica
                a["base_score"] *= (1.0 - beta_diversidade)
                b["base_score"] *= (1.0 - beta_diversidade)

        # ==========================
        # Score final
        # ==========================
        resultados: List[Dict[str, Any]] = []

        for c in enriched:
            score_final = (
                gamma_base * c["base_score"]
                + alpha_contraste * c["contraste"]
                + (1.0 - alpha_contraste - gamma_base) * c["base_score"]
            )

            resultados.append(
                {
                    "sequencia": c["sequencia"],
                    "score": float(score_final),
                    "detail": {
                        "scf_total": float(c["detail"]["scf_total"]),
                        "metricas": c["detail"]["metricas"],
                        "contraste": float(c["contraste"]),
                        "base_score": float(c["base_score"]),
                    },
                }
            )

        resultados.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": "v3.0",
            "score_mode": "contrast_dcr",
            "top": resultados[:top_n],
        }
