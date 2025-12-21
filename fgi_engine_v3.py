
# fgi_engine_v3.py

from typing import List, Dict, Any
import math


class MotorFGI_V3:
    """
    V3 – Contrast / DCR
    CONTRATO GARANTIDO:
      - engine_used
      - schema_version
      - score_model
      - top (lista ordenada)
    """

    def __init__(self):
        pass

    # ============================================================
    # MÉTRICAS BÁSICAS (sempre seguras)
    # ============================================================
    def _calc_metricas_basicas(
        self,
        seq: List[int],
        contexto_lab: Dict[str, Any],
    ) -> Dict[str, float]:

        dna = contexto_lab.get("dna_last25", {}) or {}
        freq = dna.get("frequencia", {}) or {}

        soma_freq = 0.0
        for n in seq:
            soma_freq += float(freq.get(str(n), 0.0))

        tamanho = float(len(seq)) if seq else 0.0
        freq_media = soma_freq / tamanho if tamanho > 0 else 0.0

        return {
            "frequencia_media": freq_media,
            "soma_frequencia": soma_freq,
            "tamanho": tamanho,
        }

    # ============================================================
    # DIVERSIDADE (entropia simples)
    # ============================================================
    def _calc_diversidade(self, seq: List[int]) -> float:
        if not seq:
            return 0.0

        valores = {}
        for n in seq:
            valores[n] = valores.get(n, 0) + 1

        total = float(len(seq))
        entropia = 0.0

        for c in valores.values():
            p = c / total
            if p > 0:
                entropia -= p * math.log(p)

        return entropia

    # ============================================================
    # CONTRASTE (anti-clone)
    # ============================================================
    def _calc_contraste(self, freq_media: float) -> float:
        # Quanto menor a frequência média, maior o contraste
        return 1.0 / (1.0 + freq_media)

    # ============================================================
    # SCORE FINAL DCR (blindado)
    # ============================================================
    def _calc_score_dcr(
        self,
        metricas: Dict[str, float],
        diversidade: float,
        contraste: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> float:

        base = (
            alpha * contraste +
            beta * diversidade +
            gamma * metricas.get("tamanho", 0.0)
        )

        return float(base)

    # ============================================================
    # RERANK PRINCIPAL
    # ============================================================
    def rerank(
        self,
        candidatos: List[List[int]],
        contexto_lab: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:

        # ---------------- parâmetros ----------------
        top_n = int(overrides.get("top_n", 10))

        alpha = float(overrides.get("alpha_contraste", 0.55))
        beta = float(overrides.get("beta_diversidade", 0.30))
        gamma = float(overrides.get("gamma_base", 0.15))

        resultados: List[Dict[str, Any]] = []

        # ---------------- loop seguro ----------------
        for seq in candidatos:

            if not isinstance(seq, (list, tuple)):
                continue
            if not seq:
                continue

            seq = [int(x) for x in seq]

            metricas = self._calc_metricas_basicas(seq, contexto_lab)
            diversidade = self._calc_diversidade(seq)
            contraste = self._calc_contraste(metricas["frequencia_media"])

            score = self._calc_score_dcr(
                metricas=metricas,
                diversidade=diversidade,
                contraste=contraste,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            resultados.append({
                "sequencia": seq,
                "score": score,
                "detail": {
                    "metricas": metricas,
                    "diversidade": diversidade,
                    "contraste": contraste,
                    "score_dcr": score,
                }
            })

        # ---------------- ordenação ----------------
        resultados.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": "v3.0",
            "score_model": "DCR",
            "top": resultados[:top_n],
        }
