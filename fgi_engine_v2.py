
# fgi_engine_v2.py
from typing import List, Dict, Any
import math


class MotorFGI_V2:
    """
    V2 — Direcional / SCF
    CONTRATO GARANTIDO:
      - sequencia
      - score
      - detail.metricas (dict numérico)
      - detail.scf_total (float)
    """

    def __init__(self):
        pass

    # =========================================================
    # MÉTRICAS BÁSICAS (sempre retorna dict numérico)
    # =========================================================
    def _calc_metricas_basicas(
        self,
        seq: List[int],
        contexto_lab: Dict[str, Any]
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

    # =========================================================
    # SCF TOTAL — contínuo, estável
    # =========================================================
    def _calc_scf_total(self, metricas: Dict[str, float]) -> float:
        return (
            0.6 * metricas.get("frequencia_media", 0.0)
            + 0.4 * metricas.get("tamanho", 0.0)
        )

    # =========================================================
    # RERANK PRINCIPAL
    # =========================================================
    def rerank(
        self,
        candidatos: List[List[int]],
        contexto_lab: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:

        top_n = int(overrides.get("top_n", 10))

        resultados: List[Dict[str, Any]] = []

        for seq in candidatos:
            # blindagem
            if not isinstance(seq, (list, tuple)) or not seq:
                continue

            metricas = self._calc_metricas_basicas(seq, contexto_lab)
            scf_total = float(self._calc_scf_total(metricas))

            resultados.append(
                {
                    "sequencia": [int(x) for x in seq],
                    "score": scf_total,
                    "detail": {
                        "metricas": metricas,
                        "scf_total": scf_total,
                    },
                }
            )

        # ordenação estável
        resultados.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v2",
            "schema_version": "v2.1",
            "score_mode": "scf_governing",
            "top": resultados[:top_n],
        }
