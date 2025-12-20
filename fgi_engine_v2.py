
# fgi_engine_v2.py
from typing import List, Dict, Any


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

    def _calc_metricas_basicas(self, seq: List[int], contexto_lab: Dict[str, Any]) -> Dict[str, float]:
        dna = contexto_lab.get("dna_last25", {})
        freq = dna.get("frequencia", {})

        soma_freq = sum(freq.get(str(n), 0.0) for n in seq)
        tamanho = float(len(seq))
        media = soma_freq / max(tamanho, 1.0)

        return {
            "frequencia_media": float(media),
            "soma_frequencia": float(soma_freq),
            "tamanho": tamanho,
        }

    def _calc_scf_total(self, metricas: Dict[str, float]) -> float:
        return (
            0.6 * metricas.get("frequencia_media", 0.0)
            + 0.4 * metricas.get("tamanho", 0.0)
        )

    def rerank(
        self,
        candidatos: List[List[int]],
        contexto_lab: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:

        top_n = int(overrides.get("top_n", 10))

        resultados: List[Dict[str, Any]] = []

        for seq in candidatos:
            metricas = self._calc_metricas_basicas(seq, contexto_lab)
            scf_total = self._calc_scf_total(metricas)

            resultados.append(
                {
                    "sequencia": seq,
                    "score": float(scf_total),
                    "detail": {
                        "metricas": metricas,
                        "scf_total": float(scf_total),
                    },
                }
            )

        resultados.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v2",
            "schema_version": "v2.1",
            "score_mode": "scf_governing",
            "top": resultados[:top_n],
        }
