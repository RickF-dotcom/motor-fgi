# fgi_engine_v3.py
from typing import List, Dict, Any
import math


class MotorFGI_V3:
    """
    V3 – Contraste / DCR
    Blindado contra inconsistências do V2
    """

    def __init__(self):
        pass

    # ==========================================================
    # Normalização dura do contrato vindo do V2
    # ==========================================================
    def _normalize_v2_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {}

        seq = item.get("sequencia") or item.get("sequence")
        if not isinstance(seq, (list, tuple)) or not seq:
            return {}

        detail = item.get("detail", {})
        if not isinstance(detail, dict):
            detail = {}

        # aceita metricas ou metrics
        metricas = (
            detail.get("metricas")
            or detail.get("metrics")
            or {}
        )

        if not isinstance(metricas, dict):
            metricas = {}

        # força valores numéricos
        metricas_num = {}
        for k, v in metricas.items():
            try:
                metricas_num[k] = float(v)
            except Exception:
                continue

        scf_total = detail.get("scf_total", item.get("score", 0.0))
        try:
            scf_total = float(scf_total)
        except Exception:
            scf_total = 0.0

        return {
            "sequencia": [int(x) for x in seq],
            "metricas": metricas_num,
            "scf_total": scf_total,
        }

    # ==========================================================
    # Score DCR simples, estável e contínuo
    # ==========================================================
    def _calc_dcr_score(self, metricas: Dict[str, float], scf_total: float) -> float:
        base = scf_total

        # reforços defensivos
        diversidade = metricas.get("diversidade", 0.0)
        frequencia = metricas.get("frequencia_media", 0.0)

        score = (
            0.6 * base
            + 0.25 * diversidade
            + 0.15 * frequencia
        )

        if math.isnan(score) or math.isinf(score):
            return 0.0

        return float(score)

    # ==========================================================
    # RERANK PRINCIPAL
    # ==========================================================
    def rerank(
        self,
        candidatos_v2: List[Dict[str, Any]],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:

        top_n = int(overrides.get("top_n", 10))

        if not isinstance(candidatos_v2, list) or not candidatos_v2:
            return {
                "engine_used": "v3",
                "schema_version": "v3.0",
                "error": "lista_v2_vazia",
                "top": [],
            }

        normalizados: List[Dict[str, Any]] = []

        for item in candidatos_v2:
            norm = self._normalize_v2_item(item)
            if not norm:
                continue

            score = self._calc_dcr_score(
                norm["metricas"],
                norm["scf_total"],
            )

            normalizados.append({
                "sequencia": norm["sequencia"],
                "score": score,
                "detail": {
                    "metricas": norm["metricas"],
                    "scf_total": norm["scf_total"],
                    "dcr_score": score,
                },
            })

        if not normalizados:
            return {
                "engine_used": "v3",
                "schema_version": "v3.0",
                "error": "normalizacao_falhou",
                "top": [],
            }

        normalizados.sort(key=lambda x: x["score"], reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": "v3.0",
            "score_mode": "contrast_dcr",
            "top": normalizados[:top_n],
        }
