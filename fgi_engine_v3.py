
# fgi_engine_v3.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import math


class MotorFGI_V3:
    """
    V3 — Contraste / DCR (rank final)

    CONTRATO DE ENTRADA (candidatos):
      [
        {
          "sequencia": [int, ...],
          "score": float (opcional),
          "detail": {
              "scf_total": float,                 # vindo do V2
              "metricas": {str: number, ...}      # dict numérico vindo do V2
          }
        },
        ...
      ]

    SAÍDA:
      {
        "engine_used": "v3",
        "schema_version": "v3.1",
        "score_mode": "dcr_contrast",
        "top": [
          { "sequencia": [...], "score": float, "detail": {...} },
          ...
        ]
      }
    """

    def __init__(self) -> None:
        pass

    # =========================
    # Utils
    # =========================
    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

    @staticmethod
    def _jaccard(a: List[int], b: List[int]) -> float:
        sa = set(int(x) for x in a)
        sb = set(int(x) for x in b)
        if not sa and not sb:
            return 1.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return float(inter) / float(union) if union else 0.0

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return default

    # =========================
    # Core: contraste (DCR)
    # =========================
    def _collect_metric_keys(self, candidatos: List[Dict[str, Any]]) -> List[str]:
        keys = set()
        for c in candidatos:
            detail = c.get("detail") if isinstance(c.get("detail"), dict) else {}
            metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else {}
            for k, v in metricas.items():
                if self._is_number(v):
                    keys.add(str(k))
        return sorted(keys)

    def _stats(self, candidatos: List[Dict[str, Any]], keys: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        # mean / std por métrica (std com epsilon)
        mean: Dict[str, float] = {k: 0.0 for k in keys}
        var: Dict[str, float] = {k: 0.0 for k in keys}

        if not candidatos or not keys:
            return mean, {k: 1.0 for k in keys}

        # média
        n = 0
        for c in candidatos:
            detail = c.get("detail") if isinstance(c.get("detail"), dict) else {}
            metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else {}
            ok = False
            for k in keys:
                if k in metricas and self._is_number(metricas[k]):
                    mean[k] += float(metricas[k])
                    ok = True
            if ok:
                n += 1
        if n <= 0:
            return mean, {k: 1.0 for k in keys}
        for k in keys:
            mean[k] /= float(n)

        # variância
        for c in candidatos:
            detail = c.get("detail") if isinstance(c.get("detail"), dict) else {}
            metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else {}
            for k in keys:
                if k in metricas and self._is_number(metricas[k]):
                    d = float(metricas[k]) - mean[k]
                    var[k] += d * d
        for k in keys:
            var[k] = var[k] / float(max(n, 1))

        std = {k: math.sqrt(var[k]) if var[k] > 0 else 1.0 for k in keys}
        # evita std zero
        for k in keys:
            if not math.isfinite(std[k]) or std[k] <= 0:
                std[k] = 1.0
        return mean, std

    def _contrast_score(self, metricas: Dict[str, Any], mean: Dict[str, float], std: Dict[str, float], keys: List[str]) -> float:
        # distância z-normalizada ao centróide (contraste estrutural)
        if not keys:
            return 0.0
        acc = 0.0
        used = 0
        for k in keys:
            if k in metricas and self._is_number(metricas[k]):
                z = (float(metricas[k]) - mean[k]) / std[k]
                acc += z * z
                used += 1
        if used <= 0:
            return 0.0
        return math.sqrt(acc / float(used))

    # =========================
    # Seleção com diversidade
    # =========================
    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int = 10,
        alpha_contraste: float = 0.55,
        beta_diversidade: float = 0.30,
        gamma_base: float = 0.15,
        jaccard_penalty_threshold: float = 0.75,
    ) -> Dict[str, Any]:

        # blindagem de entrada
        if not isinstance(candidatos, list) or not candidatos:
            return {
                "engine_used": "v3",
                "schema_version": "v3.1",
                "score_mode": "dcr_contrast",
                "top": [],
            }

        alpha = self._safe_float(alpha_contraste, 0.55)
        beta = self._safe_float(beta_diversidade, 0.30)
        gamma = self._safe_float(gamma_base, 0.15)
        thr = self._safe_float(jaccard_penalty_threshold, 0.75)

        # normaliza candidatos: exige sequencia + metricas + scf_total
        norm: List[Dict[str, Any]] = []
        for c in candidatos:
            if not isinstance(c, dict):
                continue
            seq = c.get("sequencia")
            if not isinstance(seq, (list, tuple)) or not seq:
                continue

            detail = c.get("detail") if isinstance(c.get("detail"), dict) else {}
            metricas = detail.get("metricas") if isinstance(detail.get("metricas"), dict) else None
            scf_total = detail.get("scf_total", c.get("score", 0.0))

            if not isinstance(metricas, dict):
                # contrato: V3 depende de metricas do V2
                raise ValueError("V3 exige metricas do V2 (detail.metricas).")

            norm.append(
                {
                    "sequencia": [int(x) for x in seq],
                    "score": self._safe_float(c.get("score", 0.0), 0.0),
                    "detail": {
                        "scf_total": self._safe_float(scf_total, 0.0),
                        "metricas": metricas,
                    },
                }
            )

        if not norm:
            return {
                "engine_used": "v3",
                "schema_version": "v3.1",
                "score_mode": "dcr_contrast",
                "top": [],
            }

        # estatísticas globais para contraste
        keys = self._collect_metric_keys(norm)
        mean, std = self._stats(norm, keys)

        # score bruto (sem diversidade ainda)
        scored: List[Dict[str, Any]] = []
        for c in norm:
            metricas = c["detail"]["metricas"]
            scf_total = self._safe_float(c["detail"].get("scf_total", 0.0), 0.0)
            contrast = self._contrast_score(metricas, mean, std, keys)

            # base: SCF total (V2), contraste: distância, diversidade entra na seleção
            base_component = scf_total
            contrast_component = contrast

            pre_score = (gamma * base_component) + (alpha * contrast_component)

            scored.append(
                {
                    "sequencia": c["sequencia"],
                    "score": float(pre_score),
                    "detail": {
                        "scf_total": float(scf_total),
                        "metricas": metricas,
                        "score_components": {
                            "base_scf": float(base_component),
                            "contraste": float(contrast_component),
                        },
                    },
                }
            )

        # ordenação inicial por contraste/base (determinística)
        scored.sort(key=lambda x: (x["score"], x["detail"]["scf_total"]), reverse=True)

        # seleção final com diversidade (penaliza redundância via Jaccard)
        top_n = int(top_n) if isinstance(top_n, int) else 10
        top_n = max(1, min(top_n, len(scored)))

        escolhidos: List[Dict[str, Any]] = []
        for cand in scored:
            if len(escolhidos) >= top_n:
                break

            if not escolhidos:
                cand["detail"]["diversidade"] = 1.0
                cand["detail"]["redundancia_max_jaccard"] = 0.0
                cand["detail"]["score_final"] = cand["score"]
                escolhidos.append(cand)
                continue

            max_j = 0.0
            for picked in escolhidos:
                j = self._jaccard(cand["sequencia"], picked["sequencia"])
                if j > max_j:
                    max_j = j

            diversidade = 1.0 - max_j  # maior é melhor
            penal = 0.0
            if max_j >= thr:
                # penaliza forte quando passa do threshold
                penal = (max_j - thr) / max(1e-9, (1.0 - thr))
                penal = max(0.0, min(1.0, penal))

            # score final incorporando diversidade e penalidade
            score_final = cand["score"] + (beta * diversidade) - (beta * penal)

            cand["detail"]["diversidade"] = float(diversidade)
            cand["detail"]["redundancia_max_jaccard"] = float(max_j)
            cand["detail"]["score_final"] = float(score_final)

            # regra: aceita, mas empurra redundantes pro fim
            cand["_score_final_internal"] = float(score_final)
            escolhidos.append(cand)

        # reordena pelo score_final
        for c in escolhidos:
            if "_score_final_internal" not in c:
                c["_score_final_internal"] = float(c["detail"].get("score_final", c["score"]))

        escolhidos.sort(key=lambda x: x["_score_final_internal"], reverse=True)
        for c in escolhidos:
            c["score"] = float(c["_score_final_internal"])
            c.pop("_score_final_internal", None)

        return {
            "engine_used": "v3",
            "schema_version": "v3.1",
            "score_mode": "dcr_contrast",
            "top": escolhidos,
        }
```0
