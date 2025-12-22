# fgi_engine_v2.py
from __future__ import annotations

from typing import List, Dict, Any, Optional


class MotorFGI_V2:
    """
    V2 — Direcional / SCF (BLINDADO)

    CONTRATO GARANTIDO (sempre retorna):
      - engine_used: "v2"
      - schema_version: "v2.2"
      - score_model: "scf_governing"
      - top: List[Dict]  (nunca None)

    Objetivo aqui é: NUNCA quebrar com 500 por falta/forma de contexto.
    Se contexto vier vazio ou diferente, usa fallback seguro.
    """

    def __init__(self) -> None:
        pass

    # =========================
    # Helpers (blindagem)
    # =========================
    @staticmethod
    def _safe_dict(v: Any) -> Dict[str, Any]:
        return v if isinstance(v, dict) else {}

    @staticmethod
    def _safe_list(v: Any) -> List[Any]:
        return v if isinstance(v, list) else []

    @staticmethod
    def _coerce_seq(seq: Any) -> Optional[List[int]]:
        """
        Garante sequência como List[int] com valores plausíveis.
        Retorna None se não der pra usar.
        """
        if not isinstance(seq, (list, tuple)) or not seq:
            return None
        out: List[int] = []
        for x in seq:
            try:
                xi = int(x)
            except Exception:
                return None
            out.append(xi)
        return out if out else None

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    # =========================
    # Contexto / DNA
    # =========================
    def _extract_freq(self, contexto_lab: Dict[str, Any]) -> Dict[str, float]:
        """
        Esperado:
          contexto_lab["dna_last25"]["frequencia"] = { "1": 3, "2": 7, ... }

        Blindagem:
          - se não existir, retorna {}
          - se valores não forem numéricos, ignora/zera
        """
        contexto = self._safe_dict(contexto_lab)
        dna_last25 = self._safe_dict(contexto.get("dna_last25"))
        freq_raw = self._safe_dict(dna_last25.get("frequencia"))

        freq: Dict[str, float] = {}
        for k, v in freq_raw.items():
            try:
                key = str(k)
                val = float(v)
            except Exception:
                continue
            freq[key] = val
        return freq

    # =========================
    # Métricas
    # =========================
    def _calc_metricas_basicas(self, seq: List[int], contexto_lab: Dict[str, Any]) -> Dict[str, float]:
        """
        Métricas mínimas e determinísticas.
        Sempre retorna dict numérico.
        """
        freq = self._extract_freq(contexto_lab)

        soma_freq = 0.0
        for n in seq:
            soma_freq += self._safe_float(freq.get(str(n), 0.0), 0.0)

        tamanho = float(len(seq))
        freq_media = (soma_freq / tamanho) if tamanho > 0 else 0.0

        # bônus simples por "range" (evita tudo colado)
        # não é sofisticado: só ajuda a estabilizar ranking sem depender de nada externo
        try:
            span = float(max(seq) - min(seq)) if seq else 0.0
        except Exception:
            span = 0.0

        return {
            "frequencia_media": float(freq_media),
            "soma_frequencia": float(soma_freq),
            "tamanho": float(tamanho),
            "span": float(span),
        }

    def _calc_scf_total(self, metrics: Dict[str, float]) -> float:
        """
        SCF total — contínuo, estável, sem explosão.
        Pesos simples.
        """
        m = self._safe_dict(metrics)
        freq_media = self._safe_float(m.get("frequencia_media"), 0.0)
        tamanho = self._safe_float(m.get("tamanho"), 0.0)
        span = self._safe_float(m.get("span"), 0.0)

        # Normalizações defensivas
        # (span típico 1..24 em lotofácil, mas aqui não assumimos nada)
        span_norm = span / (span + 10.0) if span >= 0 else 0.0

        # Score: mistura frequência + estabilidade pelo tamanho + dispersão leve
        scf = (0.65 * freq_media) + (0.25 * tamanho) + (0.10 * span_norm)
        return float(scf)

    # =========================
    # API principal
    # =========================
    def rerank(
        self,
        candidatos: List[List[int]],
        contexto_lab: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Entrada:
          candidatos: lista de sequências (List[int])
          contexto_lab: dict (pode vir vazio)
          overrides: dict (pode vir vazio)

        Saída:
          dict com top ordenado
        """
        ov = self._safe_dict(overrides)
        top_n = 10
        try:
            top_n = int(ov.get("top_n", 10))
        except Exception:
            top_n = 10
        if top_n <= 0:
            top_n = 10

        # Blindagem total do input
        cand_list = self._safe_list(candidatos)
        resultados: List[Dict[str, Any]] = []

        for raw_seq in cand_list:
            seq = self._coerce_seq(raw_seq)
            if not seq:
                continue

            metrics = self._calc_metricas_basicas(seq, self._safe_dict(contexto_lab))
            scf_total = self._calc_scf_total(metrics)

            resultados.append(
                {
                    "sequencia": [int(x) for x in seq],
                    "score": float(scf_total),
                    "detail": {
                        "metrics": {k: float(v) for k, v in metrics.items()},
                        "scf_total": float(scf_total),
                    },
                }
            )

        # Ordenação estável (se empate, mantém determinismo por sequência)
        resultados.sort(
            key=lambda x: (
                -self._safe_float(x.get("score"), 0.0),
                x.get("sequencia", []),
            )
        )

        return {
            "engine_used": "v2",
            "schema_version": "v2.2",
            "score_model": "scf_governing",
            "top": resultados[:top_n],
            "meta": {
                "received_candidates": len(cand_list),
                "scored_candidates": len(resultados),
                "top_n": top_n,
                "context_has_dna": bool(self._safe_dict(contexto_lab).get("dna_last25")),
                "context_has_freq": bool(self._extract_freq(self._safe_dict(contexto_lab))),
            },
        }
```0
