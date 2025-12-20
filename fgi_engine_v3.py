
# fgi_engine_v3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class CandidateV3:
    sequencia: List[int]
    scf_total: float
    metricas: Dict[str, float]
    score: float = 0.0  # score anterior (opcional)


class MotorFGI_V3:
    """
    V3 — Contraste / DCR (rank final)
    Objetivo: rankear candidatos (normalmente vindos do V2) com:
      - contraste (contra base/mediana do lote)
      - diversidade (penaliza redundância entre os selecionados)
      - base (scf_total ou score anterior)

    CONTRATO DE ENTRADA (aceita tolerâncias):
      candidatos: List[Dict] com chaves mínimas:
        - "sequencia": List[int]
        - "detail": {"scf_total": float, "metricas": Dict[str, number]}
      Também aceita:
        - "metricas" no topo (fallback)
        - "scf_total" no topo (fallback)
        - "score" no topo (fallback)

    GARANTIA:
      - nunca levanta exception para o app (retorna resultado estável)
      - descarta candidatos inválidos sem derrubar o servidor
    """

    def rank(
        self,
        candidatos: List[Dict[str, Any]],
        top_n: int = 10,
        alpha_contraste: float = 0.55,
        beta_diversidade: float = 0.30,
        gamma_base: float = 0.15,
        jaccard_penalty_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        top_n = int(top_n) if isinstance(top_n, (int, float, str)) else 10
        top_n = max(1, min(top_n, 200))

        # Normaliza pesos (não confia em input)
        a = float(alpha_contraste) if self._is_num(alpha_contraste) else 0.55
        b = float(beta_diversidade) if self._is_num(beta_diversidade) else 0.30
        g = float(gamma_base) if self._is_num(gamma_base) else 0.15
        s = a + b + g
        if s <= 0:
            a, b, g = 0.55, 0.30, 0.15
            s = 1.0
        a, b, g = a / s, b / s, g / s

        thr = float(jaccard_penalty_threshold) if self._is_num(jaccard_penalty_threshold) else 0.75
        thr = max(0.0, min(thr, 1.0))

        # 1) Parse defensivo dos candidatos
        parsed = self._parse_candidates(candidatos)

        if not parsed:
            return {
                "engine_used": "v3",
                "schema_version": "v3.1",
                "score_mode": "dcr_contrast",
                "top": [],
                "debug": {
                    "received": len(candidatos) if isinstance(candidatos, list) else 0,
                    "parsed": 0,
                    "reason": "no_valid_candidates",
                },
            }

        # 2) Calcula baseline de métricas para contraste (mediana por métrica)
        metric_keys = self._collect_metric_keys(parsed)
        medians = {k: self._median([c.metricas.get(k, 0.0) for c in parsed]) for k in metric_keys}

        # 3) Pre-score (sem diversidade) para ordenar candidato base
        # contraste = soma |m - mediana| (normalizado)
        pre_scored: List[Tuple[CandidateV3, float, float, float]] = []
        for c in parsed:
            contrast = self._contrast_score(c, medians)
            base = self._base_score(c)
            # diversidade será calculada no greedy
            pre = a * contrast + g * base
            pre_scored.append((c, contrast, base, pre))

        pre_scored.sort(key=lambda t: t[3], reverse=True)

        # 4) Greedy com diversidade (DCR): seleciona top_n penalizando redundância
        selected: List[Dict[str, Any]] = []
        selected_sets: List[set] = []

        # Se top_n > len(parsed), limita
        limit = min(top_n, len(pre_scored))

        for c, contrast, base, _pre in pre_scored:
            if len(selected) >= limit:
                break

            seq_set = set(c.sequencia)

            # diversidade = 1 - max_jaccard_com_selected (ou 1 se vazio)
            if not selected_sets:
                max_j = 0.0
            else:
                max_j = max(self._jaccard(seq_set, ss) for ss in selected_sets)

            diversidade = 1.0 - max_j
            # penaliza forte se acima do threshold
            penalty = 0.0
            if max_j >= thr:
                # penalidade cresce com o excesso
                penalty = (max_j - thr) / max(1e-9, (1.0 - thr))
                penalty = max(0.0, min(penalty, 1.0))

            final = (a * contrast) + (b * diversidade) + (g * base)
            final = final * (1.0 - 0.50 * penalty)

            selected.append(
                {
                    "sequencia": c.sequencia,
                    "score": float(final),
                    "detail": {
                        "score_components": {
                            "contraste": float(contrast),
                            "diversidade": float(diversidade),
                            "base": float(base),
                        },
                        "max_jaccard": float(max_j),
                        "penalty": float(penalty),
                        "scf_total": float(c.scf_total),
                        "metricas": c.metricas,
                    },
                }
            )
            selected_sets.append(seq_set)

        # 5) Ordena final por score e retorna
        selected.sort(key=lambda it: float(it.get("score", 0.0)), reverse=True)

        return {
            "engine_used": "v3",
            "schema_version": "v3.1",
            "score_mode": "dcr_contrast",
            "top": selected,
            "weights": {
                "alpha_contraste": a,
                "beta_diversidade": b,
                "gamma_base": g,
                "jaccard_penalty_threshold": thr,
            },
            "debug": {
                "received": len(candidatos) if isinstance(candidatos, list) else 0,
                "parsed": len(parsed),
                "selected": len(selected),
                "metric_keys": metric_keys[:50],
            },
        }

    # =========================
    # Internals (defensivos)
    # =========================

    def _parse_candidates(self, candidatos: Any) -> List[CandidateV3]:
        if not isinstance(candidatos, list):
            return []

        out: List[CandidateV3] = []
        for it in candidatos:
            try:
                c = self._parse_one(it)
                if c is not None:
                    out.append(c)
            except Exception:
                # nunca derruba o servidor
                continue
        return out

    def _parse_one(self, it: Any) -> Optional[CandidateV3]:
        if isinstance(it, CandidateV3):
            # garante tipos
            seq = [int(x) for x in it.sequencia] if isinstance(it.sequencia, list) else []
            if not seq:
                return None
            metricas = self._coerce_metricas(it.metricas)
            scf_total = float(it.scf_total) if self._is_num(it.scf_total) else 0.0
            score = float(it.score) if self._is_num(it.score) else 0.0
            return CandidateV3(sequencia=seq, scf_total=scf_total, metricas=metricas, score=score)

        if not isinstance(it, dict):
            return None

        seq = it.get("sequencia")
        if not isinstance(seq, (list, tuple)) or not seq:
            return None
        seq_list = [int(x) for x in seq]

        detail = it.get("detail") if isinstance(it.get("detail"), dict) else {}
        metricas_raw = None

        # metricas preferenciais: detail.metricas
        if isinstance(detail.get("metricas"), dict):
            metricas_raw = detail.get("metricas")
        elif isinstance(it.get("metricas"), dict):
            metricas_raw = it.get("metricas")
        else:
            metricas_raw = {}

        metricas = self._coerce_metricas(metricas_raw)

        # scf_total preferencial: detail.scf_total
        scf_total = detail.get("scf_total", None)
        if scf_total is None:
            scf_total = it.get("scf_total", None)
        if scf_total is None:
            scf_total = it.get("score", 0.0)

        score = it.get("score", 0.0)

        scf_total_f = float(scf_total) if self._is_num(scf_total) else 0.0
        score_f = float(score) if self._is_num(score) else 0.0

        return CandidateV3(
            sequencia=seq_list,
            scf_total=scf_total_f,
            metricas=metricas,
            score=score_f,
        )

    def _coerce_metricas(self, metricas: Any) -> Dict[str, float]:
        if not isinstance(metricas, dict):
            return {}

        out: Dict[str, float] = {}
        for k, v in metricas.items():
            if not isinstance(k, str):
                continue
            if self._is_num(v):
                out[k] = float(v)
            else:
                # tenta converter string numérica
                try:
                    out[k] = float(str(v).strip())
                except Exception:
                    continue

        # remove NaN/Inf
        clean: Dict[str, float] = {}
        for k, v in out.items():
            if self._finite(v):
                clean[k] = v
        return clean

    def _collect_metric_keys(self, parsed: List[CandidateV3]) -> List[str]:
        keys = set()
        for c in parsed:
            for k in c.metricas.keys():
                keys.add(k)
        # ordena para determinismo
        return sorted(list(keys))

    def _contrast_score(self, c: CandidateV3, medians: Dict[str, float]) -> float:
        # contraste = média do |m - mediana| nas métricas presentes
        if not medians:
            return 0.0
        diffs: List[float] = []
        for k, m0 in medians.items():
            v = c.metricas.get(k, 0.0)
            diffs.append(abs(float(v) - float(m0)))
        if not diffs:
            return 0.0
        # normaliza por (1 + mediana_abs) para estabilidade
        denom = 1.0 + self._mean([abs(float(x)) for x in medians.values()]) if medians else 1.0
        return float(self._mean(diffs) / max(1e-9, denom))

    def _base_score(self, c: CandidateV3) -> float:
        # base principal = scf_total (se existir), senão score
        base = c.scf_total if self._finite(c.scf_total) else c.score
        if not self._finite(base):
            base = 0.0
        # squash para 0..1 (sigmoid-like estável)
        return float(1.0 / (1.0 + math.exp(-float(base))))

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        uni = len(a.union(b))
        if uni <= 0:
            return 0.0
        return float(inter / uni)

    # =========================
    # Utils numéricos
    # =========================

    def _is_num(self, x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _finite(self, x: Any) -> bool:
        try:
            v = float(x)
            return math.isfinite(v)
        except Exception:
            return False

    def _mean(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(sum(xs) / max(1, len(xs)))

    def _median(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = [float(x) for x in xs if self._finite(x)]
        if not ys:
            return 0.0
        ys.sort()
        n = len(ys)
        mid = n // 2
        if n % 2 == 1:
            return ys[mid]
        return (ys[mid - 1] + ys[mid]) / 2.0
```0
