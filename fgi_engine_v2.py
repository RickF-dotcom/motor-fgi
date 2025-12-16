
# fgi_engine_v2.py
# Motor v2 (Direcional / SCF)
# Ranking forte, sem empates crônicos

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import statistics


# =========================
#   Utilidades numéricas
# =========================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    xbar = statistics.mean(xs)
    ybar = statistics.mean(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


# =========================
#   DNA helpers
# =========================

def _dig(d: Any, path: List[str]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _dna_get_stats(dna: Dict[str, Any], window: int, metric: str) -> Tuple[float, float]:
    w = str(window)

    for path in (
        ["janelas", w, metric],
        [w, metric],
        ["dna_last25", w, metric],
    ):
        base = _dig(dna, path)
        if isinstance(base, dict):
            mean = _safe_float(base.get("media", base.get("mean", 0.0)))
            std = _safe_float(base.get("desvio", base.get("std", 1.0)))
            return mean, (std if std != 0 else 1.0)

    return 0.0, 1.0


# =========================
#   Métricas
# =========================

def _metric_soma(seq): return sum(seq)
def _metric_pares(seq): return sum(1 for x in seq if x % 2 == 0)
def _metric_adj(seq):
    s = sorted(seq)
    return sum(1 for i in range(1, len(s)) if s[i] - s[i-1] == 1)
def _metric_faixa_1_13(seq): return sum(1 for x in seq if 1 <= x <= 13)
def _metric_repeticao(seq, ref): return len(set(seq) & set(ref)) if ref else 0


# =========================
#   Config
# =========================

@dataclass
class MotorV2Config:
    windows: List[int]
    dna_anchor_window: int
    top_n: int
    max_candidatos: int
    pesos_windows: Dict[str, float]
    pesos_metricas: Dict[str, float]
    redundancy_jaccard_threshold: float
    redundancy_penalty: float
    z_cap: float
    align_temperature: float


# =========================
#   Motor v2
# =========================

class MotorFGI_V2:
    METRICAS = ("soma", "pares", "adj", "faixa_1_13", "repeticao")

    def __init__(self):
        self.base_cfg = MotorV2Config(
            windows=[7, 10, 13, 25],
            dna_anchor_window=13,
            top_n=30,
            max_candidatos=3000,
            pesos_windows={"7": 0.15, "10": 0.25, "13": 0.35, "25": 0.25},
            pesos_metricas={"soma": 0.25, "pares": 0.15, "adj": 0.25, "faixa_1_13": 0.20, "repeticao": 0.15},
            redundancy_jaccard_threshold=0.80,
            redundancy_penalty=0.35,
            z_cap=3.5,
            align_temperature=1.25,
        )

    # =========================
    #   API principal
    # =========================

    def rerank(
        self,
        candidatos: List[Sequence[int]],
        contexto_lab: Dict[str, Any],
        overrides: Dict[str, Any] = None,
    ) -> Dict[str, Any]:

        overrides = overrides or {}
        cfg = self._apply_overrides(overrides)
        dna = contexto_lab.get("dna_last25", {})
        last_draw = contexto_lab.get("ultimo_concurso")

        candidatos = candidatos[:cfg.max_candidatos]

        trend = self._trend_vector(dna, cfg.windows)
        scored = []

        for seq in candidatos:
            s = sorted(seq)
            metrics = self._metrics(s, last_draw)
            zscores = self._zscores(metrics, dna, cfg)

            direcional = self._score_direcional(metrics, dna, trend, cfg)
            consistencia = self._score_consistencia(zscores, cfg)
            ancora = self._score_ancora(metrics, dna, cfg)

            # ========= SCORE FINAL (SCF MANDA) =========
            score_final = (
                0.35 * direcional +
                0.30 * consistencia +
                0.20 * ancora
            )

            scored.append((score_final, s, {
                "scf_total": score_final,
                "direcional": direcional,
                "consistencia": consistencia,
                "ancora": ancora,
                "zscores": zscores,
                "metricas": metrics,
                "redundancia_penalidade": 0.0
            }))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ========= ANTI-CLONE =========
        top = []
        selected = []

        for score, seq, detail in scored:
            if len(top) >= cfg.top_n:
                break

            max_sim = max((_jaccard(seq, s2) for s2 in selected), default=0.0)
            if max_sim >= cfg.redundancy_jaccard_threshold:
                penal = cfg.redundancy_penalty * (
                    (max_sim - cfg.redundancy_jaccard_threshold) /
                    (1.0 - cfg.redundancy_jaccard_threshold)
                )
                penal = _clamp(penal, 0.0, 0.85)
                score *= (1.0 - penal)
                detail["redundancia_penalidade"] = penal
                if penal > 0.6:
                    continue

            top.append({
                "sequencia": seq,
                "score": score,
                "detail": detail
            })
            selected.append(seq)

        return {
            "engine": "v2",
            "config_usada": cfg.__dict__,
            "top": top,
            "debug": {
                "trend_vector": trend,
                "candidatos_processados": len(candidatos)
            }
        }

    # =========================
    #   Internos
    # =========================

    def _apply_overrides(self, o):
        cfg = self.base_cfg
        return MotorV2Config(
            windows=o.get("windows", cfg.windows),
            dna_anchor_window=o.get("dna_anchor_window", cfg.dna_anchor_window),
            top_n=o.get("top_n", cfg.top_n),
            max_candidatos=o.get("max_candidatos", cfg.max_candidatos),
            pesos_windows=o.get("pesos_windows", cfg.pesos_windows),
            pesos_metricas=o.get("pesos_metricas", cfg.pesos_metricas),
            redundancy_jaccard_threshold=o.get("redundancy_jaccard_threshold", cfg.redundancy_jaccard_threshold),
            redundancy_penalty=o.get("redundancy_penalty", cfg.redundancy_penalty),
            z_cap=o.get("z_cap", cfg.z_cap),
            align_temperature=o.get("align_temperature", cfg.align_temperature),
        )

    def _metrics(self, seq, last_draw):
        return {
            "soma": _metric_soma(seq),
            "pares": _metric_pares(seq),
            "adj": _metric_adj(seq),
            "faixa_1_13": _metric_faixa_1_13(seq),
            "repeticao": _metric_repeticao(seq, last_draw),
        }

    def _zscores(self, metrics, dna, cfg):
        out = {}
        for m in self.METRICAS:
            out[m] = {}
            for w in cfg.windows:
                mean, std = _dna_get_stats(dna, w, m)
                z = (metrics[m] - mean) / std
                out[m][w] = _clamp(z, -cfg.z_cap, cfg.z_cap)
        return out

    def _trend_vector(self, dna, windows):
        xs = [float(w) for w in windows]
        return {
            m: _linear_slope(xs, [_dna_get_stats(dna, w, m)[0] for w in windows])
            for m in self.METRICAS
        }

    def _score_direcional(self, metrics, dna, trend, cfg):
        score = 0.0
        for m, w in cfg.pesos_metricas.items():
            mean, std = _dna_get_stats(dna, cfg.dna_anchor_window, m)
            delta = (metrics[m] - mean) / std
            t = trend[m]
            align = 0.5 if abs(t) < 1e-9 else _sigmoid(cfg.align_temperature * (delta if t > 0 else -delta))
            score += w * align
        return score

    def _score_consistencia(self, zscores, cfg):
        score = 0.0
        for m, w in cfg.pesos_metricas.items():
            zs = [zscores[m][w_] for w_ in cfg.windows]
            std = statistics.pstdev(zs) if len(zs) > 1 else 0.0
            score += w * (1.0 - _clamp(std / 2.0, 0.0, 1.0))
        return score

    def _score_ancora(self, metrics, dna, cfg):
        score = 0.0
        for m, w in cfg.pesos_metricas.items():
            mean, std = _dna_get_stats(dna, cfg.dna_anchor_window, m)
            z = abs((metrics[m] - mean) / std)
            score += w * math.exp(-((z - 0.75) ** 2) / 0.8)
        return score
