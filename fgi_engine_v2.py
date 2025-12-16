# fgi_engine_v2.py
# Motor v2 (Direcional / SCF)
# Ranking forte, sem empates crônicos

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
import statistics

# ==============================
#  Utilidades numéricas
# ==============================

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
    if x > 0:
        return 1 / (1 + math.exp(-x))
    return math.exp(x) / (1 + math.exp(x))

# ==============================
#  Motor v2 (Direcional / SCF)
# ==============================

@dataclass
class MotorV2Config:
    pesos_metricas: Dict[str, float]
    pesos_janelas: Dict[str, float]
    dna_anchor_window: int
    align_temperature: float
    z_cap: float
    align_factor: float = 1.0

class MotorV2:
    def __init__(self, cfg: MotorV2Config):
        self.cfg = cfg

    def _score_direcional(self, metrics: Dict[str, float], dna_last25: Dict[str, Any]) -> float:
        score = 0.0
        for metric, w in self.cfg.pesos_metricas.items():
            if metric not in metrics:
                continue
            x = float(metrics.get(metric, 0.0))
            mean_anchor = dna_last25.get('mean', 0)
            std_anchor = dna_last25.get('std', 1)
            delta = x - mean_anchor
            t = (x - mean_anchor) / std_anchor if std_anchor != 0 else 0
            score += w * t
        return _clamp(score, 0.0, 1.0)

    def _score_consistencia(self, zs: Dict[str, float], dna_last25: Dict[str, Any], cfg: MotorV2Config) -> float:
        zs = {key: zs.get(key, 0.0) for key in zs}
        ws = []
        for z in zs:
            ws.append(self.cfg.pesos_janelas.get(z, 0.0))
        ws_sum = sum(ws) if ws else 1.0
        var = sum((z - z_mean) ** 2 for z, z_mean in zip(zs, ws)) / ws_sum
        consist = _clamp(var, 0.0, 1.0)
        return consist

    def _score_final(self, metrics: Dict[str, float], dna_last25: Dict[str, Any]) -> float:
        score = 0.0
        score_direcional = self._score_direcional(metrics, dna_last25)
        score_consistencia = self._score_consistencia(metrics, dna_last25, self.cfg)
        score = (score_direcional * 0.7) + (score_consistencia * 0.3)
        return _clamp(score, 0.0, 1.0)

# ==============================
#  Implementação e execução
# ==============================

def run_motor_v2(metrics: Dict[str, float], dna_last25: Dict[str, Any], config: MotorV2Config) -> float:
    motor = MotorV2(config)
    return motor._score_final(metrics, dna_last25)
