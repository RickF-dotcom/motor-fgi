
# fgi_engine_v2.py
# Motor v2 (Direcional / SCF)
# - Re-ranqueia candidatos (vindos do Grupo de Milhões / Motor v1) com:
#   1) alinhamento direcional (tendência do regime nas janelas)
#   2) consistência multi-janela
#   3) penalização de redundância (anti-clone)
#   4) âncora temporal (DNA-13 ou DNA-14, com 25 como contexto)

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
    # sigmoid estável
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """
    Regressão linear simples: slope = cov(x,y)/var(x)
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    xbar = statistics.mean(xs)
    ybar = statistics.mean(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return (inter / uni) if uni else 0.0


# =========================
#   Extração DNA (robusta)
# =========================

def _dig(d: Any, path: List[str]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _dna_get_stats(dna_last25: Dict[str, Any], window: int, metric: str) -> Tuple[float, float]:
    """
    Tenta extrair (media, desvio) do DNA, tolerando formatos comuns.
    Se não achar, retorna (0, 1).
    Formatos aceitos (exemplos):
      dna["janelas"]["13"]["soma"]["media"]
      dna["13"]["soma"]["media"]
      dna["dna_last25"]["13"]["soma"]["media"]
    """
    w = str(window)

    base = _dig(dna_last25, ["janelas", w, metric])
    if isinstance(base, dict):
        mean = _safe_float(base.get("media", base.get("mean", base.get("avg", 0.0))), 0.0)
        std = _safe_float(base.get("desvio", base.get("std", base.get("sigma", 1.0))), 1.0)
        return mean, (std if std != 0 else 1.0)

    base = _dig(dna_last25, [w, metric])
    if isinstance(base, dict):
        mean = _safe_float(base.get("media", base.get("mean", base.get("avg", 0.0))), 0.0)
        std = _safe_float(base.get("desvio", base.get("std", base.get("sigma", 1.0))), 1.0)
        return mean, (std if std != 0 else 1.0)

    base = _dig(dna_last25, ["dna_last25", w, metric])
    if isinstance(base, dict):
        mean = _safe_float(base.get("media", base.get("mean", base.get("avg", 0.0))), 0.0)
        std = _safe_float(base.get("desvio", base.get("std", base.get("sigma", 1.0))), 1.0)
        return mean, (std if std != 0 else 1.0)

    return 0.0, 1.0


# =========================
#   Métricas da sequência
# =========================

def _metric_soma(seq: Sequence[int]) -> int:
    return int(sum(seq))


def _metric_pares(seq: Sequence[int]) -> int:
    return int(sum(1 for x in seq if x % 2 == 0))


def _metric_adjacencias(seq: Sequence[int]) -> int:
    s = sorted(seq)
    adj = 0
    for i in range(1, len(s)):
        if s[i] - s[i - 1] == 1:
            adj += 1
    return adj


def _metric_faixa_1_13(seq: Sequence[int]) -> int:
    return int(sum(1 for x in seq if 1 <= x <= 13))


def _metric_repeticao(seq: Sequence[int], ref: Optional[Sequence[int]]) -> int:
    if not ref:
        return 0
    return int(len(set(seq) & set(ref)))


# =========================
#   Config / Resultado
# =========================

@dataclass
class MotorV2Config:
    windows: List[int] = None
    dna_anchor_window: int = 13
    top_n: int = 30
    max_candidatos: int = 3000

    # Pesos por janela (consistência e zscores)
    pesos_windows: Dict[str, float] = None

    # Pesos por métrica
    pesos_metricas: Dict[str, float] = None

    # Anti-clone
    redundancy_jaccard_threshold: float = 0.80
    redundancy_penalty: float = 0.35  # multiplicativo no greedy

    # Guard rails
    z_cap: float = 3.5
    align_temperature: float = 1.25


@dataclass
class MotorV2ScoreDetail:
    scf_total: float
    direcional: float
    consistencia: float
    ancora: float
    redundancia_penalidade: float
    metricas: Dict[str, float]
    zscores: Dict[str, Dict[str, float]]  # metric -> window(str) -> z(float)


# =========================
#   Motor v2 (SCF)
# =========================

class MotorFGI_V2:
    METRICAS_SUPORTADAS = ("soma", "pares", "adj", "faixa_1_13", "repeticao")

    def __init__(self, config: Optional[MotorV2Config] = None):
        self.base_cfg = config or self._default_cfg()
        self.base_cfg = self._normalize_cfg(self.base_cfg)

    def _default_cfg(self) -> MotorV2Config:
        cfg = MotorV2Config()
        cfg.windows = [7, 10, 13, 25]
        cfg.pesos_windows = {"7": 0.15, "10": 0.25, "13": 0.35, "25": 0.25}
        cfg.pesos_metricas = {"soma": 0.25, "pares": 0.15, "adj": 0.25, "faixa_1_13": 0.20, "repeticao": 0.15}
        return cfg

    def _normalize_cfg(self, cfg: MotorV2Config) -> MotorV2Config:
        # defaults defensivos
        if cfg.windows is None:
            cfg.windows = [7, 10, 13, 25]
        if cfg.pesos_windows is None:
            cfg.pesos_windows = {str(w): 1.0 / len(cfg.windows) for w in cfg.windows}
        if cfg.pesos_metricas is None:
            cfg.pesos_metricas = {m: 1.0 / len(self.METRICAS_SUPORTADAS) for m in self.METRICAS_SUPORTADAS}

        # normaliza pesos de janela
        sw = sum(max(0.0, _safe_float(v)) for v in cfg.pesos_windows.values())
        if sw <= 0:
            cfg.pesos_windows = {str(w): 1.0 / len(cfg.windows) for w in cfg.windows}
        else:
            cfg.pesos_windows = {str(k): max(0.0, _safe_float(v)) / sw for k, v in cfg.pesos_windows.items()}

        # normaliza pesos de métrica
        sm = sum(max(0.0, _safe_float(v)) for v in cfg.pesos_metricas.values())
        if sm <= 0:
            cfg.pesos_metricas = {m: 1.0 / len(self.METRICAS_SUPORTADAS) for m in self.METRICAS_SUPORTADAS}
        else:
            cfg.pesos_metricas = {str(k): max(0.0, _safe_float(v)) / sm for k, v in cfg.pesos_metricas.items()}

        return cfg

    def _apply_overrides(self, overrides: Dict[str, Any]) -> MotorV2Config:
        base = self.base_cfg
        cfg = MotorV2Config(
            windows=list(base.windows),
            dna_anchor_window=int(base.dna_anchor_window),
            top_n=int(base.top_n),
            max_candidatos=int(base.max_candidatos),
            pesos_windows=dict(base.pesos_windows),
            pesos_metricas=dict(base.pesos_metricas),
            redundancy_jaccard_threshold=float(base.redundancy_jaccard_threshold),
            redundancy_penalty=float(base.redundancy_penalty),
            z_cap=float(base.z_cap),
            align_temperature=float(base.align_temperature),
        )

        if isinstance(overrides.get("windows"), list) and overrides["windows"]:
            cfg.windows = [int(x) for x in overrides["windows"]]

        if "dna_anchor_window" in overrides:
            cfg.dna_anchor_window = int(overrides["dna_anchor_window"])
        if "top_n" in overrides:
            cfg.top_n = int(overrides["top_n"])
        if "max_candidatos" in overrides:
            cfg.max_candidatos = int(overrides["max_candidatos"])

        if isinstance(overrides.get("pesos_windows"), dict):
            cfg.pesos_windows = {str(k): float(v) for k, v in overrides["pesos_windows"].items()}
        if isinstance(overrides.get("pesos_metricas"), dict):
            cfg.pesos_metricas = {str(k): float(v) for k, v in overrides["pesos_metricas"].items()}

        if "redundancy_jaccard_threshold" in overrides:
            cfg.redundancy_jaccard_threshold = float(overrides["redundancy_jaccard_threshold"])
        if "redundancy_penalty" in overrides:
            cfg.redundancy_penalty = float(overrides["redundancy_penalty"])
        if "z_cap" in overrides:
            cfg.z_cap = float(overrides["z_cap"])
        if "align_temperature" in overrides:
            cfg.align_temperature = float(overrides["align_temperature"])

        return self._normalize_cfg(cfg)

    def _cfg_to_dict(self, cfg: MotorV2Config) -> Dict[str, Any]:
        return {
            "windows": list(cfg.windows),
            "dna_anchor_window": int(cfg.dna_anchor_window),
            "top_n": int(cfg.top_n),
            "max_candidatos": int(cfg.max_candidatos),
            "pesos_windows": dict(cfg.pesos_windows),
            "pesos_metricas": dict(cfg.pesos_metricas),
            "redundancy_jaccard_threshold": float(cfg.redundancy_jaccard_threshold),
            "redundancy_penalty": float(cfg.redundancy_penalty),
            "z_cap": float(cfg.z_cap),
            "align_temperature": float(cfg.align_temperature),
        }

    # -------------------------
    # API principal
    # -------------------------

    def rerank(
        self,
        candidatos: List[Sequence[int]],
        contexto_lab: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retorna dict pronto pra endpoint:
          {
            "engine": "v2",
            "config_usada": {...},
            "top": [ { "sequencia": [...], "score": ..., "detail": {...} }, ... ],
            "debug": {...}
          }
        """
        contexto_lab = contexto_lab or {}
        cfg = self._apply_overrides(overrides or {})

        # disciplina operacional: corta entrada
        cand = candidatos[: max(0, int(cfg.max_candidatos))]

        dna_last25 = contexto_lab.get("dna_last25") or contexto_lab.get("dna") or {}
        last_draw = (
            contexto_lab.get("last_draw")
            or contexto_lab.get("ultimo_concurso")
            or contexto_lab.get("last_result")
            or None
        )

        trend = self._compute_trend_vector(dna_last25, cfg.windows)

        scored: List[Tuple[float, List[int], MotorV2ScoreDetail]] = []
        for seq in cand:
            s = sorted(int(x) for x in seq)
            metricas = self._compute_metrics(s, last_draw)
            zscores = self._compute_zscores(metricas, dna_last25, cfg)

            direcional = self._score_direcional(metricas, dna_last25, trend, cfg)
            consistencia = self._score_consistencia(zscores, cfg)
            ancora = self._score_ancora(metricas, dna_last25, cfg)

            scf_total = (0.45 * direcional) + (0.35 * consistencia) + (0.20 * ancora)

            detail = MotorV2ScoreDetail(
                scf_total=float(scf_total),
                direcional=float(direcional),
                consistencia=float(consistencia),
                ancora=float(ancora),
                redundancia_penalidade=0.0,
                metricas={k: float(v) for k, v in metricas.items()},
                zscores={m: {str(w): float(z) for w, z in zscores[m].items()} for m in zscores},
            )
            scored.append((float(scf_total), s, detail))

        scored.sort(key=lambda x: x[0], reverse=True)

        # greedy anti-clone
        top: List[Dict[str, Any]] = []
        selected: List[List[int]] = []

        for base_score, seq, detail in scored:
            if len(top) >= int(cfg.top_n):
                break

            max_sim = 0.0
            for s2 in selected:
                sim = _jaccard(seq, s2)
                if sim > max_sim:
                    max_sim = sim

            penal = 0.0
            final_score = base_score
            if selected and max_sim >= cfg.redundancy_jaccard_threshold:
                penal = cfg.redundancy_penalty * (
                    (max_sim - cfg.redundancy_jaccard_threshold)
                    / max(1e-9, (1.0 - cfg.redundancy_jaccard_threshold))
                )
                penal = _clamp(penal, 0.0, 0.85)
                final_score = base_score * (1.0 - penal)

            detail.redundancia_penalidade = float(penal)

            # gate: corta clones agressivos
            if penal > 0.60:
                continue

            top.append(
                {
                    "sequencia": seq,
                    "score": float(final_score),
                    "detail": {
                        "scf_total": float(detail.scf_total),
                        "direcional": float(detail.direcional),
                        "consistencia": float(detail.consistencia),
                        "ancora": float(detail.ancora),
                        "redundancia_penalidade": float(detail.redundancia_penalidade),
                        "metricas": detail.metricas,
                        "zscores": detail.zscores,
                    },
                }
            )
            selected.append(seq)

        return {
            "engine": "v2",
            "config_usada": self._cfg_to_dict(cfg),
            "top": top,
            "debug": {
                "trend_vector": {k: float(v) for k, v in trend.items()},
                "candidatos_recebidos": len(candidatos),
                "candidatos_processados": len(cand),
                "top_retornado": len(top),
            },
        }

    # -------------------------
    # Métricas / zscores / trend
    # -------------------------

    def _compute_metrics(self, seq: List[int], last_draw: Optional[Sequence[int]]) -> Dict[str, float]:
        return {
            "soma": float(_metric_soma(seq)),
            "pares": float(_metric_pares(seq)),
            "adj": float(_metric_adjacencias(seq)),
            "faixa_1_13": float(_metric_faixa_1_13(seq)),
            "repeticao": float(_metric_repeticao(seq, last_draw)),
        }

    def _compute_zscores(
        self,
        metricas: Dict[str, float],
        dna_last25: Dict[str, Any],
        cfg: MotorV2Config,
    ) -> Dict[str, Dict[int, float]]:
        out: Dict[str, Dict[int, float]] = {}
        for metric in self.METRICAS_SUPORTADAS:
            out[metric] = {}
            x = float(metricas.get(metric, 0.0))
            for w in cfg.windows:
                mean, std = _dna_get_stats(dna_last25, w, metric)
                std = std if std != 0 else 1.0
                z = (x - mean) / std
                z = _clamp(z, -cfg.z_cap, cfg.z_cap)
                out[metric][w] = float(z)
        return out

    def _compute_trend_vector(self, dna_last25: Dict[str, Any], windows: List[int]) -> Dict[str, float]:
        xs = [float(w) for w in windows]
        trend: Dict[str, float] = {}
        for metric in self.METRICAS_SUPORTADAS:
            ys: List[float] = []
            for w in windows:
                mean, _std = _dna_get_stats(dna_last25, w, metric)
                ys.append(float(mean))
            trend[metric] = float(_linear_slope(xs, ys))
        return trend

    # -------------------------
    # SCF: direcional / consistência / âncora
    # -------------------------

    def _score_direcional(
        self,
        metricas: Dict[str, float],
        dna_last25: Dict[str, Any],
        trend: Dict[str, float],
        cfg: MotorV2Config,
    ) -> float:
        score = 0.0
        for metric, w_m in cfg.pesos_metricas.items():
            if metric not in self.METRICAS_SUPORTADAS:
                continue

            x = float(metricas.get(metric, 0.0))
            mean_anchor, std_anchor = _dna_get_stats(dna_last25, cfg.dna_anchor_window, metric)
            std_anchor = std_anchor if std_anchor != 0 else 1.0
            delta = (x - mean_anchor) / std_anchor

            t = float(trend.get(metric, 0.0))
            if abs(t) < 1e-12:
                align = 0.5
            else:
                desired = 1.0 if t > 0 else -1.0
                raw = desired * delta
                align = _sigmoid(cfg.align_temperature * raw)

            score += float(w_m) * float(align)

        return float(_clamp(score, 0.0, 1.0))

    def _score_consistencia(self, zscores: Dict[str, Dict[int, float]], cfg: MotorV2Config) -> float:
        parts: List[float] = []
        for metric, w_m in cfg.pesos_metricas.items():
            if metric not in zscores:
                continue

            zs: List[float] = []
            ws: List[float] = []
            for w in cfg.windows:
                zs.append(float(zscores[metric].get(w, 0.0)))
                ws.append(float(cfg.pesos_windows.get(str(w), 0.0)))

            wsum = sum(ws) if sum(ws) > 0 else 1.0
            zbar = sum(z * w for z, w in zip(zs, ws)) / wsum
            var = sum(w * (z - zbar) ** 2 for z, w in zip(zs, ws)) / wsum
            std = math.sqrt(var)

            consist = 1.0 - _clamp(std / 2.0, 0.0, 1.0)

            mag = math.exp(-0.35 * (abs(zbar) ** 2))
            info = 1.0 - mag

            parts.append(float(w_m) * (0.70 * consist + 0.30 * info))

        total = sum(parts) if parts else 0.0
        return float(_clamp(total, 0.0, 1.0))

    def _score_ancora(self, metricas: Dict[str, float], dna_last25: Dict[str, Any], cfg: MotorV2Config) -> float:
        parts: List[float] = []
        for metric, w_m in cfg.pesos_metricas.items():
            if metric not in self.METRICAS_SUPORTADAS:
                continue
            x = float(metricas.get(metric, 0.0))
            mean, std = _dna_get_stats(dna_last25, cfg.dna_anchor_window, metric)
            std = std if std != 0 else 1.0
            z = _clamp((x - mean) / std, -cfg.z_cap, cfg.z_cap)

            az = abs(z)
            anchor_score = math.exp(-((az - 0.75) ** 2) / 0.80)
            parts.append(float(w_m) * float(anchor_score))

        total = sum(parts) if parts else 0.0
        return float(_clamp(total, 0.0, 1.0))
