# fgi_engine_v2.py
# Motor v2 (Direcional / SCF)
# - Re-ranqueia candidatos (vindos do Grupo de Milhões / Motor v1) com:
#   1) alinhamento direcional (tendência do regime nas janelas)
#   2) consistência multi-janela
#   3) penalização de redundância (anti-clone)
#   4) âncora temporal (DNA-13 ou DNA-14, com 25 como contexto)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
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
    # sigmoid estável o suficiente pro nosso range
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """
    Regressão linear simples: slope = cov(x,y)/var(x)
    Retorna 0 se não dá pra calcular.
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
    Se não achar, retorna (0, 1) pra não explodir zscore.
    Formatos aceitos (exemplos):
      dna["janelas"]["13"]["soma"]["media"]
      dna["13"]["soma"]["media"]
      dna["13"]["soma"]["mean"]
      dna["13"]["soma"] = {"media":..., "desvio":...}
    """
    w = str(window)

    # 1) dna["janelas"][w][metric][...]
    base = _dig(dna_last25, ["janelas", w, metric])
    if isinstance(base, dict):
        mean = _safe_float(base.get("media", base.get("mean", base.get("avg", 0.0))), 0.0)
        std = _safe_float(base.get("desvio", base.get("std", base.get("sigma", 1.0))), 1.0)
        return (mean, std if std != 0 else 1.0)

    # 2) dna[w][metric][...]
    base = _dig(dna_last25, [w, metric])
    if isinstance(base, dict):
        mean = _safe_float(base.get("media", base.get("mean", base.get("avg", 0.0))), 0.0)
        std = _safe_float(base.get("desvio", base.get("std", base.get("sigma", 1.0))), 1.0)
        return (mean, std if std != 0 else 1.0)

    # 3) dna["dna_last25"][w][metric][...]
    base = _dig(dna_last25, ["dna_last25", w, metric])
    if isinstance(base, dict):
        mean = _safe_float(base.get("media", base.get("mean", base.get("avg", 0.0))), 0.0)
        std = _safe_float(base.get("desvio", base.get("std", base.get("sigma", 1.0))), 1.0)
        return (mean, std if std != 0 else 1.0)

    return (0.0, 1.0)


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

    # Pesos por métrica (soma/pares/adj/faixa_1_13/repeticao)
    pesos_metricas: Dict[str, float] = None

    # Parâmetros anti-clone
    redundancy_jaccard_threshold: float = 0.80
    redundancy_penalty: float = 0.35  # penaliza score final (multiplicativo no greedy)

    # Guard rails
    z_cap: float = 3.5  # corta zscore extremo
    align_temperature: float = 1.25  # quão “agudo” é o alinhamento direcional


@dataclass
class MotorV2ScoreDetail:
    scf_total: float
    direcional: float
    consistencia: float
    ancora: float
    redundancia_penalidade: float
    metricas: Dict[str, float]
    zscores: Dict[str, Dict[str, float]]  # metric -> window -> z


# =========================
#   Motor v2 (SCF)
# =========================

class MotorFGI_V2:
    """
    Motor v2 (Direcional/SCF):
    - Entrada: candidatos (lista de sequências) + contexto do laboratório (dna_last25, last_draw, etc.)
    - Saída: top_n sequências, com detalhes de score e sem empates crônicos (anti-clone + score contínuo)
    """

    METRICAS_SUPORTADAS = ("soma", "pares", "adj", "faixa_1_13", "repeticao")

    def __init__(self, config: Optional[MotorV2Config] = None):
        self.cfg = config or MotorV2Config()
        if self.cfg.windows is None:
            self.cfg.windows = [7, 10, 13, 25]
        if self.cfg.pesos_windows is None:
            # default coerente com teu plano
            self.cfg.pesos_windows = {"7": 0.15, "10": 0.25, "13": 0.35, "25": 0.25}
        if self.cfg.pesos_metricas is None:
            self.cfg.pesos_metricas = {"soma": 0.25, "pares": 0.15, "adj": 0.25, "faixa_1_13": 0.20, "repeticao": 0.15}

        # normaliza pesos (não aceito soma maluca virar motor instável)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        # windows
        sw = sum(max(0.0, _safe_float(v)) for v in self.cfg.pesos_windows.values())
        if sw <= 0:
            self.cfg.pesos_windows = {str(w): 1.0 / len(self.cfg.windows) for w in self.cfg.windows}
        else:
            self.cfg.pesos_windows = {k: max(0.0, _safe_float(v)) / sw for k, v in self.cfg.pesos_windows.items()}

        # metricas
        sm = sum(max(0.0, _safe_float(v)) for v in self.cfg.pesos_metricas.values())
        if sm <= 0:
            self.cfg.pesos_metricas = {m: 1.0 / len(self.METRICAS_SUPORTADAS) for m in self.METRICAS_SUPORTADAS}
        else:
            self.cfg.pesos_metricas = {k: max(0.0, _safe_float(v)) / sm for k, v in self.cfg.pesos_metricas.items()}

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
        Retorna dict pronto pra responder no endpoint:
          {
            "engine": "v2",
            "config_usada": {...},
            "top": [ { "sequencia": [...], "score": ..., "detail": {...} }, ... ],
            "debug": {...}
          }
        """
        contexto_lab = contexto_lab or {}
        cfg = self._apply_overrides(overrides or {})

        # corta input (disciplina operacional)
        cand = candidatos[: max(0, int(cfg.max_candidatos))]

        dna_last25 = contexto_lab.get("dna_last25") or contexto_lab.get("dna") or {}
        last_draw = (
            contexto_lab.get("last_draw")
            or contexto_lab.get("ultimo_concurso")
            or contexto_lab.get("last_result")
            or None
        )

        # 1) score base SCF (sem anti-clone)
        scored: List[Tuple[float, List[int], MotorV2ScoreDetail]] = []
        trend = self._compute_trend_vector(dna_last25, cfg.windows)  # metric -> slope

        for seq in cand:
            s = sorted(int(x) for x in seq)
            metricas = self._compute_metrics(s, last_draw)
            zscores = self._compute_zscores(metricas, dna_last25, cfg.windows)

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
                zscores={m: {str(w): float(z) see if False else float(z) for w, z in zscores[m].items()} for m in zscores},
            )
            scored.append((float(scf_total), s, detail))

        # 2) ordena por score (desc)
        scored.sort(key=lambda x: x[0], reverse=True)

        # 3) greedy anti-clone (penalização por redundância)
        top: List[Dict[str, Any]] = []
        selected: List[List[int]] = []

        for base_score, seq, detail in scored:
            if len(top) >= int(cfg.top_n):
                break

            # calcula redundância vs selecionados
            max_sim = 0.0
            for s2 in selected:
                sim = _jaccard(seq, s2)
                if sim > max_sim:
                    max_sim = sim

            penal = 0.0
            final_score = base_score
            if selected and max_sim >= cfg.redundancy_jaccard_threshold:
                # penaliza multiplicando (mantém ranking contínuo e evita empates)
                penal = cfg.redundancy_penalty * (max_sim - cfg.redundancy_jaccard_threshold) / max(1e-9, (1.0 - cfg.redundancy_jaccard_threshold))
                penal = _clamp(penal, 0.0, 0.85)
                final_score = base_score * (1.0 - penal)

            detail.redundancia_penalidade = float(penal)

            # gate: se ficou muito penalizado, não entra
            # (isso corta “clones” mesmo que a base seja alta)
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
    # Overrides / Config
    # -------------------------

    def _apply_overrides(self, overrides: Dict[str, Any]) -> MotorV2Config:
        cfg = MotorV2Config(
            windows=list(self.cfg.windows),
            dna_anchor_window=int(self.cfg.dna_anchor_window),
            top_n=int(self.cfg.top_n),
            max_candidatos=int(self.cfg.max_candidatos),
            pesos_windows=dict(self.cfg.pesos_windows),
            pesos_metricas=dict(self.cfg.pesos_metricas),
            redundancy_jaccard_threshold=float(self.cfg.redundancy_jaccard_threshold),
            redundancy_penalty=float(self.cfg.redundancy_penalty),
            z_cap=float(self.cfg.z_cap),
            align_temperature=float(self.cfg.align_temperature),
        )

        # aplica overrides do body do Swagger (tolerante)
        if "windows" in overrides and isinstance(overrides["windows"], list) and overrides["windows"]:
            cfg.windows = [int(x) for x in overrides["windows"]]

        if "dna_anchor_window" in overrides:
            cfg.dna_anchor_window = int(overrides["dna_anchor_window"])

        if "top_n" in overrides:
            cfg.top_n = int(overrides["top_n"])

        if "max_candidatos" in overrides:
            cfg.max_candidatos = int(overrides["max_candidatos"])

        if "pesos_windows" in overrides and isinstance(overrides["pesos_windows"], dict):
            cfg.pesos_windows = {str(k): float(v) for k, v in overrides["pesos_windows"].items()}

        if "pesos_metricas" in overrides and isinstance(overrides["pesos_metricas"], dict):
            cfg.pesos_metricas = {str(k): float(v) for k, v in overrides["pesos_metricas"].items()}

        if "redundancy_jaccard_threshold" in overrides:
            cfg.redundancy_jaccard_threshold = float(overrides["redundancy_jaccard_threshold"])

        if "redundancy_penalty" in overrides:
            cfg.redundancy_penalty = float(overrides["redundancy_penalty"])

        if "z_cap" in overrides:
            cfg.z_cap = float(overrides["z_cap"])

        if "align_temperature" in overrides:
            cfg.align_temperature = float(overrides["align_temperature"])

        # normaliza pesos (pra não virar caos)
        self.cfg = cfg
        self._normalize_weights()
        # volta self.cfg ao padrão do objeto? não: esse motor é stateless por chamada, então:
        # (deixa cfg normalizado e restaura self.cfg original não é necessário aqui;
        #  mas pra não mutar o motor base em runtime, reinstanciamos a partir dele no retorno)
        # SOLUÇÃO: reconstruir cfg com pesos já normalizados e devolver.
        cfg.pesos_windows = dict(self.cfg.pesos_windows)
        cfg.pesos_metricas = dict(self.cfg.pesos_metricas)

        # restaura self.cfg base (não mutar motor global em app)
        self.cfg = MotorV2Config()
        self.cfg.windows = [7, 10, 13, 25]
        self.cfg.dna_anchor_window = 13
        self.cfg.top_n = 30
        self.cfg.max_candidatos = 3000
        self.cfg.pesos_windows = {"7": 0.15, "10": 0.25, "13": 0.35, "25": 0.25}
        self.cfg.pesos_metricas = {"soma": 0.25, "pares": 0.15, "adj": 0.25, "faixa_1_13": 0.20, "repeticao": 0.15}
        self.cfg.redundancy_jaccard_threshold = 0.80
        self.cfg.redundancy_penalty = 0.35
        self.cfg.z_cap = 3.5
        self.cfg.align_temperature = 1.25
        self._normalize_weights()

        return cfg

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
    # Cálculos SCF
    # -------------------------

    def _compute_metrics(self, seq: List[int], last_draw: Optional[Sequence[int]]) -> Dict[str, float]:
        m = {
            "soma": float(_metric_soma(seq)),
            "pares": float(_metric_pares(seq)),
            "adj": float(_metric_adjacencias(seq)),
            "faixa_1_13": float(_metric_faixa_1_13(seq)),
            "repeticao": float(_metric_repeticao(seq, last_draw)),
        }
        return m

    def _compute_zscores(
        self,
        metricas: Dict[str, float],
        dna_last25: Dict[str, Any],
        windows: List[int],
    ) -> Dict[str, Dict[int, float]]:
        out: Dict[str, Dict[int, float]] = {}
        for metric in self.METRICAS_SUPORTADAS:
            out[metric] = {}
            x = float(metricas.get(metric, 0.0))
            for w in windows:
                mean, std = _dna_get_stats(dna_last25, w, metric)
                z = (x - mean) / (std if std != 0 else 1.0)
                z = _clamp(z, -self.cfg.z_cap, self.cfg.z_cap)
                out[metric][w] = float(z)
        return out

    def _compute_trend_vector(self, dna_last25: Dict[str, Any], windows: List[int]) -> Dict[str, float]:
        """
        Trend = slope do "mean" da métrica ao longo das janelas.
        Interpretação:
          slope > 0 => regime recente empurrando métrica pra cima quando aumenta janela
          slope < 0 => empurrando pra baixo
        """
        xs = [float(w) for w in windows]
        trend: Dict[str, float] = {}
        for metric in self.METRICAS_SUPORTADAS:
            ys = []
            for w in windows:
                mean, _std = _dna_get_stats(dna_last25, w, metric)
                ys.append(float(mean))
            slope = _linear_slope(xs, ys)
            trend[metric] = float(slope)
        return trend

    def _score_direcional(
        self,
        metricas: Dict[str, float],
        dna_last25: Dict[str, Any],
        trend: Dict[str, float],
        cfg: MotorV2Config,
    ) -> float:
        """
        Alinhamento direcional:
        - compara candidato vs mean da âncora (dna_anchor_window)
        - olha o sinal do trend (slope)
        - recompensa se candidato "vai na direção" do movimento
        """
        score = 0.0
        for metric, w_m in cfg.pesos_metricas.items():
            if metric not in self.METRICAS_SUPORTADAS:
                continue
            x = float(metricas.get(metric, 0.0))
            mean_anchor, std_anchor = _dna_get_stats(dna_last25, cfg.dna_anchor_window, metric)
            std_anchor = std_anchor if std_anchor != 0 else 1.0

            # delta normalizado em relação à âncora
            delta = (x - mean_anchor) / std_anchor

            # direção esperada
            t = float(trend.get(metric, 0.0))
            # se t > 0, queremos delta positivo; se t < 0, delta negativo; se ~0, neutro
            align = 0.0
            if abs(t) < 1e-12:
                align = 0.5  # neutro: não mata o candidato, mas não dá prêmio alto
            else:
                desired = 1.0 if t > 0 else -1.0
                raw = desired * delta
                # sigmoid dá score contínuo (evita empate crônico)
                align = _sigmoid(cfg.align_temperature * raw)

            score += float(w_m) * float(align)

        return float(_clamp(score, 0.0, 1.0))

    def _score_consistencia(self, zscores: Dict[str, Dict[int, float]], cfg: MotorV2Config) -> float:
        """
        Coerência multi-janela:
        - queremos zscores "compatíveis" entre janelas (baixa variância)
        - mas sem colar tudo em 0: também premiamos magnitude moderada (informação)
        """
        parts = []
        for metric, w_m in cfg.pesos_metricas.items():
            if metric not in zscores:
                continue

            # z por janela com pesos de janela
            zs = []
            ws = []
            for w in cfg.windows:
                z = float(zscores[metric].get(w, 0.0))
                ww = float(cfg.pesos_windows.get(str(w), 0.0))
                zs.append(z)
                ws.append(ww if ww > 0 else 0.0)

            # média ponderada
            wsum = sum(ws) if sum(ws) > 0 else 1.0
            zbar = sum(z * w for z, w in zip(zs, ws)) / wsum

            # variância ponderada
            var = sum(w * (z - zbar) ** 2 for z, w in zip(zs, ws)) / wsum
            std = math.sqrt(var)

            # consistência alta = std baixo
            consist = 1.0 - _clamp(std / 2.0, 0.0, 1.0)  # std=0 => 1.0, std>=2 => 0.0

            # “informação” moderada: |zbar| muito alto tende a ser outlier; muito baixo é neutro
            mag = math.exp(-0.35 * (abs(zbar) ** 2))  # |z|=0 => 1, |z|=2 => ~0.25
            info = 1.0 - mag  # queremos algo entre 0 e 1, favorece |z| moderado

            # mistura (consistência manda)
            parts.append(float(w_m) * (0.70 * consist + 0.30 * info))

        total = sum(parts) if parts else 0.0
        return float(_clamp(total, 0.0, 1.0))

    def _score_ancora(
        self,
        metricas: Dict[str, float],
        dna_last25: Dict[str, Any],
        cfg: MotorV2Config,
    ) -> float:
        """
        Âncora temporal:
        - puxa candidato pra perto do DNA âncora em termos de zscore “bem comportado”
        - mas não igual: penaliza extremos, favorece z perto de 0..1
        """
        parts = []
        for metric, w_m in cfg.pesos_metricas.items():
            if metric not in self.METRICAS_SUPORTADAS:
                continue
            x = float(metricas.get(metric, 0.0))
            mean, std = _dna_get_stats(dna_last25, cfg.dna_anchor_window, metric)
            std = std if std != 0 else 1.0
            z = _clamp((x - mean) / std, -cfg.z_cap, cfg.z_cap)

            # recompensa z moderado: pico perto de |z| ~ 0.75
            az = abs(z)
            anchor_score = math.exp(-((az - 0.75) ** 2) / 0.80)  # suave
            parts.append(float(w_m) * float(anchor_score))

        total = sum(parts) if parts else 0.0
        return float(_clamp(total, 0.0, 1.0))
