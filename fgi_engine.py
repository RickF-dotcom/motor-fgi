
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from grupo_de_milhoes import GrupoDeMilhoes


# ============================================================
# Estruturas
# ============================================================

@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


# ============================================================
# MotorFGI
# ============================================================

class MotorFGI:
    """
    Motor de avaliação / geração de protótipos.

    - Contrato-resiliente: aceita chaves novas no request sem quebrar
    - set_dna_anchor(): recebe DNA(últimos 25) + janela (13/14) ou mix
    - Score passa a OBEDECER ao DNA (âncora), não apenas anexar contexto
    """

    def __init__(
        self,
        historico_csv: Optional[str] = None,
        universo_max: int = 25,
    ) -> None:
        self.universo_max = int(universo_max)

        self.grupo = GrupoDeMilhoes(
            universo_max=self.universo_max,
            historico_csv=Path(historico_csv) if historico_csv else None,
        )

        self.regime_padrao = "estavel"

        # Regras/limites por regime (mantém compat)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
                # força da âncora (DNA)
                "peso_anchor": 1.25,
                # escalas de distância (quanto maior, mais tolerante)
                "scales_anchor": {"soma": 18.0, "impares": 2.0, "faixa_1_13": 2.0, "adj": 4.0},
                # pesos internos da âncora
                "pesos_anchor": {"soma": 1.0, "impares": 0.7, "faixa_1_13": 0.8, "adj": 1.0},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
                "peso_anchor": 1.10,
                "scales_anchor": {"soma": 20.0, "impares": 2.5, "faixa_1_13": 2.5, "adj": 5.0},
                "pesos_anchor": {"soma": 1.0, "impares": 0.6, "faixa_1_13": 0.7, "adj": 1.0},
            },
        }

        # DNA anchor
        self._dna_last25: Optional[Dict[str, Any]] = None
        self._anchor_window: Optional[int] = None
        self._anchor_mix: Optional[Dict[int, float]] = None
        self._anchor_target: Optional[Dict[str, float]] = None
        self._dna_anchor_active: bool = False

    # ------------------------------------------------------------
    # Contrato (app.py)
    # ------------------------------------------------------------

    def set_dna_anchor(
        self,
        dna_last25: Optional[Dict[str, Any]] = None,
        window: Optional[int] = None,
        dna_anchor_mix: Optional[Dict[str, float]] = None,
        **_ignored: Any,
    ) -> None:
        """
        Define âncora:
        - window = 13 ou 14 (padrão recomendado)
        - ou dna_anchor_mix = {"13":0.5,"14":0.5}
        """
        self._dna_last25 = dna_last25 or None
        self._anchor_window = int(window) if window is not None else None

        mix_norm: Optional[Dict[int, float]] = None
        if dna_anchor_mix and isinstance(dna_anchor_mix, dict):
            tmp: Dict[int, float] = {}
            for k, v in dna_anchor_mix.items():
                try:
                    kk = int(str(k).strip())
                    vv = float(v)
                    if vv > 0:
                        tmp[kk] = vv
                except Exception:
                    continue
            if tmp:
                s = sum(tmp.values())
                if s > 0:
                    mix_norm = {kk: vv / s for kk, vv in tmp.items()}

        self._anchor_mix = mix_norm

        self._anchor_target = self._build_anchor_target()
        self._dna_anchor_active = bool(self._anchor_target)

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "window": self._anchor_window,
            "mix": self._anchor_mix,
            "target": self._anchor_target,
        }

    def _build_anchor_target(self) -> Optional[Dict[str, float]]:
        """
        Extrai o target (médias) da janela escolhida dentro de dna_last25["janelas"].
        """
        if not self._dna_last25:
            return None

        janelas = self._dna_last25.get("janelas") or self._dna_last25.get("dna", {}).get("janelas")
        if not isinstance(janelas, dict):
            return None

        def pick(win: int) -> Optional[Dict[str, Any]]:
            d = janelas.get(str(win))
            return d if isinstance(d, dict) else None

        # Mix (ex: 13/14)
        if self._anchor_mix:
            agg: Dict[str, float] = {"soma": 0.0, "impares": 0.0, "faixa_1_13": 0.0, "adj": 0.0}
            used = 0
            for w, a in self._anchor_mix.items():
                d = pick(w)
                if not d:
                    continue
                agg["soma"] += float(d.get("soma_media", 0.0)) * a
                agg["impares"] += float(d.get("impares_media", 0.0)) * a
                agg["faixa_1_13"] += float(d.get("faixa_1_13_media", 0.0)) * a
                agg["adj"] += float(d.get("adjacencias_media", 0.0)) * a
                used += 1
            if used == 0:
                return None
            return {"soma_media": agg["soma"], "impares_media": agg["impares"], "faixa_1_13_media": agg["faixa_1_13"], "adj_media": agg["adj"]}

        # Window simples
        win = self._anchor_window
        if not win:
            return None
        d = pick(win)
        if not d:
            return None

        return {
            "soma_media": float(d.get("soma_media", 0.0)),
            "impares_media": float(d.get("impares_media", 0.0)),
            "faixa_1_13_media": float(d.get("faixa_1_13_media", 0.0)),
            "adj_media": float(d.get("adjacencias_media", 0.0)),
        }

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Soma de k números sem reposição em {1..N}:
        mu = k*(N+1)/2
        var = k*(N-k)*(N+1)/12
        """
        N = self.universo_max
        k = int(k)
        mu = k * (N + 1) / 2.0
        var = k * (N - k) * (N + 1) / 12.0
        sd = math.sqrt(var) if var > 0 else 0.0
        return mu, sd

    def _count_adjacencias(self, seq: List[int]) -> int:
        if not seq:
            return 0
        s = sorted(seq)
        adj = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                adj += 1
        return adj

    def _count_faixa_1_13(self, seq: List[int]) -> int:
        return sum(1 for x in seq if 1 <= int(x) <= 13)

    def _anchor_score(self, metrics: Dict[str, float], cfg: Dict[str, Any]) -> float:
        """
        Score contínuo de proximidade ao DNA (âncora).
        Retorna [-1, +1] (aprox), depois multiplicado por peso_anchor.
        """
        if not self._dna_anchor_active or not self._anchor_target:
            return 0.0

        pesos_anchor = dict(cfg.get("pesos_anchor", {}))
        scales = dict(cfg.get("scales_anchor", {}))

        def closeness(x: float, mu: float, scale: float) -> float:
            # 1 quando igual; decai linearmente; abaixo de 0 vira negativo leve
            if scale <= 0:
                return 0.0
            d = abs(x - mu) / scale
            s = 1.0 - d
            # clamp com cauda negativa controlada
            return max(-1.0, min(1.0, s))

        s_soma = closeness(metrics["soma"], self._anchor_target["soma_media"], float(scales.get("soma", 18.0)))
        s_imp = closeness(metrics["impares"], self._anchor_target["impares_media"], float(scales.get("impares", 2.0)))
        s_f13 = closeness(metrics["faixa_1_13"], self._anchor_target["faixa_1_13_media"], float(scales.get("faixa_1_13", 2.0)))
        s_adj = closeness(metrics["adj"], self._anchor_target["adj_media"], float(scales.get("adj", 4.0)))

        total_w = float(pesos_anchor.get("soma", 1.0) + pesos_anchor.get("impares", 0.7) + pesos_anchor.get("faixa_1_13", 0.8) + pesos_anchor.get("adj", 1.0))
        if total_w <= 0:
            total_w = 1.0

        score = (
            float(pesos_anchor.get("soma", 1.0)) * s_soma
            + float(pesos_anchor.get("impares", 0.7)) * s_imp
            + float(pesos_anchor.get("faixa_1_13", 0.8)) * s_f13
            + float(pesos_anchor.get("adj", 1.0)) * s_adj
        ) / total_w

        return float(score)

    # ------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------

    def avaliar_sequencia(
        self,
        seq: List[int],
        k: int,
        regime_id: str = "estavel",
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ) -> Prototipo:
        k = int(k)
        regime_id = (regime_id or self.regime_padrao).strip().lower()
        base = self._regimes.get(regime_id, self._regimes[self.regime_padrao])

        # constraints
        z_max_soma = float(base["z_max_soma"])
        max_adj = int(base["max_adjacencias"])
        max_desvio_pares = int(base["max_desvio_pares"])

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])

        # pesos base (soma/pares/adj)
        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        seq_sorted = sorted(int(x) for x in seq)
        soma = int(sum(seq_sorted))
        pares = sum(1 for x in seq_sorted if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_sorted)
        f13 = self._count_faixa_1_13(seq_sorted)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # componentes compatíveis
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        score_adj = 0.0 if adj <= max_adj else -1.0

        # ANCHOR (DNA)
        cfg_anchor = base
        anchor_metrics = {"soma": float(soma), "impares": float(impares), "faixa_1_13": float(f13), "adj": float(adj)}
        score_anchor_raw = self._anchor_score(anchor_metrics, cfg_anchor)
        peso_anchor = float(cfg_anchor.get("peso_anchor", 1.0))
        score_anchor = peso_anchor * score_anchor_raw

        # score total = base + anchor
        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
            + score_anchor
        )

        # violações
        viol = 0
        if abs(z_soma) > z_max_soma:
            viol += 1
        if adj > max_adj:
            viol += 1
        if max_desvio_pares > 0 and desvio_pares > max_desvio_pares:
            viol += 1

        total_constraints = 3
        coerencias = max(0, total_constraints - viol)

        detalhes = {
            "k": k,
            "soma": soma,
            "mu_soma": round(mu_soma, 4),
            "sd_soma": round(sd_soma, 4),
            "z_soma": round(z_soma, 4),
            "pares": pares,
            "impares": impares,
            "faixa_1_13": f13,
            "adjacencias": adj,
            "regime": regime_id,
            "componentes": {
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "score_anchor_raw": round(score_anchor_raw, 4),
                "peso_anchor": peso_anchor,
                "score_anchor": round(score_anchor, 4),
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": z_max_soma,
                    "max_adjacencias": max_adj,
                    "max_desvio_pares": max_desvio_pares,
                },
            },
            "dna_anchor": self.get_dna_anchor(),
        }

        return Prototipo(
            sequencia=seq_sorted,
            score_total=float(round(score_total, 6)),
            coerencias=int(coerencias),
            violacoes=int(viol),
            detalhes=detalhes,
        )

    # ------------------------------------------------------------
    # Protótipos (JSON)
    # ------------------------------------------------------------

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: str = "estavel",
        max_candidatos: int = 2000,
        incluir_contexto_dna: bool = True,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        dna_anchor_window: Optional[int] = None,
        dna_anchor_mix: Optional[Dict[str, float]] = None,
        top_n: int = 30,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        # permite setar âncora por request, sem depender do app.py
        if (dna_anchor_window is not None) or (dna_anchor_mix is not None):
            self.set_dna_anchor(dna_last25=self._dna_last25, window=dna_anchor_window, dna_anchor_mix=dna_anchor_mix)

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            p = self.avaliar_sequencia(
                list(seq),
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override or {},
                constraints_override=constraints_override or {},
            )
            prototipos.append(p)

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[: max(1, top_n)]

        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active and self._dna_last25:
            contexto_lab = self._dna_last25

        return {
            "prototipos": [
                {
                    "sequencia": p.sequencia,
                    "score_total": p.score_total,
                    "coerencias": p.coerencias,
                    "violacoes": p.violacoes,
                    "detalhes": p.detalhes,
                }
                for p in prototipos
            ],
            "regime_usado": (regime_id or self.regime_padrao),
            "max_candidatos_usado": max_candidatos,
            "overrides_usados": {
                "pesos_override": pesos_override or {},
                "constraints_override": constraints_override or {},
                "dna_anchor_window": dna_anchor_window,
                "dna_anchor_mix": dna_anchor_mix,
                "top_n": top_n,
            },
            "contexto_lab": contexto_lab,
                          }
