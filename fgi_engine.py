
# fgi_engine.py
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

    Objetivo desta versão:
    - Score coerente com "similaridade" ao DNA (âncora) em vez de penalizar tudo.
    - Aceita overrides sem quebrar.
    - set_dna_anchor() compatível com app.py.
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

        # Regimes = pesos + tolerâncias (já “humanas” pra Lotofácil).
        # OBS: adjacências na vida real ficam altas; então NÃO faz sentido max_adj=3.
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "pesos": {"soma": 0.40, "pares": 0.20, "adj": 0.40},
                "tolerancias": {
                    "z_soma_max": 2.20,
                    "desvio_pares_max": 3.0,
                    "desvio_adj_max": 4.0,
                },
            },
            "tenso": {
                "pesos": {"soma": 0.35, "pares": 0.15, "adj": 0.50},
                "tolerancias": {
                    "z_soma_max": 2.70,
                    "desvio_pares_max": 4.0,
                    "desvio_adj_max": 5.0,
                },
            },
        }

        # Âncora DNA (injetada pelo app)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

        # Targets efetivos (derivados do DNA)
        self._anchor_targets: Dict[str, float] = {}

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py
    # ------------------------------------------------------------

    def set_dna_anchor(self, dna_last25: Optional[Dict[str, Any]] = None, window: int = 25, **_ignored: Any) -> None:
        """
        Recebe o DNA (do RegimeDetector) e fixa um alvo (target) baseado em uma janela.
        window recomendado: 13 ou 14 (ou 25 como fallback).
        """
        self._dna_anchor = dna_last25 or None
        self._dna_anchor_active = bool(dna_last25)

        self._anchor_targets = {}
        if not self._dna_anchor_active:
            return

        # estrutura esperada: {"origem": "...", "janelas": {"7": {...}, "12": {...}, ...}}
        janelas = (dna_last25 or {}).get("janelas") or {}
        w = str(int(window))

        # fallback em cascata: window -> 25 -> primeiro disponível
        base = janelas.get(w) or janelas.get("25")
        if not base and isinstance(janelas, dict) and len(janelas) > 0:
            # pega a primeira janela existente
            first_key = sorted(janelas.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9)[0]
            base = janelas.get(first_key)

        base = base or {}

        # targets que vamos usar no score
        self._anchor_targets = {
            "soma_media": float(base.get("soma_media", 195.0)),
            "impares_media": float(base.get("impares_media", 7.5)),
            "adjacencias_media": float(base.get("adjacencias_media", 8.0)),
        }

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "targets": self._anchor_targets if self._dna_anchor_active else None,
        }

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Aproximação robusta pra soma no universo 1..N sem reposição.
        mu = k*(N+1)/2
        var(sum) = k * var_pop * fpc
        var_pop(1..N) = (N^2 - 1)/12
        fpc = (N-k)/(N-1)
        """
        N = self.universo_max
        k = int(k)
        if k <= 0 or k > N:
            return 0.0, 1.0

        mu = k * (N + 1) / 2.0
        var_pop = (N * N - 1) / 12.0
        fpc = (N - k) / (N - 1) if N > 1 else 1.0
        var_sum = k * var_pop * fpc
        sd = math.sqrt(max(var_sum, 1e-9))
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

    def _clamp01(self, x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _similaridade_linear(self, dist: float, max_dist: float) -> float:
        """
        dist=0 -> 1.0
        dist=max_dist -> 0.0
        dist>max_dist -> 0.0
        """
        if max_dist <= 0:
            return 0.0
        return self._clamp01(1.0 - (abs(dist) / max_dist))

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
        cfg = self._regimes.get(regime_id, self._regimes[self.regime_padrao])

        pesos = dict(cfg["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        toler = dict(cfg["tolerancias"])
        if constraints_override:
            for kk, vv in constraints_override.items():
                if kk in toler:
                    toler[kk] = float(vv)

        seq_ord = sorted(int(x) for x in seq)
        soma = int(sum(seq_ord))
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        # --- targets (DNA anchor) ou fallback “teórico”
        targets = self._anchor_targets if self._dna_anchor_active else {}
        target_soma = float(targets.get("soma_media", self._sum_stats_sem_reposicao(k)[0]))
        target_impares = float(targets.get("impares_media", k / 2.0))
        target_adj = float(targets.get("adjacencias_media", 8.0))

        # --- normalizações / distâncias
        mu_teor, sd_teor = self._sum_stats_sem_reposicao(k)
        # z relativo ao TEÓRICO (pra estabilidade), mas distância relativa ao target (DNA)
        z_soma = 0.0 if sd_teor == 0 else abs((soma - mu_teor) / sd_teor)

        dist_soma = (soma - target_soma) / max(sd_teor, 1.0)
        dist_pares = (pares - (k / 2.0))
        dist_adj = (adj - target_adj)

        # --- scores (0..1)
        score_soma = self._similaridade_linear(dist_soma, max_dist=float(toler["z_soma_max"]))
        score_pares = self._similaridade_linear(dist_pares, max_dist=float(toler["desvio_pares_max"]))
        score_adj = self._similaridade_linear(dist_adj, max_dist=float(toler["desvio_adj_max"]))

        # Score total 0..1 (ponderado e normalizado)
        soma_pesos = max(1e-9, float(pesos["soma"] + pesos["pares"] + pesos["adj"]))
        score_total = (
            pesos["soma"] * score_soma +
            pesos["pares"] * score_pares +
            pesos["adj"] * score_adj
        ) / soma_pesos

        # coerência/violação: usa o mesmo limite das tolerâncias (simples e estável)
        viol = 0
        if abs(dist_soma) > float(toler["z_soma_max"]):
            viol += 1
        if abs(dist_pares) > float(toler["desvio_pares_max"]):
            viol += 1
        if abs(dist_adj) > float(toler["desvio_adj_max"]):
            viol += 1

        coerencias = 3 - viol

        detalhes = {
            "k": k,
            "soma": soma,
            "pares": pares,
            "impares": impares,
            "adjacencias": adj,
            "regime": regime_id,
            "anchor": self.get_dna_anchor(),
            "targets": {
                "soma_media": round(target_soma, 4),
                "impares_media": round(target_impares, 4),
                "adjacencias_media": round(target_adj, 4),
            },
            "distancias_norm": {
                "dist_soma": round(float(dist_soma), 6),
                "dist_pares": round(float(dist_pares), 6),
                "dist_adj": round(float(dist_adj), 6),
                "z_soma_teorico": round(float(z_soma), 6),
            },
            "componentes": {
                "score_soma": round(float(score_soma), 6),
                "score_pares": round(float(score_pares), 6),
                "score_adj": round(float(score_adj), 6),
                "pesos": pesos,
                "tolerancias": toler,
            },
        }

        return Prototipo(
            sequencia=seq_ord,
            score_total=float(round(score_total, 6)),
            coerencias=int(coerencias),
            violacoes=int(viol),
            detalhes=detalhes,
        )

    # ------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: str = "estavel",
        max_candidatos: int = 2000,
        incluir_contexto_dna: bool = True,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        top_n: int = 30,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            p = self.avaliar_sequencia(
                list(seq),
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
            )
            prototipos.append(p)

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[: max(1, top_n)]

        out: Dict[str, Any] = {
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
            },
        }

        # aqui o app injeta o contexto_lab; então a flag só fica pra compat
        if incluir_contexto_dna:
            out["dna_anchor_status"] = self.get_dna_anchor()

        return out
