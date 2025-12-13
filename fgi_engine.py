from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from grupo_de_milhoes import GrupoDeMilhoes


# =========================
# Estrutura de Protótipo
# =========================

@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


# =========================
# MotorFGI
# =========================

class MotorFGI:
    def __init__(
        self,
        historico_csv: Optional[str] = None,
        universo_max: int = 25,
    ) -> None:
        self.universo_max = universo_max

        self.grupo = GrupoDeMilhoes(
            universo_max=universo_max,
            historico_csv=Path(historico_csv) if historico_csv else None,
        )

        self.regime_padrao = "estavel"

        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
            },
        }

    # =========================
    # Helpers
    # =========================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        N = self.universo_max
        mu = k * (N + 1) / 2
        var_pop = (N * N - 1) / 12
        fpc = (N - k) / (N - 1)
        var_sum = k * var_pop * fpc
        return mu, math.sqrt(var_sum)

    def _count_adjacencias(self, seq: List[int]) -> int:
        return sum(1 for i in range(1, len(seq)) if seq[i] - seq[i - 1] == 1)

    # =========================
    # Avaliação
    # =========================

    def avaliar_sequencia(
        self,
        seq: List[int],
        regime: str,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        seq = sorted(seq)
        k = len(seq)

        cfg = self._regimes.get(regime, self._regimes[self.regime_padrao])
        pesos = dict(cfg["pesos"])
        if pesos_override:
            pesos.update(pesos_override)

        z_max_soma = cfg["z_max_soma"]
        max_adj = cfg["max_adjacencias"]
        max_desvio_pares = cfg["max_desvio_pares"]

        if constraints_override:
            z_max_soma = constraints_override.get("z_max_soma", z_max_soma)
            max_adj = constraints_override.get("max_adjacencias", max_adj)
            max_desvio_pares = constraints_override.get("max_desvio_pares", max_desvio_pares)

        soma = sum(seq)
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma)

        coerencias = 0
        violacoes = 0

        # Soma
        if z_soma <= z_max_soma:
            coerencias += 1
            score_soma = 1 - (z_soma / z_max_soma)
        else:
            violacoes += 1
            score_soma = -1

        # Pares
        alvo = k / 2
        desvio = abs(pares - alvo)
        if desvio <= max_desvio_pares:
            coerencias += 1
            score_pares = 1 - (desvio / max_desvio_pares)
        else:
            violacoes += 1
            score_pares = -1

        # Adjacências
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1 - (adj / max_adj)
        else:
            violacoes += 1
            score_adj = -1

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        return {
            "score_total": round(score_total, 6),
            "coerencias": coerencias,
            "violacoes": violacoes,
            "detalhes": {
                "k": k,
                "soma": soma,
                "mu_soma": mu_soma,
                "sd_soma": sd_soma,
                "z_soma": z_soma,
                "pares": pares,
                "impares": impares,
                "adjacencias": adj,
                "regime": regime,
                "pesos": pesos,
            },
        }

    # =========================
    # Protótipos
    # =========================

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: int = 2000,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        regime = regime_id or self.regime_padrao
        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        protos = []
        for seq in candidatos:
            av = self.avaliar_sequencia(
                seq,
                regime,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
            )
            protos.append(
                Prototipo(
                    sequencia=seq,
                    score_total=av["score_total"],
                    coerencias=av["coerencias"],
                    violacoes=av["violacoes"],
                    detalhes=av["detalhes"],
                )
            )

        protos.sort(key=lambda p: p.score_total, reverse=True)

        return {
            "prototipos": [
                {
                    "sequencia": p.sequencia,
                    "score_total": p.score_total,
                    "coerencias": p.coerencias,
                    "violacoes": p.violacoes,
                    "detalhes": p.detalhes,
                }
                for p in protos[:k]
            ],
            "regime_usado": regime,
            "max_candidatos_usado": max_candidatos,
        }
