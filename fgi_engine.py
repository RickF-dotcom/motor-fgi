from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from grupo_de_milhoes import GrupoDeMilhoes


@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


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

        self._regimes = {
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

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        N = self.universo_max
        mu = k * (N + 1) / 2
        var_pop = (N * N - 1) / 12
        fpc = (N - k) / (N - 1)
        var_sum = k * var_pop * fpc
        return mu, math.sqrt(var_sum)

    def _count_adjacencias(self, seq: List[int]) -> int:
        return sum(1 for i in range(1, len(seq)) if seq[i] - seq[i - 1] == 1)

    def avaliar_sequencia(
        self,
        seq: List[int],
        regime: str,
        pesos_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        cfg = self._regimes.get(regime, self._regimes["estavel"])
        pesos = cfg["pesos"].copy()

        if pesos_override:
            pesos.update(pesos_override)

        seq = sorted(seq)
        k = len(seq)

        soma = sum(seq)
        pares = sum(1 for x in seq if x % 2 == 0)
        adj = self._count_adjacencias(seq)

        mu, sd = self._sum_stats_sem_reposicao(k)
        z = abs((soma - mu) / sd)

        score_soma = 1 - z / cfg["z_max_soma"] if z <= cfg["z_max_soma"] else -1
        score_pares = 1 - abs(pares - k / 2) / cfg["max_desvio_pares"]
        score_adj = 1 - adj / cfg["max_adjacencias"] if adj <= cfg["max_adjacencias"] else -1

        score_total = (
            pesos["soma"] * score_soma +
            pesos["pares"] * score_pares +
            pesos["adj"] * score_adj
        )

        return {
            "score_total": round(score_total, 6),
            "coerencias": sum(s > 0 for s in [score_soma, score_pares, score_adj]),
            "violacoes": sum(s < 0 for s in [score_soma, score_pares, score_adj]),
            "detalhes": {
                "pesos_usados": pesos,
                "soma": soma,
                "pares": pares,
                "adjacencias": adj,
            },
        }

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: str,
        max_candidatos: int,
        incluir_contexto_dna: bool,
        pesos_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        candidatos = self.grupo.get_candidatos(k, max_candidatos)
        protos = []

        for seq in candidatos:
            av = self.avaliar_sequencia(seq, regime_id, pesos_override)
            protos.append({
                "sequencia": seq,
                **av
            })

        protos.sort(key=lambda x: x["score_total"], reverse=True)

        return {
            "prototipos": protos[:k],
            "regime_usado": regime_id,
        }
