
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

        self.regimes = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
            }
        }

    def _sum_stats(self, k: int) -> Tuple[float, float]:
        N = self.universo_max
        mu = k * (N + 1) / 2
        var = k * ((N * N - 1) / 12) * ((N - k) / (N - 1))
        return mu, math.sqrt(var)

    def _adj(self, seq: List[int]) -> int:
        return sum(1 for i in range(1, len(seq)) if seq[i] - seq[i - 1] == 1)

    def avaliar(
        self,
        seq: List[int],
        regime: str,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        cfg = self.regimes[regime]
        pesos = cfg["pesos"].copy()

        if pesos_override:
            pesos.update(pesos_override)

        seq = sorted(seq)
        k = len(seq)

        soma = sum(seq)
        pares = sum(1 for x in seq if x % 2 == 0)
        adj = self._adj(seq)

        mu, sd = self._sum_stats(k)
        z = abs((soma - mu) / sd)

        score_soma = 1 - z / cfg["z_max_soma"]
        score_pares = 1 - abs(pares - k / 2) / cfg["max_desvio_pares"]
        score_adj = 1 - adj / cfg["max_adjacencias"]

        score = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        return {
            "score_total": round(score, 6),
            "coerencias": 3,
            "violacoes": 0,
            "detalhes": {
                "soma": soma,
                "pares": pares,
                "adj": adj,
                "z_soma": round(z, 4),
                "pesos": pesos,
                "constraints_override": constraints_override or {},
            },
        }

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: str = "estavel",
        max_candidatos: int = 2000,
        incluir_contexto_dna: bool = True,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        candidatos = self.grupo.get_candidatos(k, max_candidatos)

        protos = []
        for seq in candidatos:
            av = self.avaliar(
                seq,
                regime_id,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
            )
            protos.append(
                {
                    "sequencia": list(seq),
                    **av,
                }
            )

        protos.sort(key=lambda x: x["score_total"], reverse=True)

        return {
            "prototipos": protos[:k],
            "regime_usado": regime_id,
            "overrides": {
                "pesos": pesos_override or {},
                "constraints": constraints_override or {},
            },
            }
