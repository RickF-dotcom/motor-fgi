
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
    Motor FGI com DNA ATIVO.

    - Score base: soma, pares, adjacências (como antes)
    - Score DNA: distância estatística em relação ao DNA(14)
    - Total = score_base + score_dna

    Compatível com Swagger e com todos os testes anteriores.
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

        # Regras por regime
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

        # DNA anchor
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_window: Optional[int] = None

    # ============================================================
    # DNA Anchor (ativo)
    # ============================================================

    def set_dna_anchor(self, dna_last25: Dict[str, Any], window: int = 14) -> None:
        self._dna_anchor = dna_last25
        self._dna_window = int(window)

    def _get_dna_metrics(self) -> Optional[Dict[str, float]]:
        if not self._dna_anchor or not self._dna_window:
            return None

        janelas = self._dna_anchor.get("janelas", {})
        win = str(self._dna_window)

        if win not in janelas:
            return None

        return janelas[win]

    # ============================================================
    # Helpers matemáticos
    # ============================================================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        N = self.universo_max
        mu = k * (N + 1) / 2.0
        var = k * (N - k) * (N + 1) / 12.0
        sd = math.sqrt(var) if var > 0 else 1.0
        return mu, sd

    def _count_adjacencias(self, seq: List[int]) -> int:
        s = sorted(seq)
        return sum(1 for i in range(1, len(s)) if s[i] == s[i - 1] + 1)

    # ============================================================
    # Avaliação de sequência
    # ============================================================

    def avaliar_sequencia(
        self,
        seq: List[int],
        k: int,
        regime_id: str = "estavel",
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Prototipo:

        regime = self._regimes.get(regime_id, self._regimes[self.regime_padrao])
        pesos = dict(regime["pesos"])
        if pesos_override:
            pesos.update(pesos_override)

        z_max = regime["z_max_soma"]
        max_adj = regime["max_adjacencias"]
        max_desvio_pares = regime["max_desvio_pares"]

        soma = sum(seq)
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = (soma - mu_soma) / sd_soma

        # -------- SCORE BASE --------
        score_soma = -max(0.0, abs(z_soma) - z_max)
        desvio_pares = abs(pares - k / 2)
        score_pares = max(0.0, 1.0 - desvio_pares / max_desvio_pares)
        score_adj = 0.0 if adj <= max_adj else -1.0

        score_base = (
            pesos["soma"] * score_soma +
            pesos["pares"] * score_pares +
            pesos["adj"] * score_adj
        )

        # -------- SCORE DNA (NOVO) --------
        dna_metrics = self._get_dna_metrics()
        score_dna = 0.0

        if dna_metrics:
            # distância normalizada simples
            dist = 0.0
            dist += abs(soma - dna_metrics["soma_media"]) / 20.0
            dist += abs(impares - dna_metrics["impares_media"]) / 5.0
            dist += abs(adj - dna_metrics["adjacencias_media"]) / 5.0

            score_dna = -dist  # quanto mais perto do DNA, menos penaliza

        score_total = score_base + score_dna

        violacoes = 0
        if abs(z_soma) > z_max:
            violacoes += 1
        if adj > max_adj:
            violacoes += 1
        if desvio_pares > max_desvio_pares:
            violacoes += 1

        coerencias = 3 - violacoes

        detalhes = {
            "k": k,
            "soma": soma,
            "mu_soma": round(mu_soma, 4),
            "sd_soma": round(sd_soma, 4),
            "z_soma": round(z_soma, 4),
            "pares": pares,
            "impares": impares,
            "adjacencias": adj,
            "regime": regime_id,
            "componentes": {
                "score_base": round(score_base, 6),
                "score_dna": round(score_dna, 6),
                "pesos": pesos,
                "dna_window": self._dna_window,
            },
        }

        return Prototipo(
            sequencia=sorted(seq),
            score_total=round(score_total, 6),
            coerencias=max(0, coerencias),
            violacoes=violacoes,
            detalhes=detalhes,
        )

    # ============================================================
    # Geração de protótipos (API)
    # ============================================================

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: str = "estavel",
        max_candidatos: int = 2000,
        incluir_contexto_dna: bool = True,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        top_n: int = 30,
    ) -> Dict[str, Any]:

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        protos: List[Prototipo] = []
        for seq in candidatos:
            protos.append(
                self.avaliar_sequencia(
                    list(seq),
                    k=k,
                    regime_id=regime_id,
                    pesos_override=pesos_override,
                    constraints_override=constraints_override,
                )
            )

        protos.sort(key=lambda p: p.score_total, reverse=True)
        protos = protos[:top_n]

        return {
            "prototipos": [
                {
                    "sequencia": p.sequencia,
                    "score_total": p.score_total,
                    "coerencias": p.coerencias,
                    "violacoes": p.violacoes,
                    "detalhes": p.detalhes,
                }
                for p in protos
            ],
            "regime_usado": regime_id,
            "max_candidatos_usado": max_candidatos,
            "dna_ativo": bool(self._dna_anchor),
            "dna_window": self._dna_window,
        }
