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
    """
    Motor principal do laboratório.

    Importante:
    - 'k' é tamanho da sequência e também a quantidade de protótipos retornados (compat do endpoint).
    - Agora aceita overrides:
        * pesos_override: altera pesos do score (soma/pares/adj)
        * constraints_override: altera limites do regime (z_max_soma, max_adjacencias, max_desvio_pares)
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

        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.20, "pares": 0.60, "adj": 0.60},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.00, "pares": 0.45, "adj": 0.45},
            },
        }

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
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

    def _count_adjacencias(self, seq_ordenada: List[int]) -> int:
        adj = 0
        for i in range(1, len(seq_ordenada)):
            if seq_ordenada[i] - seq_ordenada[i - 1] == 1:
                adj += 1
        return adj

    def _cfg_regime_com_overrides(
        self,
        regime: str,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = dict(self._regimes.get(regime, self._regimes["estavel"]))

        # pesos
        pesos = dict(base.get("pesos", {}))
        if pesos_override:
            for k, v in pesos_override.items():
                if k in ("soma", "pares", "adj"):
                    try:
                        pesos[k] = float(v)
                    except Exception:
                        pass
        base["pesos"] = pesos

        # constraints
        if constraints_override:
            for k, v in constraints_override.items():
                if k in ("z_max_soma", "max_adjacencias", "max_desvio_pares"):
                    base[k] = v

        # sanidade mínima
        base["z_max_soma"] = float(base.get("z_max_soma", 1.30))
        base["max_adjacencias"] = int(base.get("max_adjacencias", 3))
        base["max_desvio_pares"] = int(base.get("max_desvio_pares", 2))

        return base

    def avaliar_sequencia(
        self,
        seq: List[int],
        regime: str,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        seq_ord = sorted(int(x) for x in seq)
        k = len(seq_ord)

        cfg = self._cfg_regime_com_overrides(
            regime=regime,
            pesos_override=pesos_override,
            constraints_override=constraints_override,
        )
        pesos = cfg["pesos"]

        soma = sum(seq_ord)
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        coerencias = 0
        violacoes = 0

        # SOMA
        z_max = float(cfg["z_max_soma"])
        if z_soma <= z_max:
            coerencias += 1
            score_soma = 1.0 - (z_soma / max(1e-9, z_max))
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - z_max) / 1.0)

        # PARES
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        max_desvio = int(cfg["max_desvio_pares"])
        if desvio_pares <= max_desvio:
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, float(max_desvio)))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - float(max_desvio)) / 1.0)

        # ADJ
        max_adj = int(cfg["max_adjacencias"])
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, float(max_adj)))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        score_total = (
            pesos["soma"] * score_soma +
            pesos["pares"] * score_pares +
            pesos["adj"] * score_adj
        )

        detalhes = {
            "k": k,
            "soma": soma,
            "mu_soma": round(mu_soma, 4),
            "sd_soma": round(sd_soma, 4),
            "z_soma": round(z_soma, 4),
            "pares": pares,
            "impares": impares,
            "adjacencias": adj,
            "regime": regime,
            "componentes": {
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "pesos": dict(pesos),
                "constraints": {
                    "z_max_soma": cfg["z_max_soma"],
                    "max_adjacencias": cfg["max_adjacencias"],
                    "max_desvio_pares": cfg["max_desvio_pares"],
                },
            },
        }

        return {
            "score_total": float(round(score_total, 6)),
            "coerencias": int(coerencias),
            "violacoes": int(violacoes),
            "detalhes": detalhes,
        }

    def gerar_prototipos(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> List[Prototipo]:
        regime = regime_id or self.regime_padrao
        limite = int(max_candidatos or 2000)

        candidatos = self.grupo.get_candidatos(
            k=int(k),
            max_candidatos=limite,
        )

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            avaliacao = self.avaliar_sequencia(
                seq=list(seq),
                regime=regime,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
            )
            prototipos.append(
                Prototipo(
                    sequencia=list(seq),
                    score_total=float(avaliacao["score_total"]),
                    coerencias=int(avaliacao["coerencias"]),
                    violacoes=int(avaliacao["violacoes"]),
                    detalhes=dict(avaliacao["detalhes"]),
                )
            )

        # score desc, menos violações, mais coerências
        prototipos.sort(
            key=lambda p: (p.score_total, -p.violacoes, p.coerencias),
            reverse=True,
        )

        return prototipos[: int(k)]

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
        incluir_contexto_dna: bool = True,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        protos = self.gerar_prototipos(
            k=int(k),
            regime_id=regime_id,
            max_candidatos=max_candidatos,
            pesos_override=pesos_override,
            constraints_override=constraints_override,
        )

        payload = [
            {
                "sequencia": p.sequencia,
                "score_total": p.score_total,
                "coerencias": p.coerencias,
                "violacoes": p.violacoes,
                "detalhes": p.detalhes,
            }
            for p in protos
        ]

        resp: Dict[str, Any] = {
            "prototipos": payload,
            "regime_usado": regime_id or self.regime_padrao,
            "max_candidatos_usado": int(max_candidatos or 2000),
            "overrides_usados": {
                "pesos_override": pesos_override or {},
                "constraints_override": constraints_override or {},
            },
        }

        if incluir_contexto_dna:
            resp["contexto_dna"] = {
                "universo_max": self.universo_max,
                "total_sorteadas": self.grupo.total_sorteadas(),
            }

        return resp
