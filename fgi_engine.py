from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import random

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
    """
    Motor principal do laboratório.

    Correção crítica (PASSO 2):
    - Constraints NÃO SÃO "sugestões". Viram filtro duro.
      Se violar, a sequência não entra no conjunto aprovado.
    - Pesos servem apenas para ordenar ENTRE aprovados.
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

    # =========================
    # Helpers estatísticos
    # =========================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Média e desvio padrão da SOMA ao amostrar k números de 1..N (sem reposição).
        Pop variance de 1..N: (N^2 - 1)/12
        Var(sum) = k * var_pop * ((N - k)/(N - 1))
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

    def _count_adjacencias(self, seq_ordenada: List[int]) -> int:
        adj = 0
        for i in range(1, len(seq_ordenada)):
            if seq_ordenada[i] - seq_ordenada[i - 1] == 1:
                adj += 1
        return adj

    # =========================
    # Avaliação (PONTO C)
    # =========================

    def avaliar_sequencia(self, seq: List[int], regime: str) -> Dict[str, Any]:
        seq_ord = sorted(int(x) for x in seq)
        k = len(seq_ord)

        cfg = self._regimes.get(regime, self._regimes["estavel"])
        pesos = cfg["pesos"]

        soma = sum(seq_ord)
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)

        # ---------
        # FILTER DURO (aqui é a correção real)
        # ---------
        z_max_soma = float(cfg["z_max_soma"])
        max_adj = int(cfg["max_adjacencias"])
        max_desvio = int(cfg["max_desvio_pares"])

        violou_soma = z_soma > z_max_soma
        violou_adj = adj > max_adj
        violou_pares = desvio_pares > max_desvio

        aprovada = not (violou_soma or violou_adj or violou_pares)

        coerencias = 0
        violacoes = 0

        # --- SOMA
        if not violou_soma:
            coerencias += 1
            score_soma = 1.0 - (z_soma / z_max_soma)  # 1..0
        else:
            violacoes += 1
            # penaliza, mas o que manda é aprovada=False
            score_soma = -min(1.0, (z_soma - z_max_soma) / 1.0)

        # --- PARES
        if not violou_pares:
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, float(max_desvio)))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - float(max_desvio)) / 1.0)

        # --- ADJ
        if not violou_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, float(max_adj)))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        score_total = (
            pesos["soma"] * score_soma +
            pesos["pares"] * score_pares +
            pesos["adj"]  * score_adj
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
            "aprovada": aprovada,
            "componentes": {
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": z_max_soma,
                    "max_adjacencias": max_adj,
                    "max_desvio_pares": max_desvio,
                },
            },
        }

        return {
            "score_total": float(round(score_total, 6)),
            "coerencias": int(coerencias),
            "violacoes": int(violacoes),
            "aprovada": bool(aprovada),
            "detalhes": detalhes,
        }

    # =========================
    # Geração de Protótipos
    # =========================

    def gerar_prototipos(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
    ) -> List[Prototipo]:

        regime = regime_id or self.regime_padrao
        limite = int(max_candidatos or 2000)

        candidatos = self.grupo.get_candidatos(k=int(k), max_candidatos=limite)

        # Importante: candidatos vindo em ordem “baixa” tende a dar soma baixa.
        # Embaralhar aumenta MUITO a chance de pegar somas perto de mu (≈195).
        random.shuffle(candidatos)

        aprovados: List[Prototipo] = []
        rejeitados: List[Prototipo] = []

        for seq in candidatos:
            avaliacao = self.avaliar_sequencia(seq, regime)
            p = Prototipo(
                sequencia=list(seq),
                score_total=float(avaliacao["score_total"]),
                coerencias=int(avaliacao["coerencias"]),
                violacoes=int(avaliacao["violacoes"]),
                detalhes=dict(avaliacao["detalhes"]),
            )
            if avaliacao.get("aprovada"):
                aprovados.append(p)
            else:
                rejeitados.append(p)

        # Ordena aprovados por score desc
        aprovados.sort(key=lambda p: (p.score_total, -p.coerencias), reverse=True)

        # Fallback: se não tiver aprovados suficientes, completa com rejeitados (mas fica claro no detalhe)
        if len(aprovados) < int(k):
            rejeitados.sort(key=lambda p: p.score_total, reverse=True)
            aprovados.extend(rejeitados[: max(0, int(k) - len(aprovados))])

        return aprovados[: int(k)]

    # =========================
    # JSON-friendly (API)
    # =========================

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
        incluir_contexto_dna: bool = True,
    ) -> Dict[str, Any]:

        protos = self.gerar_prototipos(k=int(k), regime_id=regime_id, max_candidatos=max_candidatos)

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
        }

        if incluir_contexto_dna:
            resp["contexto_dna"] = {
                "universo_max": self.universo_max,
                "total_sorteadas": self.grupo.total_sorteadas(),
            }

        return resp
