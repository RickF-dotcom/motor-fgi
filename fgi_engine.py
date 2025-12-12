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
    """
    Motor principal do laboratório.

    Responsabilidades:
    - Carregar Grupo de Milhões
    - Avaliar sequências (PONTO C / score)
    - Gerar protótipos estruturais (LHE/LHS)

    Nota importante (por design, por enquanto):
    - 'k' é usado como TAMANHO DA SEQUÊNCIA e também como QUANTIDADE DE PROTÓTIPOS retornados.
      (Mantido assim pra compatibilidade com seu endpoint atual.)
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

        # placeholder de regime padrão
        self.regime_padrao = "estavel"

        # Configs de regime (você vai evoluir isso depois)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,          # faixa mais “apertada”
                "max_adjacencias": 3,        # penaliza sequências muito consecutivas
                "max_desvio_pares": 2,       # tolerância de pares vs k/2
                "pesos": {
                    "soma": 1.20,
                    "pares": 0.60,
                    "adj": 0.60,
                },
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {
                    "soma": 1.00,
                    "pares": 0.45,
                    "adj": 0.45,
                },
            },
        }

    # =========================
    # Helpers estatísticos (universo 1..N, sem reposição)
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
        """
        Conta quantos pares consecutivos existem: (x, x+1).
        Ex.: [1,2,3,7,9,10] -> adj=3 (1-2,2-3,9-10)
        """
        adj = 0
        for i in range(1, len(seq_ordenada)):
            if seq_ordenada[i] - seq_ordenada[i - 1] == 1:
                adj += 1
        return adj

    # =========================
    # Avaliação (PONTO C)
    # =========================

    def avaliar_sequencia(self, seq: List[int], regime: str) -> Dict[str, Any]:
        """
        Avaliação estruturada (já discriminante e dependente de k).

        O que entra agora:
        - Soma (normalizada por z-score no universo 1..N sem reposição)
        - Paridade (pares vs k/2)
        - Adjacências (consecutivos)
        """
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

        coerencias = 0
        violacoes = 0

        # --- Regras: SOMA (por z-score)
        if z_soma <= float(cfg["z_max_soma"]):
            coerencias += 1
            score_soma = 1.0 - (z_soma / float(cfg["z_max_soma"]))  # 1..0
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - float(cfg["z_max_soma"])) / 1.0)

        # --- Regras: PARES (desvio do equilíbrio)
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        if desvio_pares <= int(cfg["max_desvio_pares"]):
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, float(cfg["max_desvio_pares"])))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - float(cfg["max_desvio_pares"])) / 1.0)

        # --- Regras: ADJACÊNCIAS
        max_adj = int(cfg["max_adjacencias"])
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, float(max_adj)))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        # Score total (ponderado)
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
                "pesos": pesos,
            },
        }

        return {
            "score_total": float(round(score_total, 6)),
            "coerencias": int(coerencias),
            "violacoes": int(violacoes),
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

        candidatos = self.grupo.get_candidatos(
            k=int(k),
            max_candidatos=limite,
        )

        prototipos: List[Prototipo] = []

        for seq in candidatos:
            avaliacao = self.avaliar_sequencia(seq, regime)

            prototipos.append(
                Prototipo(
                    sequencia=list(seq),
                    score_total=float(avaliacao["score_total"]),
                    coerencias=int(avaliacao["coerencias"]),
                    violacoes=int(avaliacao["violacoes"]),
                    detalhes=dict(avaliacao["detalhes"]),
                )
            )

        # Ordenação: score desc, depois MENOS violações, depois MAIS coerências
        prototipos.sort(
            key=lambda p: (p.score_total, -p.violacoes, p.coerencias),
            reverse=True,
        )

        # Mantém compat: retorna TOP k protótipos
        return prototipos[: int(k)]

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

        protos = self.gerar_prototipos(
            k=int(k),
            regime_id=regime_id,
            max_candidatos=max_candidatos,
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
        }

        if incluir_contexto_dna:
            resp["contexto_dna"] = {
                "universo_max": self.universo_max,
                "total_sorteadas": self.grupo.total_sorteadas(),
            }

        return resp


# =========================
# TESTE LOCAL DIRETO
# =========================

if __name__ == "__main__":
    motor = MotorFGI(
        historico_csv="lotofacil_ultimos_25_concursos.csv",
        universo_max=25,
    )

    resultado = motor.gerar_prototipos_json(
        k=15,
        regime_id="estavel",
        max_candidatos=2000,
        incluir_contexto_dna=True,
    )

    print(resultado)
