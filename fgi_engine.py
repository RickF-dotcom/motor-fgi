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

    Nota (compat com seu endpoint atual):
    - 'k' é usado como TAMANHO DA SEQUÊNCIA (ex.: 15 dezenas) E também como QUANTIDADE
      de protótipos retornados (top-k).
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

        # -------------------------
        # Regimes (calibração inicial)
        # -------------------------
        # Ponto crítico corrigido:
        # - "estavel" NÃO pode ter max_adjacencias=3, porque o DNA real (últimas 25)
        #   gira perto de 8 adjacências em média. 3 quebra tudo.
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 12,     # <-- corrigido (antes 3)
                "max_desvio_pares": 2,
                "pesos": {
                    "soma": 1.20,
                    "pares": 0.60,
                    "adj": 0.60,
                },
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 14,
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

    def _merge_pesos(
        self,
        base: Dict[str, float],
        override: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        out = dict(base)
        if override:
            for k, v in override.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
        return out

    def _merge_constraints(
        self,
        base: Dict[str, Any],
        override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out = dict(base)
        if override:
            for k, v in override.items():
                out[str(k)] = v
        return out

    # =========================
    # Avaliação (PONTO C)
    # =========================

    def avaliar_sequencia(
        self,
        seq: List[int],
        regime: str,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Avaliação estruturada.

        Componentes atuais:
        - Soma (z-score no universo 1..N sem reposição)
        - Paridade (pares vs k/2)
        - Adjacências (consecutivos)
        """
        seq_ord = sorted(int(x) for x in seq)
        k = len(seq_ord)

        cfg_base = self._regimes.get(regime, self._regimes["estavel"])

        # constraints efetivas (com override)
        constraints_base = {
            "z_max_soma": float(cfg_base["z_max_soma"]),
            "max_adjacencias": int(cfg_base["max_adjacencias"]),
            "max_desvio_pares": int(cfg_base["max_desvio_pares"]),
        }
        constraints = self._merge_constraints(constraints_base, constraints_override)

        # pesos efetivos (com override)
        pesos_base = {
            "soma": float(cfg_base["pesos"]["soma"]),
            "pares": float(cfg_base["pesos"]["pares"]),
            "adj": float(cfg_base["pesos"]["adj"]),
        }
        pesos = self._merge_pesos(pesos_base, pesos_override)

        soma = sum(seq_ord)
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        coerencias = 0
        violacoes = 0

        # --- SOMA
        z_max_soma = float(constraints["z_max_soma"])
        if z_soma <= z_max_soma:
            coerencias += 1
            score_soma = 1.0 - (z_soma / max(1e-9, z_max_soma))  # 1..0
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - z_max_soma) / 1.0)

        # --- PARES
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        max_desvio_pares = int(constraints["max_desvio_pares"])
        if desvio_pares <= max_desvio_pares:
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, float(max_desvio_pares)))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - float(max_desvio_pares)) / 1.0)

        # --- ADJACÊNCIAS
        max_adj = int(constraints["max_adjacencias"])
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, float(max_adj)))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        # Score total (ponderado)
        score_total = (
            float(pesos["soma"]) * float(score_soma) +
            float(pesos["pares"]) * float(score_pares) +
            float(pesos["adj"]) * float(score_adj)
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
                "constraints": constraints,
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
                list(seq),
                regime,
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
