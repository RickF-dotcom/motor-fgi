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

    Nota (compat):
    - 'k' é TAMANHO DA SEQUÊNCIA e também QUANTIDADE DE PROTÓTIPOS retornados.
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
    # Helpers
    # =========================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Média e desvio padrão da SOMA ao amostrar k números de 1..N (sem reposição).
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

    def _count_range(self, seq_ordenada: List[int], lo: int, hi: int) -> int:
        return sum(1 for x in seq_ordenada if lo <= x <= hi)

    def _merge_pesos(self, base: Dict[str, float], override: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not override:
            return dict(base)
        out = dict(base)
        for k, v in override.items():
            if k in ("soma", "pares", "adj"):
                try:
                    out[k] = float(v)
                except Exception:
                    pass
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

        seq_ord = sorted(int(x) for x in seq)
        k = len(seq_ord)

        cfg = self._regimes.get(regime, self._regimes["estavel"])
        pesos = self._merge_pesos(cfg["pesos"], pesos_override)

        soma = sum(seq_ord)
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        faixa_1_10 = self._count_range(seq_ord, 1, 10)
        faixa_11_25 = self._count_range(seq_ord, 11, 25)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        coerencias = 0
        violacoes = 0

        # --- Soma
        if z_soma <= float(cfg["z_max_soma"]):
            coerencias += 1
            score_soma = 1.0 - (z_soma / float(cfg["z_max_soma"]))
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - float(cfg["z_max_soma"])) / 1.0)

        # --- Pares
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        if desvio_pares <= int(cfg["max_desvio_pares"]):
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, float(cfg["max_desvio_pares"])))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - float(cfg["max_desvio_pares"])) / 1.0)

        # --- Adjacências
        max_adj = int(cfg["max_adjacencias"])
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, float(max_adj)))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        # --- Constraints (apenas contam como coerência/violação e entram na ordenação)
        cons = constraints_override or {}

        def _check_minmax(key_min: str, key_max: str, valor: int) -> None:
            nonlocal coerencias, violacoes
            if key_min in cons:
                try:
                    if valor >= int(cons[key_min]):
                        coerencias += 1
                    else:
                        violacoes += 1
                except Exception:
                    pass
            if key_max in cons:
                try:
                    if valor <= int(cons[key_max]):
                        coerencias += 1
                    else:
                        violacoes += 1
                except Exception:
                    pass

        _check_minmax("min_faixa_1_10", "max_faixa_1_10", faixa_1_10)
        _check_minmax("min_faixa_11_25", "max_faixa_11_25", faixa_11_25)
        _check_minmax("min_adj", "max_adj", adj)
        _check_minmax("min_pares", "max_pares", pares)
        _check_minmax("min_soma", "max_soma", soma)

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
            "faixa_1_10": faixa_1_10,
            "faixa_11_25": faixa_11_25,
            "regime": regime,
            "componentes": {
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "pesos": pesos,
                "constraints_override": cons,
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
                seq,
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
