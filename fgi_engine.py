from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math

from grupo_de_milhoes import GrupoDeMilhoes


# ============================================================
# Estrutura de saída
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
    Gera e avalia protótipos (sequências) usando:
      - universo filtrado (GrupoDeMilhoes)
      - estatísticas de soma sem reposição
      - regras por "regime" (estavel / tenso)
      - overrides opcionais de pesos e constraints
      - HARD FILTER de adjacência no regime estável (ponto crítico)
    """

    def __init__(
        self,
        historico_csv: Optional[str] = None,
        universo_max: int = 25,
    ) -> None:
        self.universo_max = int(universo_max)

        # Universo de combinações ainda não sorteadas (Grupo de Milhões)
        self.grupo = GrupoDeMilhoes(
            universo_max=self.universo_max,
            historico_csv=historico_csv if historico_csv else None,
        )

        # Config por regime (baseline)
        self.regime_padrao = "estavel"
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,  # desvio máximo permitido vs k/2
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
                # hard filter apenas aqui:
                "hard_filter_adjacencias": True,
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
                "hard_filter_adjacencias": False,
            },
        }

    # ----------------------------
    # Estatística (soma) sem reposição
    # ----------------------------
    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Soma de k números distintos amostrados uniformemente de {1..N}, sem reposição.
        Retorna (média, desvio padrão).
        """
        N = self.universo_max
        k = int(k)

        # Média de 1..N = (N+1)/2, então soma tem média k*(N+1)/2
        mu = k * (N + 1) / 2.0

        # Variância da soma sem reposição:
        # Var(S) = k * Var(pop) * (N-k)/(N-1)
        # Var(pop) para 1..N = (N^2 - 1)/12
        var_pop = (N * N - 1) / 12.0
        fpc = (N - k) / (N - 1) if N > 1 else 0.0
        var_sum = k * var_pop * fpc
        sd = math.sqrt(max(var_sum, 0.0))

        return mu, sd

    # ----------------------------
    # Métricas simples
    # ----------------------------
    def count_adjacencias(self, seq: List[int]) -> int:
        """
        Conta adjacências como número de pares consecutivos (n, n+1) presentes.
        Ex: [1,2,3] tem 2 adjacências: (1,2) e (2,3)
        """
        s = set(seq)
        return sum(1 for n in s if (n + 1) in s)

    def count_pares(self, seq: List[int]) -> int:
        return sum(1 for n in seq if n % 2 == 0)

    # ----------------------------
    # Scoring (comportamento observado nos JSONs)
    # ----------------------------
    def _score_soma(self, z_abs: float, z_max: float) -> Tuple[float, int]:
        """
        Replica o comportamento que apareceu no JSON:
          - se z <= z_max => score positivo em [0..1]
          - se z > z_max => score = -(z - z_max)  (excesso vira negativo)
        """
        if z_abs <= z_max:
            # quanto mais perto de 0, mais "coerente"
            score = (z_max - z_abs) / z_max if z_max > 0 else 0.0
            return score, 0
        else:
            return -(z_abs - z_max), 1

    def _score_pares(self, pares: int, k: int, max_desvio: float) -> Tuple[float, int]:
        """
        Esperado ~ k/2. Se dentro do desvio, score = 1 - (diff/max_desvio).
        Se fora, score = -(diff - max_desvio).
        """
        alvo = k / 2.0
        diff = abs(pares - alvo)

        if max_desvio <= 0:
            return 0.0, 0

        if diff <= max_desvio:
            return 1.0 - (diff / max_desvio), 0
        else:
            return -(diff - max_desvio), 1

    def _score_adj(self, adj: int, max_adj: int) -> Tuple[float, int]:
        """
        Simples e consistente com os JSONs anteriores:
          - se passa do limite: score -1 e 1 violação
          - se não passa: score +1 (coerente)
        Obs: no regime estável, acima do limite vira HARD FILTER antes de chegar aqui.
        """
        if max_adj is None:
            return 0.0, 0
        if adj > max_adj:
            return -1.0, 1
        return 1.0, 0

    # ----------------------------
    # Avaliação de uma sequência
    # ----------------------------
    def avaliar_sequencia(
        self,
        seq: List[int],
        k: int,
        regime_id: str,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        pesos_metricas: Optional[Dict[str, float]] = None,
    ) -> Optional[Prototipo]:
        """
        Retorna Prototipo avaliado, ou None se a sequência for descartada (hard filter).
        """
        regime_id = (regime_id or self.regime_padrao).strip().lower()
        regime = self._regimes.get(regime_id, self._regimes[self.regime_padrao])

        # constraints e pesos efetivos
        constraints = {
            "z_max_soma": regime["z_max_soma"],
            "max_adjacencias": regime["max_adjacencias"],
            "max_desvio_pares": regime["max_desvio_pares"],
        }
        if constraints_override:
            # aceita chaves extras mas só aplica as conhecidas
            for key in ("z_max_soma", "max_adjacencias", "max_desvio_pares"):
                if key in constraints_override and constraints_override[key] is not None:
                    constraints[key] = constraints_override[key]

        pesos = dict(regime["pesos"])
        # 1) se vier pesos_metricas (novo), ele tem prioridade
        if pesos_metricas:
            for key in ("soma", "pares", "adj"):
                if key in pesos_metricas and pesos_metricas[key] is not None:
                    pesos[key] = float(pesos_metricas[key])
        # 2) depois pesos_override (legado)
        if pesos_override:
            for key in ("soma", "pares", "adj"):
                if key in pesos_override and pesos_override[key] is not None:
                    pesos[key] = float(pesos_override[key])

        # métricas
        seq_sorted = sorted(int(x) for x in seq)
        soma = sum(seq_sorted)
        pares = self.count_pares(seq_sorted)
        impares = k - pares
        adj = self.count_adjacencias(seq_sorted)

        # ===========================
        # HARD FILTER (ESTÁVEL)
        # ===========================
        if regime.get("hard_filter_adjacencias", False):
            max_adj = int(constraints["max_adjacencias"])
            if adj > max_adj:
                # DESCARTE TOTAL: não entra no ranking, não entra no JSON.
                return None

        # soma z-score
        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma
        z_abs = abs(z_soma)

        # scores + violações
        coerencias = 0
        violacoes = 0

        score_soma, v1 = self._score_soma(z_abs=z_abs, z_max=float(constraints["z_max_soma"]))
        violacoes += v1
        coerencias += 1 if v1 == 0 else 0

        score_pares, v2 = self._score_pares(
            pares=pares,
            k=k,
            max_desvio=float(constraints["max_desvio_pares"]),
        )
        violacoes += v2
        coerencias += 1 if v2 == 0 else 0

        score_adj, v3 = self._score_adj(adj=adj, max_adj=int(constraints["max_adjacencias"]))
        violacoes += v3
        coerencias += 1 if v3 == 0 else 0

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        detalhes = {
            "k": int(k),
            "soma": int(soma),
            "mu_soma": float(mu_soma),
            "sd_soma": float(sd_soma),
            "z_soma": float(z_soma),
            "pares": int(pares),
            "impares": int(impares),
            "adjacencias": int(adj),
            "regime": regime_id,
            "componentes": {
                "score_soma": float(score_soma),
                "score_pares": float(score_pares),
                "score_adj": float(score_adj),
                "pesos": {"soma": float(pesos["soma"]), "pares": float(pesos["pares"]), "adj": float(pesos["adj"])},
                "constraints": {
                    "z_max_soma": float(constraints["z_max_soma"]),
                    "max_adjacencias": int(constraints["max_adjacencias"]),
                    "max_desvio_pares": float(constraints["max_desvio_pares"]),
                    "hard_filter_adjacencias": bool(regime.get("hard_filter_adjacencias", False)),
                },
            },
        }

        return Prototipo(
            sequencia=seq_sorted,
            score_total=float(score_total),
            coerencias=int(coerencias),
            violacoes=int(violacoes),
            detalhes=detalhes,
        )

    # ----------------------------
    # Geração (top N) com suporte a overrides e janelas
    # ----------------------------
    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: str = "estavel",
        max_candidatos: int = 2000,
        incluir_contexto_dna: bool = True,
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        windows: Optional[List[int]] = None,
        pesos_windows: Optional[Dict[str, float]] = None,
        pesos_metricas: Optional[Dict[str, float]] = None,
        top_n: int = 30,
        contexto_lab: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estratégia:
          - Pega candidatos do GrupoDeMilhoes (não sorteados).
          - Avalia cada sequência (com hard filter no estável).
          - Ordena por score_total desc.
          - Retorna top_n.

        Observação:
          - parâmetros windows/pesos_windows ficam aceitos para “linha de pesquisa”
            (comparação fractal/tempo), mas aqui ainda não alteram o score diretamente.
            (a base do motor continua consistente e testável).
        """
        k = int(k)
        max_candidatos = int(max_candidatos)
        top_n = int(top_n)

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            p = self.avaliar_sequencia(
                seq=seq,
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
                pesos_metricas=pesos_metricas,
            )
            if p is not None:
                prototipos.append(p)

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[:top_n]

        payload: Dict[str, Any] = {
            "prototipos": [asdict(p) for p in prototipos],
            "regime_usado": (regime_id or self.regime_padrao),
            "max_candidatos_usado": max_candidatos,
            "overrides_usados": {
                "pesos_override": pesos_override or {},
                "constraints_override": constraints_override or {},
            },
        }

        # Mantemos esses campos para pesquisa, sem quebrar swagger
        if windows is not None:
            payload["windows"] = windows
        if pesos_windows is not None:
            payload["pesos_windows"] = pesos_windows
        if pesos_metricas is not None:
            payload["pesos_metricas"] = pesos_metricas

        # Se o app quiser anexar contexto do laboratório
        if incluir_contexto_dna and contexto_lab is not None:
            payload["contexto_lab"] = contexto_lab

        return payload
