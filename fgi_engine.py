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
    Motor de avaliação / geração de protótipos.

    Objetivo: contrato-resiliente.
    - Não quebra com parâmetros novos.
    - Exponibiliza set_dna_anchor(...) no formato que o app.py chama.
    - Permite evolução incremental sem estourar 500.

    Decisão de laboratório aplicada aqui:
    - No regime 'estavel', adjacências NÃO eliminam (não contam como violação).
      Elas apenas penalizam score.
    - Eliminação por adjacência só ocorre se houver hard_max_adjacencias (override).
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

        # Config por regime
        # adj_hard: se True, adjacência conta como violação (corte/eliminação lógica)
        #           se False, adjacência só penaliza no score (não elimina)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
                "adj_hard": False,  # <<< DECISÃO: no estável NÃO elimina
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
                "adj_hard": True,   # tenso pode ser mais rígido
            },
        }

        # DNA anchor (contexto do laboratório)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py
    # ------------------------------------------------------------

    def set_dna_anchor(self, *args: Any, **kwargs: Any) -> None:
        """
        Suporta:
          - set_dna_anchor(dna_last25=<dict>, window=<int>)
          - set_dna_anchor(dna_anchor=<dict>)
          - set_dna_anchor(<dict>)  (posicional)
        Sem acoplar score ao DNA ainda: aqui é só âncora / contexto.
        """
        payload: Optional[Dict[str, Any]] = None

        # caso posicional: set_dna_anchor(dna_dict)
        if args:
            if isinstance(args[0], dict):
                payload = args[0]
            else:
                payload = {"value": args[0]}

        # caso keyword: dna_last25 + window (forma do seu app.py)
        if "dna_last25" in kwargs:
            dna_last25 = kwargs.get("dna_last25")
            window = kwargs.get("window")
            payload = {
                "dna_last25": dna_last25,
                "window": window,
            }

        # caso keyword direto: dna_anchor
        if "dna_anchor" in kwargs and kwargs.get("dna_anchor") is not None:
            payload = kwargs.get("dna_anchor")

        self._dna_anchor = payload if payload else None
        self._dna_anchor_active = bool(self._dna_anchor)

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "payload": self._dna_anchor if self._dna_anchor_active else None,
        }

    # ------------------------------------------------------------
    # Helpers matemáticos
    # ------------------------------------------------------------

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Soma de k números amostrados sem reposição de {1..N}:
        mu = k*(N+1)/2
        var = k*(N-k)*(N+1)/12
        """
        N = self.universo_max
        k = int(k)
        mu = k * (N + 1) / 2.0
        var = k * (N - k) * (N + 1) / 12.0
        sd = math.sqrt(var) if var > 0 else 0.0
        return mu, sd

    def _count_adjacencias(self, seq: List[int]) -> int:
        if not seq:
            return 0
        s = sorted(seq)
        adj = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                adj += 1
        return adj

    # ------------------------------------------------------------
    # Avaliação (score + violações)
    # ------------------------------------------------------------

    def avaliar_sequencia(
        self,
        seq: List[int],
        k: int,
        regime_id: str = "estavel",
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ) -> Prototipo:
        k = int(k)
        regime_id = (regime_id or self.regime_padrao).strip().lower()
        base = self._regimes.get(regime_id, self._regimes[self.regime_padrao])

        # constraints efetivos
        z_max_soma = float(base["z_max_soma"])
        max_adj = int(base["max_adjacencias"])
        max_desvio_pares = int(base["max_desvio_pares"])
        adj_hard = bool(base.get("adj_hard", True))

        hard_max_adjacencias: Optional[int] = None  # override extra (hard filter)

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])

            # hard kill switch (se você quiser forçar eliminação)
            if "hard_max_adjacencias" in constraints_override and constraints_override["hard_max_adjacencias"] is not None:
                hard_max_adjacencias = int(constraints_override["hard_max_adjacencias"])

        # pesos efetivos
        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        soma = int(sum(seq))
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # score_soma
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        # score_pares
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        # score_adj (sempre penaliza se passou do max_adj)
        score_adj = 0.0 if adj <= max_adj else -1.0

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        # violações
        viol = 0
        if abs(z_soma) > z_max_soma:
            viol += 1

        # >>> decisão: adjacência só é violação se adj_hard=True
        if adj_hard and (adj > max_adj):
            viol += 1

        if max_desvio_pares > 0 and desvio_pares > max_desvio_pares:
            viol += 1

        total_constraints = 3
        coerencias = max(0, total_constraints - viol)

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
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": z_max_soma,
                    "max_adjacencias": max_adj,
                    "max_desvio_pares": max_desvio_pares,
                    "adj_hard": adj_hard,
                    "hard_max_adjacencias": hard_max_adjacencias,
                },
            },
            "dna_anchor": self.get_dna_anchor(),
        }

        return Prototipo(
            sequencia=sorted(seq),
            score_total=float(round(score_total, 6)),
            coerencias=int(coerencias),
            violacoes=int(viol),
            detalhes=detalhes,
        )

    # ------------------------------------------------------------
    # Geração de protótipos (JSON)
    # ------------------------------------------------------------

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
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """
        Retorna payload compatível com o Swagger.
        """

        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        # contexto (se estiver ativo)
        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active:
            contexto_lab = self._dna_anchor

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        # hard filter opcional
        hard_max_adjacencias: Optional[int] = None
        if constraints_override and "hard_max_adjacencias" in constraints_override and constraints_override["hard_max_adjacencias"] is not None:
            hard_max_adjacencias = int(constraints_override["hard_max_adjacencias"])

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            seq_list = list(seq)

            # se tiver hard cut, elimina aqui (antes de avaliar)
            if hard_max_adjacencias is not None:
                adj = self._count_adjacencias(seq_list)
                if adj > hard_max_adjacencias:
                    continue

            p = self.avaliar_sequencia(
                seq_list,
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override or {},
                constraints_override=constraints_override or {},
            )
            prototipos.append(p)

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[: max(1, top_n)] if prototipos else []

        return {
            "prototipos": [
                {
                    "sequencia": p.sequencia,
                    "score_total": p.score_total,
                    "coerencias": p.coerencias,
                    "violacoes": p.violacoes,
                    "detalhes": p.detalhes,
                }
                for p in prototipos
            ],
            "regime_usado": (regime_id or self.regime_padrao),
            "max_candidatos_usado": max_candidatos,
            "overrides_usados": {
                "pesos_override": pesos_override or {},
                "constraints_override": constraints_override or {},
                "windows": windows or None,
                "pesos_windows": pesos_windows or None,
                "pesos_metricas": pesos_metricas or None,
            },
            "contexto_lab": contexto_lab,
        }
