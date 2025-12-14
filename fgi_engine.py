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

    Objetivos deste arquivo:
    - contrato-resiliente (aceita overrides novos sem quebrar)
    - expõe set_dna_anchor() para o app.py não estourar 500
    - HARD FILTER de adjacência no regime 'estavel' (elimina, não penaliza)
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

        # Regras/limites por regime (compatível com seus JSONs)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,          # limite "soft" (score/violação)
                "hard_max_adjacencias": 3,     # limite HARD (eliminação) no estavel
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "hard_max_adjacencias": None,  # no tenso não elimina por padrão
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
            },
        }

        # DNA anchor (injetado pelo app quando incluir_contexto_dna=True)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py
    # ------------------------------------------------------------

    def set_dna_anchor(
        self,
        dna_anchor: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Compatível com:
          - set_dna_anchor(dna_anchor={...})
          - set_dna_anchor(dna_last25={...}, window=12)
        """
        # Formato novo (recomendado no app.py)
        dna_last25 = kwargs.get("dna_last25", None)
        window = kwargs.get("window", None)

        if dna_anchor is None and dna_last25 is not None:
            payload = {
                "dna_last25": dna_last25,
                "window": window,
            }
            self._dna_anchor = payload
            self._dna_anchor_active = True
            return

        # Formato antigo (genérico)
        self._dna_anchor = dna_anchor or None
        self._dna_anchor_active = bool(dna_anchor)

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
    # Config efetiva por regime + overrides
    # ------------------------------------------------------------

    def _effective_regime_cfg(
        self,
        regime_id: str,
        pesos_override: Optional[Dict[str, float]],
        constraints_override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        regime_id = (regime_id or self.regime_padrao).strip().lower()
        base = self._regimes.get(regime_id, self._regimes[self.regime_padrao])

        # constraints
        z_max_soma = float(base["z_max_soma"])
        max_adj = int(base["max_adjacencias"])
        max_desvio_pares = int(base["max_desvio_pares"])

        hard_max_adj = base.get("hard_max_adjacencias", None)
        hard_max_adj = None if hard_max_adj is None else int(hard_max_adj)

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])
            # HARD override (novo)
            if "hard_max_adjacencias" in constraints_override:
                v = constraints_override["hard_max_adjacencias"]
                hard_max_adj = None if v is None else int(v)

        # pesos
        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        return {
            "regime_id": regime_id,
            "z_max_soma": z_max_soma,
            "max_adjacencias": max_adj,
            "hard_max_adjacencias": hard_max_adj,
            "max_desvio_pares": max_desvio_pares,
            "pesos": pesos,
        }

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

        cfg = self._effective_regime_cfg(regime_id, pesos_override, constraints_override)
        regime_id = cfg["regime_id"]
        z_max_soma = cfg["z_max_soma"]
        max_adj = cfg["max_adjacencias"]
        max_desvio_pares = cfg["max_desvio_pares"]
        pesos = cfg["pesos"]

        soma = int(sum(seq))
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        score_adj = 0.0 if adj <= max_adj else -1.0

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        viol = 0
        if abs(z_soma) > z_max_soma:
            viol += 1
        if adj > max_adj:
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
                    "hard_max_adjacencias": cfg["hard_max_adjacencias"],
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
    # Geração de protótipos (JSON) com HARD FILTER
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
        Retorna payload compatível com o Swagger:
        - prototipos (lista)
        - regime_usado
        - max_candidatos_usado
        - overrides_usados
        - contexto_lab (quando incluir_contexto_dna=True e anchor ativo)

        Obs: windows/pesos_windows/pesos_metricas são aceitos pra não quebrar contrato.
        """
        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        pesos_override = pesos_override or {}
        constraints_override = constraints_override or {}

        cfg = self._effective_regime_cfg(regime_id, pesos_override, constraints_override)
        regime_eff = cfg["regime_id"]

        # HARD FILTER: no estavel, elimina por adjacência acima do limite HARD
        hard_max_adj = cfg.get("hard_max_adjacencias", None)
        apply_hard = (regime_eff == "estavel" and hard_max_adj is not None)

        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active:
            contexto_lab = self._dna_anchor

        prototipos: List[Prototipo] = []

        # Estratégia anti-sequidão: se filtro ficar muito agressivo, tenta buscar mais candidatos
        # (sem loop infinito).
        tries = 0
        cap = max(2000, max_candidatos)
        step = cap
        current_max = cap

        while len(prototipos) < top_n and tries < 6:
            tries += 1

            candidatos = self.grupo.get_candidatos(k=k, max_candidatos=current_max)

            prototipos = []
            for seq in candidatos:
                seq_list = list(seq)

                if apply_hard:
                    adj = self._count_adjacencias(seq_list)
                    if adj > int(hard_max_adj):
                        continue

                p = self.avaliar_sequencia(
                    seq_list,
                    k=k,
                    regime_id=regime_eff,
                    pesos_override=pesos_override,
                    constraints_override=constraints_override,
                )
                prototipos.append(p)

            if len(prototipos) >= top_n:
                break

            # aumenta o universo de busca (se o GrupoDeMilhoes respeitar isso)
            current_max = min(current_max + step, 50000)

            # se não cresceu nada, não adianta insistir
            if current_max == 50000:
                break

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
            "regime_usado": regime_eff,
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
