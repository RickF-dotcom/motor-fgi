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

    Contrato-resiliente:
    - aceita overrides novos sem quebrar (via **kwargs e campos opcionais)
    - expõe set_dna_anchor() com assinatura compatível com o app.py atual
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
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
            },
        }

        # DNA anchor (injetado pelo app quando incluir_contexto_dna=True)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py (corrige 500 de atributo/assinatura)
    # ------------------------------------------------------------

    def set_dna_anchor(
        self,
        dna_anchor: Optional[Dict[str, Any]] = None,
        *,
        dna_last25: Optional[Dict[str, Any]] = None,
        window: Optional[int] = None,
        **_ignored: Any,
    ) -> None:
        """
        Suporta os dois formatos:

        (A) formato novo (do seu app.py):
            set_dna_anchor(dna_last25=<dict>, window=12)

        (B) formato antigo:
            set_dna_anchor(dna_anchor=<dict>)

        Qualquer kwargs extra é ignorado (resiliência).
        """
        if dna_anchor is not None:
            payload = dna_anchor
        else:
            # monta payload "padrão laboratório"
            payload = {
                "dna_last25": dna_last25,
                "window": int(window) if window is not None else None,
            }

        # normaliza: se não veio nada útil, desativa
        if not payload or (isinstance(payload, dict) and all(v is None for v in payload.values())):
            self._dna_anchor = None
            self._dna_anchor_active = False
            return

        if not isinstance(payload, dict):
            # garante que não explode por tipo bizarro vindo do app
            self._dna_anchor = {"raw": payload}
            self._dna_anchor_active = True
            return

        self._dna_anchor = payload
        self._dna_anchor_active = True

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
        """
        Conta adjacências (pares consecutivos) dentro da sequência.
        Ex: [1,2,3] tem 2 adjacências (1-2, 2-3).
        """
        if not seq:
            return 0
        s = sorted(seq)
        adj = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                adj += 1
        return adj

    # ------------------------------------------------------------
    # Avaliação (score + violações) — compatível com seus JSONs
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

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])

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

        # componentes
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))
        else:
            score_pares = 0.0

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
        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active:
            contexto_lab = self._dna_anchor

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            p = self.avaliar_sequencia(
                list(seq),
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override or {},
                constraints_override=constraints_override or {},
            )
            prototipos.append(p)

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[: max(1, top_n)]

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
