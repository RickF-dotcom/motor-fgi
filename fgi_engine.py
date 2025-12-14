
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import random

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

    Foco agora: ajuste fino do pipeline (não “teoria”):
    - evita viés do get_candidatos determinístico (primeiras combinações -> altas adjacências)
    - quando detecta lote “ruim”, reamostra aleatoriamente do universo filtrando já-sorteadas
    - mantém contrato resiliente (aceita campos extras sem quebrar)
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

        # DNA anchor injetado pelo app (não acopla score ainda)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py (não quebrar)
    # ------------------------------------------------------------

    def set_dna_anchor(self, dna_anchor: Optional[Dict[str, Any]] = None, **_ignored: Any) -> None:
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

    def _ja_sorteada(self, seq: List[int]) -> bool:
        # Compat com versões diferentes do GrupoDeMilhoes
        if hasattr(self.grupo, "ja_sorteada"):
            return bool(self.grupo.ja_sorteada(seq))  # type: ignore[attr-defined]
        if hasattr(self.grupo, "_ja_saiu"):
            return bool(self.grupo._ja_saiu(seq))  # type: ignore[attr-defined]
        # fallback extremo: se não dá pra checar, assume “não sorteada”
        return False

    # ------------------------------------------------------------
    # Candidatos (ANTI-VIÉS)
    # ------------------------------------------------------------

    def _candidatos_deterministicos(self, k: int, max_candidatos: int) -> List[List[int]]:
        # tenta chamar assinatura nova (max_candidatos) ou antiga (limite)
        try:
            return [list(x) for x in self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)]  # type: ignore[arg-type]
        except TypeError:
            return [list(x) for x in self.grupo.get_candidatos(k, max_candidatos)]  # type: ignore[misc]

    def _candidatos_random(self, k: int, max_candidatos: int, seed: int = 123) -> List[List[int]]:
        rng = random.Random(int(seed))
        alvo = int(max_candidatos)

        vistos: set[Tuple[int, ...]] = set()
        out: List[List[int]] = []

        tentativas = 0
        limite_tentativas = max(20000, alvo * 250)

        universo = list(range(1, self.universo_max + 1))

        while len(out) < alvo and tentativas < limite_tentativas:
            tentativas += 1
            comb = tuple(sorted(rng.sample(universo, k)))
            if comb in vistos:
                continue
            vistos.add(comb)

            seq = list(comb)
            if self._ja_sorteada(seq):
                continue

            out.append(seq)

        return out

    def _gerar_candidatos(self, k: int, max_candidatos: int, seed: int = 123) -> List[List[int]]:
        """
        Estratégia:
        1) pega lote determinístico (rápido)
        2) se vier “podre” (muita adjacência => todo mundo viola), troca por random
        """
        lote = self._candidatos_deterministicos(k, max_candidatos)

        # Heurística objetiva: se a adj mínima já é alta (>=5), esse lote está viciado
        if lote:
            adjs = [self._count_adjacencias(s) for s in lote[: min(200, len(lote))]]
            if adjs and min(adjs) >= 5:
                return self._candidatos_random(k, max_candidatos, seed=seed)

        return lote if lote else self._candidatos_random(k, max_candidatos, seed=seed)

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

        z_max_soma = float(base["z_max_soma"])
        max_adj = int(base["max_adjacencias"])
        max_desvio_pares = int(base["max_desvio_pares"])

        # NOVO (fine-tuning): modo hard para adj
        adj_hard = False
        hard_max_adj = max_adj

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
                hard_max_adj = max_adj
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])

            if "adj_hard" in constraints_override:
                adj_hard = bool(constraints_override["adj_hard"])
            if "hard_max_adjacencias" in constraints_override:
                hard_max_adj = int(constraints_override["hard_max_adjacencias"])

        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        seq = sorted(int(x) for x in seq)
        soma = int(sum(seq))
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # score_soma (penaliza só excedente do limite)
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        # score_pares (0..1)
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        # score_adj (soft) + opcional hard
        if adj_hard and adj > hard_max_adj:
            score_adj = -1.0
        else:
            exced = max(0, adj - max_adj)
            denom = max(1.0, 2.0 * float(max_adj))
            score_adj = -min(1.0, exced / denom)

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

        coerencias = max(0, 3 - viol)

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
                    "hard_max_adjacencias": hard_max_adj,
                },
            },
            "dna_anchor": self.get_dna_anchor(),
        }

        return Prototipo(
            sequencia=seq,
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
        top_n: int = 30,
        seed: int = 123,  # aceito mesmo se o app não expõe
        **_ignored: Any,
    ) -> Dict[str, Any]:
        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active:
            contexto_lab = self._dna_anchor

        candidatos = self._gerar_candidatos(k=k, max_candidatos=max_candidatos, seed=seed)

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
                "seed": int(seed),
                "top_n": int(top_n),
            },
            "contexto_lab": contexto_lab,
    }
