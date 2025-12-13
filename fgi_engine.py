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
    - Gerar protótipos estruturais

    Nota (compat):
    - 'k' é usado como tamanho da sequência e também como quantidade retornada.
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

        # Regimes básicos
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 8,        # (você está trabalhando com DNA que aceita adj alta)
                "max_desvio_pares": 2,
                "dna_window": 12,
                "dna_dist_max": 1.6,         # escala do score_dna (ajustável)
                "dna_peso": 1.2,
                "pesos": {
                    "soma": 1.20,
                    "pares": 0.60,
                    "adj": 0.60,
                },
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 9,
                "max_desvio_pares": 3,
                "dna_window": 12,
                "dna_dist_max": 1.9,
                "dna_peso": 1.0,
                "pesos": {
                    "soma": 1.00,
                    "pares": 0.45,
                    "adj": 0.45,
                },
            },
        }

        # Contexto opcional de DNA (injeta pelo app.py antes de gerar)
        # Formato esperado:
        # { "janelas": { "12": {"soma_media":..., "impares_media":..., "faixa_1_13_media":..., "adjacencias_media":...}, ... } }
        self._dna_last25: Optional[Dict[str, Any]] = None

    # =========================
    # Injeção de contexto (opcional)
    # =========================

    def set_dna_last25(self, dna_last25: Optional[Dict[str, Any]]) -> None:
        self._dna_last25 = dna_last25

    # =========================
    # Helpers estatísticos (universo 1..N, sem reposição)
    # =========================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Média e desvio padrão da SOMA ao amostrar k números de 1..N (sem reposição).
        Var(sum) = k * var_pop * ((N - k)/(N - 1))
        Pop variance de 1..N: (N^2 - 1)/12
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

    def _count_faixa_1_13(self, seq_ordenada: List[int]) -> int:
        return sum(1 for x in seq_ordenada if 1 <= x <= 13)

    # =========================
    # DNA anchor
    # =========================

    def _get_dna_window(self, window: int) -> Optional[Dict[str, float]]:
        if not self._dna_last25:
            return None
        janelas = self._dna_last25.get("janelas") or {}
        w = str(int(window))
        d = janelas.get(w)
        if not isinstance(d, dict):
            return None

        # normaliza nomes esperados
        try:
            return {
                "soma_media": float(d["soma_media"]),
                "impares_media": float(d["impares_media"]),
                "faixa_1_13_media": float(d["faixa_1_13_media"]),
                "adjacencias_media": float(d["adjacencias_media"]),
            }
        except Exception:
            return None

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _dna_distance(
        self,
        seq_stats: Dict[str, float],
        dna: Dict[str, float],
        tols: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Distância “normalizada” por tolerâncias (quanto maior, pior).
        """
        d_soma = abs(seq_stats["soma"] - dna["soma_media"]) / max(1e-9, tols["soma"])
        d_impares = abs(seq_stats["impares"] - dna["impares_media"]) / max(1e-9, tols["impares"])
        d_faixa = abs(seq_stats["faixa_1_13"] - dna["faixa_1_13_media"]) / max(1e-9, tols["faixa_1_13"])
        d_adj = abs(seq_stats["adjacencias"] - dna["adjacencias_media"]) / max(1e-9, tols["adjacencias"])

        dist_total = d_soma + d_impares + d_faixa + d_adj
        return dist_total, {
            "d_soma": round(d_soma, 4),
            "d_impares": round(d_impares, 4),
            "d_faixa": round(d_faixa, 4),
            "d_adj": round(d_adj, 4),
        }

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

        cfg_base = self._regimes.get(regime, self._regimes["estavel"])
        cfg = dict(cfg_base)
        if constraints_override:
            cfg.update(constraints_override)

        pesos = dict(cfg.get("pesos", {}))
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos and vv is not None:
                    pesos[kk] = float(vv)

        soma = float(sum(seq_ord))
        pares = float(sum(1 for x in seq_ord if x % 2 == 0))
        impares = float(k - int(pares))
        adj = float(self._count_adjacencias(seq_ord))
        faixa_1_13 = float(self._count_faixa_1_13(seq_ord))

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        coerencias = 0
        violacoes = 0

        # --- SOMA
        z_max_soma = float(cfg["z_max_soma"])
        if z_soma <= z_max_soma:
            coerencias += 1
            score_soma = 1.0 - (z_soma / z_max_soma)
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - z_max_soma) / 1.0)

        # --- PARES
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        max_desvio_pares = float(cfg["max_desvio_pares"])
        if desvio_pares <= max_desvio_pares:
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, max_desvio_pares))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - max_desvio_pares) / 1.0)

        # --- ADJ
        max_adj = float(cfg["max_adjacencias"])
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, max_adj))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        # Score abstrato (sem DNA)
        score_abstrato = (
            float(pesos["soma"]) * score_soma +
            float(pesos["pares"]) * score_pares +
            float(pesos["adj"]) * score_adj
        )

        # =========================
        # DNA score (opcional)
        # =========================
        dna_window = int(cfg.get("dna_window", 12))
        dna_dist_max = float(cfg.get("dna_dist_max", 1.6))
        dna_peso = float(cfg.get("dna_peso", 1.2))

        dna_anchor: Dict[str, Any] = {"ativo": False, "motivo": "sem_dna_anchor"}
        score_dna = 0.0

        dna = self._get_dna_window(dna_window)
        if dna:
            # tolerâncias: as mesmas que você já está usando nos JSONs
            tols = {
                "soma": 18.0,
                "impares": 1.5,
                "faixa_1_13": 1.5,
                "adjacencias": 2.2,
            }
            seq_stats = {
                "soma": soma,
                "impares": impares,
                "faixa_1_13": faixa_1_13,
                "adjacencias": adj,
            }

            dist_total, dist_comp = self._dna_distance(seq_stats, dna, tols)

            # ✅ CORREÇÃO: score_dna SEMPRE em [-1, +1]
            # linear: dist=0 => +1 ; dist=dna_dist_max => 0 ; dist>> => vai até -1, mas nunca passa disso
            score_dna = 1.0 - (dist_total / max(1e-9, dna_dist_max))
            score_dna = self._clamp(score_dna, -1.0, 1.0)

            # coerência/violação do DNA como “regra extra”
            if dist_total <= dna_dist_max:
                coerencias += 1
            else:
                violacoes += 1

            dna_anchor = {
                "ativo": True,
                "window": dna_window,
                "seq": {
                    "soma": int(soma),
                    "impares": int(impares),
                    "faixa_1_13": int(faixa_1_13),
                    "adjacencias": int(adj),
                },
                "dna": {
                    "soma_media": dna["soma_media"],
                    "impares_media": dna["impares_media"],
                    "faixa_1_13_media": dna["faixa_1_13_media"],
                    "adjacencias_media": dna["adjacencias_media"],
                },
                "dist_comp": dist_comp,
                "dist_total": round(dist_total, 4),
                "tols": tols,
            }

        # Score final
        score_total = score_abstrato + (dna_peso * score_dna)

        detalhes = {
            "k": k,
            "soma": int(soma),
            "mu_soma": round(mu_soma, 4),
            "sd_soma": round(sd_soma, 4),
            "z_soma": round(z_soma, 4),
            "pares": int(pares),
            "impares": int(impares),
            "adjacencias": int(adj),
            "regime": regime,
            "componentes": {
                "score_abstrato": round(score_abstrato, 6),
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "score_dna": round(score_dna, 4),
                "dna_peso": round(dna_peso, 4),
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": float(cfg["z_max_soma"]),
                    "max_adjacencias": float(cfg["max_adjacencias"]),
                    "max_desvio_pares": float(cfg["max_desvio_pares"]),
                    "dna_dist_max": float(dna_dist_max),
                    "dna_window": int(dna_window),
                },
                "dna_anchor": dna_anchor,
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
