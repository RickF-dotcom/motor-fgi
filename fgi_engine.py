from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from grupo_de_milhoes import GrupoDeMilhoes


@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


class MotorFGI:
    """
    Motor principal do laboratório.

    Agora com ANCORAGEM FRACTAL:
    - além de score "abstrato" (soma/paridade/adj),
      aplicamos um score de DISTÂNCIA ao DNA-alvo (por padrão janela 12).
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

        # Âncora DNA (setada pelo app antes de gerar protótipos)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_window: int = 12

        # Configs de regimes (ajustáveis por override)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                # abstrato
                "z_max_soma": 1.30,
                "max_adjacencias": 8,        # abstrato, não é o DNA; é só uma trava
                "max_desvio_pares": 2,

                # ancoragem (DNA)
                "dna_dist_max": 1.60,        # <= isso conta como "coerente" na âncora
                "dna_peso": 1.20,            # peso do termo de ancoragem no score total
                "dna_tols": {                # tolerâncias (escala) de distância
                    "soma": 18.0,
                    "impares": 1.5,
                    "faixa_1_13": 1.5,
                    "adjacencias": 2.2,
                },
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

                "dna_dist_max": 2.10,
                "dna_peso": 1.00,
                "dna_tols": {
                    "soma": 20.0,
                    "impares": 1.8,
                    "faixa_1_13": 1.8,
                    "adjacencias": 2.6,
                },
                "pesos": {
                    "soma": 1.00,
                    "pares": 0.45,
                    "adj": 0.45,
                },
            },
        }

    # =========================
    # API de configuração (chamada pelo app)
    # =========================

    def set_dna_anchor(self, dna_last25: Dict[str, Any], window: int = 12) -> None:
        """
        Recebe o blob do RegimeDetector.extrair_dna() (dna_last25)
        e guarda a janela usada como âncora (default 12).
        """
        self._dna_anchor = dna_last25 or None
        self._dna_anchor_window = int(window)

    # =========================
    # Helpers estatísticos (1..N, sem reposição)
    # =========================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
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

    def _count_faixa_1_13(self, seq: List[int]) -> int:
        return sum(1 for x in seq if 1 <= x <= 13)

    # =========================
    # ÂNCORA: distância ao DNA(window)
    # =========================

    def _get_dna_window(self) -> Optional[Dict[str, Any]]:
        if not self._dna_anchor:
            return None
        janelas = self._dna_anchor.get("janelas") or {}
        # janelas vêm como dict com chaves "7","10","12"... (strings)
        key = str(self._dna_anchor_window)
        return janelas.get(key)

    def _dna_distance(
        self,
        seq_ord: List[int],
        regime_cfg: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Distância normalizada ao DNA alvo (janela 12 por padrão).
        Retorna:
          - dist (quanto menor melhor)
          - detalhes (para instrumentação)
        """
        dna_w = self._get_dna_window()
        if not dna_w:
            return 0.0, {"ativo": False, "motivo": "sem_dna_anchor"}

        soma = sum(seq_ord)
        impares = sum(1 for x in seq_ord if x % 2 == 1)
        faixa_1_13 = self._count_faixa_1_13(seq_ord)
        adj = self._count_adjacencias(seq_ord)

        # médias do DNA
        mu_soma = float(dna_w.get("soma_media", 0.0))
        mu_impares = float(dna_w.get("impares_media", 0.0))
        mu_faixa = float(dna_w.get("faixa_1_13_media", 0.0))
        mu_adj = float(dna_w.get("adjacencias_media", 0.0))

        tols = (regime_cfg.get("dna_tols") or {})
        tol_soma = float(tols.get("soma", 18.0)) or 18.0
        tol_impares = float(tols.get("impares", 1.5)) or 1.5
        tol_faixa = float(tols.get("faixa_1_13", 1.5)) or 1.5
        tol_adj = float(tols.get("adjacencias", 2.2)) or 2.2

        d_soma = abs(soma - mu_soma) / tol_soma
        d_impares = abs(impares - mu_impares) / tol_impares
        d_faixa = abs(faixa_1_13 - mu_faixa) / tol_faixa
        d_adj = abs(adj - mu_adj) / tol_adj

        # distância composta (L1 normalizada)
        dist = d_soma + d_impares + d_faixa + d_adj

        detalhes = {
            "ativo": True,
            "window": self._dna_anchor_window,
            "seq": {
                "soma": soma,
                "impares": impares,
                "faixa_1_13": faixa_1_13,
                "adjacencias": adj,
            },
            "dna": {
                "soma_media": round(mu_soma, 4),
                "impares_media": round(mu_impares, 4),
                "faixa_1_13_media": round(mu_faixa, 4),
                "adjacencias_media": round(mu_adj, 4),
            },
            "dist_comp": {
                "d_soma": round(d_soma, 4),
                "d_impares": round(d_impares, 4),
                "d_faixa": round(d_faixa, 4),
                "d_adj": round(d_adj, 4),
            },
            "dist_total": round(dist, 4),
            "tols": {
                "soma": tol_soma,
                "impares": tol_impares,
                "faixa_1_13": tol_faixa,
                "adjacencias": tol_adj,
            },
        }
        return float(dist), detalhes

    # =========================
    # Avaliação (PONTO C: abstrato + âncora)
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

        # overrides de constraints
        if constraints_override:
            for key, val in constraints_override.items():
                cfg[key] = val

        pesos = dict(cfg.get("pesos", {}))
        if pesos_override:
            for key, val in pesos_override.items():
                if key in pesos:
                    pesos[key] = float(val)

        soma = sum(seq_ord)
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        coerencias = 0
        violacoes = 0

        # --- SOMA (abstrato)
        z_max = float(cfg.get("z_max_soma", 1.30))
        if z_soma <= z_max:
            coerencias += 1
            score_soma = 1.0 - (z_soma / max(1e-9, z_max))
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - z_max) / 1.0)

        # --- PARES (abstrato)
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        max_desvio = float(cfg.get("max_desvio_pares", 2))
        if desvio_pares <= max_desvio:
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, max_desvio))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - max_desvio) / 1.0)

        # --- ADJ (abstrato)
        max_adj = float(cfg.get("max_adjacencias", 8))
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, max_adj))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        # Score abstrato ponderado
        score_abstrato = (
            float(pesos.get("soma", 1.0)) * score_soma +
            float(pesos.get("pares", 1.0)) * score_pares +
            float(pesos.get("adj", 1.0)) * score_adj
        )

        # --- ÂNCORA DNA (janela 12)
        dna_dist, dna_det = self._dna_distance(seq_ord, cfg)
        dna_dist_max = float(cfg.get("dna_dist_max", 1.6))
        dna_peso = float(cfg.get("dna_peso", 1.2))

        if dna_det.get("ativo"):
            # dist pequena => score positivo; dist grande => score negativo
            # normaliza pelo "dna_dist_max"
            if dna_dist <= dna_dist_max:
                coerencias += 1
                score_dna = 1.0 - (dna_dist / max(1e-9, dna_dist_max))
            else:
                violacoes += 1
                score_dna = -min(2.0, (dna_dist - dna_dist_max) / 1.0)
        else:
            score_dna = 0.0

        score_total = score_abstrato + (dna_peso * score_dna)

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
                "score_abstrato": round(score_abstrato, 6),
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "score_dna": round(score_dna, 6),
                "dna_peso": dna_peso,
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": z_max,
                    "max_adjacencias": max_adj,
                    "max_desvio_pares": max_desvio,
                    "dna_dist_max": dna_dist_max,
                    "dna_window": self._dna_anchor_window,
                },
                "dna_anchor": dna_det,
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
                "dna_anchor_window": self._dna_anchor_window,
                "dna_anchor_ativo": bool(self._get_dna_window()),
            }

        return resp
