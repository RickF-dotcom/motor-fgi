
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
    Motor de avaliação / geração de protótipos.

    Metas deste arquivo:
    - Contrato resiliente (não quebrar o app.py / Swagger)
    - Rank "de verdade" (score_adj NÃO binário)
    - Hard filter como teto de pool (não como score)
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

        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

    # -----------------------------
    # Contrato resiliente
    # -----------------------------
    def set_dna_anchor(self, dna_anchor: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Aceita:
          - set_dna_anchor({...})
          - set_dna_anchor(dna_last25=..., window=12, regime_atual=...)
        """
        if dna_anchor is None and kwargs:
            dna_anchor = dict(kwargs)
        self._dna_anchor = dna_anchor or None
        self._dna_anchor_active = bool(dna_anchor)

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {"ativo": self._dna_anchor_active, "payload": self._dna_anchor if self._dna_anchor_active else None}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
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

    def _anchor_window(self) -> Optional[int]:
        if not self._dna_anchor_active or not isinstance(self._dna_anchor, dict):
            return None
        w = self._dna_anchor.get("window")
        try:
            return int(w) if w is not None else None
        except Exception:
            return None

    def _anchor_janela(self, w: int) -> Optional[Dict[str, Any]]:
        if not self._dna_anchor_active or not isinstance(self._dna_anchor, dict):
            return None
        dna = self._dna_anchor.get("dna_last25")
        if not isinstance(dna, dict):
            return None
        janelas = dna.get("janelas")
        if not isinstance(janelas, dict):
            return None
        j = janelas.get(str(w)) or janelas.get(w)
        return j if isinstance(j, dict) else None

    def _baseline_stds(self) -> Optional[Dict[str, float]]:
        """
        Tenta aproveitar stds do regime_atual.baseline.stds quando existir.
        """
        if not self._dna_anchor_active or not isinstance(self._dna_anchor, dict):
            return None
        reg = self._dna_anchor.get("regime_atual")
        if not isinstance(reg, dict):
            return None
        baseline = reg.get("baseline")
        if not isinstance(baseline, dict):
            return None
        stds = baseline.get("stds")
        if not isinstance(stds, dict):
            return None
        out: Dict[str, float] = {}
        for k in ["soma", "impares", "adjacencias", "repeticao", "faixa_1_13"]:
            if k in stds:
                try:
                    out[k] = float(stds[k])
                except Exception:
                    pass
        return out or None

    # -----------------------------
    # Avaliação (score + violações)
    # -----------------------------
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

        # extras (hard ceiling do pool + flag)
        hard_max_adj = None
        adj_hard = False

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])

            if "hard_max_adjacencias" in constraints_override:
                hard_max_adj = int(constraints_override["hard_max_adjacencias"])
            if "adj_hard" in constraints_override:
                adj_hard = bool(constraints_override["adj_hard"])

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

        # --- metas/escala: tenta usar âncora/baseline quando existir
        mu_soma_theory, sd_soma_theory = self._sum_stats_sem_reposicao(k)

        w = self._anchor_window()
        janela = self._anchor_janela(w) if w else None
        stds = self._baseline_stds()

        # soma: alvo = soma_media da janela (se existir), sd = baseline.soma (se existir) senão teoria
        mu_soma = float(janela["soma_media"]) if janela and "soma_media" in janela else float(mu_soma_theory)
        sd_soma = float(stds["soma"]) if stds and "soma" in stds else float(sd_soma_theory)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # pares: alvo = k - impares_media (se existir), escala = baseline.impares (se existir) senão max_desvio
        pares_target = None
        if janela and "impares_media" in janela:
            try:
                pares_target = float(k) - float(janela["impares_media"])
            except Exception:
                pares_target = None
        if pares_target is None:
            pares_target = k / 2.0

        # desvio_pares "normalizado" pra virar score contínuo
        desvio_pares = abs(float(pares) - float(pares_target))
        pares_scale = float(stds["impares"]) if stds and "impares" in stds else float(max(1, max_desvio_pares))
        z_pares = 0.0 if pares_scale == 0 else desvio_pares / pares_scale

        # adj: alvo = adjacencias_media da janela (se existir), escala = baseline.adjacencias (se existir) senão 2.0
        adj_target = float(janela["adjacencias_media"]) if janela and "adjacencias_media" in janela else float(max_adj)
        adj_scale = float(stds["adjacencias"]) if stds and "adjacencias" in stds else 2.0
        z_adj = 0.0 if adj_scale == 0 else abs(float(adj) - float(adj_target)) / adj_scale

        # -----------------------------
        # SCORES (contínuos)
        # -----------------------------

        # soma: dentro do z_max_soma = 0, fora penaliza excedente (negativo)
        excedente_soma = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente_soma

        # pares: 1.0 perfeito, cai linear até 0 quando passa max_desvio_pares
        # (max_desvio_pares continua sendo o "limite operacional")
        score_pares = 1.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        # adj: aqui está o conserto do ranking
        # - se adj <= max_adj: score_adj = +1.0 (bom)
        # - se adj > max_adj: degrada até -1.0 quando chega no hard_max_adj (se existir) ou em max_adj+6
        if adj <= max_adj:
            score_adj = 1.0
        else:
            ceiling = hard_max_adj if hard_max_adj is not None else (max_adj + 6)
            ceiling = max(ceiling, max_adj + 1)
            frac = (adj - max_adj) / float(ceiling - max_adj)
            score_adj = -max(0.0, min(1.0, frac))

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        # -----------------------------
        # VIOLAÇÕES
        # -----------------------------
        viol = 0

        # z_soma violação
        if abs(z_soma) > z_max_soma:
            viol += 1

        # pares violação operacional
        if max_desvio_pares > 0 and desvio_pares > max_desvio_pares:
            viol += 1

        # adj violação:
        # - se adj_hard: viola quando passa hard_max_adj
        # - se não: viola quando passa max_adj (mas score ainda diferencia o quanto)
        if adj_hard and hard_max_adj is not None:
            if adj > hard_max_adj:
                viol += 1
        else:
            if adj > max_adj:
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
                    "adj_hard": bool(adj_hard),
                    "hard_max_adjacencias": hard_max_adj,
                },
                "anchors": {
                    "window": w,
                    "soma_target": round(mu_soma, 4),
                    "pares_target": round(float(pares_target), 4),
                    "adj_target": round(float(adj_target), 4),
                    "z_adj_norm": round(float(z_adj), 4),
                    "z_pares_norm": round(float(z_pares), 4),
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

    # -----------------------------
    # Geração
    # -----------------------------
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

        constraints_override = constraints_override or {}
        pesos_override = pesos_override or {}

        # hard ceiling do pool
        hard_max_adj = constraints_override.get("hard_max_adjacencias", None)
        hard_max_adj = int(hard_max_adj) if hard_max_adj is not None else None

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        prototipos: List[Prototipo] = []
        skip_hard = 0

        for seq in candidatos:
            seq_list = list(seq)
            if hard_max_adj is not None:
                adj = self._count_adjacencias(seq_list)
                if adj > hard_max_adj:
                    skip_hard += 1
                    continue

            p = self.avaliar_sequencia(
                seq_list,
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
            )
            prototipos.append(p)

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[: max(1, top_n)] if prototipos else []

        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active:
            contexto_lab = self._dna_anchor

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
                "pesos_override": pesos_override,
                "constraints_override": constraints_override,
                "windows": windows or None,
                "pesos_windows": pesos_windows or None,
                "pesos_metricas": pesos_metricas or None,
            },
            "contexto_lab": contexto_lab,
            "diagnostico": {
                "candidatos_total": len(candidatos),
                "candidatos_filtrados_hard": int(skip_hard),
                "candidatos_avaliados": len(prototipos) if prototipos else 0,
                "top_n": top_n,
            },
    }
