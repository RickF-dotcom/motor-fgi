
# fgi_engine.py
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
    Motor de avaliação / geração de protótipos (LHE/LHS).

    Objetivos desta versão:
    - NÃO quebrar contrato com app.py / swagger (aceita kwargs extras).
    - Evolução cirúrgica: ADJACÊNCIAS viram métrica de CONVERGÊNCIA ao DNA(âncora),
      e não apenas punição genérica.
    - Manter hard-constraints para impedir sequência degenerada.
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
                "max_desvio_pares": 2,
                "max_adjacencias": 3,          # regra antiga (sem âncora)
                "hard_max_adjacencias": 13,    # regra dura (com âncora)
                "adj_tolerancia": 2.5,         # tolerância padrão para convergência ao alvo
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_desvio_pares": 3,
                "max_adjacencias": 4,
                "hard_max_adjacencias": 13,
                "adj_tolerancia": 3.0,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
            },
        }

        # DNA anchor (injetado pelo app)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False
        self._dna_anchor_window: Optional[int] = None  # ex.: 12

    # ------------------------------------------------------------
    # Contrato resiliente: set_dna_anchor
    # ------------------------------------------------------------

    def set_dna_anchor(self, dna_anchor: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Formatos aceitos:
        1) set_dna_anchor(dna_anchor=<dict>)
        2) set_dna_anchor(dna_last25=<dict>, window=<int>)   (compat com seu app.py)
        """
        # Formato 2 (compat):
        if dna_anchor is None and ("dna_last25" in kwargs or "window" in kwargs):
            dna_last25 = kwargs.get("dna_last25")
            window = kwargs.get("window")

            if isinstance(window, (int, float, str)):
                try:
                    window = int(window)
                except Exception:
                    window = None

            if isinstance(dna_last25, dict):
                self._dna_anchor = dna_last25
                self._dna_anchor_window = window
                self._dna_anchor_active = True
                return

            self._dna_anchor = None
            self._dna_anchor_window = None
            self._dna_anchor_active = False
            return

        # Formato 1:
        self._dna_anchor = dna_anchor if isinstance(dna_anchor, dict) else None
        self._dna_anchor_active = bool(self._dna_anchor)

        w = kwargs.get("window", None)
        if isinstance(w, (int, float, str)):
            try:
                self._dna_anchor_window = int(w)
            except Exception:
                self._dna_anchor_window = None
        else:
            self._dna_anchor_window = None

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "window": self._dna_anchor_window,
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

    def _extract_adj_target_from_anchor(self) -> Optional[float]:
        """
        Tenta extrair adjacencias_media do DNA(anchor) na janela desejada.
        Esperado:
          dna = {"origem": "...", "janelas": {"12": {"adjacencias_media": ...}, ...}}
        """
        if not self._dna_anchor_active or not isinstance(self._dna_anchor, dict):
            return None

        janelas = self._dna_anchor.get("janelas")
        if not isinstance(janelas, dict) or not janelas:
            return None

        # janela preferida
        if isinstance(self._dna_anchor_window, int) and str(self._dna_anchor_window) in janelas:
            wdata = janelas.get(str(self._dna_anchor_window), {})
            if isinstance(wdata, dict) and "adjacencias_media" in wdata:
                try:
                    return float(wdata["adjacencias_media"])
                except Exception:
                    pass

        # fallback: tenta 12, depois 10, depois 25
        for fallback in ("12", "10", "25"):
            if fallback in janelas:
                wdata = janelas.get(fallback, {})
                if isinstance(wdata, dict) and "adjacencias_media" in wdata:
                    try:
                        return float(wdata["adjacencias_media"])
                    except Exception:
                        continue

        return None

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
        max_desvio_pares = int(base["max_desvio_pares"])
        max_adj_classico = int(base["max_adjacencias"])
        hard_max_adj = int(base.get("hard_max_adjacencias", max_adj_classico))
        adj_tol = float(base.get("adj_tolerancia", 2.5))

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])
            if "max_adjacencias" in constraints_override:
                max_adj_classico = int(constraints_override["max_adjacencias"])
            if "hard_max_adjacencias" in constraints_override:
                hard_max_adj = int(constraints_override["hard_max_adjacencias"])
            if "adj_tolerancia" in constraints_override:
                adj_tol = float(constraints_override["adj_tolerancia"])

        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        seq = [int(x) for x in seq]
        soma = int(sum(seq))
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # score_soma: 0 se ok, negativo se exceder
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        # score_pares: 0..1
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        # score_adj: com âncora -> convergir ao alvo do DNA(window); sem âncora -> regra antiga
        adj_target = self._extract_adj_target_from_anchor()

        if self._dna_anchor_active and adj_target is not None and adj_tol > 0:
            dist = abs(adj - float(adj_target))
            score_adj = 1.0 - (dist / float(adj_tol))
            if score_adj < -1.0:
                score_adj = -1.0
            if score_adj > 1.0:
                score_adj = 1.0
            violou_adj = adj > min(k - 1, hard_max_adj)
        else:
            score_adj = 0.0 if adj <= max_adj_classico else -1.0
            violou_adj = adj > max_adj_classico

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        # violações
        viol = 0
        if abs(z_soma) > z_max_soma:
            viol += 1
        if violou_adj:
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
                    "max_desvio_pares": max_desvio_pares,
                    "max_adjacencias": max_adj_classico,
                    "hard_max_adjacencias": hard_max_adj,
                    "adj_target": round(float(adj_target), 4) if adj_target is not None else None,
                    "adj_tolerancia": adj_tol if (adj_target is not None) else None,
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
        Payload compatível com Swagger:
        - prototipos
        - regime_usado
        - max_candidatos_usado
        - overrides_usados
        - contexto_lab (quando incluir_contexto_dna=True e âncora ativa)

        windows/pesos_windows/pesos_metricas aceitos (não usados aqui) para não quebrar contrato.
        """
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

        prototipos.sort(key=lambda x: (x.score_total, -x.violacoes, x.coerencias), reverse=True)
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
                "top_n": top_n,
            },
            "contexto_lab": contexto_lab,
        }
