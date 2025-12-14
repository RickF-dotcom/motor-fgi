
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

    Objetivo desta versão:
    - manter contrato e payloads estáveis
    - aceitar overrides sem quebrar
    - suportar âncora DNA fractal (ex: DNA(12))
    - tornar ADJACÊNCIAS uma métrica CONTÍNUA ancorada no DNA (melhora discriminação)
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

        # Regras/limites por regime (mantém compatibilidade)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,     # fallback (se não houver DNA anchor)
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,     # fallback (se não houver DNA anchor)
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45},
            },
        }

        # DNA anchor (injetado pelo app)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False

        # params derivados do DNA (quando ativo)
        self._adj_target: Optional[float] = None
        self._adj_sigma: Optional[float] = None
        self._adj_window: Optional[int] = None

    # ------------------------------------------------------------
    # Contrato: app.py chama set_dna_anchor(...)
    # ------------------------------------------------------------

    def set_dna_anchor(self, *args: Any, **kwargs: Any) -> None:
        """
        Aceita chamadas antigas e novas sem quebrar.

        Suporta:
          - set_dna_anchor(dna_anchor_dict)
          - set_dna_anchor(dna_last25=<dna_dict>, window=<int>)
        """
        dna_anchor = None
        window = None

        if args:
            # modo antigo: set_dna_anchor(dna_anchor_dict)
            dna_anchor = args[0] if len(args) >= 1 else None

        if "dna_anchor" in kwargs:
            dna_anchor = kwargs.get("dna_anchor")

        if "dna_last25" in kwargs:
            dna_anchor = kwargs.get("dna_last25")

        if "window" in kwargs:
            window = kwargs.get("window")

        if "dna_anchor_window" in kwargs:
            window = kwargs.get("dna_anchor_window")

        self._dna_anchor = dna_anchor or None
        self._dna_anchor_active = bool(self._dna_anchor)

        # deriva target/sigma de adjacências a partir do DNA de janelas
        self._adj_target = None
        self._adj_sigma = None
        self._adj_window = None

        if self._dna_anchor_active and isinstance(self._dna_anchor, dict):
            try:
                # tenta puxar do formato: dna["janelas"][str(window)]["adjacencias_media"]
                janelas = self._dna_anchor.get("janelas") or {}
                if window is None:
                    window = 12  # default prático
                wkey = str(int(window))

                bloco = janelas.get(wkey) or {}
                adj_target = bloco.get("adjacencias_media", None)

                if adj_target is not None:
                    adj_target = float(adj_target)

                    # sigma heurístico: precisa dar gradiente, não pode ser “duro”
                    # Regra: pelo menos 1.5, e proporcional ao alvo.
                    adj_sigma = max(1.5, adj_target * 0.25)

                    self._adj_target = adj_target
                    self._adj_sigma = adj_sigma
                    self._adj_window = int(window)
            except Exception:
                # se falhar, só não ativa os parâmetros derivados
                self._adj_target = None
                self._adj_sigma = None
                self._adj_window = None

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "payload": self._dna_anchor if self._dna_anchor_active else None,
            "adj_target": self._adj_target,
            "adj_sigma": self._adj_sigma,
            "adj_window": self._adj_window,
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
        s = sorted(int(x) for x in seq)
        adj = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                adj += 1
        return adj

    # ------------------------------------------------------------
    # Componentes de score
    # ------------------------------------------------------------

    def _score_adj_anchor(self, adj: int) -> Tuple[float, bool]:
        """
        Score contínuo de adjacências ancorado no DNA:
          score_adj = 1 - |adj - target|/sigma   (clamp [-1, 1])
        coerente = True se distância <= 2*sigma (janela ampla)
        """
        if self._adj_target is None or self._adj_sigma is None:
            return 0.0, True

        target = float(self._adj_target)
        sigma = float(self._adj_sigma)

        dist = abs(float(adj) - target)
        raw = 1.0 - (dist / max(1e-9, sigma))
        score = max(-1.0, min(1.0, raw))

        coerente = dist <= (2.0 * sigma)
        return float(score), bool(coerente)

    # ------------------------------------------------------------
    # Avaliação — compatível com JSONs anteriores, mas com adj melhor
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
        max_adj_fallback = int(base["max_adjacencias"])
        max_desvio_pares = int(base["max_desvio_pares"])

        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj_fallback = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])

        # pesos efetivos
        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        seq_int = [int(x) for x in seq]
        soma = int(sum(seq_int))
        pares = sum(1 for x in seq_int if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_int)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # -------------------
        # score_soma
        # -------------------
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente  # 0 se ok, negativo se estourou

        # -------------------
        # score_pares
        # -------------------
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        # -------------------
        # score_adj
        # - se houver DNA anchor ativo: score contínuo por distância ao alvo
        # - senão: fallback antigo (0 ou -1)
        # -------------------
        score_adj = 0.0
        adj_coerente = True

        if self._dna_anchor_active and self._adj_target is not None and self._adj_sigma is not None:
            score_adj, adj_coerente = self._score_adj_anchor(adj)
        else:
            score_adj = 0.0 if adj <= max_adj_fallback else -1.0
            adj_coerente = (adj <= max_adj_fallback)

        # -------------------
        # score_total
        # -------------------
        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        # -------------------
        # violações / coerências
        # -------------------
        viol = 0

        soma_ok = abs(z_soma) <= z_max_soma
        pares_ok = (max_desvio_pares <= 0) or (desvio_pares <= max_desvio_pares)

        if not soma_ok:
            viol += 1
        if not pares_ok:
            viol += 1
        if not adj_coerente:
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
                    "max_adjacencias": max_adj_fallback,
                    "max_desvio_pares": max_desvio_pares,
                    "adj_target": self._adj_target,
                    "adj_sigma": self._adj_sigma,
                    "adj_window": self._adj_window,
                },
            },
            "dna_anchor": self.get_dna_anchor(),
        }

        return Prototipo(
            sequencia=sorted(seq_int),
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
