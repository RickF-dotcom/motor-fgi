
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
    Motor de avaliação / geração de protótipos.

    Meta do laboratório neste ponto:
    - NÃO mexer no núcleo matemático do score (já validado nos JSONs)
    - ADICIONAR uma camada "gate" (filtro) ANTES do score, usando a âncora fractal DNA(window)

    Contrato-resiliente:
    - aceita parâmetros extras sem quebrar (**kwargs)
    - expõe set_dna_anchor() com múltiplas assinaturas (app antigo/novo)
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

        # DNA anchor (fractal gate)
        self._dna_anchor_payload: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False
        self._dna_anchor_window: int = 12
        self._dna_anchor_target: Optional[Dict[str, float]] = None  # métricas alvo

        # tolerâncias default do gate (conservadoras)
        self._gate_defaults: Dict[str, float] = {
            "soma_tol": 12.0,
            "impares_tol": 2.0,
            "faixa_1_13_tol": 2.0,
            "adjacencias_tol": 3.0,
        }

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py (múltiplas assinaturas)
    # ------------------------------------------------------------

    def set_dna_anchor(self, *args: Any, **kwargs: Any) -> None:
        """
        Aceita chamadas em diferentes estilos para NÃO quebrar:

        1) set_dna_anchor(dna_anchor_dict)
        2) set_dna_anchor(dna_last25=<dict>, window=12)
        3) set_dna_anchor(dna_anchor=<dict>, window=12)
        """
        dna_anchor: Optional[Dict[str, Any]] = None
        window: Optional[int] = None

        # estilo posicional: set_dna_anchor(dna_anchor_dict)
        if args and isinstance(args[0], dict):
            dna_anchor = args[0]

        # estilo keyword
        if "dna_anchor" in kwargs and isinstance(kwargs["dna_anchor"], dict):
            dna_anchor = kwargs["dna_anchor"]

        if "dna_last25" in kwargs and isinstance(kwargs["dna_last25"], dict):
            # app manda dna_last25; internamente é o payload de DNA completo
            dna_anchor = kwargs["dna_last25"]

        if "window" in kwargs:
            try:
                window = int(kwargs["window"])
            except Exception:
                window = None

        if "dna_anchor_window" in kwargs:
            try:
                window = int(kwargs["dna_anchor_window"])
            except Exception:
                window = None

        if window is None:
            window = self._dna_anchor_window

        self._dna_anchor_payload = dna_anchor or None
        self._dna_anchor_window = int(window)
        self._dna_anchor_active = bool(dna_anchor)

        # prepara o alvo do gate (extraído do dna["janelas"][window])
        self._dna_anchor_target = self._extract_anchor_target(self._dna_anchor_payload, self._dna_anchor_window)

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "window": self._dna_anchor_window if self._dna_anchor_active else None,
            "target": self._dna_anchor_target if self._dna_anchor_active else None,
        }

    def _extract_anchor_target(self, dna: Optional[Dict[str, Any]], window: int) -> Optional[Dict[str, float]]:
        """
        Espera formato do seu RegimeDetector:
        dna = {"origem": "...", "janelas": {"7": {...}, "12": {...}, ...}}
        """
        if not dna or not isinstance(dna, dict):
            return None

        janelas = dna.get("janelas")
        if not isinstance(janelas, dict):
            return None

        w_key = str(int(window))
        bloco = janelas.get(w_key)

        # tolera janelas com chave int
        if bloco is None:
            bloco = janelas.get(int(window))  # type: ignore[index]

        if not isinstance(bloco, dict):
            return None

        # alvos mínimos que conseguimos medir em qualquer sequência
        # (repeticao_media existe no DNA, mas requer sequência anterior; não usamos no gate aqui)
        target = {
            "soma_media": float(bloco.get("soma_media", 0.0)),
            "impares_media": float(bloco.get("impares_media", 0.0)),
            "faixa_1_13_media": float(bloco.get("faixa_1_13_media", 0.0)),
            "adjacencias_media": float(bloco.get("adjacencias_media", 0.0)),
        }
        return target

    # ------------------------------------------------------------
    # Helpers matemáticos
    # ------------------------------------------------------------

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Soma de k números amostrados sem reposição de {1..N}:
        mu = k*(N+1)/2
        var(sum) = k * var_pop * ((N-k)/(N-1))
        var_pop(1..N) = (N^2 - 1)/12
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

    def _count_adjacencias(self, seq: List[int]) -> int:
        if not seq:
            return 0
        s = sorted(seq)
        adj = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                adj += 1
        return adj

    def _count_faixa_1_13(self, seq: List[int]) -> int:
        return sum(1 for x in seq if 1 <= int(x) <= 13)

    # ------------------------------------------------------------
    # Gate (âncora DNA(window)) — filtro antes do score
    # ------------------------------------------------------------

    def _passa_gate_ancora(
        self,
        seq: List[int],
        gate_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Retorna (passa?, debug).

        Gate usa alvo do DNA(window) com tolerâncias.
        Se âncora não estiver ativa/extraível, passa tudo (gate desligado).
        """
        if not self._dna_anchor_active or not self._dna_anchor_target:
            return True, {"ativo": False}

        tol = dict(self._gate_defaults)
        if gate_override:
            # permite override fino via constraints_override["gate"] ou direto
            for k, v in gate_override.items():
                if k in tol:
                    try:
                        tol[k] = float(v)
                    except Exception:
                        pass

        soma = float(sum(seq))
        impares = float(sum(1 for x in seq if int(x) % 2 != 0))
        faixa_1_13 = float(self._count_faixa_1_13(seq))
        adj = float(self._count_adjacencias(seq))

        tgt = self._dna_anchor_target

        def ok(valor: float, alvo: float, t: float) -> bool:
            return abs(valor - alvo) <= t

        passa = (
            ok(soma, tgt["soma_media"], tol["soma_tol"]) and
            ok(impares, tgt["impares_media"], tol["impares_tol"]) and
            ok(faixa_1_13, tgt["faixa_1_13_media"], tol["faixa_1_13_tol"]) and
            ok(adj, tgt["adjacencias_media"], tol["adjacencias_tol"])
        )

        debug = {
            "ativo": True,
            "window": self._dna_anchor_window,
            "target": tgt,
            "tolerancias": tol,
            "valor": {
                "soma": soma,
                "impares": impares,
                "faixa_1_13": faixa_1_13,
                "adjacencias": adj,
            },
            "passa": bool(passa),
        }
        return bool(passa), debug

    # ------------------------------------------------------------
    # Avaliação (score + violações) — MANTIDA a lógica compatível
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

        seq_ord = sorted(int(x) for x in seq)
        soma = int(sum(seq_ord))
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # componentes (compat com seus JSONs)
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
                },
            },
            "dna_anchor": self.get_dna_anchor(),
        }

        return Prototipo(
            sequencia=seq_ord,
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
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """
        Pipeline correto (sem mexer no núcleo):
        1) Pega candidatos do Grupo de Milhões
        2) (NOVO) Aplica gate de âncora DNA(window) se estiver ativo
        3) Score (mesma matemática)
        4) Ordena e devolve top_n
        """
        k = int(k)
        top_n = int(top_n)
        max_candidatos = int(max_candidatos)

        # Gate override opcional vindo em constraints_override["gate"]
        gate_override: Optional[Dict[str, Any]] = None
        if constraints_override and isinstance(constraints_override, dict):
            if isinstance(constraints_override.get("gate"), dict):
                gate_override = constraints_override["gate"]

        # Importante: buscar mais candidatos quando gate estiver ativo,
        # senão você filtra demais e fica com lixo/monotonia.
        fetch_mult = 5 if self._dna_anchor_active else 1
        fetch_n = min(max(2000, max_candidatos * fetch_mult), 50000)

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=fetch_n)

        prototipos: List[Prototipo] = []
        gate_stats = {"ativo": bool(self._dna_anchor_active), "passaram": 0, "avaliados": 0, "fetch_n": fetch_n}

        for seq in candidatos:
            gate_ok, gate_dbg = self._passa_gate_ancora(list(seq), gate_override=gate_override)
            gate_stats["avaliados"] += 1

            if not gate_ok:
                continue

            gate_stats["passaram"] += 1
            p = self.avaliar_sequencia(
                list(seq),
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override or {},
                constraints_override=constraints_override or {},
            )
            # injeta debug do gate no protótipo (sem quebrar o formato)
            p.detalhes["gate_anchor"] = gate_dbg
            prototipos.append(p)

            # já para quando tiver material suficiente para ordenar
            if len(prototipos) >= max(200, max_candidatos):
                break

        prototipos.sort(key=lambda x: x.score_total, reverse=True)
        prototipos = prototipos[: max(1, top_n)]

        contexto_lab = None
        if incluir_contexto_dna and self._dna_anchor_active and self._dna_anchor_payload:
            contexto_lab = self._dna_anchor_payload

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
                "gate_defaults": self._gate_defaults,
            },
            "debug_gate": gate_stats,
            "contexto_lab": contexto_lab,
            }
