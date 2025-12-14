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

    Objetivos:
    1) Manter compatibilidade com o payload atual (Swagger/Render)
    2) Ser resiliente a novos parâmetros (aceita **kwargs)
    3) Adicionar camada discriminante: score_fractal (DNA anchor)
       sem quebrar o score estrutural existente.
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

        # Regras/limites por regime (compat com JSONs anteriores)
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

        # DNA anchor (injetado pelo app antes do /prototipos)
        self._dna_anchor_active: bool = False
        self._dna_anchor_window: Optional[int] = None

        # alvos (médias) para a janela escolhida (ex: DNA(12))
        self._anchor_targets: Dict[str, float] = {}

        # desvios (std) usados para normalização do score_fractal
        # (tentamos puxar do baseline do regime_detector; se não vier, usamos fallback)
        self._anchor_stds: Dict[str, float] = {
            "soma": 18.25,
            "impares": 1.37,
            "faixa_1_13": 1.33,
            "adjacencias": 2.01,
            "repeticao": 1.70,
        }

    # ------------------------------------------------------------
    # Contrato para o app.py (aceita assinaturas diferentes)
    # ------------------------------------------------------------

    def set_dna_anchor(self, *args: Any, **kwargs: Any) -> None:
        """
        Aceita 2 formatos (sem quebrar):

        1) set_dna_anchor(dna_anchor: dict)  [formato antigo]
        2) set_dna_anchor(dna_last25=<dict>, window=<int>)  [formato do seu app.py atual]

        O anchor usado aqui é um "alvo" de médias do DNA para uma janela (ex: 12).
        """
        dna_anchor = None
        window = None

        # Formato 1 (posicional)
        if args and isinstance(args[0], dict):
            dna_anchor = args[0]

        # Formato 2 (kwargs)
        if "dna_last25" in kwargs and isinstance(kwargs["dna_last25"], dict):
            dna_anchor = kwargs["dna_last25"]
        if "window" in kwargs:
            try:
                window = int(kwargs["window"])
            except Exception:
                window = None

        # fallback: se veio "dna_anchor" por kwargs
        if dna_anchor is None and "dna_anchor" in kwargs and isinstance(kwargs["dna_anchor"], dict):
            dna_anchor = kwargs["dna_anchor"]

        # Se não veio nada, desativa
        if not dna_anchor:
            self._dna_anchor_active = False
            self._dna_anchor_window = None
            self._anchor_targets = {}
            return

        # Tentamos localizar o dicionário de janelas em formatos possíveis:
        # - {"janelas": {...}}
        # - {"dna": {"janelas": {...}}}
        janelas = None
        if isinstance(dna_anchor.get("janelas"), dict):
            janelas = dna_anchor["janelas"]
        elif isinstance(dna_anchor.get("dna"), dict) and isinstance(dna_anchor["dna"].get("janelas"), dict):
            janelas = dna_anchor["dna"]["janelas"]

        # Se não tem janelas, ativa mas sem targets (não quebra)
        if not isinstance(janelas, dict):
            self._dna_anchor_active = True
            self._dna_anchor_window = window
            self._anchor_targets = {}
            return

        # Se window não foi passado, escolhe 12 como padrão (boa âncora)
        if window is None:
            window = 12

        win_key = str(window)
        alvo = janelas.get(win_key)

        # Se não existir exatamente, tenta fallback: 12 -> 10 -> 15 -> 25
        if not isinstance(alvo, dict):
            for alt in ("12", "10", "15", "25", "7", "20"):
                if isinstance(janelas.get(alt), dict):
                    alvo = janelas[alt]
                    window = int(alt)
                    break

        if not isinstance(alvo, dict):
            self._dna_anchor_active = True
            self._dna_anchor_window = window
            self._anchor_targets = {}
            return

        # Targets (médias) — nomes batendo com seus JSONs
        # Alguns JSONs usam "adjacencias_media" e outros "adjacencias_media" (sem acento).
        self._anchor_targets = {
            "soma": float(alvo.get("soma_media", 0.0) or 0.0),
            "impares": float(alvo.get("impares_media", 0.0) or 0.0),
            "faixa_1_13": float(alvo.get("faixa_1_13_media", 0.0) or 0.0),
            "adjacencias": float(alvo.get("adjacencias_media", 0.0) or 0.0),
            "repeticao": float(alvo.get("repeticao_media", 0.0) or 0.0),
        }

        # Se o dna_anchor também trouxer baseline stds (do regime_detector), usamos
        # Possível formato:
        # dna_anchor["regime_atual"]["baseline"]["stds"][...]
        try:
            reg = dna_anchor.get("regime_atual") or dna_anchor.get("baseline") or {}
            if isinstance(reg, dict) and isinstance(reg.get("baseline"), dict):
                b = reg["baseline"]
            else:
                b = reg
            stds = None
            if isinstance(b, dict) and isinstance(b.get("stds"), dict):
                stds = b["stds"]
            if isinstance(stds, dict):
                for k in self._anchor_stds.keys():
                    if k in stds and stds[k] is not None:
                        self._anchor_stds[k] = float(stds[k])
        except Exception:
            pass

        self._dna_anchor_active = True
        self._dna_anchor_window = window

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "window": self._dna_anchor_window,
            "targets": self._anchor_targets if self._dna_anchor_active else None,
            "stds": self._anchor_stds if self._dna_anchor_active else None,
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

    def _count_faixa_1_13(self, seq: List[int]) -> int:
        return sum(1 for x in seq if 1 <= x <= 13)

    # ------------------------------------------------------------
    # Score fractal (camada discriminante)
    # ------------------------------------------------------------

    def _score_fractal(
        self,
        seq: List[int],
        k: int,
        pesos_metricas: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Mede "proximidade" da sequência ao DNA-anchor (janela escolhida).
        Retorna (score_fractal, detalhes_fractal)

        Estratégia:
        - calcula métricas da seq
        - calcula z-diffs (diferença / std)
        - agrega por pesos
        - score = 1 - dist (clamp)  -> quanto menor a distância, maior o score
        """
        if not (self._dna_anchor_active and self._anchor_targets):
            return 0.0, {
                "ativo": False,
                "motivo": "anchor_inativo_ou_sem_targets",
            }

        seq_sorted = sorted(seq)
        soma = float(sum(seq_sorted))
        impares = float(sum(1 for x in seq_sorted if x % 2 != 0))
        faixa = float(self._count_faixa_1_13(seq_sorted))
        adj = float(self._count_adjacencias(seq_sorted))

        # repetição: sem acesso ao “último concurso” aqui, não dá pra medir de forma correta.
        # então não usamos repeticao como distância (peso 0 por padrão).
        repeticao = 0.0

        # pesos default (foco no que a gente mede bem agora)
        w = {
            "soma": 0.40,
            "impares": 0.25,
            "faixa_1_13": 0.20,
            "adjacencias": 0.15,
            "repeticao": 0.00,
        }
        if pesos_metricas:
            for kk, vv in pesos_metricas.items():
                if kk in w:
                    w[kk] = float(vv)

        # normaliza pesos (evita soma != 1)
        wsum = sum(max(0.0, v) for v in w.values())
        if wsum <= 0:
            wsum = 1.0
        w = {kk: max(0.0, vv) / wsum for kk, vv in w.items()}

        def zdiff(val: float, target: float, std: float) -> float:
            s = float(std) if std and std > 1e-9 else 1.0
            return abs(val - float(target)) / s

        z_soma = zdiff(soma, self._anchor_targets["soma"], self._anchor_stds["soma"])
        z_imp = zdiff(impares, self._anchor_targets["impares"], self._anchor_stds["impares"])
        z_faixa = zdiff(faixa, self._anchor_targets["faixa_1_13"], self._anchor_stds["faixa_1_13"])
        z_adj = zdiff(adj, self._anchor_targets["adjacencias"], self._anchor_stds["adjacencias"])
        z_rep = 0.0  # não avaliamos repetição por enquanto

        dist = (
            w["soma"] * z_soma
            + w["impares"] * z_imp
            + w["faixa_1_13"] * z_faixa
            + w["adjacencias"] * z_adj
            + w["repeticao"] * z_rep
        )

        # score_fractal: 1 - dist, clamp para não explodir
        score = 1.0 - dist
        score = max(-2.0, min(1.0, score))

        detalhes = {
            "ativo": True,
            "window": self._dna_anchor_window,
            "pesos_metricas": w,
            "targets": self._anchor_targets,
            "stds": self._anchor_stds,
            "seq_metricas": {
                "soma": soma,
                "impares": impares,
                "faixa_1_13": faixa,
                "adjacencias": adj,
                "repeticao": repeticao,
            },
            "z_diffs": {
                "soma": round(z_soma, 6),
                "impares": round(z_imp, 6),
                "faixa_1_13": round(z_faixa, 6),
                "adjacencias": round(z_adj, 6),
                "repeticao": round(z_rep, 6),
            },
            "dist": round(dist, 6),
            "score_fractal": round(score, 6),
        }
        return float(round(score, 6)), detalhes

    # ------------------------------------------------------------
    # Avaliação estrutural (compatível com JSONs anteriores)
    # ------------------------------------------------------------

    def avaliar_sequencia(
        self,
        seq: List[int],
        k: int,
        regime_id: str = "estavel",
        pesos_override: Optional[Dict[str, float]] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
        pesos_metricas: Optional[Dict[str, float]] = None,
        **_ignored: Any,
    ) -> Prototipo:
        k = int(k)
        regime_id = (regime_id or self.regime_padrao).strip().lower()
        base = self._regimes.get(regime_id, self._regimes[self.regime_padrao])

        # constraints efetivos
        z_max_soma = float(base["z_max_soma"])
        max_adj = int(base["max_adjacencias"])
        max_desvio_pares = int(base["max_desvio_pares"])

        # lambda da camada fractal (vem por constraints_override pra não quebrar schema)
        lambda_fractal = 0.45  # default bom pra começar
        if constraints_override:
            if "z_max_soma" in constraints_override:
                z_max_soma = float(constraints_override["z_max_soma"])
            if "max_adjacencias" in constraints_override:
                max_adj = int(constraints_override["max_adjacencias"])
            if "max_desvio_pares" in constraints_override:
                max_desvio_pares = int(constraints_override["max_desvio_pares"])
            if "lambda_fractal" in constraints_override:
                lambda_fractal = float(constraints_override["lambda_fractal"])

        # pesos efetivos (estruturais)
        pesos = dict(base["pesos"])
        if pesos_override:
            for kk, vv in pesos_override.items():
                if kk in pesos:
                    pesos[kk] = float(vv)

        seq_sorted = sorted(int(x) for x in seq)
        soma = int(sum(seq_sorted))
        pares = sum(1 for x in seq_sorted if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_sorted)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # ---- componentes (mantém compat com seus outputs)
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        score_adj = 0.0 if adj <= max_adj else -1.0

        score_estrutural = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
        )

        # violações estruturais
        viol = 0
        if abs(z_soma) > z_max_soma:
            viol += 1
        if adj > max_adj:
            viol += 1
        if max_desvio_pares > 0 and desvio_pares > max_desvio_pares:
            viol += 1

        total_constraints = 3
        coerencias = max(0, total_constraints - viol)

        # camada fractal
        score_fractal, detalhes_fractal = self._score_fractal(
            seq_sorted,
            k=k,
            pesos_metricas=pesos_metricas,
        )

        score_total = score_estrutural + (lambda_fractal * score_fractal)

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
                "score_estrutural": round(score_estrutural, 6),
                "score_fractal": round(score_fractal, 6),
                "lambda_fractal": round(lambda_fractal, 6),
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": z_max_soma,
                    "max_adjacencias": max_adj,
                    "max_desvio_pares": max_desvio_pares,
                    "lambda_fractal": lambda_fractal,
                },
            },
            "fractal": detalhes_fractal,
            "dna_anchor": self.get_dna_anchor(),
        }

        return Prototipo(
            sequencia=seq_sorted,
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
            # o app injeta contexto no próprio response, então aqui só garantimos compat
            contexto_lab = {"dna_anchor": self.get_dna_anchor()}

        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=max_candidatos)

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            p = self.avaliar_sequencia(
                list(seq),
                k=k,
                regime_id=regime_id,
                pesos_override=pesos_override or {},
                constraints_override=constraints_override or {},
                pesos_metricas=pesos_metricas or None,
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
