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
    - Carregar Grupo de Milhões (combinações não sorteadas)
    - Avaliar sequências (score)
    - Gerar protótipos estruturais

    IMPORTANTE:
    - k = tamanho da sequência
    - o endpoint retorna top-k protótipos por compatibilidade
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

        # Regimes base (podem ser sobrescritos via pesos_override e constraints_override)
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.20, "pares": 0.60, "adj": 0.60},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.00, "pares": 0.45, "adj": 0.45},
            },
        }

    # =========================
    # Helpers estatísticos (universo 1..N, sem reposição)
    # =========================

    def _sum_stats_sem_reposicao(self, k: int) -> Tuple[float, float]:
        """
        Média e desvio padrão da SOMA ao amostrar k números de 1..N (sem reposição).

        Pop variance de 1..N: (N^2 - 1)/12
        Var(sum) = k * var_pop * ((N - k)/(N - 1))
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

    def _count_faixa(self, seq_ordenada: List[int], a: int, b: int) -> int:
        return sum(1 for x in seq_ordenada if a <= x <= b)

    # =========================
    # Constraints (hard filter)
    # =========================

    def _constraints_ok(
        self,
        soma: int,
        pares: int,
        adj: int,
        faixa_1_10: int,
        faixa_11_25: int,
        constraints: Optional[Dict[str, Any]],
        regime_cfg: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Constraints são filtros DUROS:
        se falhar -> sequência é descartada (não vira protótipo).
        """
        if not constraints:
            return True, []

        violacoes: List[str] = []

        def _get_int(key: str) -> Optional[int]:
            v = constraints.get(key, None)
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        def _get_float(key: str) -> Optional[float]:
            v = constraints.get(key, None)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        # Soma
        min_soma = _get_int("min_soma")
        max_soma = _get_int("max_soma")
        if min_soma is not None and soma < min_soma:
            violacoes.append("min_soma")
        if max_soma is not None and soma > max_soma:
            violacoes.append("max_soma")

        # Pares
        min_pares = _get_int("min_pares")
        max_pares = _get_int("max_pares")
        if min_pares is not None and pares < min_pares:
            violacoes.append("min_pares")
        if max_pares is not None and pares > max_pares:
            violacoes.append("max_pares")

        # Adjacências (pode sobrescrever o regime)
        max_adj = _get_int("max_adjacencias")
        if max_adj is None:
            max_adj = int(regime_cfg.get("max_adjacencias", 999))
        if adj > max_adj:
            violacoes.append("max_adjacencias")

        # Faixas (as que você já tentou no Swagger)
        min_f1 = _get_int("min_faixa_1_10")
        max_f1 = _get_int("max_faixa_1_10")
        if min_f1 is not None and faixa_1_10 < min_f1:
            violacoes.append("min_faixa_1_10")
        if max_f1 is not None and faixa_1_10 > max_f1:
            violacoes.append("max_faixa_1_10")

        min_f2 = _get_int("min_faixa_11_25")
        max_f2 = _get_int("max_faixa_11_25")
        if min_f2 is not None and faixa_11_25 < min_f2:
            violacoes.append("min_faixa_11_25")
        if max_f2 is not None and faixa_11_25 > max_f2:
            violacoes.append("max_faixa_11_25")

        # z_max_soma (se quiser endurecer/afrouxar a faixa)
        z_max = _get_float("z_max_soma")
        if z_max is not None:
            # só valida formato; a checagem do z em si é no score.
            if z_max <= 0:
                violacoes.append("z_max_soma")

        return (len(violacoes) == 0), violacoes

    # =========================
    # Avaliação (score)
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

        base_cfg = dict(self._regimes.get(regime, self._regimes["estavel"]))

        # constraints_override pode sobrescrever limites do regime (ex.: max_adjacencias, z_max_soma)
        if constraints_override:
            if "max_adjacencias" in constraints_override:
                try:
                    base_cfg["max_adjacencias"] = int(constraints_override["max_adjacencias"])
                except Exception:
                    pass
            if "z_max_soma" in constraints_override:
                try:
                    base_cfg["z_max_soma"] = float(constraints_override["z_max_soma"])
                except Exception:
                    pass
            if "max_desvio_pares" in constraints_override:
                try:
                    base_cfg["max_desvio_pares"] = int(constraints_override["max_desvio_pares"])
                except Exception:
                    pass

        pesos = dict(base_cfg.get("pesos", {"soma": 1.0, "pares": 1.0, "adj": 1.0}))
        if pesos_override:
            # sobrescreve só as chaves que vierem
            for kpeso in ("soma", "pares", "adj"):
                if kpeso in pesos_override:
                    try:
                        pesos[kpeso] = float(pesos_override[kpeso])
                    except Exception:
                        pass

        soma = sum(seq_ord)
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)
        faixa_1_10 = self._count_faixa(seq_ord, 1, 10)
        faixa_11_25 = self._count_faixa(seq_ord, 11, 25)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = abs((soma - mu_soma) / sd_soma) if sd_soma > 0 else 0.0

        coerencias = 0
        violacoes = 0

        # --- SOMA (z-score)
        z_max_soma = float(base_cfg["z_max_soma"])
        if z_soma <= z_max_soma:
            coerencias += 1
            score_soma = 1.0 - (z_soma / z_max_soma)  # 1..0
        else:
            violacoes += 1
            score_soma = -min(1.0, (z_soma - z_max_soma) / 1.0)

        # --- PARES (desvio do equilíbrio)
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        max_desvio = int(base_cfg["max_desvio_pares"])
        if desvio_pares <= max_desvio:
            coerencias += 1
            score_pares = 1.0 - (desvio_pares / max(1.0, float(max_desvio)))
        else:
            violacoes += 1
            score_pares = -min(1.0, (desvio_pares - float(max_desvio)) / 1.0)

        # --- ADJACÊNCIAS
        max_adj = int(base_cfg["max_adjacencias"])
        if adj <= max_adj:
            coerencias += 1
            score_adj = 1.0 - (adj / max(1.0, float(max_adj)))
        else:
            violacoes += 1
            score_adj = -min(1.0, (adj - max_adj) / 1.0)

        score_total = (
            pesos["soma"] * score_soma +
            pesos["pares"] * score_pares +
            pesos["adj"] * score_adj
        )

        # valida constraints (hard), mas só registra no detalhe (o filtro mesmo ocorre na geração)
        ok_constraints, lista_viol = self._constraints_ok(
            soma=soma,
            pares=pares,
            adj=adj,
            faixa_1_10=faixa_1_10,
            faixa_11_25=faixa_11_25,
            constraints=constraints_override,
            regime_cfg=base_cfg,
        )

        detalhes = {
            "k": k,
            "soma": soma,
            "mu_soma": round(mu_soma, 4),
            "sd_soma": round(sd_soma, 4),
            "z_soma": round(z_soma, 4),
            "pares": pares,
            "impares": impares,
            "adjacencias": adj,
            "faixa_1_10": faixa_1_10,
            "faixa_11_25": faixa_11_25,
            "regime": regime,
            "constraints_ok": bool(ok_constraints),
            "constraints_violadas": lista_viol,
            "componentes": {
                "score_soma": round(score_soma, 4),
                "score_pares": round(score_pares, 4),
                "score_adj": round(score_adj, 4),
                "pesos": pesos,
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

        # Regime cfg base pra usar no filtro hard
        regime_cfg = dict(self._regimes.get(regime, self._regimes["estavel"]))
        if constraints_override:
            if "max_adjacencias" in constraints_override:
                try:
                    regime_cfg["max_adjacencias"] = int(constraints_override["max_adjacencias"])
                except Exception:
                    pass
            if "z_max_soma" in constraints_override:
                try:
                    regime_cfg["z_max_soma"] = float(constraints_override["z_max_soma"])
                except Exception:
                    pass

        for seq in candidatos:
            seq_ord = sorted(int(x) for x in seq)
            soma = sum(seq_ord)
            pares = sum(1 for x in seq_ord if x % 2 == 0)
            adj = self._count_adjacencias(seq_ord)
            f1 = self._count_faixa(seq_ord, 1, 10)
            f2 = self._count_faixa(seq_ord, 11, 25)

            ok, _ = self._constraints_ok(
                soma=soma,
                pares=pares,
                adj=adj,
                faixa_1_10=f1,
                faixa_11_25=f2,
                constraints=constraints_override,
                regime_cfg=regime_cfg,
            )
            if not ok:
                continue  # <<< AQUI é o filtro duro

            avaliacao = self.avaliar_sequencia(
                list(seq_ord),
                regime,
                pesos_override=pesos_override,
                constraints_override=constraints_override,
            )

            prototipos.append(
                Prototipo(
                    sequencia=list(seq_ord),
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
        }

        if incluir_contexto_dna:
            resp["contexto_dna"] = {
                "universo_max": self.universo_max,
                "total_sorteadas": self.grupo.total_sorteadas(),
            }

        return resp
