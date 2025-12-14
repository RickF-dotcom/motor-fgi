
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
import csv
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
    - manter contrato estável com app.py
    - permitir overrides (pesos/constraints) sem quebrar
    - adicionar "REPETIÇÃO FRACTAL" como eixo de ranking (quebra empates)
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

        # Regras/limites por regime
        self._regimes: Dict[str, Dict[str, Any]] = {
            "estavel": {
                "z_max_soma": 1.30,
                "max_adjacencias": 3,
                "max_desvio_pares": 2,
                "pesos": {"soma": 1.2, "pares": 0.6, "adj": 0.6, "repeticao": 0.8},
            },
            "tenso": {
                "z_max_soma": 1.70,
                "max_adjacencias": 4,
                "max_desvio_pares": 3,
                "pesos": {"soma": 1.0, "pares": 0.45, "adj": 0.45, "repeticao": 0.6},
            },
        }

        # Âncora DNA (injetada pelo app)
        self._dna_anchor: Optional[Dict[str, Any]] = None
        self._dna_anchor_active: bool = False
        self._dna_anchor_window: int = 12

        # CSV local (se existir no projeto) para capturar o último concurso real
        self._default_last25_csv = Path(__file__).resolve().parent / "lotofacil_ultimos_25_concursos.csv"

    # ------------------------------------------------------------
    # Contrato exigido pelo app.py
    # ------------------------------------------------------------

    def set_dna_anchor(self, dna_last25: Optional[Dict[str, Any]] = None, window: int = 12) -> None:
        """
        Recebe o DNA calculado pelo RegimeDetector e define a janela que será usada
        como referência fractal (ex.: 12).
        """
        self._dna_anchor = dna_last25 or None
        self._dna_anchor_active = bool(dna_last25)
        self._dna_anchor_window = int(window or 12)

    def get_dna_anchor(self) -> Dict[str, Any]:
        return {
            "ativo": self._dna_anchor_active,
            "window": self._dna_anchor_window if self._dna_anchor_active else None,
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

    def _read_last_draw_from_csv(self, path: Path) -> Optional[List[int]]:
        """
        Lê o ÚLTIMO concurso do CSV de últimos 25.
        Precisa existir no repo em produção para esta métrica funcionar.
        """
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return None

        header = [h.strip().lower() for h in rows[0]]
        data_rows = rows[1:]

        # tenta localizar colunas dezenas (dez1..dez15) etc
        dez_idxs: List[int] = []
        for i, h in enumerate(header):
            if "dez" in h or h.startswith("d") or h.startswith("n"):
                dez_idxs.append(i)

        last = data_rows[-1]
        nums: List[int] = []

        if dez_idxs:
            for i in dez_idxs:
                if i >= len(last):
                    continue
                try:
                    v = int(str(last[i]).strip())
                except Exception:
                    continue
                if 1 <= v <= self.universo_max:
                    nums.append(v)
        else:
            # fallback: pega todos inteiros válidos da linha
            for cell in last:
                cell = (cell or "").strip()
                if not cell:
                    continue
                try:
                    v = int(cell)
                except Exception:
                    continue
                if 1 <= v <= self.universo_max:
                    nums.append(v)

        nums = sorted(set(nums))
        return nums if nums else None

    # ------------------------------------------------------------
    # Score: repetição fractal (nova engrenagem)
    # ------------------------------------------------------------

    def _target_repeticao(self) -> Optional[float]:
        """
        Busca repeticao_media do DNA(janela) escolhido.
        Estrutura esperada:
          dna_last25["janelas"][str(window)]["repeticao_media"]
        """
        if not self._dna_anchor_active or not isinstance(self._dna_anchor, dict):
            return None

        janelas = self._dna_anchor.get("janelas")
        if not isinstance(janelas, dict):
            return None

        wkey = str(self._dna_anchor_window)
        bloco = janelas.get(wkey)
        if not isinstance(bloco, dict):
            return None

        rep = bloco.get("repeticao_media")
        try:
            return float(rep)
        except Exception:
            return None

    def _score_repeticao(self, seq: List[int], tol: float = 1.5) -> Tuple[float, Dict[str, Any]]:
        """
        Mede quantos números repetem em relação ao ÚLTIMO concurso real:
          rep = |seq ∩ ultimo_concurso|
        e aproxima rep do target (DNA(window).repeticao_media)
        """
        target = self._target_repeticao()
        ultimo = self._read_last_draw_from_csv(self._default_last25_csv)

        if target is None or ultimo is None:
            # Sem dados suficientes: neutro
            return 0.0, {
                "ativo": False,
                "motivo": "sem_anchor_ou_sem_csv_ultimos25",
                "rep": None,
                "target": target,
                "tol": tol,
            }

        s = set(seq)
        u = set(ultimo)
        rep = len(s.intersection(u))

        diff = abs(rep - target)
        if diff <= tol:
            score = 1.0 - (diff / tol)  # 1..0
        else:
            score = -min(1.0, (diff - tol) / tol)  # 0..-1

        return float(score), {
            "ativo": True,
            "rep": rep,
            "target": round(float(target), 4),
            "tol": tol,
            "ultimo_concurso": ultimo,
        }

    # ------------------------------------------------------------
    # Avaliação principal
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

        seq_ord = sorted(set(int(x) for x in seq))
        soma = int(sum(seq_ord))
        pares = sum(1 for x in seq_ord if x % 2 == 0)
        impares = k - pares
        adj = self._count_adjacencias(seq_ord)

        mu_soma, sd_soma = self._sum_stats_sem_reposicao(k)
        z_soma = 0.0 if sd_soma == 0 else (soma - mu_soma) / sd_soma

        # --- score_soma
        excedente = max(0.0, abs(z_soma) - z_max_soma)
        score_soma = -excedente

        # --- score_pares
        alvo_pares = k / 2.0
        desvio_pares = abs(pares - alvo_pares)
        score_pares = 0.0
        if max_desvio_pares > 0:
            score_pares = 1.0 - (desvio_pares / float(max_desvio_pares))
            score_pares = max(0.0, min(1.0, score_pares))

        # --- score_adj
        score_adj = 0.0 if adj <= max_adj else -1.0

        # --- score_repeticao (NOVO)
        score_rep, rep_dbg = self._score_repeticao(seq_ord, tol=1.5)

        score_total = (
            pesos["soma"] * score_soma
            + pesos["pares"] * score_pares
            + pesos["adj"] * score_adj
            + pesos["repeticao"] * score_rep
        )

        # violações (mantemos só as 3 duras)
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
                "score_repeticao": round(score_rep, 4),
                "pesos": pesos,
                "constraints": {
                    "z_max_soma": z_max_soma,
                    "max_adjacencias": max_adj,
                    "max_desvio_pares": max_desvio_pares,
                },
                "repeticao_dbg": rep_dbg,
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
            },
            "contexto_lab": contexto_lab,
                }
