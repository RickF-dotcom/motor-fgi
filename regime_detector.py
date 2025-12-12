from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import math


# =========================
# Helpers
# =========================

def _as_set(seq: List[int]) -> set[int]:
    return set(int(x) for x in seq)

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    # population std (stable enough for regime scoring)
    if not xs:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(v)

def _zscore(x: float, m: float, s: float) -> float:
    return (x - m) / s if s > 1e-12 else 0.0

def _read_sequences_csv(path: Path) -> List[List[int]]:
    """
    Lê CSV com concursos da Lotofácil.
    Aceita formatos comuns:
      - colunas com dezenas (ex: D1..D15)
      - ou colunas mistas onde as dezenas aparecem como 15 ints na linha
    Retorna lista de sequências (cada uma com 15 ints).
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    seqs: List[List[int]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Se não tem header, tenta processar como linhas cruas
        if header is None:
            return seqs

        # tenta detectar colunas de dezenas no header
        # pega todas colunas cujo nome contenha "dez" ou seja numérica tipo "1","2" etc.
        header_lower = [h.strip().lower() for h in header]
        dez_idx: List[int] = []

        for i, h in enumerate(header_lower):
            if "dez" in h or h in [f"d{j}" for j in range(1, 16)] or h in [str(j) for j in range(1, 16)]:
                dez_idx.append(i)

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue

            nums: List[int] = []

            if len(dez_idx) >= 15:
                # usa as 15 primeiras colunas "dezenas"
                for i in dez_idx[:15]:
                    try:
                        nums.append(int(row[i]))
                    except Exception:
                        pass
            else:
                # fallback: extrai todos ints da linha e pega os 15 últimos plausíveis (1..25)
                extracted: List[int] = []
                for c in row:
                    c = c.strip()
                    if not c:
                        continue
                    try:
                        v = int(c)
                        if 1 <= v <= 25:
                            extracted.append(v)
                    except Exception:
                        continue
                # Se vierem exatamente 15 dezenas no fim, pega elas
                if len(extracted) >= 15:
                    nums = extracted[-15:]

            nums = sorted(set(nums))
            if len(nums) == 15:
                seqs.append(nums)

    return seqs


# =========================
# Métricas
# =========================

@dataclass
class Metrics:
    soma: float
    impares: float
    faixa_1_13: float
    adjacencias: float
    repeticao: float  # repetição com o concurso anterior (na mesma janela)

def _metrics_for_window(seqs: List[List[int]]) -> Tuple[List[Metrics], Optional[Metrics]]:
    """
    Retorna:
      - metrics por concurso (na ordem do CSV)
      - metrics médio da janela
    """
    if not seqs:
        return [], None

    ms: List[Metrics] = []

    prev: Optional[List[int]] = None
    for seq in seqs:
        s = sum(seq)
        imp = sum(1 for x in seq if x % 2 == 1)
        f13 = sum(1 for x in seq if 1 <= x <= 13)

        # adjacências: conta pares consecutivos (x, x+1) dentro da própria sequência
        st = set(seq)
        adj = 0
        for x in seq:
            if (x + 1) in st:
                adj += 1

        rep = 0
        if prev is not None:
            rep = len(_as_set(seq) & _as_set(prev))

        ms.append(
            Metrics(
                soma=float(s),
                impares=float(imp),
                faixa_1_13=float(f13),
                adjacencias=float(adj),
                repeticao=float(rep),
            )
        )
        prev = seq

    avg = Metrics(
        soma=_mean([m.soma for m in ms]),
        impares=_mean([m.impares for m in ms]),
        faixa_1_13=_mean([m.faixa_1_13 for m in ms]),
        adjacencias=_mean([m.adjacencias for m in ms]),
        repeticao=_mean([m.repeticao for m in ms]),
    )
    return ms, avg


# =========================
# RegimeDetector
# =========================

class RegimeDetector:
    """
    Detector de regime baseado em:
      - DNA das últimas 25 (janelas 7/10/12/15/20/25)
      - comparação do DNA(25) com baseline histórico (se disponível)
    """

    def __init__(
        self,
        dna_path: Path,
        historico_csv: Path,
        historico_total_csv: Optional[Path] = None,
    ):
        self.dna_path = dna_path
        self.historico_csv = historico_csv
        self.historico_total_csv = historico_total_csv

    # ---------- DNA ----------
    def extrair_dna(self) -> Dict[str, Any]:
        last25 = _read_sequences_csv(self.historico_csv)
        if len(last25) < 25:
            raise ValueError(f"CSV das últimas 25 tem {len(last25)} sequências; esperado >= 25")

        # garante pegar as últimas 25 mesmo que o arquivo tenha mais
        last25 = last25[-25:]

        janelas = [7, 10, 12, 15, 20, 25]
        dna: Dict[str, Any] = {
            "origem": "ultimos_25_concursos",
            "janelas": {},
        }

        for w in janelas:
            window_seqs = last25[-w:]
            _, avg = _metrics_for_window(window_seqs)
            assert avg is not None

            dna["janelas"][str(w)] = {
                "n": w,
                "soma_media": round(avg.soma, 4),
                "impares_media": round(avg.impares, 4),
                "faixa_1_13_media": round(avg.faixa_1_13, 4),
                "adjacencias_media": round(avg.adjacencias, 4),
                "repeticao_media": round(avg.repeticao, 4),
            }

        return dna

    # ---------- Baseline ----------
    def _baseline_historico(self) -> Dict[str, Any]:
        """
        Se existir historico_total_csv (ex: todos os concursos), calcula baseline real.
        Se não existir, usa baseline fallback (valores típicos) e marca como fallback.
        """
        if self.historico_total_csv and self.historico_total_csv.exists():
            allseqs = _read_sequences_csv(self.historico_total_csv)
            if len(allseqs) < 200:
                # muito curto pra baseline macro
                return self._baseline_fallback()

            ms, avg = _metrics_for_window(allseqs)
            # stds
            soma_std = _std([m.soma for m in ms])
            imp_std = _std([m.impares for m in ms])
            f13_std = _std([m.faixa_1_13 for m in ms])
            adj_std = _std([m.adjacencias for m in ms])
            rep_std = _std([m.repeticao for m in ms])

            return {
                "fonte": "historico_total_csv",
                "n": len(allseqs),
                "medias": {
                    "soma": round(avg.soma, 6),
                    "impares": round(avg.impares, 6),
                    "faixa_1_13": round(avg.faixa_1_13, 6),
                    "adjacencias": round(avg.adjacencias, 6),
                    "repeticao": round(avg.repeticao, 6),
                },
                "stds": {
                    "soma": round(soma_std, 6),
                    "impares": round(imp_std, 6),
                    "faixa_1_13": round(f13_std, 6),
                    "adjacencias": round(adj_std, 6),
                    "repeticao": round(rep_std, 6),
                },
            }

        return self._baseline_fallback()

    def _baseline_fallback(self) -> Dict[str, Any]:
        # fallback pragmático (macro típico). Não é “verdade”, é âncora operacional.
        return {
            "fonte": "fallback",
            "n": None,
            "medias": {
                "soma": 195.04,
                "impares": 7.49,
                "faixa_1_13": 7.80,
                "adjacencias": 8.07,
                "repeticao": 9.00,
            },
            "stds": {
                "soma": 18.25,
                "impares": 1.37,
                "faixa_1_13": 1.33,
                "adjacencias": 2.01,
                "repeticao": 1.70,
            },
        }

    # ---------- Regime ----------
    def detectar_regime(self) -> Dict[str, Any]:
        dna = self.extrair_dna()
        b = self._baseline_historico()

        d25 = dna["janelas"]["25"]
        medias = b["medias"]
        stds = b["stds"]

        z = {
            "soma": _zscore(d25["soma_media"], medias["soma"], stds["soma"]),
            "impares": _zscore(d25["impares_media"], medias["impares"], stds["impares"]),
            "faixa_1_13": _zscore(d25["faixa_1_13_media"], medias["faixa_1_13"], stds["faixa_1_13"]),
            "adjacencias": _zscore(d25["adjacencias_media"], medias["adjacencias"], stds["adjacencias"]),
            "repeticao": _zscore(d25["repeticao_media"], medias["repeticao"], stds["repeticao"]),
        }

        # escore simples de "tensão" do regime: média do |z|
        tension = _mean([abs(v) for v in z.values()])

        # classificação objetiva
        if tension < 0.35:
            label = "estavel"
        elif tension < 0.75:
            label = "transicao"
        else:
            label = "ruptura"

        return {
            "regime": label,
            "tensao": round(tension, 6),
            "zscore_25": {k: round(v, 6) for k, v in z.items()},
            "baseline": b,
            "dna_25": d25,
    }
