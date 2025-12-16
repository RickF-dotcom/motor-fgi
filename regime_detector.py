# regime_detector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import math
import statistics


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(str(x).strip())
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _adjacencias(seq: List[int]) -> int:
    s = sorted(seq)
    c = 0
    for i in range(1, len(s)):
        if s[i] - s[i - 1] == 1:
            c += 1
    return c


def _faixa_1_13(seq: List[int]) -> int:
    return sum(1 for x in seq if 1 <= x <= 13)


def _pares(seq: List[int]) -> int:
    return sum(1 for x in seq if x % 2 == 0)


def _soma(seq: List[int]) -> int:
    return sum(int(x) for x in seq)


def _parse_row_numbers(row: List[str]) -> List[int]:
    nums: List[int] = []
    for cell in row:
        v = _safe_int(cell, None)
        if v is None:
            continue
        if 1 <= v <= 25:
            nums.append(v)
    nums = sorted(set(nums))
    return nums


@dataclass
class RegimeDetector:
    """
    Lê e sintetiza contexto do laboratório:
      - dna_last25: estatísticas por janelas (7/10/12/13/14/15/20/25)
      - regime_atual: zscores simples vs baseline (macro)
    """

    base_dir: Optional[Path] = None
    ultimos25_csv_name: str = "lotofacil_ultimos_25_concursos.csv"

    def __post_init__(self) -> None:
        if self.base_dir is None:
            # garante caminho estável no Render
            self.base_dir = Path(__file__).resolve().parent

    # ----------------------------
    # IO
    # ----------------------------
    def _ultimos25_path(self) -> Path:
        return (self.base_dir / self.ultimos25_csv_name).resolve()

    def _ler_ultimos25(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        path = self._ultimos25_path()
        meta: Dict[str, Any] = {"source": str(path)}
        if not path.exists():
            return [], {"_error": f"CSV não encontrado: {path}"}

        rows: List[List[int]] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                all_rows = list(reader)
        except Exception as e:
            return [], {"_error": f"Falha lendo CSV: {e}", "source": str(path)}

        if not all_rows:
            return [], {"_error": "CSV vazio", "source": str(path)}

        # tenta detectar header; se tiver texto, ignora primeira linha
        first = all_rows[0]
        has_text = any(any(ch.isalpha() for ch in (c or "")) for c in first)
        data_rows = all_rows[1:] if has_text else all_rows

        for r in data_rows:
            nums = _parse_row_numbers(r)
            if len(nums) >= 15:
                # normaliza para 15 dezenas (Lotofácil)
                rows.append(nums[:15])

        if not rows:
            return [], {"_error": "Não consegui extrair sequências do CSV", "source": str(path)}

        # garante só últimos 25 (se tiver mais)
        if len(rows) > 25:
            rows = rows[-25:]

        meta["n_rows"] = len(rows)
        return rows, meta

    # ----------------------------
    # Métricas
    # ----------------------------
    def _metrics_for_seq(self, seq: List[int], prev_seq: Optional[List[int]] = None) -> Dict[str, float]:
        rep = 0.0
        if prev_seq is not None:
            rep = float(len(set(seq) & set(prev_seq)))

        return {
            "soma": float(_soma(seq)),
            "pares": float(_pares(seq)),
            "adj": float(_adjacencias(seq)),
            "faixa_1_13": float(_faixa_1_13(seq)),
            "repeticao": float(rep),
        }

    def _window_stats(self, seqs: List[List[int]]) -> Dict[str, Any]:
        # calcula séries
        series: Dict[str, List[float]] = {
            "soma": [],
            "pares": [],
            "adj": [],
            "faixa_1_13": [],
            "repeticao": [],
        }
        prev: Optional[List[int]] = None
        for s in seqs:
            m = self._metrics_for_seq(s, prev)
            prev = s
            for k, v in m.items():
                series[k].append(float(v))

        out: Dict[str, Any] = {}
        for k, arr in series.items():
            if not arr:
                out[k] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                continue
            mean = float(statistics.mean(arr))
            std = float(statistics.pstdev(arr)) if len(arr) > 1 else 0.0
            out[k] = {
                "mean": mean,
                "std": std,
                "min": float(min(arr)),
                "max": float(max(arr)),
            }
        return out

    def _zscores(self, stats_w: Dict[str, Any], stats_base: Dict[str, Any]) -> Dict[str, float]:
        z: Dict[str, float] = {}
        for m in stats_w.keys():
            mu = float(stats_base[m]["mean"])
            sd = float(stats_base[m]["std"])
            x = float(stats_w[m]["mean"])
            if sd <= 1e-12:
                z[m] = 0.0
            else:
                z[m] = float((x - mu) / sd)
        return z

    # ----------------------------
    # API pública
    # ----------------------------
    def get_dna_last25(self, windows: Optional[List[int]] = None) -> Dict[str, Any]:
        windows = windows or [7, 10, 12, 13, 14, 15, 20, 25]

        seqs, meta = self._ler_ultimos25()
        if not seqs:
            return {"_error": meta.get("_error", "falha"), "meta": meta}

        dna: Dict[str, Any] = {"meta": meta, "windows": {}}

        for w in windows:
            if w <= 0:
                continue
            block = seqs[-w:] if len(seqs) >= w else seqs[:]
            dna["windows"][str(w)] = {
                "n": len(block),
                "stats": self._window_stats(block),
            }

        return dna

    def detectar_regime(self) -> Dict[str, Any]:
        seqs, meta = self._ler_ultimos25()
        if not seqs:
            return {"_error": meta.get("_error", "falha"), "meta": meta}

        base = self._window_stats(seqs)  # baseline = 25
        w13 = self._window_stats(seqs[-13:]) if len(seqs) >= 13 else self._window_stats(seqs)

        z = self._zscores(w13, base)

        # “tensão” simples: norma L2 dos zscores capados
        z_cap = {k: _clamp(v, -4.0, 4.0) for k, v in z.items()}
        tensao = math.sqrt(sum(float(v) ** 2 for v in z_cap.values()))

        ultimo = seqs[-1] if seqs else []
        return {
            "meta": meta,
            "ultimo_concurso": {"sequencia": ultimo},
            "baseline_25": base,
            "janela_13": w13,
            "zscores_13_vs_25": z,
            "tensao": float(tensao),
            }
