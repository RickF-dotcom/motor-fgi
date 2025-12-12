from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv

try:
    import pandas as pd  # opcional, melhora leitura de xlsx
except Exception:
    pd = None


# =========================
# Utilidades
# =========================

def _as_set(seq: List[int]) -> set[int]:
    return set(int(x) for x in seq)

def _count_parity(seq: List[int]) -> Tuple[int, int]:
    pares = sum(1 for x in seq if x % 2 == 0)
    impares = len(seq) - pares
    return pares, impares

def _count_faixas(seq: List[int]) -> Dict[str, int]:
    # Faixas simples e úteis na Lotofácil: 1-5, 6-10, 11-15, 16-20, 21-25
    bins = {
        "1_5": 0,
        "6_10": 0,
        "11_15": 0,
        "16_20": 0,
        "21_25": 0,
    }
    for x in seq:
        x = int(x)
        if 1 <= x <= 5:
            bins["1_5"] += 1
        elif 6 <= x <= 10:
            bins["6_10"] += 1
        elif 11 <= x <= 15:
            bins["11_15"] += 1
        elif 16 <= x <= 20:
            bins["16_20"] += 1
        elif 21 <= x <= 25:
            bins["21_25"] += 1
    return bins

def _seq_contigua_metric(seq: List[int]) -> Dict[str, Any]:
    s = sorted(int(x) for x in seq)
    runs = []
    run_len = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return {
        "maior_corrida": max(runs) if runs else 0,
        "qtd_corridas": len(runs),
        "corridas": runs,
    }

def _jaccard(a: List[int], b: List[int]) -> float:
    A = _as_set(a)
    B = _as_set(b)
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0

def _overlap(a: List[int], b: List[int]) -> int:
    return len(_as_set(a) & _as_set(b))


# =========================
# Leitura do histórico
# =========================

def _read_history_from_csv(path: Path) -> List[List[int]]:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for r in reader:
            nums = []
            for cell in r:
                cell = cell.strip()
                if not cell:
                    continue
                # pega só inteiros 1..25
                try:
                    v = int(float(cell))
                except Exception:
                    continue
                if 1 <= v <= 25:
                    nums.append(v)

            # sequências da Lotofácil têm 15 números
            if len(nums) == 15:
                rows.append(sorted(nums))

    return rows

def _read_history_from_xlsx(path: Path) -> List[List[int]]:
    if pd is None:
        raise RuntimeError("pandas não está disponível. Use CSV ou inclua pandas no requirements.")

    df = pd.read_excel(path)

    # Heurística: pegar colunas que parecem números do sorteio
    cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        # geralmente vem algo como "Bola1", "Bola 1", etc.
        if "bola" in s or s.startswith("b") or s.isdigit():
            cols.append(c)

    # fallback: se não achou por nome, pega colunas numéricas com bastante preenchimento
    if not cols:
        numeric_cols = []
        for c in df.columns:
            try:
                pd.to_numeric(df[c], errors="coerce")
                numeric_cols.append(c)
            except Exception:
                pass
        cols = numeric_cols

    # tenta achar as 15 colunas mais prováveis
    # pega as colunas com mais valores 1..25
    scores = []
    for c in cols:
        ser = pd.to_numeric(df[c], errors="coerce")
        ok = ser.between(1, 25).sum()
        scores.append((int(ok), c))
    scores.sort(reverse=True, key=lambda x: x[0])
    best_cols = [c for _, c in scores[:15]]

    rows: List[List[int]] = []
    for _, row in df.iterrows():
        nums = []
        for c in best_cols:
            v = row.get(c, None)
            if v is None:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            if 1 <= iv <= 25:
                nums.append(iv)
        if len(nums) == 15:
            rows.append(sorted(nums))
    return rows


# =========================
# DNA / Regime
# =========================

@dataclass
class WindowDNA:
    tamanho: int
    n: int
    soma_media: float
    soma_std: float
    pares_media: float
    pares_std: float
    jaccard_medio_consecutivo: float
    overlap_medio_consecutivo: float
    repeticao_top5: List[Tuple[int, int]]  # (numero, freq)
    ausentes: List[int]
    faixas_media: Dict[str, float]
    corridas_media: float
    maior_corrida_media: float


@dataclass
class RegimeDiagnostico:
    # scores 0..1 (quanto maior, mais “estável/consistente” no micro)
    estabilidade: float
    ruido: float
    # label interpretável
    regime: str
    # sinais principais
    sinais: Dict[str, Any]


class RegimeDetector:
    """
    Passo C:
    - extrai DNA por janelas das últimas 25 sequências reais
    - calcula um diagnóstico de regime atual
    """

    def __init__(self, history: List[List[int]]):
        if len(history) < 25:
            raise ValueError("Histórico insuficiente: precisa de pelo menos 25 sequências.")
        self.history = [sorted(list(map(int, s))) for s in history]

    @classmethod
    def from_file(cls, path: str) -> "RegimeDetector":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

        if p.suffix.lower() == ".csv":
            hist = _read_history_from_csv(p)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            hist = _read_history_from_xlsx(p)
        else:
            raise ValueError("Formato não suportado. Use .csv ou .xlsx")
        return cls(hist)

    def last_n(self, n: int) -> List[List[int]]:
        return self.history[-n:]

    def compute_window_dna(self, n: int) -> WindowDNA:
        window = self.last_n(n)

        somas = [sum(s) for s in window]
        pares = [_count_parity(s)[0] for s in window]

        # consecutivo
        jacs = []
        ovs = []
        for i in range(1, len(window)):
            jacs.append(_jaccard(window[i-1], window[i]))
            ovs.append(_overlap(window[i-1], window[i]))

        # frequências
        freq = {k: 0 for k in range(1, 26)}
        for s in window:
            for x in s:
                if 1 <= x <= 25:
                    freq[int(x)] += 1

        top5 = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ausentes = [k for k, v in freq.items() if v == 0]

        # faixas médias
        faixas_acc = {k: 0 for k in ["1_5","6_10","11_15","16_20","21_25"]}
        for s in window:
            bins = _count_faixas(s)
            for k in faixas_acc:
                faixas_acc[k] += bins[k]
        faixas_media = {k: faixas_acc[k] / n for k in faixas_acc}

        # corridas
        qtd_corridas = []
        maior_corrida = []
        for s in window:
            m = _seq_contigua_metric(s)
            qtd_corridas.append(m["qtd_corridas"])
            maior_corrida.append(m["maior_corrida"])

        def _mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs: List[float]) -> float:
            if len(xs) < 2:
                return 0.0
            mu = _mean(xs)
            var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
            return var ** 0.5

        return WindowDNA(
            tamanho=n,
            n=n,
            soma_media=_mean(somas),
            soma_std=_std(somas),
            pares_media=_mean(pares),
            pares_std=_std(pares),
            jaccard_medio_consecutivo=_mean(jacs) if jacs else 0.0,
            overlap_medio_consecutivo=_mean(ovs) if ovs else 0.0,
            repeticao_top5=top5,
            ausentes=ausentes,
            faixas_media=faixas_media,
            corridas_media=_mean(qtd_corridas),
            maior_corrida_media=_mean(maior_corrida),
        )

    def dna_last25(self, janelas: Optional[List[int]] = None) -> Dict[str, Any]:
        janelas = janelas or [7, 10, 12, 15, 20, 25]
        out = {"origem": "ultimos_25_concursos", "janelas": []}
        for n in janelas:
            wdna = self.compute_window_dna(n)
            out["janelas"].append(self._wdna_to_dict(wdna))
        return out

    def diagnosticar_regime_atual(self) -> RegimeDiagnostico:
        # usa janela 25 como base e 7/10 como “sensores” de curto prazo
        w25 = self.compute_window_dna(25)
        w10 = self.compute_window_dna(10)
        w7 = self.compute_window_dna(7)

        # Heurísticas objetivas (não místicas):
        # - estabilidade aumenta quando variações caem e overlap/jaccard sobem
        # - ruído aumenta quando curto prazo diverge do 25 e std sobe

        # Normalizações simples
        # soma_std típico (Lotofácil 15 nums) — só pra escala interna
        # não é “verdade”, é régua do laboratório
        soma_std_ref = 25.0
        pares_std_ref = 2.0

        def clamp01(x: float) -> float:
            return 0.0 if x < 0 else 1.0 if x > 1 else x

        estabilidade_base = (
            (1 - clamp01(w25.soma_std / soma_std_ref)) * 0.35 +
            (1 - clamp01(w25.pares_std / pares_std_ref)) * 0.15 +
            clamp01(w25.jaccard_medio_consecutivo) * 0.30 +
            clamp01(w25.overlap_medio_consecutivo / 15.0) * 0.20
        )

        # divergência curto vs longo
        div_soma = abs(w7.soma_media - w25.soma_media) / 60.0  # escala larga
        div_pares = abs(w7.pares_media - w25.pares_media) / 6.0
        div_jac = abs(w7.jaccard_medio_consecutivo - w25.jaccard_medio_consecutivo)

        ruido = clamp01(0.45 * div_soma + 0.25 * div_pares + 0.30 * div_jac)

        estabilidade = clamp01(estabilidade_base * (1 - 0.6 * ruido))

        # label do regime
        if estabilidade >= 0.75 and ruido <= 0.35:
            regime = "estavel"
        elif ruido >= 0.65:
            regime = "turbulento"
        else:
            regime = "transicao"

        sinais = {
            "w25": self._wdna_to_dict(w25),
            "w10": self._wdna_to_dict(w10),
            "w7": self._wdna_to_dict(w7),
            "divergencia_curto_vs_longo": {
                "div_soma": div_soma,
                "div_pares": div_pares,
                "div_jaccard": div_jac,
            },
        }

        return RegimeDiagnostico(
            estabilidade=estabilidade,
            ruido=ruido,
            regime=regime,
            sinais=sinais,
        )

    @staticmethod
    def _wdna_to_dict(w: WindowDNA) -> Dict[str, Any]:
        return {
            "tamanho": w.tamanho,
            "soma": {"media": w.soma_media, "std": w.soma_std},
            "pares": {"media": w.pares_media, "std": w.pares_std},
            "consecutivo": {
                "jaccard_medio": w.jaccard_medio_consecutivo,
                "overlap_medio": w.overlap_medio_consecutivo,
            },
            "repeticao_top5": [{"numero": n, "freq": f} for n, f in w.repeticao_top5],
            "ausentes": w.ausentes,
            "faixas_media": w.faixas_media,
            "contiguidade": {
                "corridas_media": w.corridas_media,
                "maior_corrida_media": w.maior_corrida_media,
            },
        }


# =========================
# Execução local (opcional)
# =========================
if __name__ == "__main__":
    # Ajuste para seu arquivo real
    detector = RegimeDetector.from_file("lotofacil_ultimos_25_concursos.csv")
    print(detector.dna_last25())
    print(detector.diagnosticar_regime_atual())
