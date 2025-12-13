from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import math


# =========================
# DNA temporal (por janela)
# =========================

@dataclass
class WindowDNA:
    window: int
    k: int
    n_amostras: int
    soma_mu: float
    soma_sd: float
    pares_mu: float
    pares_sd: float
    adj_mu: float
    adj_sd: float


def _safe_sd(values: List[float]) -> float:
    if not values:
        return 1.0
    if len(values) == 1:
        return 1.0
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(var, 1e-9))


def _count_adjacencias(seq_ordenada: List[int]) -> int:
    adj = 0
    for i in range(1, len(seq_ordenada)):
        if seq_ordenada[i] - seq_ordenada[i - 1] == 1:
            adj += 1
    return adj


def _parse_row_ints(row: List[str]) -> List[int]:
    nums: List[int] = []
    for cell in row:
        cell = (cell or "").strip()
        if cell.isdigit():
            nums.append(int(cell))
        else:
            # tenta extrair números em formatos tipo "01"
            try:
                v = int(cell)
                nums.append(v)
            except Exception:
                pass
    return nums


def carregar_historico(csv_path: Path, k_esperado: Optional[int] = None) -> List[List[int]]:
    """
    Lê um CSV de concursos e devolve lista de sequências (ordenadas).
    - Se k_esperado for passado, filtra linhas que não batem.
    - Caso contrário, aceita a primeira linha válida como referência de k.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Histórico não encontrado: {csv_path}")

    seqs: List[List[int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            nums = _parse_row_ints(row)
            if not nums:
                continue
            # muitas planilhas trazem concurso/data antes; pega só os 15 últimos números se sobrar demais
            # regra segura: usa todos, mas se tiver > 15, tenta manter os últimos 15
            if len(nums) > 15:
                nums = nums[-15:]

            nums = sorted(nums)
            if k_esperado is None and len(nums) >= 5:
                k_esperado = len(nums)

            if k_esperado is not None and len(nums) != k_esperado:
                continue

            seqs.append(nums)

    if not seqs:
        raise ValueError(f"Histórico vazio ou inválido: {csv_path}")

    return seqs


def extrair_dna_temporal(historico: List[List[int]], window: int) -> WindowDNA:
    """
    Calcula DNA estatístico de uma janela temporal (últimos N concursos).
    Métricas (por sequência):
      - soma
      - pares
      - adjacências
    """
    if window <= 0:
        raise ValueError("window precisa ser > 0")

    k = len(historico[-1])
    fatia = historico[-window:] if len(historico) >= window else historico[:]
    n = len(fatia)

    somas: List[float] = []
    pares: List[float] = []
    adjs: List[float] = []

    for seq in fatia:
        seq_ord = sorted(int(x) for x in seq)
        somas.append(float(sum(seq_ord)))
        pares.append(float(sum(1 for x in seq_ord if x % 2 == 0)))
        adjs.append(float(_count_adjacencias(seq_ord)))

    soma_mu = sum(somas) / n
    pares_mu = sum(pares) / n
    adj_mu = sum(adjs) / n

    return WindowDNA(
        window=window,
        k=k,
        n_amostras=n,
        soma_mu=float(round(soma_mu, 6)),
        soma_sd=float(round(_safe_sd(somas), 6)),
        pares_mu=float(round(pares_mu, 6)),
        pares_sd=float(round(_safe_sd(pares), 6)),
        adj_mu=float(round(adj_mu, 6)),
        adj_sd=float(round(_safe_sd(adjs), 6)),
    )


def _zdist(x: float, mu: float, sd: float) -> float:
    sd = sd if sd > 0 else 1.0
    return abs((x - mu) / sd)


def distancia_prototipo_para_dna(
    prototipo_seq: List[int],
    dna: WindowDNA,
    pesos_metricas: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Distância estrutural (quanto menor, mais parecido com a janela).
    Usa z-distância em soma, pares, adj.
    """
    pesos = pesos_metricas or {"soma": 1.0, "pares": 1.0, "adj": 1.0}

    seq_ord = sorted(int(x) for x in prototipo_seq)
    soma = float(sum(seq_ord))
    pares = float(sum(1 for x in seq_ord if x % 2 == 0))
    adj = float(_count_adjacencias(seq_ord))

    d_soma = _zdist(soma, dna.soma_mu, dna.soma_sd)
    d_pares = _zdist(pares, dna.pares_mu, dna.pares_sd)
    d_adj = _zdist(adj, dna.adj_mu, dna.adj_sd)

    dist_total = (
        float(pesos.get("soma", 1.0)) * d_soma +
        float(pesos.get("pares", 1.0)) * d_pares +
        float(pesos.get("adj", 1.0)) * d_adj
    )

    return {
        "window": dna.window,
        "k": dna.k,
        "metricas_proto": {"soma": soma, "pares": pares, "adj": adj},
        "dist_componentes": {"soma": round(d_soma, 6), "pares": round(d_pares, 6), "adj": round(d_adj, 6)},
        "pesos_metricas": pesos,
        "dist_total": float(round(dist_total, 6)),
    }


def rankear_prototipos_por_fractal_temporal(
    prototipos: List[Dict[str, Any]],
    historico_csv: Path,
    windows: Optional[List[int]] = None,
    pesos_windows: Optional[Dict[int, float]] = None,
    pesos_metricas: Optional[Dict[str, float]] = None,
    top_n: int = 50,
) -> Dict[str, Any]:
    """
    Entrada:
      prototipos: lista de dicts no formato que a API já retorna:
        {"sequencia":[...], "score_total":..., "coerencias":..., "violacoes":..., ...}

    Saída:
      ranking por similaridade temporal (distância menor = melhor)
    """
    windows = windows or [25, 13, 8]
    pesos_windows = pesos_windows or {25: 0.25, 13: 0.50, 8: 0.25}
    pesos_metricas = pesos_metricas or {"soma": 1.0, "pares": 1.0, "adj": 1.0}

    hist = carregar_historico(historico_csv)
    dnas = {w: extrair_dna_temporal(hist, w) for w in windows}

    scored: List[Dict[str, Any]] = []
    for p in prototipos:
        seq = p.get("sequencia") or []
        if not isinstance(seq, list) or not seq:
            continue

        detalhes_por_w: Dict[int, Dict[str, Any]] = {}
        dist_agregada = 0.0

        for w in windows:
            dna = dnas[w]
            det = distancia_prototipo_para_dna(seq, dna, pesos_metricas=pesos_metricas)
            detalhes_por_w[w] = det
            dist_agregada += float(pesos_windows.get(w, 0.0)) * float(det["dist_total"])

        item = dict(p)
        item["fractal_temporal"] = {
            "windows": windows,
            "pesos_windows": pesos_windows,
            "dist_agregada": float(round(dist_agregada, 6)),
            "detalhes": detalhes_por_w,
        }
        scored.append(item)

    scored.sort(key=lambda x: x["fractal_temporal"]["dist_agregada"])
    return {
        "status": "ok",
        "windows": windows,
        "pesos_windows": pesos_windows,
        "pesos_metricas": pesos_metricas,
        "dna": {w: dnas[w].__dict__ for w in windows},
        "top": scored[: max(1, int(top_n))],
        "total_avaliados": len(scored),
      }
