
# grupo_de_milhoes.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import csv
import itertools
import random


def _as_int_list(values: Iterable[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _norm_seq(seq: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in seq if x is not None}))


@dataclass
class GrupoMilhoes:
    """
    Grupo de Milhões = combinações que AINDA NÃO saíram.

    - Não pré-gera tudo (sob demanda)
    - get_candidatos() suporta shuffle/seed
    """
    universo_max: int = 25
    historico_csv: Optional[Path] = None
    auto_generate: bool = False

    _sorteadas_por_k: Dict[int, Set[frozenset[int]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.historico_csv is not None:
            self._carregar_historico(self.historico_csv)

    def _carregar_historico(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"csv não encontrado: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return

            header_lower = [(h or "").strip().lower() for h in header]

            dez_idxs: List[int] = []
            for i, h in enumerate(header_lower):
                if "dez" in h or h.startswith("d") or h.isdigit() or h.startswith("n"):
                    dez_idxs.append(i)

            for row in reader:
                if not row or all((c or "").strip() == "" for c in row):
                    continue

                nums: List[int] = []
                if dez_idxs:
                    for i in dez_idxs:
                        if i >= len(row):
                            continue
                        try:
                            v = int(str(row[i]).strip())
                        except Exception:
                            continue
                        if 1 <= v <= self.universo_max:
                            nums.append(v)
                else:
                    # fallback: varre tudo
                    for cell in row:
                        cell = (cell or "").strip()
                        if not cell:
                            continue
                        try:
                            v = int(cell)
                        except Exception:
                            continue
                        if 1 <= v <= self.universo_max:
                            nums.append(v)

                seq = _norm_seq(nums)
                k = len(seq)
                if k <= 0:
                    continue

                self._sorteadas_por_k.setdefault(k, set()).add(frozenset(seq))

    def total_sorteadas(self, k: Optional[int] = None) -> int:
        if k is None:
            return sum(len(s) for s in self._sorteadas_por_k.values())
        return len(self._sorteadas_por_k.get(k, set()))

    def gerar_combinacoes(self, k: int) -> Iterator[Tuple[int, ...]]:
        universo = range(1, self.universo_max + 1)
        return itertools.combinations(universo, k)

    def get_candidatos(
        self,
        k: int,
        max_candidatos: int = 2000,
        shuffle: bool = True,
        seed: Optional[int] = 1337,
    ) -> List[List[int]]:
        """
        shuffle=True por padrão, porque ordem lexicográfica gera um viés horroroso de adjacências.
        """
        if not (1 <= k <= self.universo_max):
            raise ValueError(f"k inválido: {k} (esperado 1..{self.universo_max})")

        if max_candidatos <= 0:
            return []

        sorteadas = self._sorteadas_por_k.get(k, set())

        # determinístico (streaming)
        if not shuffle:
            out: List[List[int]] = []
            for comb in self.gerar_combinacoes(k):
                if frozenset(comb) in sorteadas:
                    continue
                out.append(list(comb))
                if len(out) >= max_candidatos:
                    break
            return out

        # aleatório (amostragem por tentativas)
        rng = random.Random(seed)
        out_set: Set[Tuple[int, ...]] = set()
        tentativas = 0
        limite_tentativas = max_candidatos * 500

        while len(out_set) < max_candidatos and tentativas < limite_tentativas:
            tentativas += 1
            comb = tuple(sorted(rng.sample(range(1, self.universo_max + 1), k)))
            if frozenset(comb) in sorteadas:
                continue
            out_set.add(comb)

        return [list(t) for t in out_set]


# ============================================================
# Compatibilidade com o app.py atual (não quebrar pipeline)
# ============================================================

class GrupoMilhoes(GrupoDeMilhoes):
    """
    Alias compatível com imports existentes:
      from grupo_de_milhoes import GrupoMilhoes

    Além disso, implementa gerar_combinacoes() no formato que o app.py usa:
      candidatos = grupo.gerar_combinacoes()
      -> retorna List[List[int]]
    """
    def gerar_combinacoes(
        self,
        k: int = 15,
        max_candidatos: int = 3000,
        shuffle: bool = True,
        seed: Optional[int] = 1337,
    ) -> List[List[int]]:
        return self.get_candidatos(
            k=k,
            max_candidatos=max_candidatos,
            shuffle=shuffle,
            seed=seed,
        )
