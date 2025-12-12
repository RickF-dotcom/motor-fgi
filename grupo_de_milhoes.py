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
            # ignora lixo
            continue
    return out


def _norm_seq(seq: Sequence[int]) -> Tuple[int, ...]:
    # normaliza como tupla ordenada crescente, sem duplicatas e sem None
    return tuple(sorted({int(x) for x in seq if x is not None}))


@dataclass
class GrupoDeMilhoes:
    """
    Grupo de Milhões = universo de combinações que AINDA NÃO aconteceram.

    - Universo padrão: 1..25 (Lotofácil)
    - Lê um CSV histórico (opcional) e registra as sequências já sorteadas por tamanho k
    - Gera combinações (itertools.combinations) e filtra as que já saíram
    - Entrega candidatos via get_candidatos(k, max_candidatos)

    Observações:
    - O histórico só exclui sequências do mesmo tamanho k.
    - NÃO pré-gera milhões no boot. Tudo é sob demanda.
    - auto_generate existe pra compatibilidade com MotorFGI (não explode RAM).
    """

    universo_max: int = 25
    historico_csv: Optional[Path] = None
    auto_generate: bool = False

    # interno: mapeia k -> set de sequências sorteadas (frozenset)
    _sorteadas_por_k: Dict[int, Set[frozenset[int]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.historico_csv is not None:
            self._carregar_historico(self.historico_csv)

        # Compat: se alguém passar auto_generate=True, não pré-calcula milhões.
        # Só garante que o histórico (se existir) foi carregado.
        if self.auto_generate:
            _ = self.total_sorteadas()

    # ----------------------------
    # Histórico
    # ----------------------------
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

            # tenta detectar colunas de dezenas (dez1..dez15 etc)
            # pega colunas cujo nome contenha "dez" ou seja numérico tipo "1","2"... (alguns CSVs vêm assim)
            dez_idxs: List[int] = []
            for i, h in enumerate(header_lower):
                if (
                    "dez" in h
                    or h.startswith("d")
                    or h.isdigit()
                    or h in {"bola1", "bola2", "bola3", "bola4", "bola5", "bola6"}
                ):
                    dez_idxs.append(i)

            for row in reader:
                if not row or all((c or "").strip() == "" for c in row):
                    continue

                nums: List[int] = []

                if dez_idxs:
                    # usa colunas detectadas
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
                    # fallback: extrai todos inteiros da linha e filtra no range
                    nums = []
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

    # ----------------------------
    # Universo / geração
    # ----------------------------
    def gerar_combinacoes(self, k: int) -> Iterator[Tuple[int, ...]]:
        """
        Gera todas as combinações do universo 1..universo_max com tamanho k.
        """
        universo = range(1, self.universo_max + 1)
        return itertools.combinations(universo, k)

    def _ja_saiu(self, seq: Sequence[int]) -> bool:
        seq_norm = _norm_seq(seq)
        k = len(seq_norm)
        if k <= 0:
            return False
        return frozenset(seq_norm) in self._sorteadas_por_k.get(k, set())

    # ----------------------------
    # API principal
    # ----------------------------
    def get_candidatos(
        self,
        k: int,
        max_candidatos: int = 2000,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Retorna até max_candidatos combinações de tamanho k que AINDA NÃO saíram no histórico.

        - shuffle=False: pega as primeiras válidas em ordem lexicográfica
        - shuffle=True: amostra aleatória (sem percorrer tudo quando possível)
        """
        if not (1 <= k <= self.universo_max):
            raise ValueError(f"k inválido: {k} (esperado 1..{self.universo_max})")

        if max_candidatos <= 0:
            return []

        sorteadas = self._sorteadas_por_k.get(k, set())

        # Modo determinístico: streaming e corte
        if not shuffle:
            out: List[List[int]] = []
            for comb in self.gerar_combinacoes(k):
                if frozenset(comb) in sorteadas:
                    continue
                out.append(list(comb))
                if len(out) >= max_candidatos:
                    break
            return out

        # Modo aleatório:
        rng = random.Random(seed)

        # Para universo pequeno (Lotofácil), ainda é viável fazer amostragem por tentativas.
        # Evita materializar o universo inteiro.
        out_set: Set[Tuple[int, ...]] = set()
        tentativas = 0
        limite_tentativas = max_candidatos * 200  # segura pra não travar se estiver “apertado”

        while len(out_set) < max_candidatos and tentativas < limite_tentativas:
            tentativas += 1
            comb = tuple(sorted(rng.sample(range(1, self.universo_max + 1), k)))
            if frozenset(comb) in sorteadas:
                continue
            out_set.add(comb)

        return [list(t) for t in out_set]

    # ----------------------------
    # Utilitários
    # ----------------------------
    def info(self) -> dict:
        return {
            "universo_max": self.universo_max,
            "historico_csv": str(self.historico_csv) if self.historico_csv else None,
            "total_sorteadas": self.total_sorteadas(),
            "sorteadas_por_k": {str(k): len(v) for k, v in sorted(self._sorteadas_por_k.items())},
            }
