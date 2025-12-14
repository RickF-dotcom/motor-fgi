from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class _Sorteio:
    dezenas: Tuple[int, ...]  # sempre ordenado


class GrupoDeMilhoes:
    """
    Grupo de Milhões = todas as combinações possíveis que AINDA NÃO SAÍRAM.

    Nesta versão, a estratégia é:
    - ler o histórico (se existir)
    - construir um set das combinaações já sorteadas (como tuplas ordenadas)
    - gerar candidatos por amostragem aleatória + rejeição (sem enumerar universo inteiro)
    """

    def __init__(
        self,
        universo_max: int = 25,
        historico_csv: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.universo_max = int(universo_max)
        self.historico_csv = historico_csv
        self._rng = random.Random(seed)

        self._sorteados: Set[Tuple[int, ...]] = set()
        if self.historico_csv:
            self._sorteados = self._carregar_historico(self.historico_csv)

    # -------------------------
    # Leitura do histórico
    # -------------------------

    def _carregar_historico(self, path: Path) -> Set[Tuple[int, ...]]:
        if not path.exists():
            return set()

        sorteados: Set[Tuple[int, ...]] = set()

        def extrair_ints(row: Sequence[str]) -> List[int]:
            nums: List[int] = []
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
            return nums

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                nums = extrair_ints(row)

                # Tenta detectar dezenas da Lotofácil:
                # - se linha tiver >= 15 números válidos, pega os 15 primeiros
                # - se tiver exatamente 15, ok
                if len(nums) >= 15:
                    dezenas = tuple(sorted(nums[:15]))
                    if len(dezenas) == 15:
                        sorteados.add(dezenas)

        return sorteados

    # -------------------------
    # API pública (usada pelo MotorFGI)
    # -------------------------

    def ja_sorteada(self, seq: Sequence[int]) -> bool:
        t = tuple(sorted(int(x) for x in seq))
        return t in self._sorteados

    def get_candidatos(self, k: int, max_candidatos: int = 2000) -> List[Tuple[int, ...]]:
        """
        Retorna uma lista de candidatos (tuplas ordenadas) que NÃO estão no histórico.

        Estratégia: amostragem aleatória + rejeição.
        Isso evita o viés de devolver sempre sequências compactadas (1..10 etc).
        """
        k = int(k)
        max_candidatos = int(max_candidatos)

        if k <= 0 or k > self.universo_max:
            raise ValueError(f"k inválido: {k}")

        if max_candidatos <= 0:
            return []

        candidatos: List[Tuple[int, ...]] = []
        vistos: Set[Tuple[int, ...]] = set()

        # limite de tentativas para evitar loop infinito se o histórico for gigantesco
        # (na prática, pra Lotofácil isso é tranquilo)
        max_tentativas = max(50_000, max_candidatos * 200)

        universo = list(range(1, self.universo_max + 1))

        tentativas = 0
        while len(candidatos) < max_candidatos and tentativas < max_tentativas:
            tentativas += 1

            seq = tuple(sorted(self._rng.sample(universo, k)))

            # evita repetição na própria amostragem
            if seq in vistos:
                continue
            vistos.add(seq)

            # exclui o que já saiu
            if seq in self._sorteados:
                continue

            candidatos.append(seq)

        return candidatos
