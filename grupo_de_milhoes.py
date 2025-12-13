from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
import csv
import random


class GrupoDeMilhoes:
    """
    Representa o UNIVERSO DE COMBINAÇÕES NÃO SORTEADAS.

    Responsabilidades:
    - Carregar histórico real (CSV)
    - Gerar combinações possíveis de k números (1..N)
    - EXCLUIR todas que já foram sorteadas
    - Fornecer candidatos controlados (amostragem segura)
    """

    def __init__(
        self,
        universo_max: int = 25,
        historico_csv: Optional[Path] = None,
    ) -> None:
        self.universo_max = int(universo_max)
        self._historico: Set[Tuple[int, ...]] = set()

        if historico_csv:
            self._carregar_historico(historico_csv)

    # =========================
    # Histórico
    # =========================

    def _carregar_historico(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Histórico não encontrado: {path}")

        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                nums = sorted(int(x) for x in row if x.strip().isdigit())
                if nums:
                    self._historico.add(tuple(nums))

    def total_sorteadas(self) -> int:
        return len(self._historico)

    # =========================
    # Geração de candidatos
    # =========================

    def _todas_combinacoes(self, k: int) -> Iterable[Tuple[int, ...]]:
        universo = range(1, self.universo_max + 1)
        return combinations(universo, k)

    def get_candidatos(
        self,
        k: int,
        max_candidatos: int = 2000,
    ) -> List[List[int]]:
        """
        Retorna até `max_candidatos` combinações NÃO sorteadas.
        Usa amostragem segura se o universo for muito grande.
        """
        k = int(k)
        max_candidatos = int(max_candidatos)

        candidatos: List[Tuple[int, ...]] = []

        for comb in self._todas_combinacoes(k):
            if comb not in self._historico:
                candidatos.append(comb)

                if len(candidatos) >= max_candidatos:
                    break

        # Se ainda ficou pequeno, embaralha para evitar viés
        random.shuffle(candidatos)

        return [list(c) for c in candidatos]
