# grupo_de_milhoes.py
#
# Responsável por manter o "grupo de milhões":
# - universo total de combinações 15/25 (3.268.760 jogos)
# - MENOS todas as combinações já sorteadas
#   (vindas do historico sequencia_real.csv, se existir)
# - MENOS os concursos que você enviar via /carregar
#
# A API usada pelo FGIMotor:
#   - GrupoDeMilhoes()
#   - remover_sorteadas(concursos)
#   - sample(n)  -> devolve n jogos (listas de dezenas)

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, List
import csv
import os
import pickle
import random


ARQUIVO_PADRAO = "grupo_de_milhoes.pkl"
HISTORICO_CSV = Path(__file__).parent / "sequencia_real.csv"


class GrupoDeMilhoes:
    def __init__(self, arquivo: str = ARQUIVO_PADRAO, auto_generate: bool = True):
        self.arquivo = arquivo
        self.combos: List[int] = []

        # 1) tenta carregar do .pkl
        if os.path.exists(self.arquivo):
            self._load()
        else:
            # 2) se não existir, pode gerar universo total
            if auto_generate:
                self._gerar_universo_total()
            else:
                self.combos = []

        # 3) se existir histórico em CSV, remove sorteadas do universo
        if self.combos and HISTORICO_CSV.exists():
            self._remover_sorteadas_de_csv()

    # ----------------------------------------
    # Codificação: jogo -> bitmask (25 bits)
    # ----------------------------------------
    @staticmethod
    def _encode(jogo: Iterable[int]) -> int:
        mask = 0
        for d in jogo:
            if not 1 <= d <= 25:
                raise ValueError(f"Dezena inválida: {d}")
            mask |= 1 << (d - 1)
        return mask

    @staticmethod
    def _decode(mask: int) -> List[int]:
        return [d + 1 for d in range(25) if mask & (1 << d)]

    # ----------------------------------------
    # Persistência
    # ----------------------------------------
    def _save(self) -> None:
        with open(self.arquivo, "wb") as f:
            pickle.dump(self.combos, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self) -> None:
        with open(self.arquivo, "rb") as f:
            self.combos = pickle.load(f)

    # ----------------------------------------
    # Universo total
    # ----------------------------------------
    def _gerar_universo_total(self) -> None:
        """
        Gera TODAS as combinações de 15 dezenas entre 1..25.
        Tamanho: 3.268.760 combinações.
        Roda uma vez, salva em grupo_de_milhoes.pkl.
        """
        combos: List[int] = []
        for comb in combinations(range(1, 26), 15):
            combos.append(self._encode(comb))

        self.combos = combos
        self._save()

    # ----------------------------------------
    # Remoção de sorteadas (via CSV histórico)
    # ----------------------------------------
    def _remover_sorteadas_de_csv(self) -> None:
        """
        Lê o sequencia_real.csv (se existir) e remove TODAS
        as combinações já sorteadas do universo.
        """
        base = set(self.combos)
        alterou = False

        try:
            with HISTORICO_CSV.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    nums = [c.strip() for c in row if c.strip()]
                    if not nums:
                        continue
                    dezenas = [int(x) for x in nums[:15]]
                    dezenas_validas = [d for d in dezenas if 1 <= d <= 25]
                    if len(dezenas_validas) != 15:
                        continue
                    mask = self._encode(dezenas_validas)
                    if mask in base:
                        base.remove(mask)
                        alterou = True
        except Exception:
            # se der erro de leitura, não mata o servidor; só segue com o que tem
            return

        if alterou:
            self.combos = list(base)
            self._save()

    # ----------------------------------------
    # Remoção incremental (via /carregar)
    # ----------------------------------------
    def remover_sorteadas(self, concursos: List[List[int]]) -> None:
        """
        Remove do grupo todas as combinações que já saíram.
        Pode ser chamado várias vezes (incremental).
        """
        if not self.combos:
            return

        base = set(self.combos)
        alterou = False

        for jogo in concursos:
            dezenas_validas = [d for d in jogo if 1 <= d <= 25]
            if len(dezenas_validas) != 15:
                continue
            mask = self._encode(dezenas_validas)
            if mask in base:
                base.remove(mask)
                alterou = True

        if alterou:
            self.combos = list(base)
            self._save()

    # ----------------------------------------
    # Amostragem para o motor
    # ----------------------------------------
    def sample(self, n: int) -> List[List[int]]:
        """
        Retorna n jogos distintos do grupo (em forma de listas de dezenas).
        """
        if not self.combos:
            return []

        n = min(n, len(self.combos))
        indices = random.sample(range(len(self.combos)), n)
        return [self._decode(self.combos[i]) for i in indices]
