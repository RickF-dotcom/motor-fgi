# grupo_de_milhoes.py
#
# Responsável por:
# - gerar o universo total de combinações (25C15)
# - manter o "grupo de milhões" em disco (grupo_de_milhoes.pkl)
# - remover concursos já sorteados desse universo
# - entregar amostras de jogos para o FGIMotor

from itertools import combinations
from typing import Iterable, List
import os
import pickle
import random


ARQUIVO_PADRAO = "grupo_de_milhoes.pkl"


class GrupoDeMilhoes:
    def __init__(self, arquivo: str = ARQUIVO_PADRAO, auto_generate: bool = True):
        self.arquivo = arquivo
        self.combos: List[int] = []

        if os.path.exists(self.arquivo):
            self._load()
        else:
            if auto_generate:
                # Gera o universo total logo na primeira vez
                self._gerar_universo_total()
            else:
                self.combos = []

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
    # Universo total e remoção das sorteadas
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
            mask = self._encode(jogo)
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


# ---------------------------------------------------------
# Modo script opcional:
# python grupo_de_milhoes.py sequencia_real.csv
# (CSV com 15 colunas: d1,...,d15)
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    import csv

    if len(sys.argv) != 2:
        print(
            "Uso: python grupo_de_milhoes.py sequencia_real.csv\n"
            "CSV precisa ter 15 colunas numéricas (dezenas)."
        )
        sys.exit(1)

    csv_path = sys.argv[1]
    gm = GrupoDeMilhoes(auto_generate=True)

    concursos: List[List[int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            dezenas = [int(x) for x in row[:15]]
            concursos.append(sorted(dezenas))

    gm.remover_sorteadas(concursos)
    print(
        f"Grupo de milhões atualizado a partir de {len(concursos)} concursos. "
        f"Tamanho atual: {len(gm.combos)} combinações."
    )
