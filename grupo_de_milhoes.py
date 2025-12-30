from typing import List, Iterable, Set, Dict, Any
import random


class GrupoMilhoes:
    """
    Baseline seguro do Grupo de Milhões.
    Nenhuma dependência externa.
    Nenhuma leitura de arquivo.
    Nenhum efeito colateral no import.
    """

    def __init__(
        self,
        k: int = 15,
        min_n: int = 1,
        max_n: int = 25,
        seed: int | None = None
    ):
        self.k = k
        self.min_n = min_n
        self.max_n = max_n
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self._drawn: Set[tuple[int, ...]] = set()

    # -------------------------
    # Núcleo
    # -------------------------

    def is_drawn(self, jogo: Iterable[int]) -> bool:
        key = tuple(sorted(jogo))
        return key in self._drawn

    def add_drawn(self, jogo: Iterable[int]) -> None:
        key = tuple(sorted(jogo))
        if len(key) != self.k:
            raise ValueError("Jogo inválido: tamanho incorreto")
        self._drawn.add(key)

    def random_game(self) -> List[int]:
        nums = random.sample(range(self.min_n, self.max_n + 1), self.k)
        return sorted(nums)

    def generate_not_drawn(self, n: int = 10) -> Dict[str, Any]:
        """
        Gera jogos aleatórios que NÃO estejam marcados como sorteados.
        """
        results: List[List[int]] = []
        attempts = 0
        max_attempts = n * 20

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            jogo = self.random_game()
            if not self.is_drawn(jogo):
                results.append(jogo)

        return {
            "requested": n,
            "generated": len(results),
            "attempts": attempts,
            "games": results,
        }

    # -------------------------
    # Status
    # -------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "range": [self.min_n, self.max_n],
            "drawn_size": len(self._drawn),
            "seed": self.seed,
        }


# Alias de compatibilidade (evita NameError em imports antigos)
GrupoDeMilhoes = GrupoMilhoes
