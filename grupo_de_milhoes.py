# grupo_de_milhoes.py
from typing import List, Tuple, Iterable, Optional, Dict, Any
import random
import time

Jogo = Tuple[int, ...]


class GrupoDeMilhoes:
    """
    Representa o conjunto de todas as combinações NÃO sorteadas.
    """

    def __init__(
        self,
        jogos_drawn: Iterable[Jogo],
        min_n: int = 1,
        max_n: int = 25,
        k: int = 15,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.min_n = min_n
        self.max_n = max_n
        self.k = k
        self._drawn = set(tuple(sorted(j)) for j in jogos_drawn)

    def is_drawn(self, jogo: Iterable[int]) -> bool:
        return tuple(sorted(jogo)) in self._drawn

    def random_game(self) -> List[int]:
        return sorted(random.sample(range(self.min_n, self.max_n + 1), self.k))

    def sample_not_drawn(
        self,
        n: int = 10,
        timeout_sec: float = 2.0,
        max_attempts_factor: int = 20,
    ) -> Dict[str, Any]:
        """
        Gera jogos aleatórios NÃO sorteados (protótipos).
        """
        start = time.time()
        results = []
        seen = set()
        attempts = 0
        max_attempts = n * max_attempts_factor

        while len(results) < n:
            if attempts >= max_attempts:
                break
            if time.time() - start > timeout_sec:
                break

            attempts += 1
            jogo = tuple(self.random_game())

            if jogo in seen:
                continue

            seen.add(jogo)

            if not self.is_drawn(jogo):
                results.append(list(jogo))

        return {
            "count": len(results),
            "requested": n,
            "attempts": attempts,
            "elapsed_ms": int((time.time() - start) * 1000),
            "prototipos": results,
            "drawn_size": len(self._drawn),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "range": [self.min_n, self.max_n],
            "drawn_size": len(self._drawn),
        }


# Alias de compatibilidade (evita NameError em imports antigos)
GrupoMilhoes = GrupoDeMilhoes
