# grupo_de_milhoes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import csv
import random
import time

Jogo = Tuple[int, ...]


def _norm_jogo(nums: Iterable[int], k: int, min_n: int, max_n: int) -> Jogo:
    """Normaliza um jogo: valida faixa, remove duplicatas, ordena e retorna tuple."""
    s = set(int(x) for x in nums)
    if len(s) != k:
        raise ValueError(f"Jogo inválido: esperado {k} números distintos, veio {len(s)}")
    if any((x < min_n or x > max_n) for x in s):
        raise ValueError(f"Jogo inválido: fora do intervalo [{min_n},{max_n}]")
    return tuple(sorted(s))


def _read_csv_numbers(path: str) -> List[List[int]]:
    """
    Lê um CSV e extrai listas de inteiros por linha.
    Aceita separadores comuns e ignora colunas não-numéricas.
    """
    rows: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.reader(f, dialect)
        for r in reader:
            nums: List[int] = []
            for cell in r:
                cell = (cell or "").strip()
                if not cell:
                    continue
                # tenta extrair inteiros de células como "01", "1", "1.0"
                try:
                    if "." in cell:
                        v = int(float(cell))
                    else:
                        v = int(cell)
                    nums.append(v)
                except Exception:
                    continue
            if nums:
                rows.append(nums)
    return rows


@dataclass(frozen=True)
class GrupoMilhoesConfig:
    k: int = 15
    min_n: int = 1
    max_n: int = 25
    seed: int = 42
    max_history: int = 5000
    max_attempts_factor: int = 5000  # hard_cap ≈ n * fator


class GrupoMilhoes:
    """
    Grupo de Milhões = "tudo que ainda não saiu".
    Aqui a implementação é pragmática para API:
      - Armazena o histórico (jogos já sorteados) como set de tuples normalizadas
      - Oferece triagem e amostragem "não observada" por rejeição (sem enumerar o universo)
    """

    def __init__(self, config: Optional[GrupoMilhoesConfig] = None):
        self.cfg = config or GrupoMilhoesConfig()
        self._rng = random.Random(self.cfg.seed)
        self._drawn: Set[Jogo] = set()

    # ----------------------------
    # Carga do histórico
    # ----------------------------
    def load_from_csv(self, path: str, limit: Optional[int] = None) -> int:
        rows = _read_csv_numbers(path)
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        count = 0
        for r in rows:
            try:
                j = _norm_jogo(r[: self.cfg.k], self.cfg.k, self.cfg.min_n, self.cfg.max_n)
                self._drawn.add(j)
                count += 1
            except Exception:
                # ignora linhas ruins
                continue

        # corta histórico se crescer demais
        if len(self._drawn) > self.cfg.max_history:
            # mantém os "primeiros" arbitrariamente (set não garante ordem),
            # mas evita crescimento infinito
            self._drawn = set(list(self._drawn)[: self.cfg.max_history])
        return count

    def load_from_list(self, jogos: Iterable[Iterable[int]]) -> int:
        count = 0
        for nums in jogos:
            j = _norm_jogo(nums, self.cfg.k, self.cfg.min_n, self.cfg.max_n)
            self._drawn.add(j)
            count += 1
        if len(self._drawn) > self.cfg.max_history:
            self._drawn = set(list(self._drawn)[: self.cfg.max_history])
        return count

    # ----------------------------
    # Consultas / triagem
    # ----------------------------
    def is_drawn(self, jogo: Iterable[int]) -> bool:
        j = _norm_jogo(jogo, self.cfg.k, self.cfg.min_n, self.cfg.max_n)
        return j in self._drawn

    def filter_not_drawn(self, candidatos: Iterable[Iterable[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for c in candidatos:
            try:
                j = _norm_jogo(c, self.cfg.k, self.cfg.min_n, self.cfg.max_n)
            except Exception:
                continue
            if j not in self._drawn:
                out.append(list(j))
        return out

    # ----------------------------
    # Geração de candidatos NÃO observados
    # ----------------------------
    def _random_game(self) -> Jogo:
        nums = self._rng.sample(range(self.cfg.min_n, self.cfg.max_n + 1), self.cfg.k)
        return tuple(sorted(nums))

    def sample_not_drawn(
        self,
        n: int = 10,
        max_candidates: Optional[int] = None,
        timeout_sec: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Amostra aleatória por rejeição: gera jogos aleatórios e aceita se não observados.
        Não enumera o universo (serve bem para prototipagem).
        """
        t0 = time.time()
        target = int(n)
        if target <= 0:
            return {
                "count": 0,
                "requested": target,
                "attempts": 0,
                "elapsed_ms": 0,
                "prototipos": [],
                "drawn_size": len(self._drawn),
            }

        hard_cap = int(max_candidates) if max_candidates is not None else target * int(self.cfg.max_attempts_factor)

        results: List[List[int]] = []
        seen_local: Set[Jogo] = set()
        attempts = 0

        while len(results) < target:
            if attempts >= hard_cap:
                break
            if (time.time() - t0) > float(timeout_sec):
                break

            attempts += 1
            j = self._random_game()

            if j in seen_local:
                continue
            if j in self._drawn:
                continue

            seen_local.add(j)
            results.append(list(j))

        return {
            "count": len(results),
            "requested": target,
            "attempts": attempts,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "prototipos": results,
            "drawn_size": len(self._drawn),
        }

    def iter_not_drawn_from_pool(self, pool: Iterable[Iterable[int]]) -> Iterator[List[int]]:
        """
        Recebe um pool (ex: gerado por algum motor/heurística) e devolve só o que não saiu.
        """
        for c in pool:
            try:
                j = _norm_jogo(c, self.cfg.k, self.cfg.min_n, self.cfg.max_n)
            except Exception:
                continue
            if j not in self._drawn:
                yield list(j)

    # ----------------------------
    # Status
    # ----------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "k": self.cfg.k,
            "range": [self.cfg.min_n, self.cfg.max_n],
            "drawn_size": len(self._drawn),
            "seed": self.cfg.seed,
            "max_history": self.cfg.max_history,
        }


# Alias para compatibilidade (se algum lugar ainda usa esse nome)
GrupoDeMilhoes = GrupoMilhoes
