# grupo_de_milhoes.py
# ATHENA LABORATORIO PMF — Grupo de Milhões
# Ideia: representar o conjunto de combinações ainda NÃO observadas (não sorteadas),
# usando (1) histórico carregado e (2) geradores/triagem de candidatos (sem tentar enumerar 25C15).

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Set, Optional, Dict, Any, Iterator
from pathlib import Path
import csv
import random
import itertools
import time


Jogo = Tuple[int, ...]


def _norm_jogo(nums: Iterable[int], k: int = 15, min_n: int = 1, max_n: int = 25) -> Jogo:
    """Normaliza um jogo: valida, remove duplicatas, ordena e retorna tupla."""
    s = sorted(set(int(x) for x in nums))
    if len(s) != k:
        raise ValueError(f"Jogo inválido: esperado k={k}, veio {len(s)} itens: {s}")
    if s[0] < min_n or s[-1] > max_n:
        raise ValueError(f"Jogo fora do range [{min_n},{max_n}]: {s}")
    return tuple(s)


def _read_csv_numbers(path: Path) -> List[List[int]]:
    """
    Lê um CSV simples com dezenas.
    Aceita:
      - linhas com 15 números (separados por vírgula/; ou espaços)
      - ou colunas extras (id, data etc) desde que existam 15 inteiros na linha
    """
    rows: List[List[int]] = []
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = f.read().splitlines()

    # tenta detectar delimitador
    delimiter = ","
    if raw and (";" in raw[0] and raw[0].count(";") >= raw[0].count(",")):
        delimiter = ";"

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for r in reader:
            # extrai inteiros que existirem na linha
            ints = []
            for cell in r:
                cell = cell.strip()
                if not cell:
                    continue
                # também aceita "1 2 3 ..." numa célula só
                parts = cell.replace("\t", " ").split()
                if len(parts) > 1:
                    for p in parts:
                        if p.isdigit():
                            ints.append(int(p))
                else:
                    if cell.isdigit():
                        ints.append(int(cell))
            # tenta pegar 15 dezenas do fim (caso exista id/data no começo)
            if len(ints) >= 15:
                rows.append(ints[-15:])
    return rows


@dataclass
class GrupoMilhoesConfig:
    k: int = 15
    min_n: int = 1
    max_n: int = 25
    seed: Optional[int] = None
    # Limites para evitar travar servidor
    max_history: int = 100000  # segurança
    max_attempts_factor: int = 200  # tentativas ~ top_n * fator


class GrupoMilhoes:
    """
    Grupo de Milhões (versão pragmática):
      - Mantém um SET de jogos já observados (histórico).
      - Fornece métodos de triagem e amostragem de jogos NÃO observados.
      - Não tenta enumerar o universo completo (25C15 é grande demais).
    """

    def __init__(self, config: Optional[GrupoMilhoesConfig] = None):
        self.cfg = config or GrupoMilhoesConfig()
        self._drawn: Set[Jogo] = set()
        self._rng = random.Random(self.cfg.seed)

    # ----------------------------
    # Carregamento de histórico
    # ----------------------------
    def load_from_csv(self, csv_path: str | Path) -> Dict[str, Any]:
        p = Path(csv_path)
        rows = _read_csv_numbers(p)

        if len(rows) > self.cfg.max_history:
            rows = rows[-self.cfg.max_history :]

        added = 0
        for r in rows:
            try:
                j = _norm_jogo(r, k=self.cfg.k, min_n=self.cfg.min_n, max_n=self.cfg.max_n)
                if j not in self._drawn:
                    self._drawn.add(j)
                    added += 1
            except Exception:
                # se tiver linha zoada, ignora — melhor do que matar o serviço
                continue

        return {
            "csv": str(p),
            "rows_read": len(rows),
            "added_unique": added,
            "drawn_size": len(self._drawn),
        }

    def load_from_list(self, jogos: Iterable[Iterable[int]]) -> Dict[str, Any]:
        added = 0
        total = 0
        for nums in jogos:
            total += 1
            j = _norm_jogo(nums, k=self.cfg.k, min_n=self.cfg.min_n, max_n=self.cfg.max_n)
            if j not in self._drawn:
                self._drawn.add(j)
                added += 1
        return {"rows_read": total, "added_unique": added, "drawn_size": len(self._drawn)}

    # ----------------------------
    # Consultas
    # ----------------------------
    @property
    def drawn_size(self) -> int:
        return len(self._drawn)

    def is_drawn(self, nums: Iterable[int]) -> bool:
        j = _norm_jogo(nums, k=self.cfg.k, min_n=self.cfg.min_n, max_n=self.cfg.max_n)
        return j in self._drawn

    def filter_not_drawn(self, candidatos: Iterable[Iterable[int]]) -> List[List[int]]:
        out: List[List[int]] = []
        for c in candidatos:
            try:
                j = _norm_jogo(c, k=self.cfg.k, min_n=self.cfg.min_n, max_n=self.cfg.max_n)
                if j not in self._drawn:
                    out.append(list(j))
            except Exception:
                continue
        return out

    # ----------------------------
    # Geração de candidatos NÃO observados
    # ----------------------------
    def _random_game(self) -> Jogo:
        # sample sem repetição
        nums = self._rng.sample(range(self.cfg.min_n, self.cfg.max_n + 1), self.cfg.k)
        return tuple(sorted(nums))

    def sample_not_drawn(
        self,
        n: int = 10,
        max_candidates: Optional[int] = None,
        timeout_sec: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Amostra aleatória por rejeição: gera jogos aleatórios e aceita se não estiver no histórico.
        Isso funciona bem pra triagem (prototipagem), sem precisar enumerar nada.

        - n: quantos jogos retornar
        - max_candidates: limita tentativas (fallback)
        - timeout_sec: evita travar o servidor
        """
        t0 = time.time()
        target = int(n)
        if target <= 0:
            return {"count": 0, "prototipos": [], "attempts": 0, "elapsed_ms": 0}

        # tentativas máximas (por padrão baseado no fator)
        hard_cap = max_candidates if max_candidates is not None else target * self.cfg.max_attempts_factor

        results: List[List[int]] = []
        seen_local: Set[Jogo] = set()
        attempts = 0

        while len(results) < target:
            if attempts >= hard_cap:
                break
            if (time.time() - t0) > timeout_sec:
                break

            attempts += 1
            j = self._random_game()

            # evita duplicata local
            if j in seen_local:
                continue

            # aceita se não observado
            if j not in self._drawn:
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

    def iter_not_drawn_from_pool(
        self,
        pool: Iterable[Iterable[int]],
    ) -> Iterator[List[int]]:
        """
        Itera sobre um pool (ex: gerado por um motor/heurística) e devolve apenas os não observados.
        Ideal quando você tem um 'motor' que já gera candidatos bons, e aqui é só a triagem do grupo.
        """
        for c in pool:
            try:
                j = _norm_jogo(c, k=self.cfg.k, min_n=self.cfg.min_n, max_n=self.cfg.max_n)
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
        }


# Alias seguro pra compatibilidade com imports antigos
# (se você tinha "GrupoDeMilhoes" em algum lugar, isso evita NameError)
GrupoDeMilhoes = GrupoMilhoes
