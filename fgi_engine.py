# grupo_de_milhoes.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import csv
import itertools


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
    # normaliza como tupla ordenada crescente, sem duplicatas
    return tuple(sorted({int(x) for x in seq if x is not None}))


@dataclass
class GrupoDeMilhoes:
    """
    Grupo de Milhões = universo de combinações que ainda NÃO aconteceram.

    - Universo padrão: 1..25 (Lotofácil)
    - Lê um CSV histórico (opcional) e registra as sequências já sorteadas
    - Gera combinações (itertools.combinations) e filtra as já sorteadas
    - Entrega candidatos via get_candidatos(k, limite)

    Observação importante:
    - O histórico exclui sequências do MESMO tamanho k.
      Ex.: se seu CSV tem concursos de 15 dezenas, ele exclui apenas k=15.
    """

    universo_max: int = 25
    historico_csv: Optional[Path] = None

    # interno
    _sorteadas_por_k: Dict[int, Set[Tuple[int, ...]]] = None  # type: ignore

    def __post_init__(self) -> None:
        self._sorteadas_por_k = {}
        if self.historico_csv is not None:
            self._carregar_historico(Path(self.historico_csv))

    @property
    def universo(self) -> Tuple[int, ...]:
        return tuple(range(1, int(self.universo_max) + 1))

    # ---------------------------
    # Histórico
    # ---------------------------

    def _carregar_historico(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return

            header_lower = [(h or "").strip().lower() for h in header]

            # tenta detectar colunas de dezenas (dez1..dez15 etc) por prefixo
            dez_idxs: List[int] = []
            for i, h in enumerate(header_lower):
                if h.startswith("dez") or h.startswith("d") or h.startswith("n"):
                    dez_idxs.append(i)

            # fallback: se não achou nada, tenta pegar últimas 15 colunas
            fallback_last_n = 15

            for row in reader:
                if not row:
                    continue

                if dez_idxs:
                    dezenas_raw = [row[i] for i in dez_idxs if i < len(row)]
                else:
                    dezenas_raw = row[-fallback_last_n:] if len(row) >= fallback_last_n else row

                dezenas = _as_int_list(dezenas_raw)
                seq = _norm_seq(dezenas)

                # ignora linha inválida
                if len(seq) < 1:
                    continue

                k = len(seq)
                self._sorteadas_por_k.setdefault(k, set()).add(seq)

    def total_sorteadas(self, k: int) -> int:
        return len(self._sorteadas_por_k.get(int(k), set()))

    # ---------------------------
    # Geração / Consulta
    # ---------------------------

    def ja_sorteada(self, seq: Sequence[int]) -> bool:
        t = _norm_seq(seq)
        k = len(t)
        return t in self._sorteadas_por_k.get(k, set())

    def gerar_combinacoes(self, k: int) -> Iterable[Tuple[int, ...]]:
        k = int(k)
        if k < 1 or k > int(self.universo_max):
            raise ValueError(f"k inválido: {k} (esperado 1..{self.universo_max})")
        return itertools.combinations(self.universo, k)

    def get_candidatos(self, k: int, limite: int) -> List[Tuple[int, ...]]:
        """
        Retorna até `limite` combinações de tamanho `k` que ainda não aconteceram.

        - k = tamanho da combinação (ex: 15)
        - limite = quantidade máxima de candidatos (ex: 2000)
        """
        k = int(k)
        limite = int(limite)

        if limite <= 0:
            return []

        sorteadas = self._sorteadas_por_k.get(k, set())

        candidatos: List[Tuple[int, ...]] = []
        for comb in self.gerar_combinacoes(k):
            if comb in sorteadas:
                continue
            candidatos.append(comb)
            if len(candidatos) >= limite:
                break

        return candidatos
```0
