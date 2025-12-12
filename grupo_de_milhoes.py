# grupo_de_milhoes.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Dict
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


def _norm_seq(seq: Sequence[int]) -> tuple[int, ...]:
    # normaliza como tupla ordenada crescente, sem duplicatas
    s = sorted({int(x) for x in seq if x is not None})
    return tuple(s)


@dataclass
class GrupoDeMilhoes:
    """
    Grupo de Milhões = universo de combinações que ainda NÃO aconteceram.

    Implementação prática:
    - Universo padrão: 1..25 (Lotofácil)
    - Lê um CSV histórico (opcional) e registra as sequências já sorteadas
    - Gera combinações (itertools.combinations) e filtra as que já saíram
    - Entrega candidatos via get_candidatos(k, max_candidatos)

    Observação importante:
    - O histórico só exclui sequências do mesmo tamanho k.
      Ex.: se seu CSV tem 15 dezenas por concurso, ele exclui apenas k=15.
    """

    universo_max: int = 25
    historico_csv: Optional[Path] = None

    # interno
    _sorteadas_por_k: Dict[int, Set[frozenset[int]]] = None  # type: ignore

    def __post_init__(self) -> None:
        self._sorteadas_por_k = {}
        if self.historico_csv is not None:
            self._carregar_historico(self.historico_csv)

    # -----------------------------
    # Histórico
    # -----------------------------
    def _carregar_historico(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return

            header_lower = [h.strip().lower() for h in header]

            # tenta detectar colunas de dezenas (dez1..dez15 etc)
            dez_idxs: List[int] = []
            for i, h in enumerate(header_lower):
                # padrões comuns: "dez", "dezena", "d1", "n1", etc
                if h.startswith("dez") or h.startswith("d") or h.startswith("n"):
                    dez_idxs.append(i)

            for row in reader:
                if not row or all((c or "").strip() == "" for c in row):
                    continue

                if len(dez_idxs) >= 10:
                    nums = _as_int_list(row[i] for i in dez_idxs)
                else:
                    # fallback: tenta varrer a linha inteira e pegar só números no range 1..universo
                    nums_all = _as_int_list(row)
                    nums = [x for x in nums_all if 1 <= x <= self.universo_max]

                seq = _norm_seq(nums)
                if not seq:
                    continue

                k = len(seq)
                self._sorteadas_por_k.setdefault(k, set()).add(frozenset(seq))

    def total_sorteadas(self, k: int) -> int:
        return len(self._sorteadas_por_k.get(int(k), set()))

    # -----------------------------
    # Núcleo: geração do grupo
    # -----------------------------
    def gerar_combinacoes(self, k: int) -> Iterator[tuple[int, ...]]:
        """
        Gera combinações do universo (1..universo_max) de tamanho k,
        filtrando as que já saíram (se houver histórico carregado).
        """
        k = int(k)
        if k <= 0 or k > self.universo_max:
            raise ValueError(f"k inválido: {k}. Esperado 1..{self.universo_max}")

        ja_saiu = self._sorteadas_por_k.get(k, set())

        universo = range(1, self.universo_max + 1)
        for comb in itertools.combinations(universo, k):
            if ja_saiu and frozenset(comb) in ja_saiu:
                continue
            yield comb

    # -----------------------------
    # Interface esperada pelo MotorFGI
    # -----------------------------
    def get_candidatos(self, k: int, max_candidatos: int) -> List[List[int]]:
        """
        CONTRATO DO MotorFGI:
        retorna até max_candidatos sequências (listas de int) de tamanho k,
        vindas do Grupo de Milhões (combinações ainda não sorteadas).
        """
        k = int(k)
        max_candidatos = int(max_candidatos)
        if max_candidatos <= 0:
            return []

        candidatos: List[List[int]] = []
        for comb in self.gerar_combinacoes(k):
            candidatos.append(list(comb))
            if len(candidatos) >= max_candidatos:
                break

        return candidatos


# -----------------------------
# Teste rápido local
# -----------------------------
if __name__ == "__main__":
    # ajuste o caminho se quiser testar com histórico real
    hist = Path("lotofacil_ultimos_25_concursos.csv")
    gm = GrupoDeMilhoes(universo_max=25, historico_csv=hist if hist.exists() else None)

    print("sorteadas k=15:", gm.total_sorteadas(15))
    cands = gm.get_candidatos(k=20, max_candidatos=5)
    print("amostra k=20:", cands)
