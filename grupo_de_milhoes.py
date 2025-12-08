from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List, Set
import csv
import pickle
import random


# Cada jogo é uma tupla ordenada com 15 dezenas, ex: (1, 3, 5, ..., 25)
Combo = Tuple[int, ...]


# ------------------------------
# Funções de normalização
# ------------------------------

def normalizar_concurso(seq: Sequence[int]) -> Combo:
    """
    Recebe uma sequência de dezenas (qualquer ordem, str/int) e
    devolve uma tupla ordenada, ex: (1, 3, 7, ..., 25).
    """
    nums = [int(x) for x in seq]
    return tuple(sorted(nums))


def extrair_dezenas_de_linha(campos: Sequence[str]) -> List[int]:
    """
    Linha genérica de CSV. Varre todos os campos, pega só números 1..25.

    Serve mesmo que o CSV tenha:
      - colunas de concurso/data
      - dezenas separadas por espaço, ponto, hífen etc.
    """
    dezenas: List[int] = []
    for campo in campos:
        bruto = (
            campo.replace("-", " ")
                 .replace(".", " ")
                 .replace(";", " ")
                 .replace(",", " ")
        )
        for token in bruto.split():
            if token.isdigit():
                n = int(token)
                if 1 <= n <= 25:
                    dezenas.append(n)

    # se vier mais de 15, corta; se vier menos, essa linha será ignorada depois
    return dezenas[:15]


# ------------------------------
# Carregar histórico de sorteios
# ------------------------------

def carregar_sorteios_csv(
    caminho_csv: str | Path,
    pular_cabecalho: bool = True,
) -> Set[Combo]:
    """
    Lê um CSV com todos os sorteios reais e devolve um SET de tuplas (combo normalizado).

    Regras:
      - pega apenas números 1..25
      - se achar 15 dezenas em uma linha -> considera um concurso
      - linhas com menos de 15 dezenas são ignoradas
    """
    caminho = Path(caminho_csv)
    if not caminho.exists():
        raise FileNotFoundError(f"CSV não encontrado: {caminho}")

    sorteados: Set[Combo] = set()

    with caminho.open("r", newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        if pular_cabecalho:
            next(leitor, None)

        for linha in leitor:
            dezenas = extrair_dezenas_de_linha(linha)
            if len(dezenas) == 15:
                sorteados.add(normalizar_concurso(dezenas))

    return sorteados


# ------------------------------
# Universo total C(25, 15)
# ------------------------------

def gerar_universo_total() -> Iterable[Combo]:
    """
    Gera TODAS as combinações de 15 dezenas entre 1 e 25 (C(25,15)=3.268.760).
    Stream: não carrega tudo em memória de uma vez.
    """
    for comb in combinations(range(1, 26), 15):
        yield comb


# ------------------------------
# Construção do grupo de milhões
# ------------------------------

def construir_grupo_de_milhoes(
    csv_sorteios: str | Path,
    destino_pkl: str | Path = "grupo_de_milhoes.pkl",
    amostra: int | None = None,
) -> None:
    """
    - Lê o histórico de sorteios do CSV.
    - Gera o universo total.
    - Remove TUDO que já saiu.
    - Salva o grupo de milhões em um .pkl (lista de tuplas).

    IMPORTANTE:
      • Isso é pesado. A ideia é rodar LOCALMENTE (seu PC), NÃO no Render.
      • Depois que o .pkl estiver pronto, você sobe o arquivo pro GitHub.

    Parâmetro 'amostra':
      - Se None: salva o grupo completo (arquivo grande).
      - Se int: faz um sample aleatório do grupo (para testes).
    """
    csv_sorteios = Path(csv_sorteios)
    destino_pkl = Path(destino_pkl)

    print(f"[1/3] Carregando sorteios reais de {csv_sorteios} ...")
    sorteados = carregar_sorteios_csv(csv_sorteios)
    print(f"   -> {len(sorteados)} concursos únicos carregados.")

    print("[2/3] Construindo grupo de milhões (streaming)...")
    grupo: List[Combo] = []
    total_universo = 0
    restantes = 0

    for comb in gerar_universo_total():
        total_universo += 1
        if comb not in sorteados:
            grupo.append(comb)
            restantes += 1

    print(f"   Universo total        : {total_universo:,}")
    print(f"   Ainda não sorteados   : {restantes:,}")

    if amostra is not None and amostra < len(grupo):
        print(f"[opcional] Aplicando amostra aleatória de {amostra} jogos...")
        grupo = random.sample(grupo, amostra)

    print(f"[3/3] Salvando grupo em {destino_pkl} ...")
    with destino_pkl.open("wb") as f:
        pickle.dump(grupo, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Concluído.")


# ------------------------------
# Utilidade de leitura
# ------------------------------

def carregar_grupo_de_milhoes(path: str | Path) -> List[Combo]:
    """
    Lê o .pkl criado pela função construir_grupo_de_milhoes
    e devolve a lista de combinações.
    """
    path = Path(path)
    with path.open("rb") as f:
        grupo: List[Combo] = pickle.load(f)
    return grupo


# ------------------------------
# Interface de linha de comando (opcional)
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Constrói o grupo de milhões a partir de um CSV de sorteios reais."
    )
    parser.add_argument("csv_sorteios", help="caminho do CSV com todos os concursos reais")
    parser.add_argument("--destino", default="grupo_de_milhoes.pkl", help="arquivo .pkl de saída")
    parser.add_argument("--amostra", type=int, default=None, help="amostra aleatória opcional")
    args = parser.parse_args()

    construir_grupo_de_milhoes(
        csv_sorteios=args.csv_sorteios,
        destino_pkl=args.destino,
        amostra=args.amostra,
  )
