# fgi_engine.py
#
# Motor FGI operando sobre:
# - histórico real de concursos
# - grupo de milhões (combinações que AINDA NÃO saíram)
#
# AUTO:
# - carrega/grava grupo_de_milhoes.pkl
# - toda vez que /carregar recebe concursos,
#   essas combinações são removidas automaticamente
#   do grupo de milhões.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
import random

from grupo_de_milhoes import GrupoDeMilhoes


DEZENAS = list(range(1, 26))


@dataclass
class AnaliseEstado:
    total_concursos: int
    freq: Dict[int, int]
    frias: List[int]
    quentes: List[int]


class FGIMotor:
    def __init__(self) -> None:
        # histórico + estatísticas
        self.total_concursos: int = 0
        self.freq: Counter[int] = Counter()
        self.frias: List[int] = []
        self.quentes: List[int] = []
        self.historico: List[List[int]] = []

        # grupo de milhões (AUTO)
        self.grupo = GrupoDeMilhoes(auto_generate=True)

    # ----------------------------------------
    # Utilidades internas
    # ----------------------------------------
    @staticmethod
    def _normalizar_jogo(jogo: List[int]) -> List[int]:
        """Ordena, remove duplicadas e faz validação básica."""
        dezenas = sorted(set(int(d) for d in jogo))
        if len(dezenas) != 15:
            raise ValueError(f"Jogo inválido (precisa de 15 dezenas distintas): {jogo}")
        for d in dezenas:
            if d not in DEZENAS:
                raise ValueError(f"Dezena fora de [1,25]: {d}")
        return dezenas

    def _recalcular_frias_quentes(self) -> None:
        """Recalcula listas de frias e quentes a partir da frequência."""
        # garante todas as dezenas no dicionário
        freq_completa = {d: self.freq.get(d, 0) for d in DEZENAS}

        ordenados = sorted(freq_completa.items(), key=lambda x: x[1])
        # aqui é uma escolha de projeto: 8 frias / 8 quentes
        n = 8
        self.frias = [d for d, _ in ordenados[:n]]
        self.quentes = [d for d, _ in ordenados[-n:]]

    def _score_jogo(self, jogo: List[int]) -> float:
        """
        Score simples:
        - soma das frequências históricas
        - peso extra se usa quentes e frias
        """
        base = sum(self.freq.get(d, 0) for d in jogo)
        qtd_quentes = sum(1 for d in jogo if d in self.quentes)
        qtd_frias = sum(1 for d in jogo if d in self.frias)
        return base + 2.0 * qtd_quentes + 1.5 * qtd_frias

    # ----------------------------------------
    # API interna usada pelo FastAPI
    # ----------------------------------------
    def reset(self) -> None:
        self.total_concursos = 0
        self.freq = Counter()
        self.frias = []
        self.quentes = []
        self.historico = []

    def carregar(self, concursos: List[List[int]]) -> Dict:
        """
        Recebe uma janela de concursos (lista de jogos)
        atualiza estatísticas e remove essas combinações
        do grupo de milhões.
        """
        if not concursos:
            return self.resumo_basico()

        normalizados: List[List[int]] = []
        for jogo in concursos:
            dezenas = self._normalizar_jogo(jogo)
            normalizados.append(dezenas)

        # histórico + freq
        self.historico.extend(normalizados)
        self.total_concursos += len(normalizados)
        for dezenas in normalizados:
            self.freq.update(dezenas)

        # frias/quentes
        self._recalcular_frias_quentes()

        # AUTO: remove essas combinações do grupo de milhões
        try:
            self.grupo.remover_sorteadas(normalizados)
        except Exception:
            # não quebra a API se der algum problema no grupo
            pass

        return self.resumo_basico()

    def resumo_basico(self) -> Dict:
        return {
            "total_concursos": self.total_concursos,
            "freq": {d: self.freq.get(d, 0) for d in DEZENAS},
            "frias": self.frias,
            "quentes": self.quentes,
        }

    # ----------------------------------------
    # Geração de FGIs "finos"
    # ----------------------------------------
    def gerar_fgi_fino(
        self,
        n_resultados: int = 32,
        min_frias: int = 5,
        min_quentes: int = 4,
    ) -> List[Tuple[float, List[int]]]:
        """
        Gera N jogos candidatos a partir do grupo de milhões,
        filtrando por quentes/frias e ordenando pelo score.
        """

        if n_resultados <= 0:
            return []

        # se o grupo tiver vazio, não tem o que fazer
        if not self.grupo.combos:
            raise RuntimeError("Grupo de milhões vazio. Gere/atualize primeiro.")

        resultados: List[Tuple[float, List[int]]] = []
        vistos: set[Tuple[int, ...]] = set()

        max_tentativas = n_resultados * 800  # respiro
        tentativas = 0

        while len(resultados) < n_resultados and tentativas < max_tentativas:
            tentativas += 1
            amostra = self.grupo.sample(1)
            if not amostra:
                break

            jogo = amostra[0]
            key = tuple(jogo)
            if key in vistos:
                continue
            vistos.add(key)

            qtd_frias = sum(1 for d in jogo if d in self.frias)
            qtd_quentes = sum(1 for d in jogo if d in self.quentes)

            if qtd_frias < min_frias or qtd_quentes < min_quentes:
                continue

            score = self._score_jogo(jogo)
            resultados.append((score, jogo))

        resultados.sort(key=lambda x: x[0], reverse=True)
        return resultados[:n_resultados]

    def fgis_para_dicts(self, fgis: List[Tuple[float, List[int]]]) -> List[Dict]:
        """
        Converte a lista [(score, jogo), ...] para algo amigável em JSON.
        """
        out: List[Dict] = []
        for score, jogo in fgis:
            out.append(
                {
                    "jogo": jogo,
                    "score": score,
                }
            )
        return out
