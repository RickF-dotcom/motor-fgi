# fgi_engine.py
#
# Motor de FGIs com:
# - estatísticas finas dos últimos concursos
# - integração com o "grupo de milhões" (combinações não sorteadas)
#
# Este arquivo foi pensado para funcionar junto com:
#   - grupo_de_milhoes.py
#   - app.py (API FastAPI)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple
import statistics
import random

from grupo_de_milhoes import GrupoDeMilhoes


# -------------------------------------------------------------------
# Estruturas de dados
# -------------------------------------------------------------------


@dataclass
class AnaliseEstado:
    total_concursos: int
    freq: Dict[int, int]
    frias: List[int]
    quentes: List[int]


# -------------------------------------------------------------------
# FGIMotor
# -------------------------------------------------------------------


class FGIMotor:
    """
    Motor responsável por:
      - receber concursos reais
      - calcular estatísticas finas (freq, frias, quentes)
      - conversar com o GrupoDeMilhoes para remover sorteadas
      - gerar FGIs filtradas/ranqueadas
    """

    def __init__(self) -> None:
        # estatística dos últimos concursos carregados
        self.total_concursos: int = 0
        self.freq: Dict[int, int] = {d: 0 for d in range(1, 26)}
        self.frias: List[int] = []
        self.quentes: List[int] = []

        # janela de concursos em memória (na ordem em que chegam)
        self.concursos: List[List[int]] = []

        # grupo de milhões (universo de combinações NÃO sorteadas)
        # auto_generate=True → se não existir .pkl ele gera
        self.grupo: GrupoDeMilhoes = GrupoDeMilhoes(auto_generate=True)

    # ------------------------------------------------------------------
    # Normalização / validação de concursos
    # ------------------------------------------------------------------

    @staticmethod
    def _normaliza_jogo(jogo: Iterable[int]) -> List[int]:
        """
        Garante:
          - inteiros
          - entre 1 e 25
          - sem repetição
          - ordenado
        """
        try:
            dezenas = [int(d) for d in jogo]
        except Exception:
            raise ValueError(f"Jogo inválido (não numérico): {jogo}")

        if len(dezenas) != 15:
            raise ValueError(f"Jogo deve ter 15 dezenas, recebeu {len(dezenas)}: {jogo}")

        if any(d < 1 or d > 25 for d in dezenas):
            raise ValueError(f"Dezenas devem estar entre 1 e 25: {jogo}")

        if len(set(dezenas)) != 15:
            raise ValueError(f"Jogo possui dezenas repetidas: {jogo}")

        return sorted(dezenas)

    # ------------------------------------------------------------------
    # Carregamento de concursos + estatísticas finas
    # ------------------------------------------------------------------

    def carregar(self, concursos: List[List[int]]) -> AnaliseEstado:
        """
        Recebe uma janela de concursos e atualiza:
          - total_concursos
          - freq
          - frias / quentes
          - lista self.concursos
        Também remove esses concursos do grupo de milhões.
        """
        if not concursos:
            raise ValueError("Lista de concursos vazia.")

        concursos_norm: List[List[int]] = []

        # acumula freq e normaliza concursos
        for jogo in concursos:
            jogo_norm = self._normaliza_jogo(jogo)
            concursos_norm.append(jogo_norm)

            self.concursos.append(jogo_norm)
            self.total_concursos += 1

            for d in jogo_norm:
                self.freq[d] += 1

        # define frias/quentes via quantis da distribuição de frequências
        if self.total_concursos > 0:
            valores = list(self.freq.values())
            # quantis 25% e 75%
            q1 = statistics.quantiles(valores, n=4)[0]
            q3 = statistics.quantiles(valores, n=4)[2]

            self.frias = [d for d, f in self.freq.items() if f <= q1]
            self.quentes = [d for d, f in self.freq.items() if f >= q3]
        else:
            self.frias = []
            self.quentes = []

        # integra com o grupo de milhões (remove sorteadas do universo)
        try:
            self.grupo.remover_sorteadas(concursos_norm)
        except Exception:
            # se der problema de I/O ou outro erro, não derruba o motor
            pass

        return AnaliseEstado(
            total_concursos=self.total_concursos,
            freq=self.freq.copy(),
            frias=list(self.frias),
            quentes=list(self.quentes),
        )

    # wrapper para a API (o app.py usa ESTE nome)
    def carregar_concursos(self, concursos: List[List[int]]) -> AnaliseEstado:
        """
        Interface usada pela API. Mantém o nome explícito para clareza.
        """
        return self.carregar(concursos)

    # ------------------------------------------------------------------
    # Exposição simples de resumo (se quiser usar em outros pontos)
    # ------------------------------------------------------------------

    def resumo_basico(self) -> Dict[str, object]:
        """
        Dict pronto para serializar em JSON.
        """
        return {
            "total_concursos": self.total_concursos,
            "freq": {str(d): f for d, f in self.freq.items()},
            "frias": self.frias,
            "quentes": self.quentes,
        }

    # ------------------------------------------------------------------
    # Scoring e filtros para geração de FGIs
    # ------------------------------------------------------------------

    def _conta_frias_quentes(self, jogo: List[int]) -> Tuple[int, int]:
        set_frias = set(self.frias)
        set_quentes = set(self.quentes)

        c_f = sum(1 for d in jogo if d in set_frias)
        c_q = sum(1 for d in jogo if d in set_quentes)

        return c_f, c_q

    @staticmethod
    def _conta_consecutivos(jogo: List[int]) -> int:
        """
        Maior sequência de dezenas consecutivas dentro do jogo.
        """
        if not jogo:
            return 0

        ordenado = sorted(jogo)
        max_run = 1
        run = 1

        for i in range(1, len(ordenado)):
            if ordenado[i] == ordenado[i - 1] + 1:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 1

        return max_run

    def _score_jogo(self, jogo: List[int]) -> float:
        """
        Score simples combinando:
          - quantidade de frias/quentes
          - controle leve de consecutivos
        Quanto maior o score, mais "interessante" para estudo.
        """
        c_f, c_q = self._conta_frias_quentes(jogo)
        consec = self._conta_consecutivos(jogo)

        # pesos simples – você pode calibrar depois
        score = (
            c_f * 1.2 +
            c_q * 1.0 -
            max(0, consec - 4) * 0.5
        )
        return score

    # ------------------------------------------------------------------
    # Geração de FGIs refinadas
    # ------------------------------------------------------------------

    def gerar_fino(
        self,
        n: int = 32,
        min_frias: int = 5,
        min_quentes: int = 4,
    ) -> List[List[int]]:
        """
        Gera N jogos usando:
          - universo do GrupoDeMilhoes (não sorteados)
          - filtros de min_frias / min_quentes
          - score para ordenar

        Se o grupo de milhões estiver vazio, levanta erro.
        """
        if not self.grupo.combos:
            raise ValueError(
                "Grupo de milhões está vazio. "
                "Garanta que o arquivo .pkl foi gerado."
            )

        escolhidos: List[Tuple[float, List[int]]] = []
        tentativas = 0
        limite_tentativas = max(500, n * 50)

        while len(escolhidos) < n and tentativas < limite_tentativas:
            tentativas += 1

            # pega 1 jogo aleatório do grupo de milhões
            jogo = self.grupo.sample(1)[0]

            c_f, c_q = self._conta_frias_quentes(jogo)

            if c_f < min_frias or c_q < min_quentes:
                continue

            score = self._score_jogo(jogo)
            escolhidos.append((score, jogo))

        if not escolhidos:
            raise ValueError(
                "Não foi possível gerar jogos que respeitem os filtros "
                f"(min_frias={min_frias}, min_quentes={min_quentes})."
            )

        # ordena por score (desc) e devolve só os jogos
        escolhidos.sort(key=lambda x: x[0], reverse=True)
        jogos_ordenados = [j for _, j in escolhidos[:n]]

        return jogos_ordenados
