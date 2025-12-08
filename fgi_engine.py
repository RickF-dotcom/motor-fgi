# fgi_engine.py
#
# Motor de análise + geração de FGIs com score.
#
# Agora integrado ao GrupoDeMilhoes:
#   - universo total 15/25 menos todas as já sorteadas
#   - /carregar remove concursos do grupo
#   - gerar_fgi_fino amostra FGIs desse grupo e escolhe as melhores
#
# API usada pelo app.py:
#   - FGIMotor()
#   - carregar(concursos) -> AnaliseEstado
#   - resumo_basico() -> dict
#   - gerar_fgi_fino(n_resultados, min_frias, min_quentes)
#   - fgis_para_dicts(fgis)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import statistics

from grupo_de_milhoes import GrupoDeMilhoes


# ----------------------------------------
# Estrutura principal do motor
# ----------------------------------------

@dataclass
class AnaliseEstado:
    """Snapshot da análise depois do /carregar.

    Usado pelo app como tipo interno; a API retorna um dict.
    """
    total_concursos: int
    freq: Dict[int, int]
    frias: List[int]
    quentes: List[int]


class FGIMotor:
    """
    Motor de análise + geração de FGIs com score.

    Fluxo:
      1) carregar(concursos)  -> calcula estatísticas e
                                remove sorteadas do grupo_de_milhoes
      2) gerar_fino(...)      -> gera N jogos ranqueados por score.

    concursos: lista de listas, cada jogo com 15 dezenas de 1 a 25.
    """

    def __init__(self) -> None:
        self.grupo = GrupoDeMilhoes()
        self.reset()

    # ----------------------------------------
    # Estado / inicialização
    # ----------------------------------------

    def reset(self) -> None:
        self.concursos: List[List[int]] = []
        self.total_concursos: int = 0
        self.freq: Dict[int, int] = {d: 0 for d in range(1, 26)}
        self.frias: List[int] = []
        self.quentes: List[int] = []
        self._max_freq: int = 0

    # ----------------------------------------
    # Carregamento dos concursos
    # ----------------------------------------

    def carregar(self, concursos: List[List[int]]) -> AnaliseEstado:
        """
        Atualiza o estado do motor com uma nova janela de concursos.

        - Limpa e normaliza a entrada.
        - Calcula estatísticas de frequência.
        - Define frias/quentes com base em quantis.
        - Remove essas combinações do grupo_de_milhoes.

        Retorna AnaliseEstado.
        """
        if not concursos:
            self.reset()
            return AnaliseEstado(0, self.freq.copy(), [], [])

        # Normaliza: tira duplicados dentro do mesmo jogo,
        # ordena e ignora dezenas fora de [1,25].
        concursos_norm: List[List[int]] = []
        for jogo in concursos:
            filtrado = sorted({d for d in jogo if 1 <= d <= 25})
            if len(filtrado) == 15:
                concursos_norm.append(filtrado)

        self.reset()
        self.concursos = concursos_norm
        self.total_concursos = len(concursos_norm)

        # Frequência absoluta
        for jogo in concursos_norm:
            for d in jogo:
                self.freq[d] += 1

        self._max_freq = max(self.freq.values()) if self.total_concursos > 0 else 0

        # Define frias/quentes via quantis da distribuição de frequências
        if self.total_concursos > 0:
            valores = list(self.freq.values())
            q1 = statistics.quantiles(valores, n=4)[0]  # ~25%
            q3 = statistics.quantiles(valores, n=4)[2]  # ~75%

            self.frias = [d for d, f in self.freq.items() if f <= q1]
            self.quentes = [d for d, f in self.freq.items() if f >= q3]
        else:
            self.frias = []
            self.quentes = []

        # Remove esses concursos do grupo de milhões
        try:
            self.grupo.remover_sorteadas(concursos_norm)
        except Exception:
            # Não derruba o motor se der algum problema de I/O
            pass

        return AnaliseEstado(
            total_concursos=self.total_concursos,
            freq=self.freq.copy(),
            frias=list(self.frias),
            quentes=list(self.quentes),
        )

    # ----------------------------------------
    # Exposição simples para a API
    # ----------------------------------------

    def resumo_basico(self) -> Dict:
        """
        Dict pronto para o /carregar devolver na API.
        """
        return {
            "total_concursos": self.total_concursos,
            "freq": {str(d): f for d, f in self.freq.items()},
            "frias": self.frias,
            "quentes": self.quentes,
        }

    # ----------------------------------------
    # Scoring e filtros
    # ----------------------------------------

    def _conta_frias_quentes(self, jogo: List[int]) -> Tuple[int, int]:
        set_frias = set(self.frias)
        set_quentes = set(self.quentes)
        c_f = sum(1 for d in jogo if d in set_frias)
        c_q = sum(1 for d in jogo if d in set_quentes)
        return c_f, c_q

    @staticmethod
    def _conta_consecutivos(jogo: List[int]) -> int:
        """Maior sequência de números consecutivos no jogo."""
        if not jogo:
            return 0
        max_run = 1
        run = 1
        for a, b in zip(jogo, jogo[1:]):
            if b == a + 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        return max_run

    @staticmethod
    def _paridade_balance(jogo: List[int]) -> int:
        """Diferença absoluta entre pares e ímpares."""
        pares = sum(1 for d in jogo if d % 2 == 0)
        impares = len(jogo) - pares
        return abs(pares - impares)

    @staticmethod
    def _soma_dezenas(jogo: List[int]) -> int:
        return sum(jogo)

    def _score_jogo(self, jogo: List[int]) -> float:
        """
        Score do jogo combinando várias heurísticas simples:

        - soma das frequências (quanto mais alinhado ao histórico, maior)
        - pequeno bônus para quentes, leve penalidade para frias
        - penalidade por padrão "feio": muitos consecutivos, paridade muito torta,
          soma de dezenas muito extrema.
        """
        base_freq = sum(self.freq[d] for d in jogo)

        c_f, c_q = self._conta_frias_quentes(jogo)

        # Bônus / penalidades por frias / quentes
        bonus_quentes = c_q * 1.5
        penalty_frias = c_f * 0.5

        # Penalidades estruturais
        max_consec = self._conta_consecutivos(jogo)
        paridade = self._paridade_balance(jogo)
        soma = self._soma_dezenas(jogo)

        # Soma ideal aproximada na Lotofácil (região central ~ 180–210)
        dist_soma = abs(soma - 195)

        penalty_pattern = (
            max(0, max_consec - 3) * 3.0 +  # sequências longas
            max(0, paridade - 3) * 1.5 +    # paridade muito torta
            dist_soma / 10.0                # soma muito fora do eixo
        )

        score = base_freq + bonus_quentes - penalty_frias - penalty_pattern
        return score

    # ----------------------------------------
    # Geração fina de FGIs
    # ----------------------------------------

    def gerar_fgi_fino(
        self,
        n_resultados: int = 32,
        min_frias: int = 5,
        min_quentes: int = 4,
    ) -> List[Dict]:
        """
        Gera N FGIs já filtradas e ranqueadas.

        - Amostra um pool de jogos do grupo_de_milhoes (bem maior que N).
        - Filtra pelo mínimo de frias/quentes.
        - Aplica heurísticas de padrão.
        - Ordena pelo score e devolve os top N.
        """
        if self.total_concursos == 0:
            return []

        # Pool de amostragem: fator x em relação ao N desejado
        fator_pool = 80  # você pode ajustar depois
        pool_size = max(n_resultados * fator_pool, n_resultados * 10)

        candidatos = self.grupo.sample(pool_size)
        if not candidatos:
            return []

        fgis: List[Tuple[float, List[int], int, int]] = []

        for jogo in candidatos:
            jogo = sorted(jogo)

            c_f, c_q = self._conta_frias_quentes(jogo)
            if c_f < min_frias or c_q < min_quentes:
                continue

            # filtro de padrões básicos
            if self._conta_consecutivos(jogo) > 4:
                continue

            if self._paridade_balance(jogo) > 6:
                continue

            score = self._score_jogo(jogo)
            fgis.append((score, jogo, c_f, c_q))

        # Ordena por score decrescente
        fgis.sort(key=lambda x: x[0], reverse=True)
        fgis = fgis[:n_resultados]

        # Converte para estrutura de alto nível para a API
        resultado: List[Dict] = []
        for score, jogo, c_f, c_q in fgis:
            resultado.append(
                {
                    "dezenas": jogo,
                    "score": round(score, 3),
                    "qtd_frias": c_f,
                    "qtd_quentes": c_q,
                }
            )

        return resultado

    # ----------------------------------------
    # Utilitário para a API
    # ----------------------------------------

    @staticmethod
    def fgis_para_dicts(fgis: List[Dict]) -> List[Dict]:
        """
        Hoje os FGIs já estão em forma de dict, então apenas retorna.
        Mantemos esse método para compatibilidade com o app.py.
        """
        return fgis
