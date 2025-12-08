from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import math


# ----------------------------
# Estrutura principal do motor
# ----------------------------

@dataclass
class AnaliseEstado:
    """
    Snapshot da análise depois do /carregar.
    Usado só como tipo interno; o app FastAPI retorna um dict.
    """
    total_concursos: int
    freq: Dict[int, int]
    frias: List[int]
    quentes: List[int]


class FGIMotor:
    """
    Motor de análise + geração de FGIs com score.

    Fluxo:
      1) carregar(concursos)  -> calcula estatísticas.
      2) gerar_fino(...)      -> gera N jogos ranqueados por score.

    - concursos: lista de listas, cada jogo com 15 dezenas de 1 a 25.
    """

    def __init__(self) -> None:
        self.reset()

    # ----------------------
    # Estado / inicialização
    # ----------------------
    def reset(self) -> None:
        self.concursos: List[List[int]] = []
        self.total_concursos: int = 0

        # frequência absoluta por dezena
        self.freq: Dict[int, int] = {d: 0 for d in range(1, 26)}

        # índice do último concurso em que cada dezena saiu
        self.ultima_ocorrencia: Dict[int, int | None] = {d: None for d in range(1, 26)}

        # frias/quentes calculadas após carregar()
        self.frias: List[int] = []
        self.quentes: List[int] = []

        # cachezinhos
        self._max_freq: int = 0

    # -------------------------
    # Carregamento dos concursos
    # -------------------------
    def carregar(self, concursos: List[List[int]]) -> Dict:
        """
        Atualiza o estado do motor com uma nova janela de concursos.

        - Remove duplicadas dentro do mesmo jogo.
        - Ignora dezenas fora de [1, 25].

        Retorna um dict pronto para ser devolvido no /carregar:
        {
           "total_concursos": ...,
           "freq": { "1": x, ... },
           "frias": [ ... ],
           "quentes": [ ... ]
        }
        """
        self.reset()

        # normaliza entrada
        concursos_normalizados: List[List[int]] = []
        for jogo in concursos:
            if not jogo:
                continue
            # remove duplicadas, mantém apenas dezenas válidas
            limpo = sorted({d for d in jogo if 1 <= d <= 25})
            if len(limpo) == 15:
                concursos_normalizados.append(limpo)

        self.concursos = concursos_normalizados
        self.total_concursos = len(self.concursos)

        # frequência e última ocorrência
        for idx, jogo in enumerate(self.concursos):
            for dez in jogo:
                self.freq[dez] += 1
                self.ultima_ocorrencia[dez] = idx

        self._max_freq = max(self.freq.values()) if self.total_concursos > 0 else 0

        # define frias / quentes por percentil simples (25% mais baixas / 25% mais altas)
        if self.total_concursos > 0:
            ordenado = sorted(self.freq.items(), key=lambda kv: kv[1])  # (dez, freq)
            k = max(1, 25 // 4)  # 25% de 25 -> 6, mas garante pelo menos 1
            frias_pairs = ordenado[:k]
            quentes_pairs = ordenado[-k:]

            self.frias = sorted([d for d, _ in frias_pairs])
            self.quentes = sorted([d for d, _ in quentes_pairs])
        else:
            self.frias = []
            self.quentes = []

        # formato de saída compatível com o app atual
        return {
            "total_concursos": self.total_concursos,
            "freq": {str(d): self.freq[d] for d in range(1, 26)},
            "frias": self.frias,
            "quentes": self.quentes,
        }

    # -----------------------
    # Geração com score (FGI)
    # -----------------------
    def gerar_fino(
        self,
        n: int,
        min_frias: int = 5,
        min_quentes: int = 4,
    ) -> List[Dict]:
        """
        Gera N jogos de 15 dezenas com seleção por score.

        - Cada jogo obedece às restrições mínimas de frias/quentes.
        - Entre todos os candidatos simulados, retornamos os N com melhor score.
        """
        if self.total_concursos == 0:
            raise ValueError("Motor vazio: chame carregar() antes de gerar jogos.")

        # quantidade de amostras (candidatos) para disputar ranking
        max_amostras = max(2000, n * 250)

        # pseudo-determinismo suave
        seed_base = self.total_concursos + sum(self.freq.values())
        random.seed(seed_base)

        candidatos: List[Tuple[float, List[int]]] = []

        alvo_frias = max(0, min_frias)
        alvo_quentes = max(0, min_quentes)
        set_frias = set(self.frias)
        set_quentes = set(self.quentes)

        universo = list(range(1, 26))

        for _ in range(max_amostras):
            jogo = sorted(random.sample(universo, 15))

            # restrições duras mínimas de frias/quentes
            qtd_frias = len(set(jogo) & set_frias)
            qtd_quentes = len(set(jogo) & set_quentes)

            if qtd_frias < alvo_frias:
                continue
            if qtd_quentes < alvo_quentes:
                continue

            score = self._score_jogo(jogo, alvo_frias, alvo_quentes)
            candidatos.append((score, jogo))

        # ordena do melhor pro pior
        candidatos.sort(key=lambda x: x[0], reverse=True)

        # remove duplicados, monta saída
        resultado: List[Dict] = []
        vistos = set()

        for score, jogo in candidatos:
            chave = tuple(jogo)
            if chave in vistos:
                continue
            vistos.add(chave)

            ausentes = [d for d in universo if d not in jogo]

            resultado.append(
                {
                    "id": len(resultado) + 1,
                    "score": round(score, 5),
                    "numeros": jogo,
                    "ausentes": ausentes,
                }
            )

            if len(resultado) >= n:
                break

        return resultado

    # -----------------
    # Funções de score
    # -----------------
    def _score_jogo(
        self,
        jogo: List[int],
        alvo_frias: int,
        alvo_quentes: int,
    ) -> float:
        """
        Score composto para um jogo.
        """
        if self._max_freq == 0:
            return 0.0

        # 1) Frequência moderada
        freqs_norm = [self.freq[d] / self._max_freq for d in jogo]
        freq_score = 1.0 - self._desvio_medio(freqs_norm, alvo=0.5)

        # 2) Recência
        distancias_norm = []
        for d in jogo:
            ultima = self.ultima_ocorrencia.get(d)
            if ultima is None:
                dist = self.total_concursos
            else:
                dist = self.total_concursos - 1 - ultima
            dist_norm = dist / max(1, self.total_concursos - 1)
            distancias_norm.append(dist_norm)

        recency_score = 1.0 - self._desvio_medio(distancias_norm, alvo=0.6)

        # 3) Frias / quentes
        set_frias = set(self.frias)
        set_quentes = set(self.quentes)
        qtd_frias = len(set(jogo) & set_frias)
        qtd_quentes = len(set(jogo) & set_quentes)

        frias_score = self._score_por_alvo(qtd_frias, alvo_frias)
        quentes_score = self._score_por_alvo(qtd_quentes, alvo_quentes)

        # 4) Equilíbrio
        balance_score = self._score_balance(jogo)

        # 5) Penalidade de padrões ruins
        pattern_penalty = self._pattern_penalty(jogo)

        score = (
            0.30 * freq_score
            + 0.25 * recency_score
            + 0.15 * frias_score
            + 0.15 * quentes_score
            + 0.15 * balance_score
            - pattern_penalty
        )

        return float(score)

    @staticmethod
    def _desvio_medio(valores: List[float], alvo: float) -> float:
        if not valores:
            return 0.0
        desvios = [abs(v - alvo) for v in valores]
        return sum(desvios) / len(desvios)

    @staticmethod
    def _score_por_alvo(qtd: int, alvo: int) -> float:
        if alvo <= 0:
            return 1.0
        diff = abs(qtd - alvo)
        return max(0.0, 1.0 - diff / max(1.0, float(alvo)))

    @staticmethod
    def _score_balance(jogo: List[int]) -> float:
        if not jogo:
            return 0.0

        # regiões
        r1 = sum(1 for d in jogo if 1 <= d <= 8)
        r2 = sum(1 for d in jogo if 9 <= d <= 17)
        r3 = sum(1 for d in jogo if 18 <= d <= 25)

        alvo_regiao = len(jogo) / 3.0
        desv_regioes = (
            abs(r1 - alvo_regiao) + abs(r2 - alvo_regiao) + abs(r3 - alvo_regiao)
        ) / (3 * alvo_regiao)

        # par/ímpar
        pares = sum(1 for d in jogo if d % 2 == 0)
        impares = len(jogo) - pares
        alvo_par_impar = len(jogo) / 2.0
        desv_par_impar = (
            abs(pares - alvo_par_impar) + abs(impares - alvo_par_impar)
        ) / (2 * alvo_par_impar)

        regiao_score = max(0.0, 1.0 - desv_regioes)
        par_impar_score = max(0.0, 1.0 - desv_par_impar)

        return (regiao_score + par_impar_score) / 2.0

    @staticmethod
    def _pattern_penalty(jogo: List[int]) -> float:
        if not jogo:
            return 0.0

        # sequências consecutivas
        seq_len_max = 1
        seq_atual = 1
        for a, b in zip(jogo, jogo[1:]):
            if b == a + 1:
                seq_atual += 1
                seq_len_max = max(seq_len_max, seq_atual)
            else:
                seq_atual = 1

        penalty_seq = 0.0
        if seq_len_max >= 4:
            penalty_seq += 0.10
        if seq_len_max >= 5:
            penalty_seq += 0.05

        # concentração em faixa estreita
        amplitude = max(jogo) - min(jogo)
        penalty_faixa = 0.0
        if amplitude < 12:
            penalty_faixa = 0.10
        elif amplitude < 16:
            penalty_faixa = 0.05

        return penalty_seq + penalty_faixa
