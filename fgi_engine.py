import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple


@dataclass
class FGIResult:
    id: int
    numeros: List[int]
    ausentes: List[int]
    qtd_ausentes: int
    score: float


class FGIMotor:
    def __init__(self, total_dezenas: int = 25, dezenas_por_concurso: int = 15):
        self.total_dezenas = total_dezenas
        self.dezenas_por_concurso = dezenas_por_concurso
        self.concursos: List[List[int]] = []
        self.concursos_set: Set[Tuple[int, ...]] = set()
        self.concursos_sets: List[Set[int]] = []
        self.freq: Dict[int, int] = {}
        self.frias: List[int] = []
        self.quentes: List[int] = []

    # ---------------------------
    # CARGA E ESTATÍSTICA BÁSICA
    # ---------------------------

    def carregar_concursos(self, concursos: List[List[int]]) -> None:
        # cada concurso vem como lista de dezenas
        self.concursos = [sorted(int(d) for d in c) for c in concursos]
        self.concursos_set = {tuple(c) for c in self.concursos}
        self.concursos_sets = [set(c) for c in self.concursos]

        self._calcular_frequencias()
        self._identificar_frias_quentes()

    def _calcular_frequencias(self) -> None:
        self.freq = {d: 0 for d in range(1, self.total_dezenas + 1)}
        for concurso in self.concursos:
            for d in concurso:
                if d in self.freq:
                    self.freq[d] += 1

    def _identificar_frias_quentes(self) -> None:
        """Define frias e quentes com base na média da janela."""
        if not self.concursos:
            self.frias = []
            self.quentes = []
            return

        total_concursos = len(self.concursos)
        # média teórica de aparições por dezena na janela
        media = (total_concursos * self.dezenas_por_concurso) / self.total_dezenas

        frias = []
        quentes = []
        for d, f in self.freq.items():
            # frias: pelo menos 1 abaixo da média
            if f <= media - 1:
                frias.append(d)
            # quentes: pelo menos 1 acima da média
            elif f >= media + 1:
                quentes.append(d)

        self.frias = sorted(frias)
        self.quentes = sorted(quentes)

    # ---------------------------
    # GERAÇÃO "GROSSA" + FILTROS
    # ---------------------------

    def _score_jogo(self, numeros: List[int], qtd_frias: int, qtd_quentes: int) -> float:
        """Score combinando frias, quentes e soma de frequências."""
        soma_freq = sum(self.freq.get(d, 0) for d in numeros)

        score = (
            50 * qtd_frias +      # recompensa frias
            30 * qtd_quentes -    # recompensa quentes
            2 * soma_freq         # penaliza dezenas muito sorteadas
        )
        return float(score)

    def gerar_fgi_fino(
        self,
        n_resultados: int = 32,
        max_tentativas: int = 200_000,
        min_frias: int = 5,
        min_quentes: int = 4,
        min_pares: int = 6,
        max_pares: int = 9,
        min_baixas: int = 6,
        max_baixas: int = 9,
        max_iguais_concurso: int = 12,
    ) -> List[FGIResult]:
        """
        Gera poucos jogos já peneirados por:
        - pares/ímpares
        - baixas/altas
        - quantidade mínima de frias e quentes
        - distância dos concursos recentes
        """

        if not self.concursos:
            raise ValueError("Nenhum concurso carregado. Chame /carregar antes de gerar.")

        todas_dezenas = list(range(1, self.total_dezenas + 1))
        fgis: List[FGIResult] = []
        vistos: Set[Tuple[int, ...]] = set()

        tentativas = 0
        while len(fgis) < n_resultados and tentativas < max_tentativas:
            tentativas += 1

            # gera uma combinação aleatória
            numeros = sorted(random.sample(todas_dezenas, self.dezenas_por_concurso))
            chave = tuple(numeros)

            # descarta se já vimos ou se é igual a algum concurso real
            if chave in vistos or chave in self.concursos_set:
                continue

            # ---- filtros estruturais ----
            pares = sum(1 for d in numeros if d % 2 == 0)
            if not (min_pares <= pares <= max_pares):
                continue

            baixas = sum(1 for d in numeros if d <= 13)
            if not (min_baixas <= baixas <= max_baixas):
                continue

            qtd_frias = sum(1 for d in numeros if d in self.frias)
            if qtd_frias < min_frias:
                continue

            qtd_quentes = sum(1 for d in numeros if d in self.quentes)
            if qtd_quentes < min_quentes:
                continue

            # ---- filtro de semelhança aos concursos recentes ----
            muito_parecido = False
            set_numeros = set(numeros)
            for c_set in self.concursos_sets:
                comuns = len(set_numeros & c_set)
                if comuns >= max_iguais_concurso:
                    muito_parecido = True
                    break

            if muito_parecido:
                continue

            # passou em tudo -> calcula score
            score = self._score_jogo(numeros, qtd_frias, qtd_quentes)
            ausentes = sorted(list(set(todas_dezenas) - set_numeros))

            fgi_res = FGIResult(
                id=len(fgis) + 1,
                numeros=numeros,
                ausentes=ausentes,
                qtd_ausentes=len(ausentes),
                score=score,
            )

            fgis.append(fgi_res)
            vistos.add(chave)

        # ordena: melhor score primeiro
        fgis.sort(key=lambda x: x.score, reverse=True)
        return fgis

    # ---------------------------
    # SAÍDAS SIMPLES
    # ---------------------------

    def resumo_basico(self) -> Dict:
        return {
            "total_concursos": len(self.concursos),
            "freq": self.freq,
            "frias": self.frias,
            "quentes": self.quentes,
        }

    def fgis_para_dicts(self, fgis: List[FGIResult]):
        return [asdict(f) for f in fgis]
