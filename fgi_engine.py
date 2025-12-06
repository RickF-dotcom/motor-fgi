fgi_engine.py: |
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
          self.freq: Dict[int, int] = {}
          self.ausentes: List[int] = []
          self.concursos_set: Set[Tuple[int, ...]] = set()

      def carregar_concursos(self, concursos: List[List[int]]) -> None:
          self.concursos = [sorted([int(d) for d in c]) for c in concursos]
          self.concursos_set = {tuple(c) for c in self.concursos}
          self._calcular_frequencias()
          self._identificar_ausentes()

      def _calcular_frequencias(self) -> None:
          self.freq = {d: 0 for d in range(1, self.total_dezenas + 1)}
          for concurso in self.concursos:
              for d in concurso:
                  if d in self.freq:
                      self.freq[d] += 1

      def _identificar_ausentes(self) -> None:
          todas = set(range(1, self.total_dezenas + 1))
          usados = set()
          for concurso in self.concursos:
              usados.update(concurso)
          self.ausentes = sorted(list(todas - usados))

      def gerar_fgis(self, n_fgis: int = 30, min_ausentes: int = 4, max_tentativas: int = 50000):
          todas_dezenas = list(range(1, self.total_dezenas + 1))
          fgis = []
          fgi_vistas = set()

          tentativas = 0
          while len(fgis) < n_fgis and tentativas < max_tentativas:
              tentativas += 1

              base = []
              if self.ausentes:
                  qtd_seed = min(len(self.ausentes), min_ausentes + 1)
                  seed = random.sample(self.ausentes, qtd_seed)
                  base.extend(seed)

              restantes = [d for d in todas_dezenas if d not in base]
              random.shuffle(restantes)
              while len(base) < self.dezenas_por_concurso and restantes:
                  base.append(restantes.pop())

              base = sorted(base)
              chave = tuple(base)

              if chave in self.concursos_set:
                  continue
              if chave in fgi_vistas:
                  continue

              aus = [d for d in base if d in self.ausentes]
              if len(aus) < min_ausentes:
                  continue

              soma_freq = sum(self.freq.get(d, 0) for d in base)
              score = len(aus) * 1000 - soma_freq

              fgi_res = FGIResult(
                  id=len(fgis) + 1,
                  numeros=base,
                  ausentes=aus,
                  qtd_ausentes=len(aus),
                  score=score
              )
              fgis.append(fgi_res)
              fgi_vistas.add(chave)

          fgis.sort(key=lambda x: x.score, reverse=True)
          return fgis

      def resumo_basico(self) -> Dict:
          return {
              "total_concursos": len(self.concursos),
              "freq": self.freq,
              "ausentes": self.ausentes,
          }

      def fgis_para_dicts(self, fgis):
          return [asdict(f) for f in fgis]
