from __future__ import annotations

import os
from typing import List, NamedTuple, Tuple, Set, Dict, Any

import yaml


# ============================================================
#  LOAD DO ARQUIVO lab_config.yaml (independente do app.py)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_CONFIG_PATH = os.path.join(BASE_DIR, "lab_config.yaml")

if not os.path.exists(LAB_CONFIG_PATH):
    raise FileNotFoundError(f"Arquivo de configuração não encontrado: {LAB_CONFIG_PATH}")

with open(LAB_CONFIG_PATH, "r", encoding="utf-8") as f:
    LAB_CONFIG: Dict[str, Any] = yaml.safe_load(f)


# ============================================================
#  TIPOS BÁSICOS: Constraints e ScoreDetalhado
# ============================================================

class Constraints(NamedTuple):
    soma_min: int
    soma_max: int

    min_1_13: int
    max_1_13: int
    min_14_25: int
    max_14_25: int

    moldura_min: int
    moldura_max: int
    miolo_min: int
    miolo_max: int

    pares_proibidos: Set[Tuple[int, int]]
    trios_proibidos: Set[Tuple[int, int, int]]
    numeros_bloqueados: Set[int]


class ScoreDetalhado(NamedTuple):
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


# ============================================================
#  ENGINE PONTO C (esqueleto inicial)
# ============================================================

class PontoCEngine:
    """
    Versão 0.1: esqueleto do PONTO C.

    - Lê regimes e parâmetros do lab_config.yaml
    - Constrói Constraints básicos a partir do regime escolhido
    - Calcula um score simples de coerência de uma sequência com essas Constraints

    Depois, esse engine será enriquecido com:
      - grafo real de vínculos N/D2/D3/features/metodologias
      - cálculo de energia baseado nos vínculos
    """

    def __init__(self, lab_config: Dict[str, Any] | None = None) -> None:
        if lab_config is None:
            lab_config = LAB_CONFIG

        self.config = lab_config
        self.regimes = {r["id"]: r for r in lab_config.get("regimes", [])}
        self.ponto_c_cfg = lab_config.get("ponto_c", {})
        self.motor_fgi_cfg = lab_config.get("motor_fgi", {})
        self.estado_cfg = lab_config.get("estado", {})

        self.tamanho_jogo_default: int = int(
            self.motor_fgi_cfg.get("tamanho_jogo_padrao", 15)
        )

    # --------------------------------------------------------
    #  PUBLIC: obter constraints para um regime
    # --------------------------------------------------------
    def get_constraints(self, regime_id: str) -> Constraints:
        """
        Constrói um pacote de constraints "duros" a partir:

        - do regime (tipicamente faixa de soma)
        - de defaults do lab_config

        Nesta versão, pares/trios proibidos e numeros_bloqueados
        ainda vêm vazios (serão alimentados depois pelo grafo C real).
        """

        if regime_id not in self.regimes:
            raise ValueError(f"Regime '{regime_id}' não encontrado em lab_config.yaml")

        regime = self.regimes[regime_id]
        criterio = regime.get("criterio", {})

        # Faixa de soma vinda do regime; se não tiver, usa algo amplo
        soma_min = int(criterio.get("soma_min", 130))
        soma_max = int(criterio.get("soma_max", 230))

        # Defaults grosseiros de distribuição 1–13 / 14–25
        # (Depois isso deve ser calibrado com estatística real)
        tam = self.tamanho_jogo_default
        min_1_13 = max(0, tam // 3)      # mínimo ~1/3 do jogo
        max_1_13 = min(tam, tam - 3)     # pelo menos 3 no 14–25
        min_14_25 = max(0, tam // 3)
        max_14_25 = tam - min_1_13       # complementa

        # Moldura/miolo ainda sem refinamento. Deixa faixa larga.
        moldura_min = 0
        moldura_max = tam
        miolo_min = 0
        miolo_max = tam

        pares_proibidos: Set[Tuple[int, int]] = set()
        trios_proibidos: Set[Tuple[int, int, int]] = set()
        numeros_bloqueados: Set[int] = set()

        return Constraints(
            soma_min=soma_min,
            soma_max=soma_max,
            min_1_13=min_1_13,
            max_1_13=max_1_13,
            min_14_25=min_14_25,
            max_14_25=max_14_25,
            moldura_min=moldura_min,
            moldura_max=moldura_max,
            miolo_min=miolo_min,
            miolo_max=miolo_max,
            pares_proibidos=pares_proibidos,
            trios_proibidos=trios_proibidos,
            numeros_bloqueados=numeros_bloqueados,
        )

    # --------------------------------------------------------
    #  PUBLIC: score de coerência de uma sequência com as constraints
    # --------------------------------------------------------
    def score_sequence(self, seq: List[int], regime_id: str) -> ScoreDetalhado:
        """
        Calcula um score simples de coerência da sequência 'seq'
        com as constraints derivadas do regime escolhido.

        Versão 0.1: o score é baseado em:
          - soma dentro/fora da faixa
          - distribuição 1–13 vs 14–25
          - presença de números bloqueados
          - presença de pares/trios proibidos

        Isso é um "stub" que mais tarde será substituído
        por um cálculo de energia baseado no grafo C completo.
        """

        constraints = self.get_constraints(regime_id)
        detalhes: Dict[str, Any] = {}
        coerencias = 0
        violacoes = 0

        # ----------------------------------------------------
        # Soma
        # ----------------------------------------------------
        soma_seq = sum(seq)
        dentro_faixa_soma = constraints.soma_min <= soma_seq <= constraints.soma_max
        detalhes["soma"] = {
            "valor": soma_seq,
            "min": constraints.soma_min,
            "max": constraints.soma_max,
            "coerente": dentro_faixa_soma,
        }
        if dentro_faixa_soma:
            coerencias += 1
        else:
            violacoes += 1

        # ----------------------------------------------------
        # Distribuição 1–13 vs 14–25
        # ----------------------------------------------------
        c_1_13 = sum(1 for x in seq if 1 <= x <= 13)
        c_14_25 = len(seq) - c_1_13

        ok_1_13 = constraints.min_1_13 <= c_1_13 <= constraints.max_1_13
        ok_14_25 = constraints.min_14_25 <= c_14_25 <= constraints.max_14_25

        detalhes["faixa_1_13_14_25"] = {
            "contagem_1_13": c_1_13,
            "contagem_14_25": c_14_25,
            "min_1_13": constraints.min_1_13,
            "max_1_13": constraints.max_1_13,
            "min_14_25": constraints.min_14_25,
            "max_14_25": constraints.max_14_25,
            "coerente_1_13": ok_1_13,
            "coerente_14_25": ok_14_25,
        }

        coerencias += int(ok_1_13) + int(ok_14_25)
        violacoes += int(not ok_1_13) + int(not ok_14_25)

        # ----------------------------------------------------
        # Números bloqueados
        # ----------------------------------------------------
        bloqueados_presentes = [x for x in seq if x in constraints.numeros_bloqueados]
        detalhes["numeros_bloqueados"] = {
            "bloqueados": list(constraints.numeros_bloqueados),
            "presentes_na_seq": bloqueados_presentes,
        }
        if bloqueados_presentes:
            violacoes += len(bloqueados_presentes)
        else:
            # se não há bloqueados na sequência, conta como 1 coerência
            coerencias += 1

        # ----------------------------------------------------
        # Pares proibidos
        # ----------------------------------------------------
        pares = {
            (min(a, b), max(a, b))
            for i, a in enumerate(seq)
            for b in seq[i + 1 :]
        }
        pares_proibidos_usados = [
            p for p in pares if p in constraints.pares_proibidos
        ]
        detalhes["pares_proibidos"] = {
            "proibidos": [list(p) for p in constraints.pares_proibidos],
            "presentes_na_seq": [list(p) for p in pares_proibidos_usados],
        }
        if pares_proibidos_usados:
            violacoes += len(pares_proibidos_usados)
        else:
            coerencias += 1

        # ----------------------------------------------------
        # Trios proibidos (ainda vazio, mas estrutura pronta)
        # ----------------------------------------------------
        trios = {
            tuple(sorted((a, b, c)))
            for i, a in enumerate(seq)
            for j, b in enumerate(seq[i + 1 :], start=i + 1)
            for c in seq[j + 1 :]
        }
        trios_proibidos_usados = [
            t for t in trios if t in constraints.trios_proibidos
        ]
        detalhes["trios_proibidos"] = {
            "proibidos": [list(t) for t in constraints.trios_proibidos],
            "presentes_na_seq": [list(t) for t in trios_proibidos_usados],
        }
        if trios_proibidos_usados:
            violacoes += len(trios_proibidos_usados)
        else:
            coerencias += 1

        # ----------------------------------------------------
        # Score total simples
        # ----------------------------------------------------
        score_total = float(coerencias - violacoes)

        return ScoreDetalhado(
            score_total=score_total,
            coerencias=coerencias,
            violacoes=violacoes,
            detalhes=detalhes,
        )


# Pequeno teste manual (não será executado no Render, mas ajuda localmente)
if __name__ == "__main__":
    engine = PontoCEngine()
    exemplo_seq = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 23, 24, 25]
    resultado = engine.score_sequence(exemplo_seq, regime_id="R2")
    print("Score exemplo:", resultado)
