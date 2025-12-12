from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from grupo_de_milhoes import GrupoDeMilhoes


# =========================
# Estrutura de Protótipo
# =========================

@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


# =========================
# MotorFGI
# =========================

class MotorFGI:
    """
    Motor principal do laboratório.

    Responsabilidades:
    - Carregar Grupo de Milhões
    - Avaliar sequências (PONTO C / score)
    - Gerar protótipos estruturais (LHE/LHS)
    """

    def __init__(
        self,
        historico_csv: Optional[str] = None,
        universo_max: int = 25,
    ) -> None:
        self.universo_max = universo_max

        self.grupo = GrupoDeMilhoes(
            universo_max=universo_max,
            historico_csv=Path(historico_csv) if historico_csv else None,
        )

        # placeholder de regime padrão
        self.regime_padrao = "estavel"

    # =========================
    # Avaliação (PONTO C)
    # =========================

    def avaliar_sequencia(self, seq: List[int], regime: str) -> Dict[str, Any]:
        """
        Avaliação mínima estruturada.
        Aqui é onde seu PONTO C cresce depois.
        """
        soma = sum(seq)
        pares = sum(1 for x in seq if x % 2 == 0)
        impares = len(seq) - pares

        violacoes = 0
        coerencias = 0

        if 170 <= soma <= 220:
            coerencias += 1
        else:
            violacoes += 1

        detalhes = {
            "soma": soma,
            "pares": pares,
            "impares": impares,
            "regime": regime,
        }

        score_total = coerencias - violacoes

        return {
            "score_total": float(score_total),
            "coerencias": coerencias,
            "violacoes": violacoes,
            "detalhes": detalhes,
        }

    # =========================
    # Geração de Protótipos
    # =========================

    def gerar_prototipos(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
    ) -> List[Prototipo]:

        regime = regime_id or self.regime_padrao
        limite = max_candidatos or 2000

        candidatos = self.grupo.get_candidatos(
            k=k,
            max_candidatos=limite,
        )

        prototipos: List[Prototipo] = []

        for seq in candidatos:
            avaliacao = self.avaliar_sequencia(seq, regime)

            prototipos.append(
                Prototipo(
                    sequencia=seq,
                    score_total=avaliacao["score_total"],
                    coerencias=avaliacao["coerencias"],
                    violacoes=avaliacao["violacoes"],
                    detalhes=avaliacao["detalhes"],
                )
            )

        # ordenação estrutural
        prototipos.sort(
            key=lambda p: (p.score_total, -p.violacoes),
            reverse=True,
        )

        return prototipos[:k]

    # =========================
    # JSON-friendly (API)
    # =========================

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
        incluir_contexto_dna: bool = True,
    ) -> Dict[str, Any]:

        protos = self.gerar_prototipos(
            k=k,
            regime_id=regime_id,
            max_candidatos=max_candidatos,
        )

        payload = [
            {
                "sequencia": p.sequencia,
                "score_total": p.score_total,
                "coerencias": p.coerencias,
                "violacoes": p.violacoes,
                "detalhes": p.detalhes,
            }
            for p in protos
        ]

        resp: Dict[str, Any] = {
            "prototipos": payload,
            "regime_usado": regime_id or self.regime_padrao,
            "max_candidatos_usado": max_candidatos or 2000,
        }

        if incluir_contexto_dna:
            resp["contexto_dna"] = {
                "universo_max": self.universo_max,
                "total_sorteadas": self.grupo.total_sorteadas(),
            }

        return resp


# =========================
# TESTE LOCAL DIRETO
# =========================

if __name__ == "__main__":
    motor = MotorFGI(
        historico_csv="lotofacil_ultimos_25_concursos.csv",
        universo_max=25,
    )

    resultado = motor.gerar_prototipos_json(
        k=5,
        regime_id="estavel",
        max_candidatos=2000,
        incluir_contexto_dna=True,
    )

    print(resultado)
