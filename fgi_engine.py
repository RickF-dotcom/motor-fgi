from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from grupo_de_milhoes import GrupoDeMilhoes
from ponto_c_engine import PontoCEngine, ScoreDetalhado


# ============================================================
#  Estrutura de saída: Prototipo
# ============================================================

@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


# ============================================================
#  MotorFGI (NOVA VERSÃO) – Decoder do PONTO C
# ============================================================

class MotorFGI:
    """
    MotorFGI v0.1

    Papel desta versão:

      - NÃO pensa em estatística própria.
      - NÃO define regra de FGI.

    Quem pensa:
      - PontoCEngine (grafo C + constraints).

    Função do MotorFGI:
      - consumir o GrupoDeMilhoes (combinações não sorteadas),
      - usar o PontoCEngine para:
          * obter constraints por regime,
          * calcular score de coerência de sequência,
      - devolver PROTÓTIPOS estruturais (FGIs) ordenados por score.
    """

    def __init__(
        self,
        ponto_c: Optional[PontoCEngine] = None,
        grupo: Optional[GrupoDeMilhoes] = None,
        regime_id_padrao: Optional[str] = None,
    ) -> None:
        # Engine do PONTO C (se não vier de fora, cria um novo)
        self.ponto_c: PontoCEngine = ponto_c or PontoCEngine()

        # Config geral vinda do lab_config.yaml via PontoCEngine
        self.lab_config: Dict[str, Any] = self.ponto_c.config
        self.motor_cfg: Dict[str, Any] = self.lab_config.get("motor_fgi", {})
        self.busca_cfg: Dict[str, Any] = self.motor_cfg.get("busca", {})

        # Grupo de milhões (universo de combinações não sorteadas)
        # auto_generate=True → se não existir .pkl ele gera
        self.grupo: GrupoDeMilhoes = grupo or GrupoDeMilhoes(auto_generate=True)

        # Parâmetros padrão
        self.tamanho_jogo_padrao: int = int(
            self.motor_cfg.get("tamanho_jogo_padrao", 15)
        )
        self.qtd_prototipos_padrao: int = int(
            self.motor_cfg.get("qtd_prototipos_padrao", 50)
        )
        self.max_candidatos_avaliados_padrao: int = int(
            self.busca_cfg.get("max_candidatos_avaliados", 50000)
        )
        self.regime_padrao: str = (
            regime_id_padrao
            or self.motor_cfg.get("usar_regime_padrao", "R2")
        )

    # --------------------------------------------------------
    #  API principal: gerar protótipos estruturais
    # --------------------------------------------------------
    def gerar_prototipos(
        self,
        k: Optional[int] = None,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
    ) -> List[Prototipo]:
        """
        Gera até k protótipos estruturais (FGIs) usando:

          - universo do GrupoDeMilhoes (combinações não sorteadas),
          - constraints do PontoCEngine para o regime escolhido,
          - score de coerência do PontoCEngine.

        Retorna uma lista de Prototipo, ordenada por score_total (desc).
        """

        if not self.grupo.combos:
            raise ValueError(
                "Grupo de milhões está vazio. "
                "Garanta que o arquivo grupo_de_milhoes.pkl foi gerado."
            )

        # Parâmetros efetivos
        k = int(k or self.qtd_prototipos_padrao)
        regime_id = regime_id or self.regime_padrao

        # número máximo de candidatos para amostragem
        max_cand_cfg = self.max_candidatos_avaliados_padrao
        n_candidatos = int(max_candidatos or max_cand_cfg)
        n_candidatos = max(1, min(n_candidatos, len(self.grupo.combos)))

        # Amostra candidatos do grupo de milhões
        candidatos: List[List[int]] = self.grupo.sample(n_candidatos)

        avaliados: List[Prototipo] = []

        for seq in candidatos:
            # Garante tamanho de jogo esperado
            if len(seq) != self.tamanho_jogo_padrao:
                continue

            score: ScoreDetalhado = self.ponto_c.score_sequence(seq, regime_id)
            prot = Prototipo(
                sequencia=seq,
                score_total=score.score_total,
                coerencias=score.coerencias,
                violacoes=score.violacoes,
                detalhes=score.detalhes,
            )
            avaliados.append(prot)

        if not avaliados:
            raise ValueError(
                "Nenhum candidato pôde ser avaliado. "
                "Verifique grupo de milhões e configurações do laboratório."
            )

        # Ordena por score_total (decrescente) e pega top-k
        avaliados.sort(key=lambda p: p.score_total, reverse=True)
        return avaliados[:k]

    # --------------------------------------------------------
    #  Versão amigável para API (JSON-ready)
    # --------------------------------------------------------
    def gerar_prototipos_json(
        self,
        k: Optional[int] = None,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrapper que transforma Prototipo em dict pronto para JSON.
        Ideal para uso direto no FastAPI.
        """
        prototipos = self.gerar_prototipos(
            k=k,
            regime_id=regime_id,
            max_candidatos=max_candidatos,
        )
        retorno: List[Dict[str, Any]] = []

        for p in prototipos:
            retorno.append(
                {
                    "sequencia": p.sequencia,
                    "score_total": p.score_total,
                    "coerencias": p.coerencias,
                    "violacoes": p.violacoes,
                    "detalhes": p.detalhes,
                }
            )

        return retorno


# ------------------------------------------------------------
# Teste rápido local (não afeta o Render)
# ------------------------------------------------------------
if __name__ == "__main__":
    motor = MotorFGI()
    protos = motor.gerar_prototipos(k=5, regime_id="R2")
    for i, p in enumerate(protos, start=1):
        print(f"# Protótipo {i}")
        print("Sequência:", p.sequencia)
        print("Score:", p.score_total, "Coerências:", p.coerencias, "Violações:", p.violacoes)
        print("-" * 40)
