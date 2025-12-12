from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from grupo_de_milhoes import GrupoDeMilhoes
from ponto_c_engine import PontoCEngine, ScoreDetalhado


# ==========================================================
# Estrutura de saída: Protótipo (LHE / FGI estrutural)
# ==========================================================
@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


# ==========================================================
# Util: caminhos e carga de YAML
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
LAB_CONFIG_PATH = BASE_DIR / "lab_config.yaml"
DNA_LAST25_PATH = BASE_DIR / "dna_last25.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido (esperado dict no topo): {path}")
    return data


def _load_yaml_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


# ==========================================================
# MotorFGI (Decoder do PONTO C + Grupo de Milhões + DNA)
# ==========================================================
class MotorFGI:
    """
    MotorFGI v0.2 (com DNA opcional)

    Papel:
      - Consome o GrupoDeMilhoes (combinações não sorteadas)
      - Usa PontoCEngine para calcular coerência/violação por regime
      - Ordena e devolve protótipos estruturais (LHEs)
      - Injeta DNA (últimos 25) como "contexto de calibração" (não bloqueia nada ainda)
        -> neste estágio, DNA entra como METADADO e parâmetro de ajuste futuro
    """

    def __init__(
        self,
        ponto_c: Optional[PontoCEngine] = None,
        grupo: Optional[GrupoDeMilhoes] = None,
        regime_id_padrao: Optional[str] = None,
        dna_path: Optional[str] = None,
        lab_config_path: Optional[str] = None,
    ) -> None:
        # Config base (lab_config.yaml)
        cfg_path = Path(lab_config_path) if lab_config_path else LAB_CONFIG_PATH
        self.lab_config: Dict[str, Any] = _load_yaml(cfg_path)

        # Pedaços relevantes do config
        self.motor_cfg: Dict[str, Any] = self.lab_config.get("motor_fgi", {}) or {}
        self.busca_cfg: Dict[str, Any] = self.motor_cfg.get("busca", {}) or {}

        # Engine do PONTO C (se não vier, cria com lab_config carregado)
        self.ponto_c: PontoCEngine = ponto_c or PontoCEngine(self.lab_config)

        # Grupo de Milhões (se não vier, cria)
        # Importante: GrupoDeMilhoes pode ter assinatura (auto_generate=True/False)
        # Mantemos o padrão já usado no teu código.
        self.grupo: GrupoDeMilhoes = grupo or GrupoDeMilhoes(auto_generate=True)

        # Parâmetros padrão
        self.regime_padrao: str = (
            regime_id_padrao
            or self.motor_cfg.get("usar_regime_padrao")
            or "R2"
        )
        self.qtd_prototipos_padrao: int = int(self.motor_cfg.get("qtd_prototipos_padrao", 50))
        self.max_candidatos_padrao: int = int(self.busca_cfg.get("max_candidatos_avaliados", 50000))

        # DNA (opcional): por padrão tenta dna_last25.yaml na raiz
        resolved_dna_path = Path(dna_path) if dna_path else DNA_LAST25_PATH
        self.dna: Dict[str, Any] = _load_yaml_optional(resolved_dna_path)

        # Guarda metadado útil
        self.dna_info: Dict[str, Any] = self.dna.get("dna", {}) if isinstance(self.dna.get("dna", {}), dict) else {}

    # ----------------------------------------------------------
    # Helpers de calibração: por enquanto DNA é CONTEXTO
    # ----------------------------------------------------------
    def get_dna_contexto(self) -> Dict[str, Any]:
        """
        Retorna o 'DNA' como contexto de calibração.
        Não aplica filtro duro ainda (fase de engenharia).
        """
        if not self.dna_info:
            return {
                "ativo": False,
                "motivo": "dna_last25.yaml ausente ou vazio",
            }

        janelas = self.dna_info.get("janelas", [])
        metricas = self.dna_info.get("metricas_ativas", [])
        avaliacao = self.dna_info.get("avaliacao", {})
        uso_no_motor = self.dna_info.get("uso_no_motor", {})

        return {
            "ativo": True,
            "origem": self.dna_info.get("origem"),
            "descricao": self.dna_info.get("descricao"),
            "janelas": janelas,
            "metricas_ativas": metricas,
            "avaliacao": avaliacao,
            "uso_no_motor": uso_no_motor,
        }

    # ----------------------------------------------------------
    # API principal: gerar protótipos
    # ----------------------------------------------------------
    def gerar_prototipos(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
    ) -> List[Prototipo]:
        """
        Gera K protótipos (LHEs) avaliando candidatos do Grupo de Milhões
        e ranqueando pelo score do PONTO C.

        Observação:
          - Nesta fase, DNA ainda não muda constraints.
          - DNA entra como contexto e vai servir de calibração depois.
        """
        regime = regime_id or self.regime_padrao
        limite = int(max_candidatos if max_candidatos is not None else self.max_candidatos_padrao)

        candidatos = self.grupo.get_candidatos(limite)  # esperado: List[List[int]] ou iterável
        prototipos: List[Prototipo] = []

        for seq in candidatos:
            score: ScoreDetalhado = self.ponto_c.score_sequence(seq, regime)
            prototipos.append(
                Prototipo(
                    sequencia=list(seq),
                    score_total=float(score.score_total),
                    coerencias=int(score.coerencias),
                    violacoes=int(score.violacoes),
                    detalhes=dict(score.detalhes),
                )
            )

        # Ordena: score_total desc, depois menos violações, depois mais coerências
        prototipos.sort(key=lambda p: (p.score_total, -p.violacoes, p.coerencias), reverse=True)

        # Retorna topo K
        return prototipos[: int(k)]

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
        incluir_contexto_dna: bool = True,
    ) -> Dict[str, Any]:
        """
        Versão JSON-friendly para o endpoint /prototipos.

        Mantém compatibilidade com o que você já está usando,
        mas permite incluir "contexto_dna" no retorno.
        """
        protos = self.gerar_prototipos(k=k, regime_id=regime_id, max_candidatos=max_candidatos)

        payload_protos = [
            {
                "sequencia": p.sequencia,
                "score_total": p.score_total,
                "coerencias": p.coerencias,
                "violacoes": p.violacoes,
                "detalhes": p.detalhes,
            }
            for p in protos
        ]

        if not incluir_contexto_dna:
            return {"prototipos": payload_protos}

        return {
            "prototipos": payload_protos,
            "contexto_dna": self.get_dna_contexto(),
            "regime_usado": regime_id or self.regime_padrao,
            "max_candidatos_usado": int(max_candidatos if max_candidatos is not None else self.max_candidatos_padrao),
        }


# ==========================================================
# Teste rápido local
# ==========================================================
if __name__ == "__main__":
    motor = MotorFGI()
    out = motor.gerar_prototipos_json(k=5, regime_id="R2", max_candidatos=2000)
    print(out)
