# fgi_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

from grupo_de_milhoes import GrupoDeMilhoes

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


BASE_DIR = Path(__file__).resolve().parent
LAB_CONFIG_PATH = BASE_DIR / "lab_config.yaml"
DNA_LAST25_PATH = BASE_DIR / "dna_last25.json"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}


@dataclass
class Score:
    score_total: float
    coerencias: int = 0
    violacoes: int = 0
    detalhes: Dict[str, Any] = None  # type: ignore


@dataclass
class Prototipo:
    sequencia: List[int]
    score_total: float
    coerencias: int
    violacoes: int
    detalhes: Dict[str, Any]


class MotorFGI:
    """
    MotorFGI (base) para o laboratório:
    - Puxa candidatos do GrupoDeMilhoes
    - Calcula score (placeholder aqui; você troca pelo PONTO C real)
    - Ordena e retorna protótipos
    """

    def __init__(
        self,
        historico_csv: Optional[str] = None,
        universo_max: int = 25,
        auto_generate: bool = False,
        lab_config_path: Optional[str] = None,
        dna_last25_path: Optional[str] = None,
    ) -> None:
        self.lab_config_path = Path(lab_config_path) if lab_config_path else LAB_CONFIG_PATH
        self.dna_last25_path = Path(dna_last25_path) if dna_last25_path else DNA_LAST25_PATH

        self.lab_config = _load_yaml(self.lab_config_path)
        self._dna_last25 = _load_json(self.dna_last25_path)

        self.grupo = GrupoDeMilhoes(
            universo_max=int(universo_max),
            historico_csv=Path(historico_csv) if historico_csv else None,
            auto_generate=bool(auto_generate),
        )

    # ----------------------------
    # Contexto (DNA)
    # ----------------------------
    def get_dna_contexto(self) -> Dict[str, Any]:
        return dict(self._dna_last25) if isinstance(self._dna_last25, dict) else {}

    # ----------------------------
    # Score (placeholder)
    # ----------------------------
    def _score_sequence(self, seq: List[int], regime_id: Optional[str]) -> Score:
        # placeholder mínimo (não quebra o fluxo)
        coerencias = 0
        violacoes = 0
        detalhes: Dict[str, Any] = {"regime_id": regime_id}

        if len(seq) != len(set(seq)):
            violacoes += 1

        if all(1 <= x <= self.grupo.universo_max for x in seq):
            coerencias += 1

        score_total = float(coerencias) - float(violacoes) * 10.0
        return Score(score_total=score_total, coerencias=coerencias, violacoes=violacoes, detalhes=detalhes)

    # ----------------------------
    # API principal
    # ----------------------------
    def gerar_prototipos(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
    ) -> List[Prototipo]:
        k = int(k)
        limite = int(max_candidatos) if max_candidatos is not None else 2000

        # ✅ CORREÇÃO CRÍTICA:
        # k é k; limite é limite. Não inverter.
        candidatos = self.grupo.get_candidatos(k=k, max_candidatos=limite)

        prototipos: List[Prototipo] = []
        for seq in candidatos:
            sc = self._score_sequence(seq, regime_id=regime_id)
            prototipos.append(
                Prototipo(
                    sequencia=list(seq),
                    score_total=float(sc.score_total),
                    coerencias=int(sc.coerencias),
                    violacoes=int(sc.violacoes),
                    detalhes=dict(sc.detalhes) if sc.detalhes else {},
                )
            )

        prototipos.sort(key=lambda p: (-p.score_total, p.violacoes, -p.coerencias))
        return prototipos[:k]

    def gerar_prototipos_json(
        self,
        k: int,
        regime_id: Optional[str] = None,
        max_candidatos: Optional[int] = None,
        incluir_contexto_dna: bool = True,
    ) -> Dict[str, Any]:
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
            "regime_usado": regime_id,
            "max_candidatos_usado": int(max_candidatos) if max_candidatos is not None else 2000,
        }
