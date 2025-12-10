# maturacao.py
# Módulo de maturação para domínio discreto 1..25

from typing import List, Dict, Literal, Tuple
import math


EstadoMaturacao = Literal["EXAUSTO", "NEUTRO", "MATURANDO", "MADURO_FORTE"]


def _normalizar_lista(valores: List[float]) -> List[float]:
    """
    Normaliza uma lista de valores para [0, 1].
    Se todos forem iguais, retorna 0.5 para tudo (evita divisão por zero).
    """
    if not valores:
        return []
    vmin = min(valores)
    vmax = max(valores)
    if math.isclose(vmin, vmax):
        return [0.5 for _ in valores]
    return [(v - vmin) / (vmax - vmin) for v in valores]


def _frequencias_por_elemento(historico: List[List[int]], tamanho_janela: int) -> Dict[int, int]:
    """
    Conta quantas vezes cada elemento 1..25 aparece na última 'tamanho_janela' entradas.
    Se não houver dados suficientes, usa todo o histórico disponível.
    """
    if not historico:
        return {d: 0 for d in range(1, 26)}

    janela = historico[-tamanho_janela:] if len(historico) >= tamanho_janela else historico[:]
    contagem = {d: 0 for d in range(1, 26)}
    for concurso in janela:
        for d in concurso:
            if 1 <= d <= 25:
                contagem[d] += 1
    return contagem


def _recencia_e_ausencia(historico: List[List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Calcula, para cada d em 1..25:
    - recencia[d]  = há quantos concursos ele não aparece (0 se apareceu no último)
    - run_ausente[d] = tamanho da sequência atual de ausência (igual à recência, dado o formato)
    """
    recencia = {d: 0 for d in range(1, 26)}
    run_ausente = {d: 0 for d in range(1, 26)}

    for d in range(1, 26):
        distancia = 0
        for concursos_retro in range(len(historico) - 1, -1, -1):
            concurso = historico[concursos_retro]
            distancia += 1
            if d in concurso:
                break
        else:
            distancia = len(historico) + 1
        recencia[d] = distancia
        run_ausente[d] = distancia

    return recencia, run_ausente


def _frequencia_anterior(historico: List[List[int]], tamanho_janela: int) -> Dict[int, int]:
    """
    Frequência em uma janela anterior à atual, para ter 'trend'.
    Ex.: se tamanho_janela=10, olha os 10 concursos imediatamente anteriores aos últimos 10.
    Se não houver espaço suficiente, retorna zeros.
    """
    if len(historico) < 2 * tamanho_janela:
        return {d: 0 for d in range(1, 26)}

    inicio = len(historico) - 2 * tamanho_janela
    fim = len(historico) - tamanho_janela
    janela = historico[inicio:fim]
    contagem = {d: 0 for d in range(1, 26)}
    for concurso in janela:
        for d in concurso:
            if 1 <= d <= 25:
                contagem[d] += 1
    return contagem


def calcular_maturacao(
    historico: List[List[int]],
    janela_longa: int = 25,
    janela_curta: int = 10,
) -> Dict[int, Dict[str, float | str]]:
    """
    Calcula o score de maturação M(d) e o estado categórico para cada d em 1..25.

    Retorna um dict:
    {
      1: {"score": float, "estado": "NEUTRO"},
      2: {"score": float, "estado": "MADURO_FORTE"},
      ...
    }
    """
    if not historico:
        return {
            d: {"score": 0.5, "estado": "NEUTRO"}
            for d in range(1, 26)
        }

    freq_L = _frequencias_por_elemento(historico, janela_longa)
    freq_S = _frequencias_por_elemento(historico, janela_curta)
    freq_S_prev = _frequencia_anterior(historico, janela_curta)

    recencia, run_ausente = _recencia_e_ausencia(historico)

    total_L = sum(freq_L.values()) or 1
    freq_relativa = {d: freq_L[d] / total_L for d in range(1, 26)}

    trend = {d: freq_S[d] - freq_S_prev[d] for d in range(1, 26)}

    lista_A = [run_ausente[d] for d in range(1, 26)]
    lista_R = [recencia[d] for d in range(1, 26)]
    lista_F = [1.0 - freq_relativa[d] for d in range(1, 26)]
    lista_C = [float(janela_curta - freq_S[d]) for d in range(1, 26)]
    lista_T = [max(0.0, float(-trend[d])) for d in range(1, 26)]

    A_norm = _normalizar_lista(lista_A)
    R_norm = _normalizar_lista(lista_R)
    F_norm = _normalizar_lista(lista_F)
    C_norm = _normalizar_lista(lista_C)
    T_norm = _normalizar_lista(lista_T)

    maturacao_bruta = {}
    for idx, d in enumerate(range(1, 26)):
        A = A_norm[idx]
        R = R_norm[idx]
        F = F_norm[idx]
        C = C_norm[idx]
        T = T_norm[idx]

        M = (
            0.30 * A +
            0.25 * R +
            0.20 * F +
            0.15 * C +
            0.10 * T
        )
        maturacao_bruta[d] = M

    scores = [maturacao_bruta[d] for d in range(1, 26)]
    ordenados = sorted(scores)

    def _percentil(valor: float) -> float:
        pos = sum(1 for v in ordenados if v <= valor)
        return pos / len(ordenados)

    resultado: Dict[int, Dict[str, float | str]] = {}
    for d in range(1, 26):
        score = maturacao_bruta[d]
        p = _percentil(score)

        if p < 0.25:
            estado: EstadoMaturacao = "EXAUSTO"
        elif p < 0.60:
            estado = "NEUTRO"
        elif p < 0.85:
            estado = "MATURANDO"
        else:
            estado = "MADURO_FORTE"

        resultado[d] = {
            "score": float(round(score, 4)),
            "estado": estado,
        }

    return resultado


def score_maturacao_jogo(
    jogo: List[int],
    maturacao_por_elemento: Dict[int, Dict[str, float | str]],
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> float:
    """
    Calcula um score de maturação para um jogo (lista de inteiros).

    - média dos scores individuais
    - soma de MADURO_FORTE
    - penalização de EXAUSTO
    """
    if not jogo:
        return 0.0

    scores = []
    maduros = 0
    exaustos = 0

    for d in jogo:
        info = maturacao_por_elemento.get(d)
        if not info:
            continue
        s = float(info["score"])
        estado = str(info["estado"])
        scores.append(s)
        if estado == "MADURO_FORTE":
            maduros += 1
        elif estado == "EXAUSTO":
            exaustos += 1

    if not scores:
        return 0.0

    media = sum(scores) / len(scores)
    return (
        alpha * media +
        beta * (maduros / len(jogo)) -
        gamma * (exaustos / len(jogo))
  )
