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
