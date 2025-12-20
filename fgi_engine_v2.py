        if alvo is None:
            # proxy: ancorar em “meio” do universo (15 números ~ média 13) => soma ~ 195
            alvo = 195.0
        # distância normalizada
        dist = abs(soma - alvo) / max(alvo, 1.0)
        ancora = 1.0 - _clip(dist, 0.0, 1.0)

        return {
            "direcional": float(_clip(direcional, 0.0, 1.0)),
            "consistencia": float(_clip(consistencia, 0.0, 1.0)),
            "ancora": float(_clip(ancora, 0.0, 1.0)),
        }

    def _scf_total(self, metricas: Dict[str, float], pesos_metricas: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Retorna:
          - scf_total
          - raw_components
          - weighted_components
        """
        raw = {k: float(metricas.get(k, 0.0)) for k in pesos_metricas.keys()}
        weighted = {k: raw[k] * float(pesos_metricas.get(k, 1.0)) for k in raw.keys()}
        total = float(sum(weighted.values()))
        return total, raw, weighted

    def rerank(self, candidatos: List[List[int]], contexto_lab: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overrides = overrides or {}

        top_n = int(overrides.get("top_n", 30))
        max_candidatos = int(overrides.get("max_candidatos", 3000))

        windows = overrides.get("windows", list(self.default_windows))
        windows = [int(w) for w in windows] if isinstance(windows, list) else list(self.default_windows)

        dna_anchor_window = int(overrides.get("dna_anchor_window", self.default_dna_anchor_window))

        pesos_windows = overrides.get("pesos_windows", dict(self.default_pesos_windows))
        pesos_metricas = overrides.get("pesos_metricas", dict(self.default_pesos_metricas))

        redundancy_jaccard_threshold = float(overrides.get("redundancy_jaccard_threshold", self.default_redundancy_jaccard_threshold))
        redundancy_penalty = float(overrides.get("redundancy_penalty", self.default_redundancy_penalty))

        z_cap = float(overrides.get("z_cap", self.default_z_cap))
        align_temperature = float(overrides.get("align_temperature", self.default_align_temperature))

        # corta candidatos (segurança)
        pool = candidatos[:max_candidatos]

        scored: List[Dict[str, Any]] = []
        for seq in pool:
            seq = [int(x) for x in seq]
            metricas = self._metricas_base(seq, contexto_lab, windows, dna_anchor_window)

            scf_total, raw_components, weighted_components = self._scf_total(metricas, pesos_metricas)

            # ajuste leve: temperature (só escala)
            if align_temperature > 0:
                scf_total = scf_total / align_temperature

            # z_cap: clip do score total (evita explosão)
            scf_total = _clip(scf_total, -abs(z_cap), abs(z_cap))

            scored.append(
                {
                    "sequencia": seq,
                    "score": float(scf_total),
                    "detail": {
                        "metricas": metricas,  # <<< O QUE O V3 EXIGE
                        "scf_total": float(scf_total),
                        "score_raw_components": raw_components,
                        "score_weighted_components": weighted_components,
                        "redundancy_penalty": 0.0,
                    },
                }
            )

        # ordena antes de diversidade
        scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        # diversidade / anti-redundância
        selected: List[Dict[str, Any]] = []
        for item in scored:
            if len(selected) >= top_n:
                break

            pen = 0.0
            for prev in selected:
                j = _jaccard(item["sequencia"], prev["sequencia"])
                if j >= redundancy_jaccard_threshold:
                    pen += redundancy_penalty * (j - redundancy_jaccard_threshold) / max(1e-9, (1.0 - redundancy_jaccard_threshold))

            # aplica penalidade
            if pen > 0:
                new_score = float(item["score"]) - float(pen)
                item["score"] = new_score
                item["detail"]["scf_total"] = float(new_score)
                item["detail"]["redundancy_penalty"] = float(pen)

            selected.append(item)

        # reordena pós-penalidade
        selected.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        return {
            "engine_used": "v2",
            "top": selected,
            "config_used": {
                "top_n": top_n,
                "max_candidatos": max_candidatos,
                "windows": windows,
                "dna_anchor_window": dna_anchor_window,
                "pesos_windows": pesos_windows,
                "pesos_metricas": pesos_metricas,
                "redundancy_jaccard_threshold": redundancy_jaccard_threshold,
                "redundancy_penalty": redundancy_penalty,
                "z_cap": z_cap,
                "align_temperature": align_temperature,
            },
        }
