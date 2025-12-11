from fastapi import FastAPI
from fastapi.responses import JSONResponse
import yaml
import os

# ============================================================
#  LOAD DO ARQUIVO lab_config.yaml
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_CONFIG_PATH = os.path.join(BASE_DIR, "lab_config.yaml")

if not os.path.exists(LAB_CONFIG_PATH):
    raise FileNotFoundError(
        f"Arquivo de configuração não encontrado: {LAB_CONFIG_PATH}"
    )

with open(LAB_CONFIG_PATH, "r", encoding="utf-8") as f:
    LAB_CONFIG = yaml.safe_load(f)

# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="ATHENAH LABORATORIO PMFC",
    description="Backend do MotorFGI + PONTO C + Config YAML",
    version="0.1.0",
)

# ============================================================
#  ENDPOINT: Retorna o YAML carregado
# ============================================================

@app.get("/lab/config")
def get_lab_config():
    """
    Retorna o conteúdo completo do arquivo lab_config.yaml
    como JSON.
    """
    return JSONResponse(content=LAB_CONFIG)

# ============================================================
#  ENDPOINT DE STATUS (opcional)
# ============================================================

@app.get("/lab/status")
def status():
    return {
        "status": "online",
        "versao_laboratorio": LAB_CONFIG.get("versao_laboratorio", "desconhecida")
    }

# ============================================================
#  OBSERVAÇÃO
# ============================================================
# O restante do laboratório (motor FGI, Ponto C, metodologias etc.)
# será acoplado nos próximos passos. Aqui você já tem:
#   ✔ leitura automática do lab_config.yaml
#   ✔ endpoint /lab/config funcionando
#   ✔ estrutura limpa e completa sem depender de código prévio
