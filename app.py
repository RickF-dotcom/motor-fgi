name: app.py
content: |
  from fastapi import FastAPI
  from pydantic import BaseModel
  from fgi_engine import (
      extract_recent_absences,
      group_frequency,
      generate_fgis
  )

  app = FastAPI(title="Motor FGI – V1")

  class DrawData(BaseModel):
      draws: list[list[int]]
      quantity: int = 10

  @app.post("/fgi")
  def create_fgi(data: DrawData):
      draws = data.draws
      qty = data.quantity

      abs_10, abs_15, abs_20, abs_25 = extract_recent_absences(draws)
      freq = group_frequency(draws)

      fgis = generate_fgis(
          qty=qty,
          freq=freq,
          abs10=abs_10,
          abs15=abs_15,
          abs20=abs_20,
          abs25=abs_25
      )

      return {
          "absences": {
              "last10": abs_10,
              "last15": abs_15,
              "last20": abs_20,
              "last25": abs_25,
          },
          "frequency": freq,
          "fgis": fgis,
      }
