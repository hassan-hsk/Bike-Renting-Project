from pydantic import BaseModel

class PredictionInput(BaseModel):
    season: int          # 1 = Spring, 2 = Summer, etc.
    yr: int              # 0 = 2011, 1 = 2012
    mnth: int            # 1 - 12
    holiday: int         # 0 or 1
    weekday: int         # 0 (Sunday) to 6 (Saturday)
    workingday: int      # 0 or 1
    weathersit: int      # 1 to 4
    temp: float          # Normalized: (real + 8) / 47
    atemp: float         # Normalized: (real + 16) / 66
    hum: float           # Normalized: % / 100
    windspeed: float     # Normalized: km/h / 67
    day: int             # Day of the month (1-31)
    dayofweek: int       # 0 (Sunday) to 6 (Saturday)
    model: str           # "Random Forest", "Decision Tree", etc.
