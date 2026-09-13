import pandas as pd
from typing import Dict, Tuple

class FXConverter:
    def __init__(self, exchange_rates_path: str = "dataset/exchange_rates.csv"):
        self.rates: Dict[Tuple[str, str, str], float] = {}
        df = pd.read_csv(exchange_rates_path)
        for _, row in df.iterrows():
            from_curr = str(row['from_currency']).strip()
            to_curr = str(row['to_currency']).strip()
            rate_date = str(row['rate_date']).strip()
            rate = float(row['rate'])
            self.rates[(from_curr, to_curr, rate_date)] = rate

    def convert(self, amount: float, from_curr: str, to_curr: str, rate_date: str) -> float:
        from_curr = str(from_curr).strip()
        to_curr = str(to_curr).strip()
        rate_date = str(rate_date).strip()
        
        if from_curr == to_curr or not from_curr or not to_curr:
            return amount
            
        key = (from_curr, to_curr, rate_date)
        if key in self.rates:
            return amount * self.rates[key]
            
        inv_key = (to_curr, from_curr, rate_date)
        if inv_key in self.rates:
            return amount / self.rates[inv_key]
            
        # Fallback: if rate date doesn't match exactly, find closest date
        matching_dates = [k for k in self.rates if k[0] == from_curr and k[1] == to_curr]
        if matching_dates:
            matching_dates.sort(key=lambda k: abs((pd.to_datetime(k[2]) - pd.to_datetime(rate_date)).days))
            return amount * self.rates[matching_dates[0]]
            
        return amount
