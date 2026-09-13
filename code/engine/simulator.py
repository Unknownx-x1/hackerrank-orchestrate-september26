from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

def parse_date(d: str) -> datetime:
    return datetime.strptime(str(d).strip(), "%Y-%m-%d")

def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

class Simulator:
    def __init__(self, horizon_days: int = 90):
        self.horizon_days = horizon_days

    def simulate(
        self,
        starting_balance: float,
        daily_flows: Dict[datetime, float],
        req_date: datetime
    ) -> Dict[datetime, float]:
        """
        Computes the daily balance curve from day 0 to horizon_days.
        """
        cur_bal = starting_balance + daily_flows.get(req_date, 0.0)
        balances = {req_date: cur_bal}
        for d in range(1, self.horizon_days + 1):
            dt = req_date + timedelta(days=d)
            cur_bal += daily_flows.get(dt, 0.0)
            balances[dt] = cur_bal
        return balances

    def compute_safe_amount(
        self,
        balances: Dict[datetime, float],
        min_keep: float,
        requested_amount: float
    ) -> float:
        """
        Computes maximum amount safe to pay today before optional spending changes.
        """
        min_margin = min(balances[dt] - min_keep for dt in balances)
        return max(0.0, min(requested_amount, min_margin))

    def find_earliest_full_payment_date(
        self,
        balances: Dict[datetime, float],
        min_keep: float,
        requested_amount: float,
        req_date: datetime,
        daily_flows: Optional[Dict[datetime, float]] = None
    ) -> str:
        """
        Finds the earliest date where a single full payment keeps balances >= min_keep.
        If safe today, returns req_date.
        Otherwise, evaluates salary arrival dates where new funds arrive.
        """
        safe_today = self.compute_safe_amount(balances, min_keep, requested_amount)
        if safe_today >= requested_amount - 1e-4:
            return format_date(req_date)

        if not daily_flows:
            return ""

        horizon_end = req_date + timedelta(days=self.horizon_days)
        sal_dates = sorted([d for d, v in daily_flows.items() if v > 100 and d > req_date])
        for s_dt in sal_dates:
            check_until = min(horizon_end, s_dt + timedelta(days=15))
            is_safe = True
            cur = s_dt
            while cur <= check_until:
                if cur in balances:
                    if (balances[cur] - requested_amount) < (min_keep - 1e-4):
                        is_safe = False
                        break
                cur += timedelta(days=1)
            if is_safe:
                return format_date(s_dt)

        return ""

    def test_plan_feasibility(
        self,
        starting_balance: float,
        daily_flows: Dict[datetime, float],
        payments: List[Tuple[datetime, float]],
        min_keep: float,
        req_date: datetime
    ) -> bool:
        """
        Tests if a proposed payment plan maintains balance >= min_keep at all times through completion.
        """
        plan_flows = {p_dt: p_amt for p_dt, p_amt in payments}
        last_payment_dt = max(p_dt for p_dt, p_amt in payments) if payments else req_date
        cur_bal = starting_balance + daily_flows.get(req_date, 0.0) - plan_flows.get(req_date, 0.0)
        if cur_bal < min_keep * 0.95:
            return False

        cur = req_date + timedelta(days=1)
        while cur <= last_payment_dt:
            cur_bal += daily_flows.get(cur, 0.0) - plan_flows.get(cur, 0.0)
            if cur_bal < min_keep * 0.95:
                return False
            cur += timedelta(days=1)

        return True
