import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

def parse_date(d: str) -> datetime:
    return datetime.strptime(str(d).strip(), "%Y-%m-%d")

def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

class RecurrenceEngine:
    def __init__(self):
        pass

    def build_schedule(
        self,
        user_id: str,
        req_date: datetime,
        horizon_days: int,
        past_events: pd.DataFrame,
        scheduled_events: pd.DataFrame,
        msg_info: Dict[str, Any],
        protected_cats: set
    ) -> Tuple[Dict[datetime, float], Dict[str, Dict[str, Any]], set]:
        """
        Builds daily net cash flows for t in [0, horizon_days]
        Returns:
            daily_flows: dict of date -> net cash flow (positive credit, negative debit)
            flexible_events: dict of event_id -> flexible event metadata
            known_salary_dates: set of (year, month) with confirmed salary
        """
        end_date = req_date + timedelta(days=horizon_days)
        daily_flows = {req_date + timedelta(days=d): 0.0 for d in range(horizon_days + 1)}
        flexible_events = {}
        known_salary_dates = set()

        # 1. Scheduled future events from dataset
        for _, ev in scheduled_events.iterrows():
            ev_dt = parse_date(ev['event_date'])
            if ev_dt in daily_flows:
                amt = float(ev['amount'])
                if ev['direction'] == 'debit':
                    daily_flows[ev_dt] -= amt
                elif ev['direction'] == 'credit':
                    if ev['category'] == 'salary':
                        known_salary_dates.add((ev_dt.year, ev_dt.month))
                    daily_flows[ev_dt] += amt

        # 2. Check if salary has ended or is one-off
        salary_stopped = msg_info.get('salary_stopped', False)
        # Check event descriptions for terminal salary
        sal_evs = past_events[past_events['category'] == 'salary'].sort_values('event_date')
        if len(sal_evs) > 0:
            last_sal_desc = str(sal_evs.iloc[-1]['description']).lower()
            if any(term in last_sal_desc for term in ['final employer payroll', 'terminal payroll', 'contract ended']):
                salary_stopped = True

        req_date_str = format_date(req_date)

        # 3. Monthly recurring expenses
        monthly_cats = [
            'rent', 'housing', 'utilities', 'education', 'debt_repayment', 'insurance',
            'healthcare', 'family_support', 'gym', 'music_subscription', 'delivery_membership',
            'cloud_storage', 'streaming', 'entertainment', 'shopping'
        ]

        for cat in monthly_cats:
            cat_evs = past_events[past_events['category'] == cat].sort_values('event_date')
            if len(cat_evs) >= 1:
                latest = cat_evs.iloc[-1]
                # Track flexible candidates
                if latest['flexibility'] in ['stoppable', 'reducible', 'reducible_or_stoppable']:
                    min_amt = float(latest['minimum_allowed_amount']) if pd.notna(latest['minimum_allowed_amount']) else 0.0
                    flexible_events[latest['event_id']] = {
                        'category': cat,
                        'event_id': latest['event_id'],
                        'description': latest['description'],
                        'flexibility': latest['flexibility'],
                        'amount': float(latest['amount']),
                        'minimum_allowed_amount': min_amt
                    }

                is_active = (cat in protected_cats)
                if not is_active:
                    continue

                dom = parse_date(latest['event_date']).day
                if cat in ['rent', 'housing']:
                    amt = float(latest['amount']) * msg_info.get('rent_multiplier', 1.0)
                elif latest['flexibility'] == 'fixed':
                    amt = float(latest['amount'])
                else:
                    # Conservative estimate for variable expenses
                    amt = float(cat_evs['amount'].tail(3).max())

                cur = req_date
                import calendar
                while cur <= end_date:
                    max_d = calendar.monthrange(cur.year, cur.month)[1]
                    target_dt = datetime(cur.year, cur.month, min(dom, max_d))

                    if target_dt == req_date:
                        # If due today, check if settled today already
                        settled_today = past_events[(past_events['category'] == cat) & (past_events['event_date'] == req_date_str)]
                        if len(settled_today) == 0:
                            daily_flows[target_dt] -= amt
                    elif req_date < target_dt <= end_date:
                        daily_flows[target_dt] -= amt

                    if cur.month == 12:
                        cur = datetime(cur.year + 1, 1, 1)
                    else:
                        cur = datetime(cur.year, cur.month + 1, 1)

        # 4. Weekly / bi-weekly recurring expenses
        interval_cats = ['groceries', 'transport', 'dining']
        for cat in interval_cats:
            cat_evs = past_events[past_events['category'] == cat].sort_values('event_date')
            if len(cat_evs) >= 2:
                latest_ev = cat_evs.iloc[-1]
                if latest_ev['flexibility'] in ['stoppable', 'reducible', 'reducible_or_stoppable']:
                    min_amt = float(latest_ev['minimum_allowed_amount']) if pd.notna(latest_ev['minimum_allowed_amount']) else 0.0
                    flexible_events[latest_ev['event_id']] = {
                        'category': cat,
                        'event_id': latest_ev['event_id'],
                        'description': latest_ev['description'],
                        'flexibility': latest_ev['flexibility'],
                        'amount': float(latest_ev['amount']),
                        'minimum_allowed_amount': min_amt
                    }

                is_active = (cat in protected_cats)
                if not is_active:
                    continue

                dates = [parse_date(d) for d in cat_evs['event_date']]
                if len(dates) >= 3:
                    diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                    cadence = int(round(np.median(diffs[-5:])))
                else:
                    cadence = 7
                if cadence < 5:
                    cadence = 7

                # Exclude one-time massive bulk outliers from recurring rate
                amts = cat_evs['amount'].tolist()
                med_amt = float(np.median(amts))
                normal_amts = [a for a in amts[-5:] if a <= 2.2 * med_amt]
                amt = float(np.max(normal_amts)) if normal_amts else med_amt

                latest_dt = dates[-1]
                next_dt = latest_dt + timedelta(days=cadence)
                while next_dt <= end_date:
                    if next_dt == req_date:
                        settled_today = past_events[(past_events['category'] == cat) & (past_events['event_date'] == req_date_str)]
                        if len(settled_today) == 0 and next_dt in daily_flows:
                            daily_flows[next_dt] -= amt
                    elif next_dt > req_date and next_dt in daily_flows:
                        daily_flows[next_dt] -= amt
                    next_dt += timedelta(days=cadence)

        # 5. Recurring Salary
        if not salary_stopped:
            sched_sal = scheduled_events[scheduled_events['category'] == 'salary']
            if msg_info.get('salary_override_amount') is not None:
                sal_amt = msg_info['salary_override_amount']
            elif len(sched_sal) > 0:
                sal_amt = float(sched_sal.iloc[-1]['amount'])
            elif len(sal_evs) > 0:
                # Use regular salary, excluding one-time bonuses, arrears, or commissions
                regular_sal = sal_evs[~sal_evs['description'].str.contains(r'bonus|arrears|prorated|commission|overtime', case=False, na=False)]
                if len(regular_sal) > 0:
                    sal_amt = float(regular_sal.iloc[-1]['amount'])
                else:
                    sal_amt = float(sal_evs.iloc[-1]['amount'])
            else:
                sal_amt = 0.0

            if sal_amt > 0:
                if msg_info.get('salary_override_date'):
                    sal_dom = parse_date(msg_info['salary_override_date']).day
                elif len(sched_sal) > 0:
                    sal_dom = parse_date(sched_sal.iloc[-1]['event_date']).day
                elif len(sal_evs) > 0:
                    regular_sal = sal_evs[~sal_evs['description'].str.contains(r'bonus|arrears|prorated|commission|overtime', case=False, na=False)]
                    if len(regular_sal) > 0:
                        doms = [parse_date(d).day for d in regular_sal['event_date']]
                        sal_dom = int(pd.Series(doms).mode()[0])
                    else:
                        sal_dom = parse_date(sal_evs.iloc[-1]['event_date']).day
                else:
                    sal_dom = 15

                cur = req_date
                import calendar
                while cur <= end_date:
                    max_d = calendar.monthrange(cur.year, cur.month)[1]
                    target_dt = datetime(cur.year, cur.month, min(sal_dom, max_d))

                    if req_date < target_dt <= end_date:
                        if (target_dt.year, target_dt.month) not in known_salary_dates:
                            daily_flows[target_dt] += sal_amt

                    if cur.month == 12:
                        cur = datetime(cur.year + 1, 1, 1)
                    else:
                        cur = datetime(cur.year, cur.month + 1, 1)

        return daily_flows, flexible_events, known_salary_dates
