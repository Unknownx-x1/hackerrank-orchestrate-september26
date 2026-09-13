import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from models import AffordabilityStatus, PaymentMethod, SpendingChange, DecisionResult

def parse_date(d: str) -> datetime:
    return datetime.strptime(str(d).strip(), "%Y-%m-%d")

def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

class PlanOptimizer:
    def __init__(self):
        pass

    def evaluate(
        self,
        req_id: str,
        req_date_str: str,
        req_amt: float,
        deadline_str: str,
        allows_partial: bool,
        user_methods: List[str],
        max_inst_months: Optional[float],
        b0: float,
        min_keep: float,
        safe_amt: float,
        earliest_full_date_str: str,
        daily_flows: Dict[datetime, float],
        flexible_events: Dict[str, Dict[str, Any]],
        options_for_req: pd.DataFrame,
        stop_categories: List[str],
        reduce_categories: List[str],
        simulator: Any
    ) -> DecisionResult:
        req_date = parse_date(req_date_str)
        deadline = parse_date(deadline_str)
        candidates = []

        # Candidate 1: full_payment today
        if "full_payment" in user_methods and safe_amt >= req_amt - 1e-4:
            plan_str = f"{req_date_str}:{req_amt:.2f}".rstrip('0').rstrip('.') if '.' in f"{req_amt:.2f}" else f"{req_date_str}:{int(req_amt)}"
            # format cleanly
            amt_str = f"{req_amt:.2f}".rstrip('0').rstrip('.')
            plan_str = f"{req_date_str}:{amt_str}"
            candidates.append({
                'method': PaymentMethod.FULL_PAYMENT,
                'status': AffordabilityStatus.AFFORDABLE_NOW,
                'plan': plan_str,
                'earliest_date': req_date_str,
                'spending_changes': "none",
                'total_paid': req_amt,
                'start_date': req_date,
                'num_payments': 1,
                'completes_by_deadline': True,
                'has_spending_changes': False,
                'option_id': "0"
            })

        # Candidate 2: partial_payment
        if (allows_partial and "partial_payment" in user_methods and 
            0 < safe_amt < req_amt and earliest_full_date_str):
            earliest_dt = parse_date(earliest_full_date_str)
            if earliest_dt <= deadline:
                rem_amt = req_amt - safe_amt
                safe_str = f"{safe_amt:.2f}".rstrip('0').rstrip('.')
                rem_str = f"{rem_amt:.2f}".rstrip('0').rstrip('.')
                plan_str = f"{req_date_str}:{safe_str}|{earliest_full_date_str}:{rem_str}"
                candidates.append({
                    'method': PaymentMethod.PARTIAL_PAYMENT,
                    'status': AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                    'plan': plan_str,
                    'earliest_date': earliest_full_date_str,
                    'spending_changes': "none",
                    'total_paid': req_amt,
                    'start_date': req_date,
                    'num_payments': 2,
                    'completes_by_deadline': True,
                    'has_spending_changes': False,
                    'option_id': "1"
                })

        # Candidate 3: installments from request_payment_options
        if "installments" in user_methods and len(options_for_req) > 0:
            for _, opt in options_for_req.iterrows():
                if str(opt['payment_method']).strip() != 'installments':
                    continue
                opt_id = str(opt['payment_option_id']).strip()
                n_payments = int(opt['number_of_payments'])
                p_amt = float(opt['payment_amount'])
                first_date_str = str(opt['first_payment_date']).strip()
                freq_days = int(opt['payment_frequency_days']) if pd.notna(opt['payment_frequency_days']) else 30
                total_payable = float(opt['total_payable_amount'])

                # Check max_installment_months constraint
                if max_inst_months is not None and max_inst_months > 0:
                    # Duration in months approximately
                    total_days = (n_payments - 1) * freq_days
                    duration_months = total_days / 30.0
                    if duration_months > max_inst_months + 0.1:
                        continue

                # Generate payment schedule
                p_first_dt = parse_date(first_date_str)
                payments = []
                for i in range(n_payments):
                    p_dt = p_first_dt + timedelta(days=i * freq_days)
                    payments.append((p_dt, p_amt))

                last_payment_dt = payments[-1][0]
                completes_by_deadline = (last_payment_dt <= deadline)

                # Test plan feasibility
                is_safe = simulator.test_plan_feasibility(
                    b0, daily_flows, payments, min_keep, req_date
                )

                if is_safe:
                    # Format plan string
                    plan_parts = []
                    for p_dt, amt in payments:
                        amt_s = f"{amt:.2f}".rstrip('0').rstrip('.')
                        plan_parts.append(f"{format_date(p_dt)}:{amt_s}")
                    plan_str = "|".join(plan_parts)

                    candidates.append({
                        'method': PaymentMethod.INSTALLMENTS,
                        'status': AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                        'plan': plan_str,
                        'earliest_date': earliest_full_date_str if earliest_full_date_str else "",
                        'spending_changes': "none",
                        'total_paid': total_payable,
                        'start_date': p_first_dt,
                        'num_payments': n_payments,
                        'completes_by_deadline': completes_by_deadline,
                        'has_spending_changes': False,
                        'option_id': opt_id
                    })

        # Candidate 4: spending changes for full_payment today
        if "full_payment" in user_methods and safe_amt < req_amt:
            deficit = req_amt - safe_amt
            # Look for stoppable/reducible actions
            change_candidates = []
            for ev_id, ev in flexible_events.items():
                cat = ev['category']
                if ev['flexibility'] in ['stoppable', 'reducible_or_stoppable'] and cat in stop_categories:
                    change_candidates.append({
                        'action': 'stop',
                        'event_id': ev_id,
                        'savings': ev['amount'],
                        'new_amount': None
                    })
                elif ev['flexibility'] in ['reducible', 'reducible_or_stoppable'] and cat in reduce_categories:
                    savings = ev['amount'] - ev['minimum_allowed_amount']
                    if savings > 0:
                        change_candidates.append({
                            'action': 'reduce_to',
                            'event_id': ev_id,
                            'savings': savings,
                            'new_amount': ev['minimum_allowed_amount']
                        })

            # Try 1, 2, or 3 changes
            best_changes = None
            from itertools import combinations
            for k in range(1, min(4, len(change_candidates) + 1)):
                for combo in combinations(change_candidates, k):
                    # Events must be unique
                    ev_set = set(c['event_id'] for c in combo)
                    if len(ev_set) < k:
                        continue
                    tot_sav = sum(c['savings'] for c in combo)
                    if tot_sav >= deficit - 1e-4:
                        best_changes = list(combo)
                        break
                if best_changes:
                    break

            if best_changes:
                parts = []
                for c in best_changes:
                    if c['action'] == 'stop':
                        parts.append(f"stop:{c['event_id']}")
                    else:
                        amt_s = f"{c['new_amount']:.2f}".rstrip('0').rstrip('.')
                        parts.append(f"reduce_to:{c['event_id']}:{amt_s}")
                spending_str = "|".join(parts)
                amt_s = f"{req_amt:.2f}".rstrip('0').rstrip('.')
                plan_str = f"{req_date_str}:{amt_s}"

                candidates.append({
                    'method': PaymentMethod.FULL_PAYMENT,
                    'status': AffordabilityStatus.AFFORDABLE_WITH_PLAN,
                    'plan': plan_str,
                    'earliest_date': earliest_full_date_str if earliest_full_date_str else "",
                    'spending_changes': spending_str,
                    'total_paid': req_amt,
                    'start_date': req_date,
                    'num_payments': 1,
                    'completes_by_deadline': True,
                    'has_spending_changes': True,
                    'option_id': "99"
                })

        # Candidate 5: wait (full payment later)
        if "full_payment" in user_methods and earliest_full_date_str:
            earliest_dt = parse_date(earliest_full_date_str)
            if earliest_dt > req_date and earliest_dt <= deadline:
                amt_s = f"{req_amt:.2f}".rstrip('0').rstrip('.')
                plan_str = f"{earliest_full_date_str}:{amt_s}"
                candidates.append({
                    'method': PaymentMethod.WAIT,
                    'status': AffordabilityStatus.AFFORDABLE_LATER,
                    'plan': plan_str,
                    'earliest_date': earliest_full_date_str,
                    'spending_changes': "none",
                    'total_paid': req_amt,
                    'start_date': earliest_dt,
                    'num_payments': 1,
                    'completes_by_deadline': True,
                    'has_spending_changes': False,
                    'option_id': "999"
                })

        # Ranking function implementing the 6 challenge tie-breakers:
        # 1. Complete full request by desired_completion_date (True first)
        # 2. Require no spending changes (False has_spending_changes first)
        # 3. Minimize total amount paid
        # 4. Start payment earlier (earlier start_date)
        # 5. Use fewer payments (smaller num_payments)
        # 6. Lowest payment_option_id
        if candidates:
            def sort_key(cand):
                return (
                    0 if cand['completes_by_deadline'] else 1,
                    1 if cand['has_spending_changes'] else 0,
                    cand['total_paid'],
                    cand['start_date'],
                    cand['num_payments'],
                    cand['option_id']
                )
            candidates.sort(key=sort_key)
            best = candidates[0]
            return DecisionResult(
                request_id=req_id,
                amount_safe_to_pay=safe_amt,
                affordability_status=best['status'],
                recommended_payment_method=best['method'],
                payment_plan=best['plan'],
                earliest_date_for_full_payment=best['earliest_date'],
                spending_changes_needed=best['spending_changes'],
                decision_explanation=""
            )

        # Fallback: not_recommended
        return DecisionResult(
            request_id=req_id,
            amount_safe_to_pay=safe_amt,
            affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
            recommended_payment_method=PaymentMethod.NOT_RECOMMENDED,
            payment_plan="none",
            earliest_date_for_full_payment=earliest_full_date_str if earliest_full_date_str else "",
            spending_changes_needed="none",
            decision_explanation=""
        )
