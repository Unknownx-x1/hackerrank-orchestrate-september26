from typing import Dict, Any, Optional
from datetime import datetime

def format_currency_amount(amount: float, currency: str) -> str:
    # Format according to standard comma formatting
    if currency == "IDR":
        return f"{currency} {amount:,.0f}"
    elif currency == "INR":
        # Indian formatting or standard formatting
        if amount.is_integer():
            return f"{currency} {int(amount):,}"
        else:
            return f"{currency} {amount:,.2f}"
    else:
        if amount.is_integer():
            return f"{currency} {int(amount):,}"
        else:
            return f"{currency} {amount:,.2f}"

def format_display_date(date_str: str) -> str:
    # Converts YYYY-MM-DD to "15 November 2019" or "8 August 2025"
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return f"{dt.day} {dt.strftime('%B %Y')}"
    except:
        return date_str

class ExplanationGenerator:
    def __init__(self):
        pass

    def generate(
        self,
        method: str,
        status: str,
        plan: str,
        safe_amt: float,
        req_amt: float,
        currency: str,
        min_keep: float,
        deadline_str: str,
        spending_changes: str,
        flexible_events: Dict[str, Dict[str, Any]],
        earliest_date_str: str
    ) -> str:
        curr_min = format_currency_amount(min_keep, currency)
        curr_req = format_currency_amount(req_amt, currency)
        curr_safe = format_currency_amount(safe_amt, currency)

        # 1. Full payment today
        if method == "full_payment":
            if spending_changes and spending_changes != "none":
                # Detail spending changes
                change_descs = []
                for ch in spending_changes.split("|"):
                    parts = ch.split(":")
                    action = parts[0]
                    ev_id = parts[1]
                    ev_desc = flexible_events.get(ev_id, {}).get('description', 'subscription')
                    if action == "stop":
                        change_descs.append(f"Stop the {ev_desc.lower()}")
                    elif action == "reduce_to":
                        new_a = float(parts[2])
                        change_descs.append(f"reduce the {ev_desc.lower()} to {format_currency_amount(new_a, currency)}")
                changes_text = " and ".join(change_descs)
                return f"{changes_text}, then pay {curr_req} today. This leaves at least {curr_min} available."
            else:
                return f"Pay {curr_req} today. This leaves at least {curr_min} available over the next 90 days."

        # 2. Partial payment
        elif method == "partial_payment":
            # parse two payments from plan
            plan_parts = plan.split("|")
            p1_parts = plan_parts[0].split(":")
            p2_parts = plan_parts[1].split(":")
            p1_amt = format_currency_amount(float(p1_parts[1]), currency)
            p2_amt = format_currency_amount(float(p2_parts[1]), currency)
            p2_date = format_display_date(p2_parts[0])
            return f"Pay {p1_amt} today and the remaining {p2_amt} on {p2_date}. This completes the full request and keeps the {curr_min} minimum protected."

        # 3. Installments
        elif method == "installments":
            plan_parts = plan.split("|")
            n_inst = len(plan_parts)
            first_p = plan_parts[0].split(":")
            inst_amt = format_currency_amount(float(first_p[1]), currency)
            start_date = format_display_date(first_p[0])
            return f"Use {n_inst} installments of {inst_amt}, starting {start_date}. This leaves at least {curr_min} available."

        # 4. Wait
        elif method == "wait":
            wait_date = format_display_date(earliest_date_str)
            return f"Pay {curr_req} in full on {wait_date}. Paying earlier would take the balance below the {curr_min} minimum."

        # 5. Not recommended
        else:
            disp_deadline = format_display_date(deadline_str)
            if safe_amt > 0:
                return f"Do not proceed with the {curr_req} request. Although {curr_safe} is available today, the full amount cannot be completed safely within 90 days."
            else:
                return f"Do not make this payment by {disp_deadline}. None of the available options keeps the {curr_min} minimum protected."
