from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import date, datetime

class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"

class PaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"

@dataclass
class SpendingChange:
    action: str  # "stop" or "reduce_to"
    event_id: str
    new_amount: Optional[float] = None

    def to_str(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        elif self.action == "reduce_to":
            amt_str = f"{self.new_amount:.2f}" if self.new_amount is not None else "0"
            if amt_str.endswith(".00"):
                amt_str = amt_str[:-3]
            elif amt_str.endswith("0") and "." in amt_str:
                amt_str = amt_str[:-1]
            return f"reduce_to:{self.event_id}:{amt_str}"
        return ""

@dataclass
class DecisionResult:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: str  # "YYYY-MM-DD:amount|..." or "none"
    earliest_date_for_full_payment: str  # "YYYY-MM-DD" or ""
    spending_changes_needed: str  # "stop:..." or "none"
    decision_explanation: str

@dataclass
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[float]

@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # debit, credit, non_cash
    amount: float
    currency: str
    event_date: str
    settlement_date: str
    status: str  # settled, pending, scheduled, cancelled, failed, unrealized
    linked_event_id: Optional[str] = None
    flexibility: str = "fixed"  # fixed, reducible, stoppable, reducible_or_stoppable
    minimum_allowed_amount: Optional[float] = None

@dataclass
class RequestPaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # full_payment, installments
    payment_amount: float
    number_of_payments: int
    first_payment_date: str
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float

@dataclass
class UserRequest:
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: float
    desired_completion_date: str
    allows_partial_payment: bool
    request_text: str
