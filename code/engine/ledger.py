import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from engine.fx import FXConverter
from perception.message_analyzer import MessageAnalyzer

def parse_date(d: str) -> datetime:
    return datetime.strptime(str(d).strip(), "%Y-%m-%d")

class FinancialLedger:
    def __init__(
        self,
        events_path: str = "dataset/financial_events.csv",
        profiles_path: str = "dataset/financial_profiles.csv",
        rates_path: str = "dataset/exchange_rates.csv",
        messages_path: str = "dataset/messages.csv",
        images_json_path: str = "code/perception/extracted_images.json"
    ):
        self.events_df = pd.read_csv(events_path)
        self.profiles_df = pd.read_csv(profiles_path)
        self.fx = FXConverter(rates_path)
        self.msg_analyzer = MessageAnalyzer(messages_path)
        
        # Load verified image amounts
        try:
            with open(images_json_path, 'r', encoding='utf-8') as f:
                self.images_data = json.load(f)
        except:
            self.images_data = {}

        # Fill missing amounts
        for idx, row in self.events_df[self.events_df['amount'].isna()].iterrows():
            ev_id = str(row['event_id']).strip()
            if ev_id in self.images_data:
                self.events_df.loc[idx, 'amount'] = self.images_data[ev_id]['amount']

    def get_user_state(self, user_id: str, req_date_str: str, req_id: Optional[str] = None) -> Dict[str, Any]:
        req_date = parse_date(req_date_str)
        prof_rows = self.profiles_df[self.profiles_df['user_id'] == user_id]
        if len(prof_rows) == 0:
            raise ValueError(f"Profile not found for user {user_id}")
        prof = prof_rows.iloc[0]

        home_curr = str(prof['home_currency']).strip()
        b0 = float(prof['current_available_balance'])
        min_keep = float(prof['minimum_balance_to_keep'])
        user_methods = str(prof['payment_methods_user_will_consider']).split('|') if pd.notna(prof['payment_methods_user_will_consider']) else []
        
        max_inst = prof['max_installment_months']
        max_inst_months = float(max_inst) if pd.notna(max_inst) and str(max_inst).strip() != '' else None

        prot_cats = set(str(prof['expense_categories_to_protect']).split('|')) if pd.notna(prof['expense_categories_to_protect']) else set()
        stop_cats = set(str(prof['expense_categories_user_is_willing_to_stop']).split('|')) if pd.notna(prof['expense_categories_user_is_willing_to_stop']) else set()
        reduce_cats = set(str(prof['expense_categories_user_is_willing_to_reduce']).split('|')) if pd.notna(prof['expense_categories_user_is_willing_to_reduce']) else set()

        u_events = self.events_df[self.events_df['user_id'] == user_id].copy()
        msg_info = self.msg_analyzer.analyze_user_messages(user_id, req_id)

        # Convert foreign currency events to home_currency
        for idx, row in u_events.iterrows():
            c = str(row['currency']).strip()
            s_date = str(row['settlement_date']).strip() if pd.notna(row['settlement_date']) else str(row['event_date']).strip()
            amt = float(row['amount']) if pd.notna(row['amount']) else 0.0
            if c != home_curr:
                u_events.loc[idx, 'amount'] = self.fx.convert(amt, c, home_curr, s_date)
                u_events.loc[idx, 'currency'] = home_curr

        # Ignore events flagged by messages
        if msg_info['ignore_events']:
            u_events = u_events[~u_events['event_id'].isin(msg_info['ignore_events'])]

        # Filter cash-only and valid records
        valid_events = u_events[~u_events['status'].isin(['cancelled', 'failed', 'unrealized'])]
        valid_events = valid_events[valid_events['direction'].isin(['debit', 'credit'])]

        past_events = valid_events[(valid_events['status'] == 'settled') & (pd.to_datetime(valid_events['event_date']) <= req_date)]
        
        # Pending debits on or before request_date
        pending_debits = valid_events[(valid_events['status'] == 'pending') & (valid_events['direction'] == 'debit') & (pd.to_datetime(valid_events['event_date']) <= req_date)]
        pending_reserve = pending_debits['amount'].sum()

        scheduled_future = valid_events[(valid_events['status'] == 'scheduled') & (pd.to_datetime(valid_events['event_date']) >= req_date)]

        return {
            'user_id': user_id,
            'home_currency': home_curr,
            'b0': b0,
            'pending_reserve': pending_reserve,
            'effective_b0': b0 - pending_reserve,
            'min_keep': min_keep,
            'user_methods': user_methods,
            'max_inst_months': max_inst_months,
            'prot_cats': prot_cats,
            'stop_cats': stop_cats,
            'reduce_cats': reduce_cats,
            'past_events': past_events,
            'scheduled_future': scheduled_future,
            'msg_info': msg_info
        }
