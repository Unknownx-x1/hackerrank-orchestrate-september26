import re
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

class MessageAnalyzer:
    def __init__(self, messages_path: str = "dataset/messages.csv"):
        self.messages_df = pd.read_csv(messages_path)

    def get_user_messages(self, user_id: str, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        # Filter messages for this user or request
        cond = (self.messages_df['user_id'] == user_id)
        if request_id:
            cond = cond | (self.messages_df['request_id'] == request_id)
        sub = self.messages_df[cond].sort_values('sent_at')
        return sub.to_dict('records')

    def analyze_user_messages(self, user_id: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        msgs = self.get_user_messages(user_id, request_id)
        info = {
            'salary_override_amount': None,
            'salary_override_date': None,
            'salary_stopped': False,
            'rent_multiplier': 1.0,
            'ignore_events': set(),
            'notes': []
        }

        for m in msgs:
            text = str(m.get('message_text', ''))
            
            # Check seasonal contract end / unconfirmed gig payouts / salary stopped
            if re.search(r'(seasonal contract has ended|no off-season income|kontrak musiman berakhir|payout is still pending|can change until the payout is closed|balance isn.?t withdrawable)', text, re.I):
                info['salary_stopped'] = True
                info['notes'].append("Salary stopped or unconfirmed gig payout")

            # Check salary amount override (English)
            # e.g., "temporary monthly pay is EUR 1037.52", "salary is reduced to EUR 1422.85", "salary of EUR 2717 resumes on 2025-08-15", "first salary will be EUR 1661"
            m_sal_eur = re.search(r'(?:pay is|reduced to|salary of|salary will be|regular salary for the next payroll is)\s*(?:EUR|USD|ZAR)\s*([\d,]+(?:\.\d+)?)', text, re.I)
            if m_sal_eur:
                val = float(m_sal_eur.group(1).replace(',', ''))
                info['salary_override_amount'] = val
                info['notes'].append(f"Salary override: {val}")

            # Check salary amount override (Indonesian)
            # e.g., "Gaji bulanan Anda naik menjadi IDR 42750000", "pembayaran faktur sebesar IDR 30780000"
            # Exclude when commissions are unapproved/pending per challenge rules
            if not re.search(r'(?:komisi.*belum disetujui|transaksi yang masih berjalan)', text, re.I):
                m_sal_idr = re.search(r'(?:naik menjadi|sebesar)\s*IDR\s*([\d,]+(?:\.\d+)?)', text, re.I)
                if m_sal_idr:
                    val = float(m_sal_idr.group(1).replace(',', ''))
                    info['salary_override_amount'] = val
                    info['notes'].append(f"Salary override IDR: {val}")

            # Check salary date override
            # e.g., "confirmed salary is now expected on 2024-09-23", "Penyelesaian diperkirakan pada 2025-08-15"
            m_date = re.search(r'(?:expected on|berlaku mulai|resumes on|credit date is|pada)\s*(\d{4}-\d{2}-\d{2})', text, re.I)
            if m_date:
                d_str = m_date.group(1)
                info['salary_override_date'] = d_str
                info['notes'].append(f"Salary date override: {d_str}")

            # Check rent increase
            # e.g., "The renewed lease increases monthly rent by 12%"
            m_rent = re.search(r'increases monthly rent by (\d+)%', text, re.I)
            if m_rent:
                pct = float(m_rent.group(1))
                info['rent_multiplier'] = 1.0 + (pct / 100.0)
                info['notes'].append(f"Rent multiplier: {info['rent_multiplier']}")

            # Check related events to ignore (e.g. unconfirmed refunds, prizes, bonuses)
            ev_id = m.get('related_event_id')
            if pd.notna(ev_id) and str(ev_id).strip():
                ev_str = str(ev_id).strip()
                if re.search(r'(not reached your account|still pending|processing|not been credited|displayed market value|unrealized)', text, re.I):
                    info['ignore_events'].add(ev_str)
                    info['notes'].append(f"Ignored unconfirmed event: {ev_str}")

        return info
