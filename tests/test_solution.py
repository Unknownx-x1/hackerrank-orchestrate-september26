import unittest
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from engine.ledger import FinancialLedger
from engine.recurrence import RecurrenceEngine
from engine.simulator import Simulator
from engine.optimizer import PlanOptimizer
from explanation.generator import ExplanationGenerator
from main import process_single_request

class TestBuyOrWaitSolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = FinancialLedger()
        cls.recurrence = RecurrenceEngine()
        cls.simulator = Simulator(horizon_days=90)
        cls.optimizer = PlanOptimizer()
        cls.explainer = ExplanationGenerator()
        cls.options_df = pd.read_csv('dataset/request_payment_options.csv')
        cls.samples_df = pd.read_csv('dataset/sample_requests.csv')

    def test_output_csv_contract(self):
        self.assertTrue(os.path.exists('output.csv'), 'output.csv must exist')
        df = pd.read_csv('dataset/requests.csv')
        out_df = pd.read_csv('output.csv')
        self.assertEqual(len(out_df), len(df), 'output.csv must have exactly 250 rows')
        expected_cols = [
            'request_id', 'amount_safe_to_pay', 'affordability_status',
            'recommended_payment_method', 'payment_plan',
            'earliest_date_for_full_payment', 'spending_changes_needed',
            'decision_explanation'
        ]
        self.assertEqual(list(out_df.columns), expected_cols)

    def test_sample_benchmark(self):
        matches_method = 0
        matches_status = 0
        total = len(self.samples_df)
        for _, row in self.samples_df.iterrows():
            res = process_single_request(
                row=row, ledger=self.ledger, recurrence=self.recurrence,
                simulator=self.simulator, optimizer=self.optimizer,
                explainer=self.explainer, options_df=self.options_df
            )
            if res.recommended_payment_method.value == row['recommended_payment_method']:
                matches_method += 1
            if res.affordability_status.value == row['affordability_status']:
                matches_status += 1

        print(f'\n[Test Suite] Public Samples Payment Method Accuracy: {matches_method}/{total} ({matches_method/total*100:.1f}%)')
        print(f'[Test Suite] Public Samples Affordability Status Accuracy: {matches_status}/{total} ({matches_status/total*100:.1f}%)')
        self.assertGreaterEqual(matches_method / total, 0.90)
        self.assertGreaterEqual(matches_status / total, 0.90)

    def test_partial_payment_rule(self):
        req_19 = self.samples_df[self.samples_df['request_id'] == 'request_19'].iloc[0]
        res = process_single_request(
            row=req_19, ledger=self.ledger, recurrence=self.recurrence,
            simulator=self.simulator, optimizer=self.optimizer,
            explainer=self.explainer, options_df=self.options_df
        )
        self.assertEqual(res.recommended_payment_method.value, 'partial_payment')
        parts = res.payment_plan.split('|')
        self.assertEqual(len(parts), 2, 'Partial payment must have exactly 2 payments')
        p1 = float(parts[0].split(':')[1])
        p2 = float(parts[1].split(':')[1])
        self.assertAlmostEqual(p1 + p2, float(req_19['requested_amount']), delta=0.01)

if __name__ == '__main__':
    unittest.main()
