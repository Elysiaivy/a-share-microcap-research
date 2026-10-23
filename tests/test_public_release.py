import unittest,tempfile,json
from pathlib import Path
from microcap.preflight import require_full_inputs
from scripts.verify_public import verify_results

class PublicReleaseTests(unittest.TestCase):
    def test_public_results_reconcile(self):
        self.assertEqual(verify_results()['winner'],'R001_100')

    def test_missing_full_inputs_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'result recomputation only'):
                require_full_inputs(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_online_does_not_bypass_historical_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                require_full_inputs(Path(d),online=True)

    def test_public_notebook_has_no_raw_file_dependency(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'research.ipynb').read_text())
        src='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertNotIn('data/prepared',src)
        self.assertNotIn('results/trades.csv',src)
