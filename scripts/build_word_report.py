"""Single entry point for the latest 14-page report from published result tables.

Run build_word_figures.py first if figures must be regenerated. PDF export uses
LibreOffice separately; see RUN_GUIDE.md. No strategy evaluation is performed.
"""
from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/_report_body.py'), run_name='__main__')
runpy.run_path(str(ROOT/'scripts/_report_applicability.py'), run_name='__main__')

import json, hashlib
for old in ['results/word_report_manifest.json','results/applicability/document_manifest.json']:
    (ROOT/old).unlink(missing_ok=True)
paths = [ROOT/'config.json', *sorted((ROOT/'results').rglob('*.csv')), *sorted((ROOT/'results/word_figures').glob('*.png'))]
paths += [ROOT/'scripts'/name for name in ['build_word_report.py','_report_body.py','_report_applicability.py']]
paths += [ROOT/'results'/name for name in ['selection.json','paired_tests.json','quality.json','verification.json','slide_payload.json','supplement/manifest.json','applicability/summary.json']]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'results/report_sources.json').write_text(json.dumps({'scope':'latest Word built from public result artifacts; historical data validation is separate','inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths},'output':{'path':'docs/research_report.docx','sha256':sha(ROOT/'docs/research_report.docx')}},ensure_ascii=False,indent=2)+'\n')
