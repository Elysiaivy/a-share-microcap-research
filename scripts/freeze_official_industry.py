"""Freeze early CSRC industry tables; parse with the bundled pdfplumber runtime.

This is a one-off acquisition command, not a backtest dependency. Publication
dates come from each CSRC article, not the period printed inside its PDF.
"""
from pathlib import Path
import hashlib
import json
import re
from urllib.request import urlopen
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data/repair_sources/point_in_time/official_industry'
SOURCES = [
    ('2014Q3', '2014-09-30', '2014-10-13', '1452017', '1616066704471_41584'),
    ('2014Q4', '2014-12-31', '2015-01-22', '1452016', '1616066703162_60057'),
    ('2015Q1', '2015-03-31', '2015-05-04', '1452015', '1616066702277_76098'),
    ('2015Q2', '2015-06-30', '2015-07-20', '1452014', '1616066701142_55290'),
    ('2015Q3', '2015-09-30', '2015-10-27', '1452013', '1616066699533_72958'),
    ('2015Q4', '2015-12-31', '2016-03-04', '1452012', '1616066698254_18672'),
    ('2016Q1', '2016-03-31', '2016-05-06', '1452011', '1616066696488_16479'),
]


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest, records = [], []
    for period, observed, published, article, file in SOURCES:
        page = f'https://www.csrc.gov.cn/csrc/c100103/c{article}/content.shtml'
        url = f'https://www.csrc.gov.cn/csrc/c100103/c{article}/{article}/files/{file}.pdf'
        path = TARGET / f'{period}.pdf'
        if not path.exists():
            with urlopen(url, timeout=45) as response:
                content = response.read()
            if not content.startswith(b'%PDF'):
                raise ValueError('Official source did not return a PDF')
            path.write_bytes(content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sector, category, label = None, None, None
        parsed, printed = [], set()
        with pdfplumber.open(path) as doc:
            for page_number, pdf_page in enumerate(doc.pages, 1):
                text = pdf_page.extract_text() or ''
                printed.update(re.findall(r'(?<!\d)\d{6}(?!\d)', text))
                for table in pdf_page.extract_tables():
                    for row in table:
                        if len(row) != 5 or not re.fullmatch(r'\d{6}', str(row[3]).strip()):
                            continue
                        a, b, c, code, name = row
                        if a:
                            match = re.search(r'[（(]([A-S])[）)]', a)
                            if match:
                                sector = match.group(1)
                        if b:
                            category = re.sub(r'\s+', '', b)
                        if c:
                            label = re.sub(r'\s+', '', c)
                        if not sector or not category or not label:
                            raise ValueError(f'Unresolved industry heading: {period} {row}')
                        code = str(code).strip()
                        parsed.append(dict(period=period, observation_date=observed,
                            publication_date=published, ts_code=code + ('.SH' if code.startswith(('6','9')) else '.SZ'),
                            name=re.sub(r'\s+', '', str(name)), industry=label,
                            industry_code=category, industry_sector=sector,
                            is_financial=int(sector == 'J'), source_sha256=digest,
                            source_url=url, publication_url=page, page=page_number))
        parsed_codes = {row['ts_code'].split('.')[0] for row in parsed}
        if printed != parsed_codes or len(parsed_codes) != len(parsed):
            raise ValueError(f'Incomplete or duplicate extraction {period}: {len(parsed_codes)} / {len(printed)}')
        import pandas as pd
        frame = pd.DataFrame(parsed)
        frame.to_csv(TARGET / f'{period}.csv', index=False)
        manifest.append(dict(period=period, observation_date=observed, publication_date=published,
            path=str(path.relative_to(ROOT)), sha256=digest, url=url, publication_url=page,
            rows=len(parsed), parsed_path=str((TARGET/f'{period}.csv').relative_to(ROOT)),
            parsed_sha256=hashlib.sha256((TARGET/f'{period}.csv').read_bytes()).hexdigest()))
        records.extend(parsed)
        print(period, len(parsed), 'financial', int(frame.is_financial.sum()), flush=True)
    import pandas as pd
    pd.DataFrame(records).to_csv(TARGET/'classifications.csv', index=False)
    (TARGET/'manifest.json').write_text(json.dumps(dict(sources=manifest,
        policy='As-of last officially published classification, usable after publication; never backdate to quarter end',
        limitation='Official classifications are quarterly and can lag economic changes. Tushare daily historical snapshots take priority when available.'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
