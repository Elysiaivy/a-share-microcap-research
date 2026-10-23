"""Supplement as-of capitalization evidence without mutating prepared panels.

Historical industry now comes from the dated sources used by `microcap prepare`.
This compatibility entry point only extends the current capitalization ledger.
Run `python3 -m microcap prepare` afterwards to recertify prepared inputs.
"""
from supplement_asof_cap import main

if __name__ == '__main__':
    main()
