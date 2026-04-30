# Nifty 500 — EMA 200/400 1H Backtest Aggregate

_Generated: 2026-05-01 IST (cycle-breakdown re-run)_

Source: Yahoo chart API, 1H bars, 720 calendar days each.
Backtest harness enforces target exits (1:3 RR equity, 5000pt index)
and EMA400 close-exits.

## Headline

| Metric | Value |
|--------|-------|
| Symbols in cache | 504 |
| Symbols processed (Yahoo returned data) | 476 |
| Symbols skipped (no Yahoo 1H data) | 28 |
| Symbols with trades | 458 |
| Symbols flat / no-trade | 18 |
| Stock-level profitable | 246 |
| Stock-level losing | 212 |
| Total closed trades | 3491 |
| Trade-level winners | 1129 |
| Trade-level win rate | **32.3%** |
| Target hits | 1000 |
| EMA400 close-exits | 2491 |
| Total P&L per unit | **13557.22** |

## Top 10 P&L

| Symbol | Signals | Closed | Winners | Tgt | EMA | P&L |
|--------|---------|--------|---------|-----|-----|-----|
| HONAUT.NS | 36 | 9 | 5 | 4 | 5 | 8496.49 |
| ABBOTINDIA.NS | 48 | 7 | 3 | 3 | 4 | 2590.75 |
| ABB.NS | 51 | 12 | 3 | 3 | 9 | 2082.08 |
| SOLARINDS.NS | 28 | 5 | 3 | 3 | 2 | 1773.91 |
| PERSISTENT.NS | 45 | 9 | 4 | 3 | 6 | 1376.55 |
| GILLETTE.NS | 27 | 5 | 3 | 1 | 4 | 1311.08 |
| GODFRYPHLP.NS | 38 | 8 | 5 | 5 | 3 | 1259.67 |
| THERMAX.NS | 54 | 13 | 7 | 6 | 7 | 1202.00 |
| AJANTPHARM.NS | 32 | 8 | 6 | 6 | 2 | 1144.70 |
| KAYNES.NS | 33 | 6 | 2 | 2 | 4 | 1124.55 |

## Bottom 10 P&L

| Symbol | Signals | Closed | Winners | Tgt | EMA | P&L |
|--------|---------|--------|---------|-----|-----|-----|
| 3MINDIA.NS | 46 | 9 | 0 | 0 | 9 | -6476.05 |
| PAGEIND.NS | 33 | 6 | 0 | 0 | 6 | -5518.65 |
| SHREECEM.NS | 50 | 12 | 4 | 2 | 10 | -2536.72 |
| DIXON.NS | 30 | 6 | 1 | 0 | 6 | -2108.00 |
| FORCEMOT.NS | 23 | 3 | 0 | 0 | 3 | -2051.05 |
| FLUOROCHEM.NS | 73 | 13 | 0 | 0 | 13 | -1219.65 |
| SUPREMEIND.NS | 32 | 5 | 0 | 0 | 5 | -1165.25 |
| LINDEINDIA.NS | 47 | 10 | 2 | 2 | 8 | -1159.23 |
| SUNDARMFIN.NS | 47 | 9 | 2 | 1 | 8 | -1105.40 |
| APARINDS.NS | 24 | 4 | 1 | 1 | 3 | -1100.00 |

## Per-stock detail

Each per-stock report at `exports/backtests/nifty500_full/<symbol>.md`
now includes a **Strategy Cycles** section grouping signals into BUY
and SELL cycles with the full stage chain (trend ID → first alert →
retest 1 → first entry → retest 2 → second entry → exit), each row
showing time, price, EMA200, EMA400, and note.

Full per-symbol table in `nifty500_full/_summary.md`.
