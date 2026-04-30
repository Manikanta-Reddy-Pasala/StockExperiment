# Fyers Nifty 500 — EMA 200/400 1H Backtest

_Generated: 2026-05-01 IST (Fyers production data, container 77.42.45.12)_

Source: **Fyers API** via `Historical1HService` flow, 1H bars,
720 calendar days, chunked 95d per call. Same auth/token flow as production.

## Headline

| Metric | Fyers | Yahoo (prior) | Δ |
|--------|-------|--------------|---|
| Symbols processed | 498 | 476 | +22 |
| Total trades | 2695 | 3491 | -796 |
| Winners | 852 | 1129 | -277 |
| Win rate | **31.6%** | 32.3% | -0.7pp |
| Target hits | 758 | 1000 | -242 |
| EMA exits | 1937 | 2491 | -554 |
| Sum P&L per unit | 9429.11 | 13557.24 | -4128.13 |
| Profitable stocks | 247 | 246 | +1 |
| Losing stocks | 236 | 212 | +24 |

## Why fewer trades than Yahoo

Fyers intraday history caps at ~2 years vs Yahoo's ~3 years.
Per symbol Fyers returned ~3409 bars vs Yahoo ~5004 bars.
Less history → fewer crossovers → fewer trades. Win rate stable at ~32%.

## Top 10 P&L (Fyers)

| Symbol | Signals | Closed | Winners | Tgt | EMA | P&L |
|--------|---------|--------|---------|-----|-----|-----|
| MRF.NS | 25 | 4 | 2 | 1 | 3 | 9287.68 |
| ABBOTINDIA.NS | 31 | 5 | 4 | 4 | 1 | 6041.16 |
| NEULANDLAB.NS | 22 | 4 | 4 | 2 | 2 | 2152.23 |
| ABB.NS | 31 | 8 | 2 | 2 | 6 | 1961.22 |
| SOLARINDS.NS | 28 | 5 | 3 | 3 | 2 | 1773.57 |
| ZFCVINDIA.NS | 32 | 6 | 2 | 2 | 4 | 1708.48 |
| INDIGO.NS | 43 | 9 | 4 | 3 | 6 | 1600.69 |
| HONAUT.NS | 20 | 5 | 2 | 1 | 4 | 1566.12 |
| SHREECEM.NS | 29 | 7 | 3 | 2 | 5 | 1415.53 |
| GODFRYPHLP.NS | 19 | 4 | 3 | 3 | 1 | 1092.89 |

## Bottom 10 P&L (Fyers)

| Symbol | Signals | Closed | Winners | Tgt | EMA | P&L |
|--------|---------|--------|---------|-----|-----|-----|
| BOSCHLTD.NS | 47 | 8 | 0 | 0 | 8 | -7848.70 |
| PTCIL.NS | 40 | 7 | 0 | 0 | 7 | -4416.40 |
| PAGEIND.NS | 20 | 3 | 0 | 0 | 3 | -4084.65 |
| DIXON.NS | 31 | 6 | 1 | 0 | 6 | -2107.05 |
| ULTRACEMCO.NS | 55 | 9 | 2 | 1 | 8 | -1347.12 |
| APARINDS.NS | 24 | 4 | 1 | 1 | 3 | -1094.45 |
| GILLETTE.NS | 16 | 3 | 1 | 0 | 3 | -875.80 |
| KAYNES.NS | 23 | 3 | 0 | 0 | 3 | -871.40 |
| TIMKEN.NS | 39 | 9 | 0 | 0 | 9 | -869.05 |
| SUNDARMFIN.NS | 42 | 8 | 2 | 1 | 7 | -855.53 |

## Per-stock detail

Per-stock reports at `exports/backtests/fyers_nifty500_full/<symbol>.md`,
each with **Strategy Cycles** breakdown (BUY + SELL stages).

## Validation
- Universe: 504 NSE Nifty 500 (official list)
- Source: Fyers API `interval='1h'` (production token, user_id=1)
- Auth flow: untouched (same path used by live trading)
- Container: trading_system_app on 77.42.45.12 (production stack)
- Production state: untouched (containers run baked image, host code
  changes don't affect them)
