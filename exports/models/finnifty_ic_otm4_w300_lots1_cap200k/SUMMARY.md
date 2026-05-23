# finnifty_ic_otm4_w300_lots1

FinNifty monthly Iron Condor rescaled to ₹200,000 investment capital
(peak-safe — every backtested entry was openable at the live broker).

## Strategy

- **Underlying:** FINNIFTY (Nifty Financial Services Index)
- **Setup:** Iron Condor monthly expiry, OTM4 body, W300 wings
- **Position size:** 1 lot (auto-sized from peak margin ≤ capital)
- **Stop:** 3× entry credit OR hold to expiry
- **Capital:** ₹200,000
- **Slippage:** realistic tiered (1× ATM → 15× >6% OTM)
- **Source:** rescaled from finnifty_ic_otm4_w300_lots5 backtest

## Result at ₹200,000

- **Started with:** ₹200,000
- **Ended with:** ₹595,937
- **Total profit:** ₹395,937
- **Total return:** +198.0 %
- **CAGR:** **+43.4 %**
- **Trades:** 35
- **Win rate:** 77.1 %
- **Avg P&L / trade:** ₹+11,312
- **Max drawdown:** -6.2 %

## Margin (SPAN+exposure approx — sweep.compute_ic_margin)

| Metric | Value |
|---|---:|
| Lots | **1** |
| Avg margin / trade | ₹72,507 |
| Peak margin / trade | ₹120,476 |
| Configured capital | ₹200,000 |
| Capital / peak ratio | 1.66× |

> ✅ Peak margin ≤ capital — every monthly entry was openable at the
> live broker. No skipped cycles.

## Yearly

| Year | Trades | Wins | WR | P&L | ROI |
|---|---:|---:|---:|---:|---:|
| 2023 | 8 | 5 | 62.5 % | ₹68,763 | +34.38 % |
| 2024 | 12 | 11 | 91.7 % | ₹86,897 | +43.45 % |
| 2025 | 12 | 9 | 75.0 % | ₹129,912 | +64.96 % |
| 2026 | 3 | 2 | 66.7 % | ₹110,362 | +55.18 % |

## Exit reasons

| Reason | Count | Avg P&L | Total P&L |
|---|---:|---:|---:|
| EXPIRY | 28 | ₹+17,072 | ₹+478,018 |
| SL | 7 | ₹-11,726 | ₹-82,081 |

## Files

| File | Description |
|---|---|
| `SUMMARY.md` | This document |
| `trades.csv` | 35 rows, one per IC trade. Includes margin per trade |
| `orders.csv` | 280 rows = 35 trades × 4 legs × 2 phases |
