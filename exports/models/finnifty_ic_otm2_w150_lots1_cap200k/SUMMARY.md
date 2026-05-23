# finnifty_ic_otm2_w150_lots1

FinNifty monthly Iron Condor rescaled to ₹200,000 investment capital
(peak-safe — every backtested entry was openable at the live broker).

## Strategy

- **Underlying:** FINNIFTY (Nifty Financial Services Index)
- **Setup:** Iron Condor monthly expiry, OTM2 body, W150 wings
- **Position size:** 1 lot (auto-sized from peak margin ≤ capital)
- **Stop:** 3× entry credit OR hold to expiry
- **Capital:** ₹200,000
- **Slippage:** realistic tiered (1× ATM → 15× >6% OTM)
- **Source:** rescaled from finnifty_ic_otm2_w150_lots5 backtest

## Result at ₹200,000

- **Started with:** ₹200,000
- **Ended with:** ₹541,231
- **Total profit:** ₹341,231
- **Total return:** +170.6 %
- **CAGR:** **+39.5 %**
- **Trades:** 36
- **Win rate:** 77.8 %
- **Avg P&L / trade:** ₹+9,479
- **Max drawdown:** -20.3 %

## Margin (SPAN+exposure approx — sweep.compute_ic_margin)

| Metric | Value |
|---|---:|
| Lots | **1** |
| Avg margin / trade | ₹63,832 |
| Peak margin / trade | ₹114,416 |
| Configured capital | ₹200,000 |
| Capital / peak ratio | 1.75× |

> ✅ Peak margin ≤ capital — every monthly entry was openable at the
> live broker. No skipped cycles.

## Yearly

| Year | Trades | Wins | WR | P&L | ROI |
|---|---:|---:|---:|---:|---:|
| 2023 | 8 | 7 | 87.5 % | ₹71,009 | +35.50 % |
| 2024 | 12 | 9 | 75.0 % | ₹98,143 | +49.07 % |
| 2025 | 12 | 9 | 75.0 % | ₹86,935 | +43.47 % |
| 2026 | 4 | 3 | 75.0 % | ₹85,142 | +42.57 % |

## Exit reasons

| Reason | Count | Avg P&L | Total P&L |
|---|---:|---:|---:|
| EXPIRY | 31 | ₹+14,747 | ₹+457,162 |
| SL | 5 | ₹-23,186 | ₹-115,931 |

## Files

| File | Description |
|---|---|
| `SUMMARY.md` | This document |
| `trades.csv` | 36 rows, one per IC trade. Includes margin per trade |
| `orders.csv` | 288 rows = 36 trades × 4 legs × 2 phases |
