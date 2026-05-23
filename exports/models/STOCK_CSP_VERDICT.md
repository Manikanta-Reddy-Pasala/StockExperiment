# Stock CSP wheel — verdict (2026-05-23)

**Status: ABANDONED. Pivot back to equity momentum.**

## What was tested

Formula-driven Cash-Secured Put wheel on 19 NSE F&O stocks with daily
options data (ASIANPAINT, HDFCBANK, RELIANCE, LT, TCS, ITC, WIPRO,
HINDUNILVR, SBIN, MARUTI, ICICIBANK, BAJFINANCE, KOTAKBANK, AXISBANK,
INFY, SUNPHARMA, ADANIENT, BHARTIARTL, ADANIPORTS).

Strict point-in-time backtest (no look-forward). Formula:
```
trend_z  = (close - SMA200) / SMA200   # bullish only
iv_proxy = realised_vol(60d)
liq_z    = log10(avg ATM-PE vol, 30d)
score    = 0.4 * trend_z + 0.4 * iv_proxy + 0.2 * liq_z
```

Pick top-5 stocks by score each monthly cycle, sell 3% OTM put, exit at
50% credit captured OR expiry. Window: 2023-05-15 → 2026-05-15, ₹2L capital.

## Variants tested

| Variant | Trades | WR % | CAGR | DD % |
|---|---:|---:|---:|---:|
| Naked CSP (no defense) | 22 | 90.9 | **-11.4 %** | -38.4 |
| Put spread (W3% wing) | ~26 | ~85 | ~+0.3 % | n/a |
| Put spread + SL 2.5× | ~26 | ~73 | ~-0.3 % | n/a |
| Naked CSP + SL 2.5× | 5 | 80.0 | +1.2 % | -0.1 |
| Put spread, no banks | 31 | 80.6 | +0.7 % | -6.9 |

**All variants ≤ +1.2 % CAGR.** Bond fund beats every one of them.

## Why it fails

1. **Universe too narrow** — only 19 large-caps. High-WR is illusion; the rare blow-up (MARUTI -₹81k, SBIN -₹165k) wipes 20 winners.
2. **2023-2026 was a low-IV bull market** — premiums thin, put spreads collect ₹3-5/unit credit. Wing cost + slippage eats edge.
3. **Spread defense kills the credit** — wing cost ≈ 50 % of short premium. Net theta capture too small to overcome occasional ITM exits.
4. **Stop-loss kills marginal winners** — 2.5× SL trips on intraday spikes that would have reverted by expiry.

## Decision

**Don't trade stock options income at current data + market regime.**

For future re-attempt, would need:
- Wider option-data universe (full Nifty 100 = 5× more stocks; bigger non-correlated pool)
- Higher-IV regime detection (skip entries when realised vol < threshold)
- Weekly cycles instead of monthly (more compounding, lower exposure per cycle)
- Stock-specific lot sizing (today's universal 18 % SPAN approximation is rough)

## Live recommendation (unchanged from earlier sessions)

**`momentum_n100_top5_max1`** — equity momentum rotation on Nifty 100.
- Live backtest: +87 % CAGR, -6 % DD, walk-forward validated
- Already running production via `tools/live/fyers_executor.py`
- Real-money cap ₹30k per model enforced

## Files

- `tools/models/stock_csp_wheel/backtest.py` — formula-driven backtester (keep for reference even though strategy abandoned; CLI flags `--wing-otm-pct`, `--stop-mult`, `--max-per-symbol-pct`, `--exclude` are reusable)
- `exports/models/stock_csp_wheel_cap*` — saved trade ledgers for each variant
- This file — verdict + pivot decision
