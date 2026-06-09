# Intraday 15-min bars — MACD day-trade test set

Real Fyers 15-minute OHLCV bars (NSE cash), pulled 2026-06-09.

| File | Symbol | Bars | Range |
|---|---|---|---|
| RELIANCE_NSE_15m.csv | RELIANCE | 1500 | 2026-03-06 → 2026-06-08 |
| HDFCBANK_NSE_15m.csv | HDFCBANK | 1500 | 2026-03-06 → 2026-06-08 |
| ICICIBANK_NSE_15m.csv | ICICIBANK | 1500 | 2026-03-06 → 2026-06-08 |
| INFY_NSE_15m.csv | INFY | 1500 | 2026-03-06 → 2026-06-08 |
| TCS_NSE_15m.csv | TCS | 1500 | 2026-03-06 → 2026-06-08 |
| SBIN_NSE_15m.csv | SBIN | 1500 | 2026-03-06 → 2026-06-08 |
| AXISBANK_NSE_15m.csv | AXISBANK | 1500 | 2026-03-06 → 2026-06-08 |

Columns: `datetime` (IST, +0530), `open, high, low, close, volume`.
TATAMOTORS: no F&O/cash 15m returned by Fyers for this window (skipped).

## MACD day-trade backtest result (`video_strategy_notes.md` rules, applied intraday)
Rules: MACD(12,26,9) cross above/below signal while below/above zero, 200-period(15m) MA
trend filter, entry next bar open (PIT), stop = 200-MA level, target 1.5 R:R, square-off
15:15 IST, 6 bps/side cost.

| Stock | LONG total | SHORT total |
|---|---|---|
| RELIANCE | +0.4% | +0.1% |
| HDFCBANK | −0.3% | −3.3% |
| ICICIBANK | −1.9% | −0.6% |
| INFY | −1.8% | −0.6% |
| TCS | −0.5% | +3.0% |
| SBIN | +1.1% | −2.8% |
| AXISBANK | −2.7% | −1.2% |
| **AVERAGE** | **−0.8%** | **−0.8%** |

**Conclusion: NO EDGE.** Both directions net negative over ~3 months (≈ −3%/yr before
slippage). Per-trade expectancy negative on most names; win-rate scattered 25–75% =
random. Consistent with the daily-timeframe test (long 1.8% CAGR / 36% DD, short −7.4%)
and with the archived ORB intraday momentum model (no edge). The video's "extremely high
probability of success" claim does not hold on real data.

See `backtest_intraday.py` to reproduce.
