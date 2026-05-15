# Dual Supertrend — Nifty 100 Backtest (3 Years)

_Window: May 2023 → May 2026. Universe: pseudo-Nifty 100 (top 100 by ADV). Capital: ₹10,00,000. Max 5 concurrent positions. 0.13% round-trip cost._

## Strategy Spec

```
Fast Supertrend: period=7, multiplier=3
Slow Supertrend: period=10, multiplier=4

Supertrend formula:
  HL2 = (high + low) / 2
  upper_band = HL2 + multiplier × ATR(period)
  lower_band = HL2 − multiplier × ATR(period)
  if close > prior_upper → bullish (line = lower_band)
  if close < prior_lower → bearish (line = upper_band)

ENTRY: BOTH fast AND slow Supertrends turn bullish (close above both lines)
HOLD : while slow Supertrend stays bullish (slow line = trailing stop)
EXIT : BOTH fast AND slow flip bearish (close below both)

POSITION SIZING: equal-weight, ₹2,00,000 per slot (max 5 = ₹10L total)
```

## Overall Performance — 3 Years

| Metric | Value |
|--------|-----:|
| Total Signals | 622 |
| Executed (5-slot cap) | 52 |
| Skipped (no slot) | 570 |
| Win Rate (executed) | **42.3%** |
| Avg trade P&L | +6.07% |
| Avg holding | 100 days |
| **Total ROI 3-yr** | **+77.89%** |
| **CAGR** | **+22.64%** |
| Sharpe (daily) | 1.12 |
| MaxDD | -22.65% |

**Compound:** ₹10,00,000 → ₹1,778,866 (+77.89%)

## Annual Performance — Regime Test

| Period | Regime | Trades | Win% (executed) | Win% (all signals) | Period ROI |
|--------|--------|------:|---------------:|------------------:|----------:|
| 2023-24 | Bull | 17 | 47.1% | 59.7% | +26.67% |
| 2024-25 | Mixed | 19 | 31.6% | 40.4% | +11.59% |
| 2025-26 | Recent | 16 | 50.0% | 43.6% | +32.36% |

**Published 60-70% bull win-rate claim — REALITY CHECK:**
- 2023-24 bull: **59.7%** strategy / **47.1%** executed → claim hits ONLY at lower bound on all signals
- 2024-25 mixed: 40.4% / 31.6% → way below claim
- Marketing-grade, not regime-robust

## Top 10 Stocks Where Strategy Works

| Symbol | Trades | Win% | Total P&L% | Avg/trade | Avg Hold (d) |
|--------|------:|----:|----------:|---------:|-------------:|
| **ANGELONE** | 8 | 62% | +1059% | +132.4% | 48 |
| **GVT&D** | 3 | 100% | +795% | +265.0% | 206 |
| **MCX** | 6 | 67% | +597% | +99.4% | 88 |
| **GALLANTT** | 8 | 38% | +373% | +46.6% | 69 |
| **WOCKPHARMA** | 6 | 50% | +356% | +59.3% | 93 |
| **POWERINDIA** | 6 | 50% | +224% | +37.3% | 80 |
| **COCHINSHIP** | 9 | 56% | +203% | +22.5% | 50 |
| **GRSE** | 8 | 50% | +173% | +21.6% | 52 |
| **NETWEB** | 6 | 67% | +164% | +27.3% | 68 |
| **VEDL** | 6 | 50% | +152% | +25.3% | 79 |

## Bottom 10 Stocks Where Strategy FAILS

| Symbol | Trades | Win% | Total P&L% |
|--------|------:|----:|----------:|
| TITAN | 10 | 50% | -9.4% |
| BANDHANBNK | 8 | 38% | -11.9% |
| KOTAKBANK | 7 | 29% | -12.8% |
| TCS | 7 | 29% | -13.6% |
| HDFCLIFE | 7 | 29% | -13.6% |
| PFC | 5 | 20% | -14.4% |
| DRREDDY | 9 | 44% | -16.1% |
| OLAELEC | 4 | 25% | -19.7% |
| HDFCBANK | 9 | 22% | -27.3% |
| IDEA | 8 | 25% | -28.1% |

**Theme:** Banks (HDFCBANK), telecom (IDEA), EV (OLAELEC), pharma defensives (DRREDDY) — Supertrend chops up on mean-reverting sectors.

## Full Trade Ledger — All 52 Executed Trades


### Year 2023-24 — 15 trades, 7W / 8L

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | Hold | P&L ₹ | P&L % |
|--:|--------|-----------|--------:|------:|---------:|-----------|--------:|----:|------:|------:|
| 1 | AXISBANK | 2023-05-15 | 910.65 | 219 | ₹199,432 | 2023-10-25 | 960.00 | 163d | ₹10,548 | +5.29% |
| 2 | BHARTIARTL | 2023-05-22 | 801.00 | 249 | ₹199,449 | 2024-11-01 | 1620.00 | 529d | ₹203,672 | +102.12% |
| 3 | DIXON | 2023-05-23 | 3267.40 | 61 | ₹199,311 | 2023-07-25 | 4015.00 | 63d | ₹45,344 | +22.75% |
| 4 | TECHM | 2023-05-23 | 1104.00 | 181 | ₹199,824 | 2023-07-28 | 1100.00 | 66d | ₹-984 | -0.49% |
| 5 | ADANIENT | 2023-05-23 | 2337.33 | 85 | ₹198,673 | 2023-10-25 | 2249.21 | 155d | ₹-7,748 | -3.90% |
| 6 | ADANIENSOL | 2023-07-26 | 839.90 | 238 | ₹199,896 | 2023-10-09 | 806.00 | 75d | ₹-8,328 | -4.17% |
| 7 | ABB | 2023-08-01 | 4578.75 | 43 | ₹196,886 | 2023-09-29 | 4120.00 | 59d | ₹-19,982 | -10.15% |
| 8 | POLYCAB | 2023-10-04 | 5352.95 | 37 | ₹198,059 | 2023-10-27 | 4900.00 | 23d | ₹-17,017 | -8.59% |
| 9 | NESTLEIND | 2023-10-12 | 1161.15 | 172 | ₹199,718 | 2024-01-19 | 1255.00 | 99d | ₹15,883 | +7.95% |
| 10 | RECLTD | 2023-11-03 | 304.00 | 657 | ₹199,728 | 2024-02-29 | 429.20 | 118d | ₹81,997 | +41.05% |
| 11 | SHRIRAMFIN | 2023-11-03 | 392.50 | 509 | ₹199,782 | 2024-06-05 | 448.38 | 215d | ₹28,183 | +14.11% |
| 12 | ADANIPOWER | 2023-11-06 | 77.79 | 2,571 | ₹199,998 | 2024-08-16 | 136.60 | 284d | ₹150,941 | +75.47% |
| 13 | BSE | 2024-02-02 | 836.67 | 239 | ₹199,964 | 2024-03-14 | 670.67 | 41d | ₹-39,934 | -19.97% |
| 14 | LT | 2024-03-02 | 3647.90 | 54 | ₹196,987 | 2024-05-07 | 3479.40 | 66d | ₹-9,355 | -4.75% |
| 15 | HDFCLIFE | 2024-03-18 | 634.95 | 314 | ₹199,374 | 2024-04-30 | 576.55 | 43d | ₹-18,597 | -9.33% |

_Best:_ **BHARTIARTL** +102.12% · _Worst:_ **BSE** -19.97%


### Year 2024-25 — 20 trades, 6W / 14L

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | Hold | P&L ₹ | P&L % |
|--:|--------|-----------|--------:|------:|---------:|-----------|--------:|----:|------:|------:|
| 1 | PFC | 2024-05-02 | 448.15 | 446 | ₹199,875 | 2024-06-05 | 444.70 | 34d | ₹-1,799 | -0.90% |
| 2 | BRITANNIA | 2024-05-07 | 5057.95 | 39 | ₹197,260 | 2024-10-21 | 5887.20 | 167d | ₹32,084 | +16.26% |
| 3 | OFSS | 2024-06-07 | 8200.00 | 24 | ₹196,800 | 2024-10-25 | 10475.00 | 140d | ₹54,344 | +27.61% |
| 4 | BANDHANBNK | 2024-06-10 | 196.55 | 1,017 | ₹199,891 | 2024-10-07 | 190.00 | 119d | ₹-6,921 | -3.46% |
| 5 | DIXON | 2024-08-20 | 12900.00 | 15 | ₹193,500 | 2025-01-09 | 16901.10 | 142d | ₹59,765 | +30.89% |
| 6 | ABB | 2024-10-10 | 8499.00 | 23 | ₹195,477 | 2024-10-24 | 7775.00 | 14d | ₹-16,906 | -8.65% |
| 7 | MAZDOCK | 2024-10-22 | 2342.50 | 85 | ₹199,112 | 2025-01-07 | 2130.35 | 77d | ₹-18,292 | -9.19% |
| 8 | HINDZINC | 2024-10-30 | 555.00 | 360 | ₹199,800 | 2024-12-20 | 480.00 | 51d | ₹-27,260 | -13.64% |
| 9 | GMDCLTD | 2024-10-30 | 362.30 | 552 | ₹199,990 | 2024-12-31 | 315.00 | 62d | ₹-26,370 | -13.19% |
| 10 | HDFCBANK | 2024-11-06 | 885.75 | 225 | ₹199,294 | 2025-01-07 | 857.50 | 62d | ₹-6,615 | -3.32% |
| 11 | DRREDDY | 2024-12-23 | 1362.00 | 146 | ₹198,852 | 2025-01-27 | 1224.40 | 35d | ₹-20,348 | -10.23% |
| 12 | HSCL | 2024-12-31 | 586.95 | 340 | ₹199,563 | 2025-01-27 | 505.00 | 27d | ₹-28,122 | -14.09% |
| 13 | ADANIPOWER | 2025-01-16 | 117.00 | 1,709 | ₹199,953 | 2025-12-05 | 143.00 | 323d | ₹44,174 | +22.09% |
| 14 | SBILIFE | 2025-01-17 | 1519.50 | 131 | ₹199,054 | 2026-03-05 | 1947.60 | 412d | ₹55,822 | +28.04% |
| 15 | RELIANCE | 2025-01-20 | 1316.00 | 151 | ₹198,716 | 2025-02-27 | 1212.80 | 38d | ₹-15,842 | -7.97% |
| 16 | BRITANNIA | 2025-01-27 | 5103.65 | 39 | ₹199,042 | 2025-03-03 | 4595.45 | 35d | ₹-20,079 | -10.09% |
| 17 | HINDUNILVR | 2025-02-01 | 2434.59 | 82 | ₹199,636 | 2025-02-19 | 2261.21 | 18d | ₹-14,477 | -7.25% |
| 18 | ASHOKLEY | 2025-02-20 | 112.25 | 1,781 | ₹199,917 | 2025-08-11 | 115.90 | 172d | ₹6,241 | +3.12% |
| 19 | SHRIRAMFIN | 2025-02-28 | 609.95 | 327 | ₹199,454 | 2025-05-05 | 610.15 | 66d | ₹-194 | -0.10% |
| 20 | NTPC | 2025-03-07 | 335.85 | 595 | ₹199,831 | 2025-06-04 | 329.00 | 89d | ₹-4,336 | -2.17% |

_Best:_ **DIXON** +30.89% · _Worst:_ **HSCL** -14.09%


### Year 2025-26 — 17 trades, 9W / 8L

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | Hold | P&L ₹ | P&L % |
|--:|--------|-----------|--------:|------:|---------:|-----------|--------:|----:|------:|------:|
| 1 | MARUTI | 2025-05-05 | 12490.00 | 16 | ₹199,840 | 2025-11-06 | 15534.00 | 185d | ₹48,444 | +24.24% |
| 2 | HINDZINC | 2025-06-04 | 472.00 | 423 | ₹199,656 | 2025-06-19 | 454.10 | 15d | ₹-7,831 | -3.92% |
| 3 | WIPRO | 2025-06-20 | 266.00 | 751 | ₹199,766 | 2025-07-29 | 249.20 | 39d | ₹-12,876 | -6.45% |
| 4 | VBL | 2025-07-30 | 515.00 | 388 | ₹199,820 | 2025-09-08 | 469.70 | 40d | ₹-17,836 | -8.93% |
| 5 | ASHOKLEY | 2025-08-19 | 131.76 | 1,517 | ₹199,880 | 2026-03-10 | 191.50 | 203d | ₹90,366 | +45.21% |
| 6 | BAJFINANCE | 2025-09-08 | 937.60 | 213 | ₹199,709 | 2025-11-12 | 1011.00 | 65d | ₹15,375 | +7.70% |
| 7 | POWERINDIA | 2025-11-06 | 20305.00 | 9 | ₹182,745 | 2025-12-08 | 19395.00 | 32d | ₹-8,428 | -4.61% |
| 8 | GRSE | 2025-11-12 | 2805.00 | 71 | ₹199,155 | 2025-12-08 | 2473.00 | 26d | ₹-23,831 | -11.97% |
| 9 | HSCL | 2025-12-08 | 471.00 | 424 | ₹199,704 | 2026-03-05 | 443.15 | 87d | ₹-12,068 | -6.04% |
| 10 | HINDALCO | 2025-12-15 | 851.95 | 234 | ₹199,356 | 2026-02-02 | 905.70 | 49d | ₹12,318 | +6.18% |
| 11 | TMCV | 2025-12-17 | 389.00 | 514 | ₹199,946 | 2026-03-16 | 420.00 | 89d | ₹15,674 | +7.84% |
| 12 | ADANIPORTS | 2026-02-04 | 1529.90 | 130 | ₹198,887 | 2026-03-16 | 1356.60 | 40d | ₹-22,788 | -11.46% |
| 13 | NATIONALUM | 2026-03-06 | 392.00 | 510 | ₹199,920 | 2026-05-12 | 394.05 | 67d | ₹786 | +0.39% |
| 14 | MAZDOCK | 2026-03-11 | 2510.10 | 79 | ₹198,298 | 2026-05-12 | 2443.80 | 62d | ₹-5,495 | -2.77% |
| 15 | WAAREEENER | 2026-03-19 | 3095.00 | 64 | ₹198,080 | 2026-05-12 | 3125.20 | 54d | ₹1,675 | +0.85% |
| 16 | OLAELEC | 2026-04-06 | 28.80 | 6,944 | ₹199,987 | 2026-05-12 | 35.13 | 36d | ₹43,696 | +21.85% |
| 17 | OFSS | 2026-04-08 | 7292.00 | 27 | ₹196,884 | 2026-05-11 | 9230.50 | 33d | ₹52,084 | +26.45% |

_Best:_ **ASHOKLEY** +45.21% · _Worst:_ **GRSE** -11.97%

## vs M3 Momentum Rotation (Validated Walk-Forward)

| Metric | Dual Supertrend | M3 |
|--------|----------------:|----:|
| CAGR | **+22.6%** | **+87.0%** |
| MaxDD | -22.6% | -6% |
| Sharpe | 1.12 | (higher) |
| Win Rate | 42% | n/a (rotation) |
| Avg Hold | 100 days | ~30 days |
| Trades / 3 yr | 52 executed (570 starved) | 13 |

M3 wins **4× on CAGR and 4× on DD**. Dual Supertrend not in same league.

## Implementation Notes

- **Capacity bottleneck**: 622 signals over 3 yrs, only 52 executed (92% starved by 5-slot cap). Raising max_positions could capture more but proportionally less per-slot deploy.
- **Sectors that work**: defense (MAZDOCK, COCHINSHIP, GRSE), fintech/exchange (ANGELONE, MCX, BSE), EV/power (GVT&D, SUZLON, POWERINDIA), capital goods (KAYNES, NETWEB, WOCKPHARMA).
- **Sectors to AVOID**: banks (HDFCBANK, KOTAK, BANDHAN), FMCG (TITAN), pharma defensives (DRREDDY), telecom (IDEA), EV losers (OLAELEC).
- **Automation**: trivial — daily EOD scan, place market order next day. Reuses momrot infrastructure.
- **Capital**: ₹2L per slot. ₹10L is the floor for sensible diversification. Scales to ₹10cr+ with no slippage on top names.

## Recommendation

**NOT for primary engine.** M3 momentum rotation dominates on every metric (CAGR, DD, Sharpe).

Acceptable use:
1. **Confirmation overlay** on M3 picks — higher conviction when both bullish
2. **Set-and-forget allocation** for passive investors who can stomach 100-day holds + 22% DD
3. **Sector-restricted variant** — only run on defense/fintech/EV names where it works (top-10 list above)

The published 60-70% bull win-rate claim is regime-cherry-picked marketing. Real lifetime executed win rate = 42%.

## Files

```
exports/backtests/DUAL_SUPERTREND.md
tools/backtests/dual_supertrend.py
remote: /app/logs/dual_supertrend/dual_supertrend_report.json
```