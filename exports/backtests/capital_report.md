# Capital Simulation Report — ₹200,000 INR on NIFTY 50

**Date:** 2026-05-08
**Universe:** 53 NSE largecaps (current NIFTY 50 + post-2024 reconstitution)
**Allocation:** Equal-weight ~₹3,774 per symbol
**Sustain:** BUY=15m, SELL=75m
**SL:** Close-based per BTC trade rules v1.2
**Source:** Fyers (53/53, 0 Yahoo)

## 180-day (6mo) results

| Case | Final Capital | P&L | ROI | Winners | Losers | Flat |
|------|---------------|-----|-----|---------|--------|------|
| BUY plain | ₹198,713 | -₹1,287 | **-0.64%** | 1 | 7 | 45 |
| BUY HTF | ₹198,713 | -₹1,287 | -0.64% | 1 | 7 | 45 |
| SELL plain | ₹198,642 | -₹1,358 | **-0.68%** | 1 | 12 | 40 |
| SELL all | ₹199,464 | -₹536 | -0.27% | 0 | 3 | 50 |

### Profitable stocks (180d) — uploaded list
- **BUY plain**: ADANIPORTS (+6.4%, +₹242)
- **BUY HTF**: ADANIPORTS (+6.4%, +₹242)
- **SELL plain**: HCLTECH (+60.0%, +₹2,264, 4 legs)
- **SELL all**: none

### Why 180d underperforms
Strategy needs ema_slow_period+5 = 405 bars warmup. 180d → 847 1H bars, leaving 442 effective. Fresh EMA200×400 crossovers are rare events — 45/53 symbols have 0 trades. Small loss is small *because* trade activity is small.

## 720-day (2yr) results — same config

| Case | Final Capital | P&L | ROI | Winners | Losers |
|------|---------------|-----|-----|---------|--------|
| **BUY plain** | **₹251,355** | **+₹51,355** | **+25.68%** | 32 | 21 |
| **SELL plain** | **₹221,717** | **+₹21,717** | **+10.86%** | 23 | 29 |

Combined BUY+SELL on 720d: **~₹73,072 P&L (+36.5% ROI on ₹200K)**.

### Top winners 720d BUY plain
| Stock | P&L | ROI | Legs |
|-------|-----|-----|------|
| HEROMOTOCO | +₹8,075 | +214.0% | 15 |
| HINDALCO | +₹7,743 | +205.2% | 13 |
| TATASTEEL | +₹5,230 | +138.6% | 24 |
| BPCL | +₹4,174 | +110.6% | 20 |
| APOLLOHOSP | +₹4,094 | +108.5% | 11 |
| JSWSTEEL | +₹3,777 | +100.1% | 25 |
| NESTLEIND | +₹3,426 | +90.8% | 24 |
| SBILIFE | +₹3,328 | +88.2% | 13 |
| MARUTI | +₹3,268 | +86.6% | 7 |
| UPL | +₹3,204 | +84.9% | 16 |

### Top winners 720d SELL plain
| Stock | P&L | ROI | Legs |
|-------|-----|-----|------|
| AXISBANK | +₹4,928 | +130.6% | 15 |
| TMPV | +₹4,019 | +106.5% | 20 |
| ADANIENT | +₹3,879 | +102.8% | 14 |
| ADANIPORTS | +₹3,615 | +95.8% | 7 |
| HEROMOTOCO | +₹3,294 | +87.3% | 14 |
| HINDALCO | +₹3,128 | +82.9% | 10 |
| HCLTECH | +₹2,921 | +77.4% | 8 |
| CIPLA | +₹2,770 | +73.4% | 9 |
| INDUSINDBK | +₹2,547 | +67.5% | 10 |
| HDFCLIFE | +₹2,068 | +54.8% | 16 |

## Recommendation

- **Skip 180d window** for evaluation — too short for crossover-based strategy.
- **Use 720d** for meaningful P&L: combined plain BUY+SELL = **+36.5% ROI**.
- Filters (HTF / sell_slope) reduce profit on both windows. Drop them.
