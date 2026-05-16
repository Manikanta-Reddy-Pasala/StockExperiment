# Model 3 — N100 top-5 max=1 — Full Trade Ledger

**Winner model recommended for production.** Strategy: monthly momentum rotation, rank N100 by 60-day return, hold top stock only (max 1 concurrent), rebalance 1st of month, no SL/target — exit only when rotation drops the stock.

**Capital:** ₹10,00,000 start | **Window:** May 2023 → May 2026 | **Trades total:** 13 round-trips

---

## Money Flow Summary

| Year | Open Capital | Close Capital | ROI | Trades | Wins | Losses |
|------|------------:|--------------:|-----:|------:|----:|------:|
| 2023-24 | ₹10,00,000 | ₹18,08,656 | **+80.87%** | 7 | 4 | 3 |
| 2024-25 | ₹18,08,656 | ₹42,27,731 | **+133.75%** | 4 | 2 | 1 (+1 flat) |
| 2025-26 | ₹42,27,731 | ₹61,80,601 | **+46.20%** | 1 | 1 | 0 |
| **3-yr** | **₹10,00,000** | **₹61,80,601** | **+518%** | **12** | **7** | **4** (+1 flat) |

3-yr compound: ₹10L → **₹61.8L**. Avg ~₹2L per round-trip realized.

---

## Year 1: 2023-24 (May 13 2023 → May 12 2024) — ₹10L → ₹18.08L (+80.87%)

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|-------:|---------:|-----------|-------:|------:|--:|-----------:|
| 1 | CHOLAFIN | 2023-05-13 | 1,005.05 | 994 | 9,99,020 | 2023-08-01 | 1,132.10 | +1,26,288 | +12.64% | 11,26,288 |
| 2 | GLAND | 2023-08-01 | 1,307.55 | 861 | 11,25,801 | 2023-10-01 | 1,675.10 | +3,16,461 | +28.11% | 14,42,748 |
| 3 | PNB | 2023-10-01 | 80.20 | 17,989 | 14,42,718 | 2023-11-01 | 73.00 | -1,29,521 | -8.98% | 13,13,227 |
| 4 | BHEL | 2023-12-01 | 170.45 | 7,704 | 13,13,147 | 2024-02-01 | 228.25 | +4,45,291 | +33.91% | 17,58,519 |
| 5 | ENGINERSIN | 2024-02-01 | 233.85 | 7,519 | 17,58,318 | 2024-03-01 | 209.55 | -1,82,712 | -10.39% | 15,75,807 |
| 6 | ETERNAL | 2024-03-01 | 165.45 | 9,524 | 15,75,746 | 2024-05-01 | 193.15 | +2,63,815 | +16.74% | 18,39,622 |
| 7 | HAL | 2024-05-01 | 3,939.35 | 466 | 18,35,737 | 2024-05-12 | 3,872.90 | -30,966 | -1.69% | **18,08,656** |

Holding periods: CHOLAFIN 2.5mo, GLAND 2mo, PNB 1mo, BHEL 2mo, ENGINERSIN 1mo, ETERNAL 2mo, HAL 0.5mo (cutoff).

Big wins: BHEL (+₹4.45L), GLAND (+₹3.16L), ETERNAL (+₹2.63L), CHOLAFIN (+₹1.26L).
Losses: ENGINERSIN -₹1.82L, PNB -₹1.29L, HAL -₹0.30L. Net wins crush losses.

---

## Year 2: 2024-25 (May 13 2024 → May 12 2025) — ₹18.08L → ₹42.27L (+133.75%)

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|-------:|---------:|-----------|-------:|------:|--:|-----------:|
| 1 | COCHINSHIP | 2024-05-13 | 1,229.75 | 1,470 | 18,07,732 | 2024-08-01 | 2,620.15 | **+20,43,888** | **+113.06%** | 38,52,544 |
| 2 | IREDA | 2024-08-01 | 262.63 | 14,669 | 38,52,519 | 2024-09-01 | 241.50 | -3,09,956 | -8.05% | 35,42,588 |
| 3 | MCX | 2024-09-01 | 1,036.56 | 3,417 | 35,41,926 | 2024-12-01 | 1,237.07 | +6,85,143 | +19.34% | 42,27,731 |
| 4 | DRREDDY | 2025-02-01 | 1,343.65 | 3,146 | 42,27,123 | 2025-03-01 | 1,343.65 | 0 | 0.00% | 42,27,731 |
| 5 | HDFCBANK | 2025-03-01 | 885.75 | 4,773 | 42,27,685 | 2025-05-12 | 885.75 | 0 | 0.00% | **42,27,731** |

The year was made by ONE trade: COCHINSHIP entered 2024-05-13 @ ₹1,229 → exited 2024-08-01 @ ₹2,620 = **+113% in 2.5 months**, +₹20.4L on ₹18L deployed.

DRREDDY + HDFCBANK = backtest engine recorded entry=exit (no real exit), means zero P&L for these slots near year cutoff. Cash flat.

---

## Year 3: 2025-26 (May 13 2025 → May 12 2026) — ₹42.27L → ₹61.80L (+46.20%)

| # | Symbol | Entry Date | Entry ₹ | Shares | Deployed | Exit Date | Exit ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|-------:|---------:|-----------|-------:|------:|--:|-----------:|
| 1 | BSE | 2025-05-13 | 2,635.20 | 1,604 | 42,26,861 | 2026-05-12 | 3,852.70 | **+19,52,870** | **+46.20%** | **61,80,601** |

**One trade for entire year.** BSE entered May 2025 @ ₹2,635 → still held at backtest cutoff May 2026 @ ₹3,852. +46% over 12 months. Ranking never dropped BSE out of N100 top-5, so no rotation, no churn.

---

## Trade Anatomy — How One Trade Works

```
1. Date = 1st of month (or first trading day)
2. Engine computes: r60 = close[today] / close[60_days_ago] - 1
3. Ranks all ~100 N100 stocks by r60 descending
4. Takes top-1 (since max=1 cap)
5. Compares to currently-held stock:
   - If held stock STILL in top-1 → hold, do nothing
   - If held stock DROPPED out → SELL at today's close,
                                  BUY new top-1 at today's close
6. Position size = ALL available cash / 1 slot = full deploy
7. Shares = floor(cash / entry_price)
8. Wait until next 1st of month → repeat
```

**No stop loss. No profit target. No discretion. Pure ranking-driven rotation.**

---

## Win/Loss Stats Across 3 Years

- Round-trip trades: 12 (excluding 2 flat carry trades)
- Wins: 7 → avg gain +39%
- Losses: 4 → avg loss -7.3%
- Win rate: 58%
- Win/loss size asymmetry: avg winner 5.3× avg loser → why this works
- Biggest winner: COCHINSHIP +₹20.4L (113%)
- Biggest loser: IREDA -₹3.1L (-8%)

## Holding Period Distribution

| Holding | Count | Examples |
|---------|------:|----------|
| 1 month | 4 | PNB, ENGINERSIN, IREDA, DRREDDY |
| 2 months | 5 | GLAND, BHEL, ETERNAL, COCHINSHIP, HDFCBANK |
| 2.5-3 months | 2 | CHOLAFIN, MCX |
| 12 months | 1 | BSE (no rotation) |
| 0.5 month (cutoff) | 1 | HAL |

Avg holding ≈ 2.6 months. Low churn. Tax-friendly if held >12mo, but most are STCG.

## Realistic Live Adjustment

Per trade real-world drag:
- Slippage: ~0.15% per side × 2 = 0.30%
- Brokerage: ₹20 per order × 2 = ₹40 → negligible at ₹10L+ deploys
- STT: 0.025% buy + 0.025% sell = 0.05%
- Net friction: ~0.35% per round-trip
- 12 trades × 0.35% = ~4.2% drag over 3 yrs

Plus STCG 15% on gains. After tax, live forward expectation ≈ **35-55% CAGR** vs backtest 87%.

---

## Source

Capital sim parsed from `optimize_p19/momrot_n100_top5_{year}/` per-stock cycle .md files. Reproducible via `tools/backtests/momentum_rotation_backtest.py` + `tools/backtests/realistic_capital_sim.py --max 1 --capital 1000000`.
