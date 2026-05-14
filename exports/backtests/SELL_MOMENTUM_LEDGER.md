# Sell Momentum (SHORT) — Backtest Ledger

_Strategy: N100 momentum rotation, SHORT the bottom-5 by 60d return (worst momentum). Monthly rebalance, max-concurrent=1, cash-secured shorts._

**Fixed capital: ₹10,00,000 locked. Window: May 2023 → May 2026.**

> ⚠️ India retail CANNOT short equity delivery (CNC) — only intraday MIS or stock futures. Backtest assumes ideal short fills — for research only.

## Money Flow Summary

| Year | Open | Close | ROI | Trades | W / L |
|------|----:|----:|----:|---:|:---|
| 2023-24 | ₹1,000,000 | ₹333,946 | **-66.61%** | 12 | 4 / 8 |
| 2024-25 | ₹333,946 | ₹276,947 | **-17.07%** | 8 | 4 / 2 |
| 2025-26 | ₹276,947 | ₹276,947 | **+0.00%** | 1 | 0 / 0 |
| **3-yr Avg** | ₹10,00,000 | ₹276,947 | **-27.89%** | **21** | **8 / 10** |

**3-yr compound:** ₹10L → ₹276,947 (-72.31%)


---

## Year 2023-24 — -66.61% (12 short trades)

| # | Symbol | Short Date | Short ₹ | Qty | Margin | Cover Date | Cover ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|----:|-------:|-----------|--------:|------:|--:|-----------:|
| 1 | SCI | 2023-05-13 | 96.75 | 10,335 | ₹999,911 | 2023-06-01 | 94.85 | ₹19,637 | +1.96% | ₹1,019,636 |
| 2 | DRREDDY | 2023-06-01 | 900.21 | 1,132 | ₹1,019,038 | 2023-07-01 | 1031.92 | ₹-149,096 | -14.63% | ₹870,541 |
| 3 | KOTAKBANK | 2023-07-01 | 369.31 | 2,357 | ₹870,464 | 2023-08-01 | 371.31 | ₹-4,714 | -0.54% | ₹865,827 |
| 4 | PERSISTENT | 2023-08-01 | 2372.03 | 365 | ₹865,791 | 2023-09-01 | 2685.35 | ₹-114,362 | -13.21% | ₹751,465 |
| 5 | INDIGO | 2023-09-01 | 2435.00 | 308 | ₹749,980 | 2023-10-01 | 2381.20 | ₹16,570 | +2.21% | ₹768,035 |
| 6 | ABB | 2023-10-01 | 4098.25 | 187 | ₹766,373 | 2023-11-01 | 4109.30 | ₹-2,066 | -0.27% | ₹765,969 |
| 7 | HFCL | 2023-11-01 | 65.20 | 11,747 | ₹765,904 | 2024-01-01 | 84.15 | ₹-222,606 | -29.06% | ₹543,363 |
| 8 | MARUTI | 2024-01-01 | 10302.30 | 52 | ₹535,720 | 2024-02-01 | 10186.90 | ₹6,001 | +1.12% | ₹549,364 |
| 9 | POLYCAB | 2024-02-01 | 4343.15 | 126 | ₹547,237 | 2024-03-01 | 4749.85 | ₹-51,244 | -9.36% | ₹498,120 |
| 10 | ANGELONE | 2024-03-01 | 278.27 | 1,790 | ₹498,103 | 2024-04-01 | 304.50 | ₹-46,952 | -9.43% | ₹451,168 |
| 11 | MAZDOCK | 2024-04-01 | 932.05 | 484 | ₹451,112 | 2024-05-01 | 1174.47 | ₹-117,331 | -26.01% | ₹333,837 |
| 12 | PERSISTENT | 2024-05-01 | 3368.60 | 99 | ₹333,491 | 2024-05-12 | 3367.50 | ₹109 | +0.03% | ₹333,946 |

_Best:_ **SCI** ₹19,637 (+1.96%) &middot; _Worst:_ **HFCL** ₹-222,606 (-29.06%)


---

## Year 2024-25 — -17.07% (8 short trades)

| # | Symbol | Short Date | Short ₹ | Qty | Margin | Cover Date | Cover ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|----:|-------:|-----------|--------:|------:|--:|-----------:|
| 1 | PERSISTENT | 2024-05-13 | 3367.50 | 99 | ₹333,382 | 2024-07-01 | 4241.45 | ₹-86,521 | -25.95% | ₹247,425 |
| 2 | BSE | 2024-07-01 | 860.85 | 287 | ₹247,064 | 2024-08-01 | 852.15 | ₹2,497 | +1.01% | ₹249,922 |
| 3 | ADANIENT | 2024-08-01 | 3072.69 | 81 | ₹248,888 | 2024-09-01 | 2927.21 | ₹11,784 | +4.73% | ₹261,706 |
| 4 | COCHINSHIP | 2024-09-01 | 1886.65 | 138 | ₹260,358 | 2024-12-01 | 1577.30 | ₹42,690 | +16.40% | ₹304,396 |
| 5 | BRITANNIA | 2024-12-01 | 4941.15 | 61 | ₹301,410 | 2025-02-01 | 4698.10 | ₹14,826 | +4.92% | ₹319,222 |
| 6 | HFCL | 2025-02-01 | 113.35 | 2,816 | ₹319,194 | 2025-03-01 | 113.35 | ₹0 | +0.00% | ₹319,222 |
| 7 | M&M | 2025-03-01 | 2585.10 | 123 | ₹317,967 | 2025-05-01 | 2928.80 | ₹-42,275 | -13.30% | ₹276,947 |
| 8 | HDFCAMC | 2025-05-01 | 2131.40 | 129 | ₹274,951 | 2025-05-12 | 2131.40 | ₹0 | +0.00% | ₹276,947 |

_Best:_ **COCHINSHIP** ₹42,690 (+16.40%) &middot; _Worst:_ **PERSISTENT** ₹-86,521 (-25.95%)


---

## Year 2025-26 — +0.00% (1 short trades)

| # | Symbol | Short Date | Short ₹ | Qty | Margin | Cover Date | Cover ₹ | P&L ₹ | % | Cash After |
|--:|--------|-----------|--------:|----:|-------:|-----------|--------:|------:|--:|-----------:|
| 1 | RECLTD | 2025-05-13 | 409.95 | 675 | ₹276,716 | 2026-05-12 | 409.95 | ₹0 | +0.00% | ₹276,947 |

_Best:_ **RECLTD** ₹0 (+0.00%) &middot; _Worst:_ **RECLTD** ₹0 (+0.00%)


---

## Trade Statistics

- Total short trades: **21**
- Win rate: **38.1%** (8W / 10L)
- Avg winner: ₹14,264
- Avg loser: ₹-83,717

## Verdict

SHORT-only momentum **LOSES** over 3 years: **-72.31% compound**.

Shorting bottom-momentum failed because:
- 2023-25 Indian bull market = even worst-momentum stocks rallied with the tide
- Mean reversion at oversold levels hurt shorts
- Asymmetric risk: shorts lose unbounded if stock rallies, gain capped at -100%

**Long-side momentum is the way.** See `BEST_RETURNS.md` for trailing_sl 8% long strategy = +242% compound.

## Real-World Blockers

1. NSE equity CNC (delivery) shorting **prohibited for retail**. Only intraday MIS.
2. Monthly shorts need stock futures (F&O ~180 names, not all N100) — different instrument, margin 20-30%, roll cost.
3. Stock Lending Borrowing (SLB) for >1 day shorts: thin liquidity, ~150 stocks, fees 4-8% annual.
4. Backtest assumed ideal fills at close — real shorts slip far worse on illiquid bottom-N names.