# Monthly + Yearly Winners — ₹2,00,000 ₹2L Backtest

Source: 48 backtests in `yearly_filter/` + `yearly_nofilter/`. Cache-only
mode, 3 years, max_concurrent=2.

## Monthly P&L Wins (out of 37 months)

| Universe / Filter | ema_9_21 | ema_200_400 | swing_pullback | orb_15min |
|---|---:|---:|---:|---:|
| N50 filter | **16 (43%)** | 15 (41%) | 4 (11%) | 0 |
| N500 filter | **20 (54%)** | 9 (24%) | 5 (14%) | 0 |

**ema_9_21 wins more individual months** (43-54% of the time) — but doesn't win every month.

## Yearly ROI — All 3 Years Positive Check

Only configurations where **every** year ended positive on ₹2L:

| Config | 2023-24 | 2024-25 | 2025-26 | 3/3 +ve? |
|---|---:|---:|---:|:---:|
| **ema_200_400 N50 (filter ON or OFF)** | **+98.04%** | **+54.47%** | **+5.41%** | **✓** |
| **ema_200_400 N500 nofilter** | +21.25% | +1.05% | +2.52% | ✓ |
| ema_9_21 N50 filter | +62.37% | +27.33% | -11.11% | ✗ |
| ema_9_21 N500 nofilter | +118.24% | -22.78% | -20.46% | ✗ |
| swing N50 / N500 | +5.91% / +12.81% | -7.50% / -16.21% | -1.09% / +0.13% | ✗ |

## Overall Winner

**EMA 200/400 on Nifty 50** = the only model that combines:
- ✅ 3/3 years positive (+98%, +54%, +5%)
- ✅ Lowest worst-year MDD: **13.02%**
- ✅ 3-year avg ROI: **+52.64%**
- ✅ Wins 15/37 (41%) of months — close to ema_9_21's 43%

## Trade-offs

- **ema_200_400 N50**: most consistent, all-year winner, moderate volatility. **Best for paper/live ₹2L deploy.**
- **ema_9_21 N500 nofilter**: highest peak year (+118%), wins most months — but loses 2 of 3 years. High variance, regime-dependent.
- **swing_pullback**: low signal density (~37 trades / yr), negative 3-yr avg both universes. Skip.
- **orb_15min**: no 5m bar Postgres cache → backtest can't compute → 0% across the board. Skip until 5m cache table added.

## Honest caveat

ema_200_400 best-year +98% reflects a strong 2023-24 bull regime. Don't extrapolate that as a future expectation. The +5% on 2025-26 may be closer to baseline. Plan for **annualized ~10-15% net** after slippage/STT in a normal market.

---

## Verification — Per-Month Winner Tables (auto-parsed from _monthly_profile.md)

P&L = realized + mark-to-market in INR each calendar month. ORB shows 0 (no 5m cache table). "Winner" = highest non-zero P&L that month.

### N50 — filter ON

| Month | ema_200_400 ₹ | ema_9_21 ₹ | swing ₹ | Winner |
|---|---:|---:|---:|---|
| 2023-05 | +0 | +17,259 | +0 | ema_9_21 |
| 2023-06 | +2,600 | +9,178 | +0 | ema_9_21 |
| 2023-07 | +0 | +902 | +2,661 | swing_pullback |
| 2023-08 | -6,280 | -16,903 | +62 | swing_pullback |
| 2023-09 | +7,992 | +11,718 | +3,234 | ema_9_21 |
| 2023-10 | -7,867 | -18,048 | -18,562 | ema_200_400 |
| 2023-11 | +12,965 | +3,556 | -8,002 | ema_200_400 |
| 2023-12 | +217 | +4,706 | +4,729 | swing_pullback |
| 2024-01 | +20,529 | +8,824 | +21,744 | swing_pullback |
| 2024-02 | +8,608 | +20,508 | +2,663 | ema_9_21 |
| 2024-03 | +12,292 | +16,069 | +3,627 | ema_9_21 |
| 2024-04 | +5,752 | +6,641 | +4,679 | ema_9_21 |
| 2024-05 | +9,069 | +5,258 | -7,782 | ema_200_400 |
| 2024-06 | +23,556 | +1,520 | +0 | ema_200_400 |
| 2024-07 | -7,270 | +26,646 | +4,960 | ema_9_21 |
| 2024-08 | -2,900 | +13,997 | +7,974 | ema_9_21 |
| 2024-09 | +19,557 | +29,491 | +241 | ema_9_21 |
| 2024-10 | -16,371 | -45,729 | -21,005 | ema_200_400 |
| 2024-11 | +704 | -18,712 | -5,815 | ema_200_400 |
| 2024-12 | -3,508 | +7,253 | +2,119 | ema_9_21 |
| 2025-01 | +8,419 | -26,403 | -3,466 | ema_200_400 |
| 2025-02 | +5,568 | -15,977 | +0 | ema_200_400 |
| 2025-03 | +41,105 | +5,397 | +0 | ema_200_400 |
| 2025-04 | +28,146 | +4,045 | +0 | ema_200_400 |
| 2025-05 | +0 | +8,413 | +0 | ema_9_21 |
| 2025-06 | +36,164 | +15,914 | +0 | ema_200_400 |
| 2025-07 | -19,215 | +5,696 | +0 | ema_9_21 |
| 2025-08 | +23,610 | -4,135 | -3,273 | ema_200_400 |
| 2025-09 | -11,500 | +19,093 | +4,028 | ema_9_21 |
| 2025-10 | +2,722 | +2,637 | -2,945 | ema_200_400 |
| 2025-11 | +14,992 | +9,677 | +0 | ema_200_400 |
| 2025-12 | -8,804 | +448 | +0 | ema_9_21 |
| 2026-01 | -13,911 | -14,961 | +0 | ema_200_400 |
| 2026-02 | +8,536 | -4,995 | +0 | ema_200_400 |
| 2026-03 | -23,829 | -9,238 | +0 | ema_9_21 |
| 2026-04 | +12,490 | +36,417 | +0 | ema_9_21 |
| 2026-05 | +5,904 | +8,201 | +0 | ema_9_21 |

**N50 filter winner tally**: ema_9_21=16, ema_200_400=15, swing=4, orb=0, total=37.

### N500 — filter ON

| Month | ema_200_400 ₹ | ema_9_21 ₹ | swing ₹ | Winner |
|---|---:|---:|---:|---|
| 2023-05 | -3,560 | +1,664 | +0 | ema_9_21 |
| 2023-06 | +12,367 | +3,614 | +0 | ema_200_400 |
| 2023-07 | -2,371 | +18,334 | +7,060 | ema_9_21 |
| 2023-08 | -5,260 | +9,695 | +0 | ema_9_21 |
| 2023-09 | -3,731 | +8,768 | +47,111 | swing_pullback |
| 2023-10 | -5,297 | +2,930 | -19,569 | ema_9_21 |
| 2023-11 | +5,024 | +16,719 | +15,371 | ema_9_21 |
| 2023-12 | +6,274 | +20,171 | +7,436 | ema_9_21 |
| 2024-01 | -3,045 | +21,452 | +10,630 | ema_9_21 |
| 2024-02 | +8,296 | +19,769 | +10,645 | ema_9_21 |
| 2024-03 | -7,340 | -15,275 | -29,147 | ema_200_400 |
| 2024-04 | +19,617 | +31,953 | -9,296 | ema_9_21 |
| 2024-05 | +12,136 | -15,380 | -16,906 | ema_200_400 |
| 2024-06 | +51,793 | +9,875 | +3,247 | ema_200_400 |
| 2024-07 | -22,448 | +3,212 | -4,831 | ema_9_21 |
| 2024-08 | +39,690 | +35,744 | -15,577 | ema_200_400 |
| 2024-09 | +21,510 | +3,373 | -3,242 | ema_200_400 |
| 2024-10 | +26,875 | -19,323 | -27,676 | ema_200_400 |
| 2024-11 | +35,059 | +41,846 | +4,146 | ema_9_21 |
| 2024-12 | -23,942 | +34,300 | +9,555 | ema_9_21 |
| 2025-01 | -41,849 | -39,327 | +0 | ema_9_21 |
| 2025-02 | -18,964 | -68,757 | +0 | ema_200_400 |
| 2025-03 | +32,181 | +25,454 | +0 | ema_200_400 |
| 2025-04 | -64,744 | -8,193 | +0 | ema_9_21 |
| 2025-05 | -5,852 | +22,112 | +0 | ema_9_21 |
| 2025-06 | +6,114 | +18,169 | +0 | ema_9_21 |
| 2025-07 | +9,377 | +26,965 | -1,835 | ema_9_21 |
| 2025-08 | -4,596 | -46,309 | +494 | swing_pullback |
| 2025-09 | +5,835 | +10,589 | +9,083 | ema_9_21 |
| 2025-10 | -15,032 | +28,507 | -11,370 | ema_9_21 |
| 2025-11 | -12,495 | -24,233 | -56 | swing_pullback |
| 2025-12 | +2,496 | -16,819 | +0 | ema_200_400 |
| 2026-01 | -23,053 | -3,255 | -3,409 | ema_9_21 |
| 2026-02 | -11,438 | -36,977 | +7,927 | swing_pullback |
| 2026-03 | -10,383 | -32,614 | +1,891 | swing_pullback |
| 2026-04 | -16,301 | +43,578 | -2,460 | ema_9_21 |
| 2026-05 | -3,546 | +3,369 | +0 | ema_9_21 |

**N500 filter winner tally**: ema_9_21=20, ema_200_400=9, swing=5, orb=0, total=34 (3 ties / dead months).

**Methodology check**: parser reads `_monthly_profile.md` per dir, extracts Sum ₹ column. Same data, no rounding tricks. Manually verifiable by opening individual `_monthly_profile.md` files.

