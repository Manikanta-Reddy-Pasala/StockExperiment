# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4990 bars)
- **Last close:** 214.49
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 10 |
| PENDING | 27 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 20 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 18 / 10
- **Target hits / Stop hits / Partials:** 0 / 21 / 7
- **Avg / median % per leg:** 4.23% / 5.98%
- **Sum % (uncompounded):** 118.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 16 | 66.7% | 0 | 18 | 6 | 4.22% | 101.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.42% | -0.8% |
| BUY @ 3rd Alert (retest2) | 22 | 16 | 72.7% | 0 | 16 | 6 | 4.65% | 102.2% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 4.30% | 17.2% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.76% | -9.5% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 13.37% | 26.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.59% | -10.4% |
| retest2 (combined) | 24 | 18 | 75.0% | 0 | 17 | 7 | 5.37% | 128.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 117.35 | 122.81 | 122.81 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 14:15:00 | 126.30 | 122.60 | 122.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 126.45 | 122.67 | 122.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 128.90 | 129.29 | 126.81 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-21 09:15:00 | 130.30 | 129.30 | 126.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:15:00 | 130.10 | 129.31 | 126.85 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-12-21 12:15:00 | 130.45 | 129.32 | 126.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 13:15:00 | 130.10 | 129.33 | 126.89 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 129.55 | 133.64 | 130.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 129.55 | 133.64 | 130.96 | SL hit (close<ema400) qty=1.00 sl=130.96 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 129.55 | 133.64 | 130.96 | SL hit (close<ema400) qty=1.00 sl=130.96 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-19 09:15:00 | 133.55 | 133.48 | 130.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 133.40 | 133.47 | 130.98 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-24 10:15:00 | 133.05 | 133.38 | 131.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-24 11:15:00 | 132.25 | 133.36 | 131.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-24 12:15:00 | 133.30 | 133.36 | 131.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 13:15:00 | 135.15 | 133.38 | 131.14 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:15:00 | 153.41 | 140.92 | 137.51 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 09:15:00 | 155.42 | 143.01 | 138.95 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 13:15:00 | 144.00 | 145.08 | 140.54 | SL hit (close<ema200) qty=0.50 sl=145.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 13:15:00 | 144.00 | 145.08 | 140.54 | SL hit (close<ema200) qty=0.50 sl=145.08 alert=retest2 |

### Cycle 3 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 163.76 | 167.92 | 167.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 163.22 | 167.87 | 167.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 154.60 | 153.65 | 157.46 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-09-17 11:15:00 | 152.42 | 153.66 | 157.18 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-17 12:15:00 | 153.19 | 153.66 | 157.16 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-18 09:15:00 | 152.22 | 153.62 | 157.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:15:00 | 151.68 | 153.60 | 157.04 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-20 13:15:00 | 151.84 | 153.15 | 156.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 14:15:00 | 151.98 | 153.14 | 156.50 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 159.06 | 153.22 | 156.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 159.06 | 153.22 | 156.39 | SL hit (close>ema400) qty=1.00 sl=156.39 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 159.06 | 153.22 | 156.39 | SL hit (close>ema400) qty=1.00 sl=156.39 alert=retest1 |

### Cycle 4 — BUY (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 10:15:00 | 164.35 | 158.70 | 158.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 155.10 | 158.78 | 158.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 154.25 | 158.73 | 158.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 153.47 | 153.33 | 155.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 156.37 | 153.37 | 155.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 156.37 | 153.37 | 155.46 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-07 11:15:00 | 154.06 | 153.39 | 155.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 151.92 | 153.38 | 155.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 129.13 | 140.27 | 144.41 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 134.08 | 133.17 | 138.15 | SL hit (close>ema200) qty=0.50 sl=133.17 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 153.13 | 138.14 | 138.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 154.60 | 142.54 | 140.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 138.53 | 142.39 | 142.40 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 145.05 | 142.33 | 142.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 147.71 | 142.47 | 142.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 154.90 | 155.32 | 151.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 152.08 | 155.17 | 151.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 152.08 | 155.17 | 151.38 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-16 10:15:00 | 153.15 | 154.92 | 151.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 153.65 | 154.91 | 151.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 153.10 | 154.29 | 151.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 152.72 | 154.27 | 151.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 12:15:00 | 152.37 | 154.09 | 151.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 152.96 | 154.08 | 151.56 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 154.22 | 154.04 | 151.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 155.66 | 154.06 | 151.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 157.30 | 160.10 | 157.45 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-08-04 12:15:00 | 159.50 | 159.59 | 157.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 159.25 | 159.59 | 157.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 156.61 | 159.45 | 157.46 | SL hit (close<static) qty=1.00 sl=156.97 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-07 13:15:00 | 158.37 | 159.38 | 157.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 159.88 | 159.39 | 157.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 12:15:00 | 158.02 | 159.34 | 157.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 158.21 | 159.33 | 157.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-11 12:15:00 | 158.40 | 159.25 | 157.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 158.45 | 159.24 | 157.52 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 157.12 | 159.40 | 157.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 155.95 | 159.30 | 157.72 | SL hit (close<static) qty=1.00 sl=156.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 155.95 | 159.30 | 157.72 | SL hit (close<static) qty=1.00 sl=156.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 155.95 | 159.30 | 157.72 | SL hit (close<static) qty=1.00 sl=156.97 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-20 10:15:00 | 161.52 | 159.08 | 157.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:15:00 | 161.90 | 159.11 | 157.76 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 156.36 | 159.32 | 158.06 | SL hit (close<static) qty=1.00 sl=156.52 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 09:15:00 | 160.94 | 158.39 | 157.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 163.60 | 158.44 | 157.77 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 176.70 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 175.63 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 175.90 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 169.44 | 169.59 | 166.11 | SL hit (close<ema200) qty=0.50 sl=169.59 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 169.44 | 169.59 | 166.11 | SL hit (close<ema200) qty=0.50 sl=169.59 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 169.44 | 169.59 | 166.11 | SL hit (close<ema200) qty=0.50 sl=169.59 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:15:00 | 179.01 | 171.47 | 167.99 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 176.64 | 176.71 | 172.36 | SL hit (close<ema200) qty=0.50 sl=176.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 161.79 | 170.51 | 170.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 161.79 | 170.51 | 170.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 14:15:00 | 160.68 | 170.33 | 170.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.95 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 180.74 | 170.67 | 170.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 200.54 | 201.77 | 192.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 190.83 | 201.17 | 193.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 190.83 | 201.17 | 193.54 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 194.82 | 200.43 | 193.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 194.00 | 200.36 | 193.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-12 14:15:00 | 193.69 | 199.40 | 193.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-12 15:15:00 | 193.33 | 199.34 | 193.53 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 185.70 | 199.20 | 193.49 | SL hit (close<static) qty=1.00 sl=187.22 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-17 14:15:00 | 195.22 | 197.18 | 192.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 195.20 | 197.16 | 192.95 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 197.99 | 196.69 | 193.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 198.45 | 196.71 | 193.04 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 187.00 | 196.12 | 192.96 | SL hit (close<static) qty=1.00 sl=187.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 187.00 | 196.12 | 192.96 | SL hit (close<static) qty=1.00 sl=187.22 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 194.82 | 195.70 | 192.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 195.80 | 195.70 | 192.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 194.30 | 195.70 | 192.96 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 197.79 | 195.36 | 192.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 197.19 | 195.37 | 192.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 188.87 | 195.33 | 193.04 | SL hit (close<static) qty=1.00 sl=192.73 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 195.95 | 195.11 | 193.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 195.72 | 195.12 | 193.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-07 12:15:00 | 195.85 | 195.15 | 193.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:15:00 | 196.48 | 195.16 | 193.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-21 10:15:00 | 130.10 | 2024-01-18 09:15:00 | 129.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-12-21 13:15:00 | 130.10 | 2024-01-18 09:15:00 | 129.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-01-19 10:15:00 | 133.40 | 2024-03-04 09:15:00 | 153.41 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-24 13:15:00 | 135.15 | 2024-03-07 09:15:00 | 155.42 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-19 10:15:00 | 133.40 | 2024-03-13 13:15:00 | 144.00 | STOP_HIT | 0.50 | 7.95% |
| BUY | retest2 | 2024-01-24 13:15:00 | 135.15 | 2024-03-13 13:15:00 | 144.00 | STOP_HIT | 0.50 | 6.55% |
| SELL | retest1 | 2024-09-18 10:15:00 | 151.68 | 2024-09-24 09:15:00 | 159.06 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest1 | 2024-09-20 14:15:00 | 151.98 | 2024-09-24 09:15:00 | 159.06 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2024-11-07 12:15:00 | 151.92 | 2025-01-10 09:15:00 | 129.13 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 12:15:00 | 151.92 | 2025-01-31 10:15:00 | 134.08 | STOP_HIT | 0.50 | 11.74% |
| BUY | retest2 | 2025-06-16 11:15:00 | 153.65 | 2025-08-07 10:15:00 | 156.61 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-06-20 11:15:00 | 152.72 | 2025-08-14 13:15:00 | 155.95 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-06-23 13:15:00 | 152.96 | 2025-08-14 13:15:00 | 155.95 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-06-24 10:15:00 | 155.66 | 2025-08-14 13:15:00 | 155.95 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-08-04 13:15:00 | 159.25 | 2025-08-26 12:15:00 | 156.36 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-08-07 14:15:00 | 159.88 | 2025-10-09 09:15:00 | 176.70 | PARTIAL | 0.50 | 10.52% |
| BUY | retest2 | 2025-08-08 13:15:00 | 158.21 | 2025-10-09 09:15:00 | 175.63 | PARTIAL | 0.50 | 11.01% |
| BUY | retest2 | 2025-08-11 13:15:00 | 158.45 | 2025-10-09 09:15:00 | 175.90 | PARTIAL | 0.50 | 11.02% |
| BUY | retest2 | 2025-08-07 14:15:00 | 159.88 | 2025-10-14 12:15:00 | 169.44 | STOP_HIT | 0.50 | 5.98% |
| BUY | retest2 | 2025-08-08 13:15:00 | 158.21 | 2025-10-14 12:15:00 | 169.44 | STOP_HIT | 0.50 | 7.10% |
| BUY | retest2 | 2025-08-11 13:15:00 | 158.45 | 2025-10-14 12:15:00 | 169.44 | STOP_HIT | 0.50 | 6.94% |
| BUY | retest2 | 2025-08-20 11:15:00 | 161.90 | 2025-10-28 09:15:00 | 179.01 | PARTIAL | 0.50 | 10.57% |
| BUY | retest2 | 2025-08-20 11:15:00 | 161.90 | 2025-11-13 14:15:00 | 176.64 | STOP_HIT | 0.50 | 9.10% |
| BUY | retest2 | 2025-09-03 10:15:00 | 163.60 | 2025-12-09 12:15:00 | 161.79 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-10 11:15:00 | 194.00 | 2026-03-13 09:15:00 | 185.70 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2026-03-17 15:15:00 | 195.20 | 2026-03-23 15:15:00 | 187.00 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-03-20 10:15:00 | 198.45 | 2026-03-23 15:15:00 | 187.00 | STOP_HIT | 1.00 | -5.77% |
| BUY | retest2 | 2026-03-25 10:15:00 | 195.80 | 2026-04-02 09:15:00 | 188.87 | STOP_HIT | 1.00 | -3.54% |
