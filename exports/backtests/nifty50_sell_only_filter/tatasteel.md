# TATASTEEL (TATASTEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4989 bars)
- **Last close:** 215.47
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 24 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 20 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 20 / 6
- **Target hits / Stop hits / Partials:** 2 / 17 / 7
- **Avg / median % per leg:** 7.76% / 10.52%
- **Sum % (uncompounded):** 201.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 19 | 79.2% | 2 | 16 | 6 | 7.78% | 186.7% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.65% | 1.3% |
| BUY @ 3rd Alert (retest2) | 22 | 17 | 77.3% | 2 | 14 | 6 | 8.43% | 185.4% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 7.50% | 15.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 0 | 1 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.65% | 1.3% |
| retest2 (combined) | 24 | 18 | 75.0% | 2 | 15 | 7 | 8.35% | 200.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 11:15:00 | 126.05 | 122.49 | 122.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 14:15:00 | 126.30 | 122.60 | 122.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 128.90 | 129.29 | 126.78 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-12-21 09:15:00 | 130.30 | 129.30 | 126.80 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:15:00 | 130.10 | 129.31 | 126.82 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-12-21 12:15:00 | 130.45 | 129.32 | 126.85 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 13:15:00 | 130.10 | 129.33 | 126.86 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 129.55 | 133.64 | 130.95 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 130.95 | 133.64 | 130.95 | SL hit qty=1.00 sl=130.95 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 130.95 | 133.64 | 130.95 | SL hit qty=1.00 sl=130.95 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-19 09:15:00 | 133.55 | 133.48 | 130.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 133.40 | 133.47 | 130.97 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-24 10:15:00 | 133.05 | 133.38 | 131.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-24 11:15:00 | 132.25 | 133.36 | 131.09 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-24 12:15:00 | 133.30 | 133.36 | 131.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 13:15:00 | 135.15 | 133.38 | 131.12 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-03-04 09:15:00 | 153.41 | 140.92 | 137.50 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-03-07 09:15:00 | 155.42 | 143.01 | 138.94 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-05-21 11:15:00 | 173.42 | 163.35 | 157.92 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-05-23 14:15:00 | 175.69 | 164.94 | 159.19 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-07-29 11:15:00 | 163.76 | 167.92 | 167.93 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 10:15:00 | 164.35 | 158.70 | 158.69 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 155.10 | 158.78 | 158.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 154.25 | 158.73 | 158.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 153.47 | 153.33 | 155.46 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 156.37 | 153.37 | 155.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 156.37 | 153.37 | 155.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-07 11:15:00 | 154.06 | 153.39 | 155.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 151.92 | 153.38 | 155.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-10 09:15:00 | 129.13 | 140.27 | 144.41 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-06 13:15:00 | 151.92 | 136.75 | 137.43 | SL hit qty=0.50 sl=151.92 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 153.13 | 138.14 | 138.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 154.60 | 142.54 | 140.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.53 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-04-23 10:15:00 | 138.53 | 142.39 | 142.40 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 5 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 145.05 | 142.33 | 142.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 147.71 | 142.47 | 142.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 154.90 | 155.32 | 151.32 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 152.08 | 155.17 | 151.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 152.08 | 155.17 | 151.38 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-16 10:15:00 | 153.15 | 154.92 | 151.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 153.65 | 154.91 | 151.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 153.10 | 154.29 | 151.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 152.72 | 154.27 | 151.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 12:15:00 | 152.37 | 154.09 | 151.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 152.96 | 154.08 | 151.56 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 154.22 | 154.04 | 151.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 155.66 | 154.06 | 151.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 157.30 | 160.10 | 157.45 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-04 12:15:00 | 159.50 | 159.59 | 157.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 159.25 | 159.59 | 157.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 156.97 | 159.48 | 157.47 | SL hit qty=1.00 sl=156.97 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-07 13:15:00 | 158.37 | 159.38 | 157.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 159.88 | 159.39 | 157.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 12:15:00 | 158.02 | 159.34 | 157.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 158.21 | 159.33 | 157.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-11 12:15:00 | 158.40 | 159.25 | 157.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 158.45 | 159.24 | 157.52 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 157.12 | 159.40 | 157.74 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 156.97 | 159.40 | 157.74 | SL hit qty=1.00 sl=156.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 156.97 | 159.40 | 157.74 | SL hit qty=1.00 sl=156.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 156.97 | 159.40 | 157.74 | SL hit qty=1.00 sl=156.97 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-20 10:15:00 | 161.52 | 159.08 | 157.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:15:00 | 161.90 | 159.11 | 157.76 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 156.52 | 159.37 | 158.08 | SL hit qty=1.00 sl=156.52 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 09:15:00 | 160.94 | 158.39 | 157.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 163.60 | 158.44 | 157.77 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-09 09:15:00 | 176.70 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-09 09:15:00 | 175.63 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-09 09:15:00 | 175.90 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-28 09:15:00 | 179.01 | 171.47 | 167.99 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-12-09 12:15:00 | 161.79 | 170.51 | 170.54 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 180.74 | 170.67 | 170.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 200.54 | 201.77 | 192.99 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 190.83 | 201.17 | 193.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 190.83 | 201.17 | 193.54 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 194.82 | 200.43 | 193.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 194.00 | 200.36 | 193.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-12 14:15:00 | 193.69 | 199.40 | 193.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-12 15:15:00 | 193.33 | 199.34 | 193.53 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 187.22 | 199.20 | 193.49 | SL hit qty=1.00 sl=187.22 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-17 14:15:00 | 195.22 | 197.18 | 192.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 195.20 | 197.16 | 192.95 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 197.99 | 196.69 | 193.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 198.45 | 196.71 | 193.04 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 187.22 | 196.64 | 193.11 | SL hit qty=1.00 sl=187.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 187.22 | 196.64 | 193.11 | SL hit qty=1.00 sl=187.22 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 194.82 | 195.70 | 192.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 195.80 | 195.70 | 192.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 194.30 | 195.70 | 192.96 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 197.79 | 195.36 | 192.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 197.19 | 195.37 | 192.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 192.73 | 195.33 | 193.04 | SL hit qty=1.00 sl=192.73 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 195.95 | 195.11 | 193.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 195.72 | 195.12 | 193.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-07 12:15:00 | 195.85 | 195.15 | 193.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:15:00 | 196.48 | 195.16 | 193.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-21 10:15:00 | 130.10 | 2024-01-18 09:15:00 | 130.95 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest1 | 2023-12-21 13:15:00 | 130.10 | 2024-01-18 09:15:00 | 130.95 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2024-01-19 10:15:00 | 133.40 | 2024-03-04 09:15:00 | 153.41 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-24 13:15:00 | 135.15 | 2024-03-07 09:15:00 | 155.42 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-19 10:15:00 | 133.40 | 2024-05-21 11:15:00 | 173.42 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-01-24 13:15:00 | 135.15 | 2024-05-23 14:15:00 | 175.69 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2024-11-07 12:15:00 | 151.92 | 2025-01-10 09:15:00 | 129.13 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 12:15:00 | 151.92 | 2025-03-06 13:15:00 | 151.92 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-06-16 11:15:00 | 153.65 | 2025-08-07 09:15:00 | 156.97 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2025-06-20 11:15:00 | 152.72 | 2025-08-14 09:15:00 | 156.97 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2025-06-23 13:15:00 | 152.96 | 2025-08-14 09:15:00 | 156.97 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest2 | 2025-06-24 10:15:00 | 155.66 | 2025-08-14 09:15:00 | 156.97 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-08-04 13:15:00 | 159.25 | 2025-08-26 10:15:00 | 156.52 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-08-07 14:15:00 | 159.88 | 2025-10-09 09:15:00 | 176.70 | PARTIAL | 0.50 | 10.52% |
| BUY | retest2 | 2025-08-08 13:15:00 | 158.21 | 2025-10-09 09:15:00 | 175.63 | PARTIAL | 0.50 | 11.01% |
| BUY | retest2 | 2025-08-11 13:15:00 | 158.45 | 2025-10-09 09:15:00 | 175.90 | PARTIAL | 0.50 | 11.02% |
| BUY | retest2 | 2025-08-20 11:15:00 | 161.90 | 2025-10-28 09:15:00 | 179.01 | PARTIAL | 0.50 | 10.57% |
| BUY | retest2 | 2025-08-07 14:15:00 | 159.88 | 2025-12-31 09:15:00 | 179.75 | STOP_HIT | 0.50 | 12.43% |
| BUY | retest2 | 2025-08-08 13:15:00 | 158.21 | 2025-12-31 09:15:00 | 179.75 | STOP_HIT | 0.50 | 13.61% |
| BUY | retest2 | 2025-08-11 13:15:00 | 158.45 | 2025-12-31 09:15:00 | 179.75 | STOP_HIT | 0.50 | 13.44% |
| BUY | retest2 | 2025-08-20 11:15:00 | 161.90 | 2025-12-31 09:15:00 | 179.75 | STOP_HIT | 0.50 | 11.03% |
| BUY | retest2 | 2025-09-03 10:15:00 | 163.60 | 2025-12-31 09:15:00 | 179.75 | STOP_HIT | 1.00 | 9.87% |
| BUY | retest2 | 2026-03-10 11:15:00 | 194.00 | 2026-03-13 09:15:00 | 187.22 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-03-17 15:15:00 | 195.20 | 2026-03-23 09:15:00 | 187.22 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2026-03-20 10:15:00 | 198.45 | 2026-03-23 09:15:00 | 187.22 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest2 | 2026-03-25 10:15:00 | 195.80 | 2026-04-02 09:15:00 | 192.73 | STOP_HIT | 1.00 | -1.57% |
