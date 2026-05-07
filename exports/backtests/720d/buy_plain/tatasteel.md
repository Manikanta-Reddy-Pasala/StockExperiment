# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 217.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 7 |
| PENDING | 21 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 19 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 18 / 6
- **Target hits / Stop hits / Partials:** 0 / 17 / 7
- **Avg / median % per leg:** 5.77% / 7.10%
- **Sum % (uncompounded):** 138.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 18 | 75.0% | 0 | 17 | 7 | 5.77% | 138.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 10.79% | 21.6% |
| BUY @ 3rd Alert (retest2) | 22 | 16 | 72.7% | 0 | 16 | 6 | 5.32% | 117.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 10.79% | 21.6% |
| retest2 (combined) | 22 | 16 | 72.7% | 0 | 16 | 6 | 5.32% | 117.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 151.50 | 137.99 | 137.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 09:15:00 | 153.13 | 138.14 | 138.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.50 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 145.30 | 142.30 | 142.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 147.74 | 142.47 | 142.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 154.85 | 155.31 | 151.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 152.08 | 155.16 | 151.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 152.08 | 155.16 | 151.37 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-16 10:15:00 | 153.14 | 154.92 | 151.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 153.69 | 154.91 | 151.41 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 153.05 | 154.28 | 151.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 152.72 | 154.26 | 151.53 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 13:15:00 | 152.94 | 154.07 | 151.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 152.37 | 154.05 | 151.55 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 154.21 | 154.03 | 151.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 155.64 | 154.05 | 151.59 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 157.30 | 160.09 | 157.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-04 12:15:00 | 159.50 | 159.59 | 157.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 159.25 | 159.58 | 157.33 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 156.55 | 159.39 | 157.45 | SL hit (close<static) qty=1.00 sl=156.56 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-07 13:15:00 | 158.37 | 159.38 | 157.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 159.88 | 159.39 | 157.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 12:15:00 | 158.02 | 159.34 | 157.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 158.21 | 159.33 | 157.49 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 15:15:00 | 158.25 | 159.30 | 157.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 158.09 | 159.29 | 157.50 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 3960m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 157.57 | 159.26 | 157.50 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-11 12:15:00 | 158.38 | 159.25 | 157.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 158.44 | 159.24 | 157.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 157.05 | 159.38 | 157.73 | SL hit (close<static) qty=1.00 sl=157.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 155.95 | 159.30 | 157.72 | SL hit (close<static) qty=1.00 sl=156.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 155.95 | 159.30 | 157.72 | SL hit (close<static) qty=1.00 sl=156.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 155.95 | 159.30 | 157.72 | SL hit (close<static) qty=1.00 sl=156.56 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 12:15:00 | 158.46 | 159.05 | 157.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 158.69 | 159.05 | 157.69 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 156.88 | 159.37 | 158.08 | SL hit (close<static) qty=1.00 sl=157.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-02 11:15:00 | 158.98 | 158.36 | 157.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 12:15:00 | 158.78 | 158.37 | 157.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 14:15:00 | 158.44 | 158.36 | 157.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 158.45 | 158.36 | 157.72 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-09 09:15:00 | 176.74 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-09 09:15:00 | 175.63 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-09 09:15:00 | 175.23 | 168.52 | 165.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 169.44 | 169.59 | 166.11 | SL hit (close<ema200) qty=0.50 sl=169.59 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 169.44 | 169.59 | 166.11 | SL hit (close<ema200) qty=0.50 sl=169.59 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 169.44 | 169.59 | 166.11 | SL hit (close<ema200) qty=0.50 sl=169.59 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-28 09:15:00 | 178.99 | 171.46 | 167.98 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-29 09:15:00 | 182.60 | 172.17 | 168.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-29 09:15:00 | 182.22 | 172.17 | 168.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 176.67 | 176.70 | 172.35 | SL hit (close<ema200) qty=0.50 sl=176.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 176.67 | 176.70 | 172.35 | SL hit (close<ema200) qty=0.50 sl=176.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 176.67 | 176.70 | 172.35 | SL hit (close<ema200) qty=0.50 sl=176.70 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 174.18 | 176.67 | 172.38 | EMA400 retest candle locked |

### Cycle 3 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 181.45 | 171.06 | 170.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 182.79 | 184.32 | 179.46 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 187.02 | 184.35 | 179.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 14:15:00 | 188.15 | 184.39 | 179.59 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-25 10:15:00 | 216.37 | 198.56 | 190.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 200.53 | 201.87 | 193.26 | SL hit (close<ema200) qty=0.50 sl=201.87 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 190.89 | 201.25 | 193.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 194.74 | 200.50 | 193.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 194.04 | 200.44 | 193.70 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-12 14:15:00 | 193.71 | 199.46 | 193.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-12 15:15:00 | 193.50 | 199.40 | 193.74 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 185.67 | 199.26 | 193.70 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-17 14:15:00 | 195.24 | 197.22 | 193.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 195.20 | 197.20 | 193.14 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 197.99 | 196.73 | 193.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 198.44 | 196.75 | 193.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 187.00 | 196.16 | 193.12 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 187.00 | 196.16 | 193.12 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 194.80 | 195.73 | 193.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 195.77 | 195.73 | 193.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 194.31 | 195.74 | 193.12 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 197.79 | 195.39 | 193.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 197.13 | 195.40 | 193.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 188.82 | 195.35 | 193.18 | SL hit (close<static) qty=1.00 sl=192.64 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 195.97 | 195.14 | 193.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 195.72 | 195.14 | 193.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-07 12:15:00 | 195.85 | 195.17 | 193.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:15:00 | 196.48 | 195.18 | 193.27 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 11:15:00 | 153.69 | 2025-08-07 12:15:00 | 156.55 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-06-20 11:15:00 | 152.72 | 2025-08-14 10:15:00 | 157.05 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest2 | 2025-06-23 14:15:00 | 152.37 | 2025-08-14 13:15:00 | 155.95 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-06-24 10:15:00 | 155.64 | 2025-08-14 13:15:00 | 155.95 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-08-04 13:15:00 | 159.25 | 2025-08-14 13:15:00 | 155.95 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-08-07 14:15:00 | 159.88 | 2025-08-26 10:15:00 | 156.88 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-08 13:15:00 | 158.21 | 2025-10-09 09:15:00 | 176.74 | PARTIAL | 0.50 | 11.71% |
| BUY | retest2 | 2025-08-11 09:15:00 | 158.09 | 2025-10-09 09:15:00 | 175.63 | PARTIAL | 0.50 | 11.09% |
| BUY | retest2 | 2025-08-11 13:15:00 | 158.44 | 2025-10-09 09:15:00 | 175.23 | PARTIAL | 0.50 | 10.59% |
| BUY | retest2 | 2025-08-08 13:15:00 | 158.21 | 2025-10-14 12:15:00 | 169.44 | STOP_HIT | 0.50 | 7.10% |
| BUY | retest2 | 2025-08-11 09:15:00 | 158.09 | 2025-10-14 12:15:00 | 169.44 | STOP_HIT | 0.50 | 7.18% |
| BUY | retest2 | 2025-08-11 13:15:00 | 158.44 | 2025-10-14 12:15:00 | 169.44 | STOP_HIT | 0.50 | 6.94% |
| BUY | retest2 | 2025-08-19 13:15:00 | 158.69 | 2025-10-28 09:15:00 | 178.99 | PARTIAL | 0.50 | 12.79% |
| BUY | retest2 | 2025-09-02 12:15:00 | 158.78 | 2025-10-29 09:15:00 | 182.60 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-09-02 15:15:00 | 158.45 | 2025-10-29 09:15:00 | 182.22 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-08-19 13:15:00 | 158.69 | 2025-11-13 14:15:00 | 176.67 | STOP_HIT | 0.50 | 11.33% |
| BUY | retest2 | 2025-09-02 12:15:00 | 158.78 | 2025-11-13 14:15:00 | 176.67 | STOP_HIT | 0.50 | 11.27% |
| BUY | retest2 | 2025-09-02 15:15:00 | 158.45 | 2025-11-13 14:15:00 | 176.67 | STOP_HIT | 0.50 | 11.50% |
| BUY | retest1 | 2026-02-02 14:15:00 | 188.15 | 2026-02-25 10:15:00 | 216.37 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-02-02 14:15:00 | 188.15 | 2026-03-04 09:15:00 | 200.53 | STOP_HIT | 0.50 | 6.58% |
| BUY | retest2 | 2026-03-10 11:15:00 | 194.04 | 2026-03-13 09:15:00 | 185.67 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2026-03-17 15:15:00 | 195.20 | 2026-03-23 15:15:00 | 187.00 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-03-20 10:15:00 | 198.44 | 2026-03-23 15:15:00 | 187.00 | STOP_HIT | 1.00 | -5.76% |
| BUY | retest2 | 2026-03-25 10:15:00 | 195.77 | 2026-04-02 09:15:00 | 188.82 | STOP_HIT | 1.00 | -3.55% |
