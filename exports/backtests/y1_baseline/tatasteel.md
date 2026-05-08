# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 214.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 9 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 7 |
| PARTIAL | 1 |
| TARGET_HIT | 4 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 4 / 4 / 1
- **Avg / median % per leg:** 2.93% / 5.00%
- **Sum % (uncompounded):** 26.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 4 | 4 | 1 | 2.93% | 26.4% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 3 | 4 | 0 | 1.63% | 11.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 7 | 3 | 42.9% | 3 | 4 | 0 | 1.63% | 11.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 162.09 | 170.68 | 170.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 14:15:00 | 160.68 | 170.33 | 170.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.99 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 180.01 | 170.37 | 170.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 180.41 | 170.47 | 170.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 182.79 | 184.32 | 179.47 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 187.02 | 184.35 | 179.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 14:15:00 | 188.15 | 184.39 | 179.60 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 09:15:00 | 197.56 | 184.51 | 179.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-10 09:15:00 | 206.97 | 188.27 | 182.50 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 190.89 | 201.25 | 193.79 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 194.74 | 200.50 | 193.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 194.04 | 200.44 | 193.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-12 14:15:00 | 193.71 | 199.46 | 193.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-12 15:15:00 | 193.50 | 199.40 | 193.75 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 185.67 | 199.26 | 193.71 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-17 14:15:00 | 195.24 | 197.22 | 193.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 195.20 | 197.20 | 193.14 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 197.99 | 196.73 | 193.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 198.44 | 196.75 | 193.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 187.00 | 196.16 | 193.13 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 187.00 | 196.16 | 193.13 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 194.80 | 195.73 | 193.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 195.77 | 195.73 | 193.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 194.31 | 195.74 | 193.12 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 197.79 | 195.39 | 193.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 197.13 | 195.40 | 193.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 188.82 | 195.35 | 193.18 | SL hit (close<static) qty=1.00 sl=192.64 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 195.97 | 195.14 | 193.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 195.72 | 195.14 | 193.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-07 12:15:00 | 195.85 | 195.17 | 193.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:15:00 | 196.48 | 195.18 | 193.27 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-28 09:15:00 | 215.35 | 204.27 | 199.44 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 09:15:00 | 215.29 | 204.27 | 199.44 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 09:15:00 | 216.13 | 204.27 | 199.44 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-02 14:15:00 | 188.15 | 2026-02-03 09:15:00 | 197.56 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-02 14:15:00 | 188.15 | 2026-02-10 09:15:00 | 206.97 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-10 11:15:00 | 194.04 | 2026-03-13 09:15:00 | 185.67 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2026-03-17 15:15:00 | 195.20 | 2026-03-23 15:15:00 | 187.00 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-03-20 10:15:00 | 198.44 | 2026-03-23 15:15:00 | 187.00 | STOP_HIT | 1.00 | -5.76% |
| BUY | retest2 | 2026-03-25 10:15:00 | 195.77 | 2026-04-02 09:15:00 | 188.82 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2026-04-01 10:15:00 | 197.13 | 2026-04-28 09:15:00 | 215.35 | TARGET_HIT | 1.00 | 9.24% |
| BUY | retest2 | 2026-04-06 13:15:00 | 195.72 | 2026-04-28 09:15:00 | 215.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 13:15:00 | 196.48 | 2026-04-28 09:15:00 | 216.13 | TARGET_HIT | 1.00 | 10.00% |
