# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 380.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 28 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 3 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 18
- **Target hits / Stop hits / Partials:** 0 / 21 / 3
- **Avg / median % per leg:** 0.19% / -1.42%
- **Sum % (uncompounded):** 4.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 6 | 25.0% | 0 | 21 | 3 | 0.19% | 4.6% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 10.72% | 64.3% |
| BUY @ 3rd Alert (retest2) | 18 | 0 | 0.0% | 0 | 18 | 0 | -3.31% | -59.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 10.72% | 64.3% |
| retest2 (combined) | 18 | 0 | 0.0% | 0 | 18 | 0 | -3.31% | -59.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 384.92 | 356.14 | 356.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 385.35 | 369.82 | 364.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 383.10 | 383.55 | 375.38 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-05 10:15:00 | 387.44 | 383.33 | 376.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:15:00 | 389.39 | 383.39 | 376.14 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-07 14:15:00 | 386.99 | 383.73 | 376.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 15:15:00 | 386.40 | 383.75 | 376.95 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-11 12:15:00 | 385.96 | 383.93 | 377.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 13:15:00 | 388.34 | 383.98 | 377.46 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 11:15:00 | 447.80 | 414.60 | 401.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 11:15:00 | 444.36 | 414.60 | 401.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 11:15:00 | 446.59 | 414.60 | 401.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 413.00 | 427.83 | 412.82 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 413.00 | 427.83 | 412.82 | SL hit (close<ema200) qty=0.50 sl=427.83 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 413.00 | 427.83 | 412.82 | SL hit (close<ema200) qty=0.50 sl=427.83 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 413.00 | 427.83 | 412.82 | SL hit (close<ema200) qty=0.50 sl=427.83 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 431.38 | 425.17 | 413.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 431.86 | 425.24 | 413.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 411.30 | 422.94 | 415.33 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 10:15:00 | 426.82 | 417.54 | 414.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 426.96 | 417.63 | 414.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 11:15:00 | 427.40 | 420.29 | 416.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 427.70 | 420.37 | 416.79 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-18 15:15:00 | 426.98 | 421.33 | 417.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 429.60 | 421.41 | 417.65 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 423.00 | 428.70 | 423.31 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-04 14:15:00 | 425.98 | 428.58 | 423.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 426.00 | 428.55 | 423.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-25 14:15:00 | 424.96 | 432.54 | 428.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-25 15:15:00 | 424.24 | 432.46 | 428.21 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.12 | 428.06 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.12 | 428.06 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.12 | 428.06 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.12 | 428.06 | SL hit (close<static) qty=1.00 sl=422.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 425.12 | 403.11 | 405.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 428.10 | 403.36 | 405.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 422.04 | 407.44 | 407.57 | SL hit (close<static) qty=1.00 sl=422.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-09 10:15:00 | 426.64 | 407.63 | 407.67 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 2 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.36 | 407.83 | 407.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.10 | 408.05 | 407.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.80 | 424.03 | 417.87 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 419.94 | 423.84 | 417.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 419.94 | 423.84 | 417.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-03 12:15:00 | 422.00 | 423.74 | 418.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 423.52 | 423.74 | 418.05 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 417.36 | 423.30 | 418.16 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-21 13:15:00 | 421.86 | 420.64 | 418.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-21 14:15:00 | 417.54 | 420.61 | 418.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-27 10:15:00 | 423.68 | 419.96 | 418.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 425.02 | 420.01 | 418.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 424.20 | 429.57 | 427.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 423.90 | 429.51 | 427.02 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 422.30 | 428.98 | 426.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 422.60 | 428.91 | 426.80 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 427.20 | 428.89 | 426.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-19 12:15:00 | 428.20 | 428.89 | 426.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-19 13:15:00 | 427.60 | 428.87 | 426.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-20 09:15:00 | 429.70 | 428.85 | 426.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 428.60 | 428.85 | 426.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-20 12:15:00 | 427.80 | 428.83 | 426.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 13:15:00 | 427.90 | 428.82 | 426.85 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 418.80 | 428.56 | 426.76 | SL hit (close<static) qty=1.00 sl=421.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 418.80 | 428.56 | 426.76 | SL hit (close<static) qty=1.00 sl=421.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 405.90 | 427.49 | 426.37 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 405.90 | 427.49 | 426.37 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 405.90 | 427.49 | 426.37 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-09 09:15:00 | 429.65 | 419.82 | 422.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 10:15:00 | 425.55 | 419.88 | 422.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-09 13:15:00 | 428.35 | 420.10 | 422.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 428.45 | 420.18 | 422.26 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-12 12:15:00 | 429.45 | 421.81 | 422.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 428.00 | 421.87 | 422.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 423.85 | 422.01 | 422.99 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 420.55 | 422.00 | 422.98 | SL hit (close<static) qty=1.00 sl=421.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 420.55 | 422.00 | 422.98 | SL hit (close<static) qty=1.00 sl=421.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-16 14:15:00 | 425.80 | 421.98 | 422.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 426.15 | 422.03 | 422.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 421.95 | 422.02 | 422.94 | SL hit (close<static) qty=1.00 sl=422.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 426.30 | 422.20 | 422.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 425.70 | 422.23 | 422.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-19 10:15:00 | 425.00 | 422.28 | 423.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-19 11:15:00 | 422.30 | 422.28 | 423.00 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 422.30 | 422.28 | 423.00 | SL hit (close<static) qty=1.00 sl=422.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 427.30 | 422.09 | 422.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 428.60 | 422.15 | 422.89 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 422.80 | 423.06 | 423.31 | SL hit (close<static) qty=1.00 sl=422.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 14:15:00 | 425.45 | 423.09 | 423.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 15:15:00 | 423.65 | 423.10 | 423.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-26 09:15:00 | 426.25 | 423.13 | 423.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 424.95 | 423.15 | 423.35 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 423.30 | 423.15 | 423.35 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-26 13:15:00 | 422.10 | 423.15 | 423.35 | SL hit (close<static) qty=1.00 sl=422.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-03-05 11:15:00 | 389.39 | 2025-04-21 11:15:00 | 447.80 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-07 15:15:00 | 386.40 | 2025-04-21 11:15:00 | 444.36 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-11 13:15:00 | 388.34 | 2025-04-21 11:15:00 | 446.59 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-05 11:15:00 | 389.39 | 2025-05-05 09:15:00 | 413.00 | STOP_HIT | 0.50 | 6.06% |
| BUY | retest1 | 2025-03-07 15:15:00 | 386.40 | 2025-05-05 09:15:00 | 413.00 | STOP_HIT | 0.50 | 6.88% |
| BUY | retest1 | 2025-03-11 13:15:00 | 388.34 | 2025-05-05 09:15:00 | 413.00 | STOP_HIT | 0.50 | 6.35% |
| BUY | retest2 | 2025-05-12 10:15:00 | 431.86 | 2025-05-22 09:15:00 | 411.30 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-06-09 11:15:00 | 426.96 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2025-06-16 12:15:00 | 427.70 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.93% |
| BUY | retest2 | 2025-06-19 09:15:00 | 429.60 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -7.34% |
| BUY | retest2 | 2025-07-04 15:15:00 | 426.00 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.55% |
| BUY | retest2 | 2025-10-06 10:15:00 | 428.10 | 2025-10-09 09:15:00 | 422.04 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-11-03 13:15:00 | 423.52 | 2025-11-06 11:15:00 | 417.36 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-27 11:15:00 | 425.02 | 2026-01-21 10:15:00 | 418.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-16 10:15:00 | 423.90 | 2026-01-21 10:15:00 | 418.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-19 10:15:00 | 422.60 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2026-01-20 10:15:00 | 428.60 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest2 | 2026-01-20 13:15:00 | 427.90 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-02-09 14:15:00 | 428.45 | 2026-02-13 13:15:00 | 420.55 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-02-12 13:15:00 | 428.00 | 2026-02-13 13:15:00 | 420.55 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-16 15:15:00 | 426.15 | 2026-02-17 09:15:00 | 421.95 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-18 15:15:00 | 425.70 | 2026-02-19 11:15:00 | 422.30 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-23 10:15:00 | 428.60 | 2026-02-25 12:15:00 | 422.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-26 10:15:00 | 424.95 | 2026-02-26 13:15:00 | 422.10 | STOP_HIT | 1.00 | -0.67% |
