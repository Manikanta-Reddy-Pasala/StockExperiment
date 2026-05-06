# WIPRO (WIPRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 199.12
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 16 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 7 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 15
- **Target hits / Stop hits / Partials:** 0 / 15 / 1
- **Avg / median % per leg:** -1.79% / -2.69%
- **Sum % (uncompounded):** -28.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 0 | 15 | 1 | -1.79% | -28.7% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.75% | -26.2% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.27% | -2.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.75% | -26.2% |
| retest2 (combined) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.27% | -2.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 209.98 | 201.36 | 201.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 210.65 | 201.62 | 201.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 252.23 | 253.87 | 242.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-06 15:15:00 | 256.58 | 253.86 | 243.27 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:15:00 | 256.60 | 253.88 | 243.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 257.35 | 254.23 | 243.98 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 258.42 | 254.27 | 244.05 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-14 14:15:00 | 259.12 | 254.45 | 245.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 15:15:00 | 258.98 | 254.50 | 245.33 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-15 12:15:00 | 256.65 | 254.53 | 245.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 13:15:00 | 257.25 | 254.55 | 245.58 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-21 09:15:00 | 251.80 | 253.84 | 246.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 10:15:00 | 251.80 | 253.82 | 246.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-22 09:15:00 | 244.82 | 253.54 | 246.34 | SL hit qty=1.00 sl=244.82 alert=retest2 |
| CROSSOVER_SKIP | 2024-04-18 12:15:00 | 225.00 | 242.68 | 242.76 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 248.38 | 235.04 | 235.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 253.18 | 238.18 | 236.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 255.40 | 261.20 | 251.52 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 12:15:00 | 251.60 | 260.98 | 251.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 251.60 | 260.98 | 251.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 257.92 | 258.91 | 251.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 259.33 | 258.92 | 251.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 250.77 | 259.38 | 253.17 | SL hit qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-16 11:15:00 | 255.57 | 253.50 | 251.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 12:15:00 | 256.55 | 253.53 | 251.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-26 15:15:00 | 295.03 | 279.41 | 273.70 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-03-06 15:15:00 | 285.20 | 299.05 | 299.11 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 256.55 | 287.32 | 292.64 | SL hit qty=0.50 sl=256.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-12 12:15:00 | 255.81 | 249.15 | 261.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 13:15:00 | 257.70 | 249.24 | 261.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-13 14:15:00 | 250.77 | 249.58 | 260.84 | SL hit qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 255.88 | 249.94 | 260.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 256.39 | 250.01 | 260.30 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 256.08 | 250.07 | 260.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 250.77 | 250.54 | 259.61 | SL hit qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-11 12:15:00 | 259.44 | 249.93 | 255.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 259.60 | 250.03 | 255.30 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 255.97 | 250.93 | 255.51 | SL hit qty=1.00 sl=255.97 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 265.01 | 258.50 | 258.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 266.11 | 258.92 | 258.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-16 13:15:00 | 262.45 | 261.40 | 260.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 262.75 | 261.42 | 260.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-17 15:15:00 | 258.50 | 261.41 | 260.47 | SL hit qty=1.00 sl=258.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-18 09:15:00 | 267.85 | 261.47 | 260.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 267.75 | 261.53 | 260.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-21 10:15:00 | 262.35 | 261.78 | 260.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-21 11:15:00 | 261.80 | 261.78 | 260.71 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 258.50 | 261.59 | 260.67 | SL hit qty=1.00 sl=258.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.93 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 4 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.33 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.79 | 257.19 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 263.70 | 261.82 | 257.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 13:15:00 | 263.65 | 261.83 | 257.28 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 13:15:00 | 262.90 | 261.91 | 257.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 14:15:00 | 264.25 | 261.94 | 257.51 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 268.95 | 261.96 | 257.72 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 267.75 | 262.02 | 257.77 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 257.97 | 262.18 | 257.97 | SL hit qty=1.00 sl=257.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 257.97 | 262.18 | 257.97 | SL hit qty=1.00 sl=257.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 257.97 | 262.18 | 257.97 | SL hit qty=1.00 sl=257.97 alert=retest1 |
| CROSSOVER_SKIP | 2026-01-27 15:15:00 | 235.55 | 254.62 | 254.62 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-07 09:15:00 | 256.60 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2024-03-11 12:15:00 | 258.42 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest1 | 2024-03-14 15:15:00 | 258.98 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest1 | 2024-03-15 13:15:00 | 257.25 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2024-03-21 10:15:00 | 251.80 | 2024-03-22 09:15:00 | 244.82 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-07-26 10:15:00 | 259.33 | 2024-08-02 14:15:00 | 250.77 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-08-16 12:15:00 | 256.55 | 2024-11-26 15:15:00 | 295.03 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-08-16 12:15:00 | 256.55 | 2025-03-19 09:15:00 | 256.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-05-12 13:15:00 | 257.70 | 2025-05-13 14:15:00 | 250.77 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-05-15 14:15:00 | 256.39 | 2025-05-20 13:15:00 | 250.77 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-11 13:15:00 | 259.60 | 2025-06-13 09:15:00 | 255.97 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-16 14:15:00 | 262.75 | 2025-07-17 15:15:00 | 258.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-18 10:15:00 | 267.75 | 2025-07-23 09:15:00 | 258.50 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest1 | 2026-01-12 13:15:00 | 263.65 | 2026-01-19 09:15:00 | 257.97 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest1 | 2026-01-13 14:15:00 | 264.25 | 2026-01-19 09:15:00 | 257.97 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2026-01-16 10:15:00 | 267.75 | 2026-01-19 09:15:00 | 257.97 | STOP_HIT | 1.00 | -3.65% |
