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
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 11 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 0 / 7 / 1
- **Avg / median % per leg:** 0.62% / -1.58%
- **Sum % (uncompounded):** 4.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.62% | 4.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.62% | 4.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.62% | 4.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 346.65 | 362.51 | 362.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 345.75 | 361.76 | 362.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 353.00 | 352.52 | 356.36 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 354.80 | 352.54 | 356.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 354.80 | 352.54 | 356.31 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-28 11:15:00 | 353.13 | 353.41 | 356.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 13:15:00 | 351.47 | 353.37 | 356.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-06 09:15:00 | 353.10 | 352.92 | 355.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-06 10:15:00 | 355.00 | 352.94 | 355.55 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 357.40 | 353.11 | 355.55 | SL hit (close>static) qty=1.00 sl=357.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-12 14:15:00 | 353.31 | 354.22 | 355.85 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-12 15:15:00 | 354.60 | 354.22 | 355.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-13 09:15:00 | 352.61 | 354.21 | 355.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-13 11:15:00 | 354.44 | 354.17 | 355.80 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-12-19 09:15:00 | 350.55 | 354.91 | 356.00 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 352.11 | 354.84 | 355.96 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-27 11:15:00 | 352.00 | 353.59 | 355.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 351.47 | 353.54 | 355.06 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-30 12:15:00 | 351.76 | 353.47 | 354.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 14:15:00 | 348.79 | 353.37 | 354.91 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 356.93 | 353.35 | 354.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 357.01 | 353.39 | 354.89 | SL hit (close>static) qty=1.00 sl=357.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 357.01 | 353.39 | 354.89 | SL hit (close>static) qty=1.00 sl=357.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 357.01 | 353.39 | 354.89 | SL hit (close>static) qty=1.00 sl=357.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-10 13:15:00 | 352.76 | 355.62 | 355.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 351.90 | 355.54 | 355.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-14 14:15:00 | 349.72 | 354.63 | 355.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-15 09:15:00 | 356.29 | 354.62 | 355.29 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 357.50 | 354.65 | 355.30 | SL hit (close>static) qty=1.00 sl=357.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-17 10:15:00 | 351.67 | 355.19 | 355.54 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 12:15:00 | 351.76 | 355.12 | 355.50 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 384.04 | 355.30 | 355.58 | SL hit (close>static) qty=1.00 sl=357.15 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.78 | 424.37 | 424.40 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.22 | 425.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 406.50 | 423.79 | 424.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-27 14:15:00 | 414.65 | 422.89 | 423.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 412.10 | 422.72 | 423.12 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 4020m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-02 09:15:00 | 350.29 | 384.42 | 398.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 14:15:00 | 380.25 | 379.38 | 394.24 | SL hit (close>ema200) qty=0.50 sl=379.38 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-28 13:15:00 | 351.47 | 2024-12-09 09:15:00 | 357.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-12-19 11:15:00 | 352.11 | 2024-12-31 11:15:00 | 357.01 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-12-27 13:15:00 | 351.47 | 2024-12-31 11:15:00 | 357.01 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-12-30 14:15:00 | 348.79 | 2024-12-31 11:15:00 | 357.01 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-01-10 15:15:00 | 351.90 | 2025-01-15 10:15:00 | 357.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-01-17 12:15:00 | 351.76 | 2025-01-20 09:15:00 | 384.04 | STOP_HIT | 1.00 | -9.18% |
| SELL | retest2 | 2026-03-02 09:15:00 | 412.10 | 2026-04-02 09:15:00 | 350.29 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 412.10 | 2026-04-08 14:15:00 | 380.25 | STOP_HIT | 0.50 | 7.73% |
