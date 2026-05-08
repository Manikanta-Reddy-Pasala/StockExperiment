# TITAN (TITAN)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 4517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 8 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 5 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 3 / 4 / 2
- **Avg / median % per leg:** 3.10% / 4.39%
- **Sum % (uncompounded):** 27.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 3 | 4 | 2 | 3.10% | 27.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 5 | 1 | 20.0% | 1 | 4 | 0 | -0.42% | -2.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 5 | 1 | 20.0% | 1 | 4 | 0 | -0.42% | -2.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3698.00 | 3522.99 | 3522.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 3741.10 | 3532.05 | 3526.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.29 | 3727.78 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-09 11:15:00 | 3841.70 | 3811.74 | 3737.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 12:15:00 | 3867.60 | 3812.29 | 3738.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-11 13:15:00 | 3839.40 | 3815.84 | 3745.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 14:15:00 | 3845.10 | 3816.13 | 3745.76 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 09:15:00 | 4037.36 | 3887.16 | 3811.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:15:00 | 4060.98 | 3888.87 | 3813.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-07 09:15:00 | 4254.36 | 3942.86 | 3853.92 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-07 09:15:00 | 4229.61 | 3942.86 | 3853.92 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.00 | 4056.22 | 3959.04 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-01 12:15:00 | 4044.90 | 4031.75 | 3957.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 4087.30 | 4032.30 | 3958.52 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 3944.00 | 4031.09 | 3958.64 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 4069.00 | 4024.99 | 3958.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 4085.10 | 4025.58 | 3958.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 3946.40 | 4154.57 | 4112.31 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 4013.00 | 4121.34 | 4097.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 4059.90 | 4120.73 | 4097.75 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 4024.50 | 4116.17 | 4096.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 4024.80 | 4115.26 | 4095.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 3936.70 | 4107.20 | 4092.36 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 3936.70 | 4107.20 | 4092.36 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 4072.00 | 4098.60 | 4088.64 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-02 14:15:00 | 4100.40 | 4092.87 | 4086.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-02 15:15:00 | 4080.00 | 4092.74 | 4086.07 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-06 09:15:00 | 4173.10 | 4093.54 | 4086.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 4175.00 | 4094.35 | 4086.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-05-08 14:15:00 | 4592.50 | 4342.53 | 4265.09 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-09 12:15:00 | 3867.60 | 2025-12-31 09:15:00 | 4037.36 | PARTIAL | 0.50 | 4.39% |
| BUY | retest1 | 2025-12-11 14:15:00 | 3845.10 | 2025-12-31 10:15:00 | 4060.98 | PARTIAL | 0.50 | 5.61% |
| BUY | retest1 | 2025-12-09 12:15:00 | 3867.60 | 2026-01-07 09:15:00 | 4254.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-12-11 14:15:00 | 3845.10 | 2026-01-07 09:15:00 | 4229.61 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 13:15:00 | 4087.30 | 2026-02-01 15:15:00 | 3944.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2026-02-03 10:15:00 | 4085.10 | 2026-03-23 09:15:00 | 3946.40 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-25 10:15:00 | 4059.90 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-03-27 11:15:00 | 4024.80 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-04-06 10:15:00 | 4175.00 | 2026-05-08 14:15:00 | 4592.50 | TARGET_HIT | 1.00 | 10.00% |
