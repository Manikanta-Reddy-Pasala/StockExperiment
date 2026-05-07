# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 6685.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 28 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 3 |
| ENTRY2 | 16 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 16
- **Target hits / Stop hits / Partials:** 0 / 19 / 3
- **Avg / median % per leg:** 2.36% / -0.92%
- **Sum % (uncompounded):** 51.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 0 | 19 | 3 | 2.36% | 51.8% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 12.52% | 75.1% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.45% | -23.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 12.52% | 75.1% |
| retest2 (combined) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.45% | -23.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 6150.00 | 5820.92 | 5819.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 6201.00 | 5712.65 | 5731.50 | Break + close above crossover candle high |

### Cycle 2 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 6067.00 | 5751.83 | 5750.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 6116.00 | 5766.43 | 5758.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 5897.50 | 5905.42 | 5839.61 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 5948.50 | 5905.85 | 5840.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 12:15:00 | 5950.00 | 5906.29 | 5840.71 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 10:15:00 | 5990.50 | 5910.51 | 5844.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 11:15:00 | 5996.50 | 5911.37 | 5845.23 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 15:15:00 | 5950.00 | 5912.90 | 5847.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:15:00 | 6136.00 | 5915.12 | 5848.75 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-12 09:15:00 | 6842.50 | 6439.41 | 6231.66 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 6494.50 | 6498.58 | 6302.62 | SL hit (close<ema200) qty=0.50 sl=6498.58 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-03 09:15:00 | 6895.97 | 6586.65 | 6405.50 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 09:15:00 | 7056.40 | 6645.20 | 6454.35 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 6700.50 | 6706.66 | 6512.40 | SL hit (close<ema200) qty=0.50 sl=6706.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 6700.50 | 6706.66 | 6512.40 | SL hit (close<ema200) qty=0.50 sl=6706.66 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6581.50 | 6694.16 | 6568.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-29 11:15:00 | 6607.00 | 6690.14 | 6568.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 6661.50 | 6689.85 | 6569.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 6468.50 | 6680.04 | 6574.37 | SL hit (close<static) qty=1.00 sl=6555.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 11:15:00 | 6595.00 | 6058.66 | 6144.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 6592.50 | 6063.97 | 6146.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 6515.00 | 6083.05 | 6154.68 | SL hit (close<static) qty=1.00 sl=6555.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 6607.50 | 6145.48 | 6182.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-16 10:15:00 | 6569.50 | 6149.70 | 6184.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 6606.50 | 6154.25 | 6186.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 6602.00 | 6158.70 | 6188.24 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-17 10:15:00 | 6604.00 | 6180.06 | 6198.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 6634.00 | 6184.58 | 6200.50 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 6618.00 | 6228.92 | 6222.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.72 | 6400.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 6405.50 | 6490.96 | 6407.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 6405.50 | 6490.96 | 6407.18 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-25 13:15:00 | 6438.50 | 6473.18 | 6404.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-25 14:15:00 | 6410.00 | 6472.55 | 6404.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 6474.00 | 6471.95 | 6405.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 6480.50 | 6472.04 | 6405.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 6374.50 | 6470.29 | 6413.11 | SL hit (close<static) qty=1.00 sl=6400.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-02 10:15:00 | 6425.00 | 6469.84 | 6413.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 11:15:00 | 6409.50 | 6469.24 | 6413.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 6425.00 | 6464.65 | 6412.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-03 12:15:00 | 6415.50 | 6464.16 | 6412.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 13:15:00 | 6431.50 | 6463.84 | 6412.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 6460.50 | 6463.80 | 6413.09 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-05 10:15:00 | 6432.50 | 6462.88 | 6415.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 6440.00 | 6462.65 | 6415.23 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 6384.00 | 6461.50 | 6416.05 | SL hit (close<static) qty=1.00 sl=6400.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 6384.00 | 6461.50 | 6416.05 | SL hit (close<static) qty=1.00 sl=6400.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 6483.00 | 6437.97 | 6408.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 6452.50 | 6438.11 | 6408.53 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 6427.00 | 6437.85 | 6408.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 6456.50 | 6437.88 | 6409.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 10:15:00 | 6429.00 | 6437.79 | 6409.10 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 6389.00 | 6436.82 | 6409.45 | SL hit (close<static) qty=1.00 sl=6400.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 6565.50 | 6416.35 | 6401.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 6515.50 | 6417.33 | 6402.55 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 6390.50 | 6438.59 | 6416.46 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-30 15:15:00 | 6480.50 | 6431.05 | 6413.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-31 09:15:00 | 6388.00 | 6430.62 | 6413.84 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 6506.00 | 6418.23 | 6409.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 6543.50 | 6419.48 | 6410.01 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 6399.00 | 6465.48 | 6435.88 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-12 13:15:00 | 6445.00 | 6463.84 | 6435.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:15:00 | 6490.50 | 6464.11 | 6435.91 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 6391.50 | 6463.06 | 6435.94 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-13 14:15:00 | 6439.50 | 6461.65 | 6435.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-13 15:15:00 | 6421.50 | 6461.25 | 6435.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-26 09:15:00 | 6551.50 | 6232.27 | 6267.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-26 10:15:00 | 6425.00 | 6234.19 | 6268.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-26 14:15:00 | 6473.00 | 6241.95 | 6271.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 6455.00 | 6244.07 | 6272.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 6395.50 | 6256.34 | 6278.14 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6323.50 | 6257.01 | 6278.37 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-02 11:15:00 | 6380.00 | 6259.07 | 6279.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-02 12:15:00 | 6342.00 | 6259.89 | 6279.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-02 13:15:00 | 6384.50 | 6261.13 | 6280.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:15:00 | 6402.00 | 6262.54 | 6280.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 6259.00 | 6263.86 | 6281.12 | SL hit (close<static) qty=1.00 sl=6263.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-06 09:15:00 | 6418.50 | 6271.58 | 6283.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 10:15:00 | 6395.50 | 6272.82 | 6284.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 6244.00 | 6277.02 | 6286.27 | SL hit (close<static) qty=1.00 sl=6263.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-10 09:15:00 | 6432.50 | 6280.30 | 6287.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:15:00 | 6446.50 | 6281.96 | 6288.40 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 15:15:00 | 6379.00 | 6287.61 | 6291.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 6380.00 | 6288.53 | 6291.55 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 6450.00 | 6134.09 | 6168.32 | Break + close above crossover candle high |

### Cycle 5 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 6444.50 | 6198.32 | 6198.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 6468.00 | 6205.80 | 6202.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-09 12:15:00 | 5950.00 | 2025-06-12 09:15:00 | 6842.50 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-05-09 12:15:00 | 5950.00 | 2025-06-19 14:15:00 | 6494.50 | STOP_HIT | 0.50 | 9.15% |
| BUY | retest1 | 2025-05-12 11:15:00 | 5996.50 | 2025-07-03 09:15:00 | 6895.97 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-05-13 09:15:00 | 6136.00 | 2025-07-08 09:15:00 | 7056.40 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-05-12 11:15:00 | 5996.50 | 2025-07-14 09:15:00 | 6700.50 | STOP_HIT | 0.50 | 11.74% |
| BUY | retest1 | 2025-05-13 09:15:00 | 6136.00 | 2025-07-14 09:15:00 | 6700.50 | STOP_HIT | 0.50 | 9.20% |
| BUY | retest2 | 2025-07-29 12:15:00 | 6661.50 | 2025-08-01 09:15:00 | 6468.50 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-10-13 12:15:00 | 6592.50 | 2025-10-14 09:15:00 | 6515.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-16 12:15:00 | 6602.00 | 2025-10-20 12:15:00 | 6593.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-10-17 11:15:00 | 6634.00 | 2025-10-20 12:15:00 | 6593.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-11-26 10:15:00 | 6480.50 | 2025-12-02 09:15:00 | 6374.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-12-03 14:15:00 | 6460.50 | 2025-12-08 10:15:00 | 6384.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-12-05 11:15:00 | 6440.00 | 2025-12-08 10:15:00 | 6384.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-11 12:15:00 | 6452.50 | 2025-12-15 09:15:00 | 6389.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-19 10:15:00 | 6515.50 | 2025-12-29 09:15:00 | 6390.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-01-06 10:15:00 | 6543.50 | 2026-01-12 09:15:00 | 6399.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-01-12 14:15:00 | 6490.50 | 2026-01-13 11:15:00 | 6391.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-26 15:15:00 | 6455.00 | 2026-02-27 15:15:00 | 6395.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-03-02 14:15:00 | 6402.00 | 2026-03-04 09:15:00 | 6259.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-06 10:15:00 | 6395.50 | 2026-03-09 09:15:00 | 6244.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-03-10 10:15:00 | 6446.50 | 2026-03-11 15:15:00 | 6330.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-03-11 09:15:00 | 6380.00 | 2026-03-11 15:15:00 | 6330.00 | STOP_HIT | 1.00 | -0.78% |
