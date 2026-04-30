# ABB India Ltd. (ABB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 7259.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 3 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 1961.22
- **Avg P&L per closed trade:** 245.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 8129.10 | 7906.04 | 7905.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 8365.95 | 7910.62 | 7907.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 8213.45 | 8233.33 | 8095.91 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 7341.50 | 7987.49 | 7990.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 7320.90 | 7980.86 | 7987.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 7347.70 | 7321.60 | 7574.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 12:15:00 | 7105.10 | 7518.64 | 7579.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 10:15:00 | 5616.50 | 5316.92 | 5567.74 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 6040.00 | 5617.45 | 5615.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 6047.00 | 5702.36 | 5661.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5885.00 | 5922.32 | 5816.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 5962.50 | 5921.29 | 5818.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 5897.50 | 5956.52 | 5864.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-02 14:15:00 | 5907.50 | 5956.03 | 5865.06 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 5875.00 | 5951.70 | 5865.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-04 09:15:00 | 5887.00 | 5950.21 | 5866.10 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-04 11:15:00 | 5859.50 | 5948.57 | 5866.11 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 5687.50 | 5819.20 | 5819.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5663.00 | 5813.26 | 5816.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 5233.30 | 5204.92 | 5363.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 5149.00 | 5269.09 | 5352.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5316.50 | 5244.11 | 5322.45 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-07 14:15:00 | 5212.50 | 5244.53 | 5320.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 5227.50 | 5212.45 | 5273.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-28 10:15:00 | 5222.00 | 5212.55 | 5272.77 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 5269.50 | 5211.49 | 5270.14 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 11:15:00 | 5294.00 | 5212.31 | 5270.26 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 5662.00 | 5141.64 | 5139.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 5830.00 | 5148.49 | 5143.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 12:15:00 | 5997.00 | 6017.35 | 5770.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-24 09:15:00 | 6114.50 | 6019.06 | 5776.52 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-20 12:15:00 | 7105.10 | 2025-01-30 11:15:00 | 5680.82 | TARGET | 1424.28 |
| BUY | 2025-06-20 10:15:00 | 5962.50 | 2025-07-04 11:15:00 | 5859.50 | EXIT_EMA400 | -103.00 |
| BUY | 2025-07-02 14:15:00 | 5907.50 | 2025-07-04 11:15:00 | 5859.50 | EXIT_EMA400 | -48.00 |
| BUY | 2025-07-04 09:15:00 | 5887.00 | 2025-07-04 11:15:00 | 5859.50 | EXIT_EMA400 | -27.50 |
| SELL | 2025-09-26 09:15:00 | 5149.00 | 2025-10-29 11:15:00 | 5294.00 | EXIT_EMA400 | -145.00 |
| SELL | 2025-10-07 14:15:00 | 5212.50 | 2025-10-29 11:15:00 | 5294.00 | EXIT_EMA400 | -81.50 |
| SELL | 2025-10-28 10:15:00 | 5222.00 | 2025-10-29 11:15:00 | 5294.00 | EXIT_EMA400 | -72.00 |
| BUY | 2026-03-24 09:15:00 | 6114.50 | 2026-04-20 09:15:00 | 7128.44 | TARGET | 1013.94 |
