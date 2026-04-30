# InterGlobe Aviation Ltd. (INDIGO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4300.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 1600.69
- **Avg P&L per closed trade:** 177.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 4024.05 | 4597.03 | 4598.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 3928.05 | 4493.20 | 4543.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4230.15 | 4200.82 | 4344.51 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 4668.00 | 4388.94 | 4388.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 13:15:00 | 4701.20 | 4392.05 | 4389.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 4453.00 | 4458.59 | 4427.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 11:15:00 | 4478.60 | 4458.79 | 4427.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 4410.10 | 4458.77 | 4428.13 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 4060.00 | 4401.89 | 4402.36 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 15:15:00 | 4492.50 | 4329.15 | 4328.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 09:15:00 | 4559.00 | 4331.44 | 4329.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 09:15:00 | 4319.15 | 4364.87 | 4347.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-04 09:15:00 | 4544.90 | 4369.60 | 4350.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 4854.15 | 4826.55 | 4656.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-07 13:15:00 | 4947.65 | 4829.82 | 4661.77 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 5030.00 | 5206.69 | 4993.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 15:15:00 | 5120.00 | 5198.75 | 4995.47 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5275.00 | 5423.03 | 5266.23 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 10:15:00 | 5305.00 | 5421.86 | 5266.42 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-13 11:15:00 | 5243.00 | 5420.08 | 5266.30 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 5597.00 | 5728.73 | 5729.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 15:15:00 | 5591.00 | 5726.06 | 5727.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 5714.00 | 5700.91 | 5713.63 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 5891.00 | 5724.45 | 5724.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 5905.00 | 5726.24 | 5725.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 5740.50 | 5764.65 | 5746.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-27 09:15:00 | 5810.00 | 5763.84 | 5746.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 5750.00 | 5769.42 | 5749.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-28 14:15:00 | 5816.50 | 5769.89 | 5750.30 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-30 12:15:00 | 5734.00 | 5772.69 | 5752.90 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 5604.50 | 5736.66 | 5736.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 5564.50 | 5734.95 | 5735.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 5734.50 | 5725.87 | 5731.10 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 5925.00 | 5736.40 | 5736.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 5932.00 | 5738.34 | 5737.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 12:15:00 | 5752.50 | 5761.58 | 5749.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-18 13:15:00 | 5778.00 | 5761.75 | 5749.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 14:15:00 | 5740.50 | 5761.53 | 5749.72 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 10:15:00 | 5298.50 | 5752.16 | 5752.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 5158.00 | 5727.21 | 5739.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4914.80 | 4877.93 | 5088.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-19 14:15:00 | 4812.20 | 4918.93 | 5037.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 4640.00 | 4349.65 | 4573.72 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-01-03 11:15:00 | 4478.60 | 2025-01-06 09:15:00 | 4410.10 | EXIT_EMA400 | -68.50 |
| BUY | 2025-03-04 09:15:00 | 4544.90 | 2025-03-20 09:15:00 | 5126.96 | TARGET | 582.06 |
| BUY | 2025-05-09 15:15:00 | 5120.00 | 2025-05-12 09:15:00 | 5493.60 | TARGET | 373.60 |
| BUY | 2025-04-07 13:15:00 | 4947.65 | 2025-06-13 11:15:00 | 5243.00 | EXIT_EMA400 | 295.35 |
| BUY | 2025-06-13 10:15:00 | 5305.00 | 2025-06-13 11:15:00 | 5243.00 | EXIT_EMA400 | -62.00 |
| BUY | 2025-10-27 09:15:00 | 5810.00 | 2025-10-30 12:15:00 | 5734.00 | EXIT_EMA400 | -76.00 |
| BUY | 2025-10-28 14:15:00 | 5816.50 | 2025-10-30 12:15:00 | 5734.00 | EXIT_EMA400 | -82.50 |
| BUY | 2025-11-18 13:15:00 | 5778.00 | 2025-11-18 14:15:00 | 5740.50 | EXIT_EMA400 | -37.50 |
| SELL | 2026-02-19 14:15:00 | 4812.20 | 2026-03-09 09:15:00 | 4136.02 | TARGET | 676.18 |
