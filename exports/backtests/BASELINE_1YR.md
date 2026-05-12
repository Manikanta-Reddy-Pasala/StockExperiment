# Baseline 1-Year Backtest Results

**Window:** 2025-05-12 → 2026-05-12 (12 months)
**Capital:** ₹2,00,000 (locked, no add/withdraw)
**Max concurrent:** 2 positions
**Universe:** Nifty 50 + Nifty 500
**Penny filter:** ON (close < ₹50 excluded) and OFF
**Run date:** 2026-05-12 (completed ~16:00 IST)

## Results matrix

| Model | N50 filter | N500 filter | N50 nofilter | N500 nofilter |
|-------|-----------:|------------:|-------------:|--------------:|
| ema_200_400    | **+7.30%** |  -33.53%    | KILLED (-9)  | KILLED (-9)   |
| ema_9_21       |    -7.52%  |  -28.31%    |  -7.52%      |    +2.84%     |
| swing_pullback |    -1.09%  |   +0.13%    |  -1.09%      |    +0.13%     |
| orb_15min      |     0.00%* |    0.00%*   |   0.00%*     |    0.00%*     |

\* ORB took **zero trades** — 5-min bar cache empty for the window. Strategy never fires.

## Detail per row

### Penny filter ON (`y1_filter`)

| Model | Universe | Taken | Skip | Final₹ | ROI% | MaxDD% | Open@End |
|-------|----------|------:|-----:|-------:|-----:|-------:|---------:|
| ema_200_400 | N50  |  54 |   337 | 214,595 | **+7.30** | 12.77 | 2 |
| ema_200_400 | N500 |  90 |  3233 | 132,933 | -33.53    | 34.94 | 2 |
| ema_9_21    | N50  | 213 |  2720 | 184,953 | -7.52     | 26.74 | 2 |
| ema_9_21    | N500 | 208 | 25890 | 143,389 | -28.31    | 41.33 | 2 |
| swing_pullback | N50  | 3 |  0 | 197,810 | -1.09 | 2.02 | 0 |
| swing_pullback | N500 | 18 | 8 | 200,266 | +0.13 | 9.83 | 2 |
| orb_15min   | N50  | 0 | 0 | 200,000 | 0.00 | 0.00 | 0 |
| orb_15min   | N500 | 0 | 0 | 200,000 | 0.00 | 0.00 | 0 |

### Penny filter OFF (`y1_nofilter`)

| Model | Universe | Taken | Final₹ | ROI% | MaxDD% |
|-------|----------|------:|-------:|-----:|-------:|
| ema_200_400 | N50  | KILLED-9 (OOM) | - | - | - |
| ema_200_400 | N500 | KILLED-9 (OOM) | - | - | - |
| ema_9_21    | N50  | 213 | 184,953 | -7.52  | 26.74 |
| ema_9_21    | N500 | 216 | 205,689 | +2.84  | 31.91 |
| swing_pullback | N50  | 3  | 197,810 | -1.09 | 2.02 |
| swing_pullback | N500 | 18 | 200,266 | +0.13 | 9.83 |
| orb_15min   | both | 0 | 200,000 | 0.00 | 0.00 |

## Key findings

1. **Target gap is 10x.** Best result = ema_200_400 N50 filter at +7.30%/yr. User's stated target was 5-10%/mo (= 60-120%/yr). Current setup is 8-16x short of the floor target.

2. **ORB never fired.** Zero trades across all 4 ORB runs. Root cause = 5-min bar cache empty for the window (only 1H, 15m, and daily are cached). Either prefetch 5m data into `historical_data_5m` or skip ORB.

3. **EMA200/400 nofilter OOM killed.** Penny filter ON survives, OFF gets killed. Likely many low-priced stocks trip the strategy state machine memory growth. Need investigation if we want nofilter coverage.

4. **EMA9/21 generates 200+ trades/yr** but loses money on both universes — overtrades + slippage drag.

5. **Swing pullback is too conservative** — only 3-18 trades/yr. Stage 2 + RSI + breakout + ADV filters compound to near-zero opportunity.

## Recommended directions

**Path A — Realistic target (recommended):**
- Drop the 60-120%/yr headline.
- Best honest pitch: 7-15%/yr with EMA200/400 N50 filter + risk overlay + fine-tuning.
- Match SIP/index baseline (~12-14% Nifty 50 long-term) + some alpha edge.

**Path B — Fine-tune sweep (parallel work):**
- HTF filter (50-DMA / 200-DMA trend gate)
- Tighter stops (1.5 ATR vs current 2.0 ATR)
- Position sizing tied to volatility
- min_crossover_gap sweep on EMA9/21 (currently 0)
- Expected upside: maybe +5-10% on best case, not 10x.

**Path C — Higher-frequency / leverage:**
- F&O / index options strategies
- Breaks ₹2L lock constraint (margin requirements)
- Order-of-magnitude higher risk envelope
- Not recommended without explicit capital re-scope.

## Next steps (suggested)

1. Fix ORB by prefetching 5m bars (`prefetch_ohlcv.py --intervals 5m`).
2. Investigate EMA200/400 nofilter OOM (memory profile per-stock loop).
3. Run fine-tune sweep on EMA200/400 N50 filter (the only positive baseline).
4. Live paper-trade EMA200/400 N50 filter for 1-2 months to verify before any real money.

## Raw data location

- Local: `exports/backtests/y1_baseline/y1_filter/` + `y1_nofilter/`
- Prod: `trading_system_app:/app/exports/backtests/y1_filter/` + `y1_nofilter/`
- 112 MB local, per-stock detail in `*.md` per-symbol files.
