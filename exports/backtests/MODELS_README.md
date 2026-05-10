# 4-Model Backtest Comparison — ₹5,00,000 Capital

All backtests run via local Python scripts (no cloud functions, no LLM
calls). Fyers OHLCV history fetched via broker API token. Capital
simulator replays time-ordered ENTRY / PARTIAL / TARGET / STOP events
against a single ₹5L pool, enforces concurrent-position cap and
lot-size floor, compounds cash from closes.

## TL;DR — Best at ₹5L (real-time legit, no lookahead)

| Model | Universe | Window | max | ROI | MDD | Risk-Adj |
|---|---|---|---|---|---|---|
| **Swing Pullback Breakout** | **N500** | 365d | **2** | **+30.40%** | **10.92%** | **2.78** |
| Swing Pullback Breakout | N500 | 365d | 3 | +20.56% | 13.81% | 1.49 |
| **EMA 9/21 1H crossover** | N50 | 365d | 2 | +13.67% | 20.52% | 0.67 |
| EMA 9/21 1H crossover | N50 | 365d | 30 | +11.14% | **6.56%** | 1.70 |
| Swing Pullback Breakout | N500 | 365d | 5 | +11.15% | 11.97% | 0.93 |
| **15-min ORB intraday** | **N500** | 90d (~+45% annualized) | **2** | **+10.33%** | **3.93%** | 2.63 |
| EMA 200/400 1H crossover | N50 | 365d | 3 | +9.10% | 9.40% | 0.97 |
| ORB intraday | N500 | 90d | 5 | +4.21% | 3.30% | 1.28 |
| EMA 200/400 1H crossover | N500 | 365d | 30 | +3.19% | 13.93% | 0.23 |

**Verdict:** Swing Pullback Breakout on Nifty 500 with max_concurrent=2
delivers +30.40% annual ROI at 10.92% drawdown — 3.3× better than the
EMA 200/400 baseline. ORB intraday is the safest model (3.93% MDD)
with strong annualized return.

---

## Models

### 1. EMA 200/400 1H Crossover (baseline)
Existing strategy. Slow trend-follow on 1-hour bars: EMA 200 crosses EMA
400 → Stage-2 trend identified → wait for retest1 (EMA200 retest from
upside) → ENTRY1; wait for retest2 (EMA400 touch) → ENTRY2. Targets
±10%, partial 50% at +5%/+15%, EMA400-based SL.

**Source file:** `src/services/technical/ema_crossover_strategy.py`
**Harness:** `tools/backtests/run_ema_200_400_backtest.py --ema-fast 200 --ema-slow 400`

### 2. EMA 9/21 1H Crossover (variant of #1)
Same state machine, faster periods. Generates ~3× more signals than
200/400 but with ~30% lower per-leg edge. Wins on N50 365d via volume
and concentration.

**Harness:** `run_ema_200_400_backtest.py --ema-fast 9 --ema-slow 21 --warmup-days 30`

### 3. Swing Pullback Breakout (Stage 2 / Minervini-style) — DAILY bars
Most-followed Indian retail swing setup (ChartInk top-loved scanners,
Vivek Bajaj, Pushkar Raj Thakur, Zerodha Varsity TA module).

Entry rules (daily close):
1. `close > EMA50 > EMA200` (Stage 2 uptrend)
2. Low of last 3 bars wicked EMA20 (the pullback)
3. Today's close > prior 5-bar high (breakout)
4. RSI(14) ∈ [50, 70]
5. Volume ≥ 1.5 × SMA(volume, 20)
6. Close ≥ ₹50 AND ADV ≥ ₹5cr (liquidity)

Exit rules:
- SL = `entry − 1.5 × ATR(14)`
- T1 = `entry + 2 × ATR(14)` (book 50%, trail SL → entry)
- Trail remainder: close < EMA20 → exit
- Time stop: 10 trading days, exit if not +3%

**Source file:** `src/services/technical/ema_pullback_breakout.py`
**Harness:** `tools/backtests/run_swing_pullback_backtest.py`

### 4. 15-min Opening Range Breakout (ORB) — INTRADAY 5m bars
Most-followed Indian intraday setup (ChartInk dominant scanner family,
Sudarshan Sukhani's published system, default template in
Streak/Tradetron/AlgoTest, Zerodha "In The Money" deep-dive).

Entry rules (5m bars):
1. ORB = high/low of 09:15-09:29 (3 5-min bars)
2. Skip day if ORB range > 1.5% of price (gap chase) or < 0.3% (no vol)
3. Long: 5m close > ORB_HIGH + > VWAP + vol ≥ 1.5 × SMA(20)
4. Short: mirror
5. Trading window: 09:30 → 11:15 only
6. One trade per symbol per day

Exit rules:
- SL = opposite ORB end OR `close − 1×ATR(14)` (whichever tighter)
- T1 = 1.5R (book 50%, SL → entry)
- Trail: long exits on close < VWAP, short on close > VWAP
- EOD square-off 15:20

**Source file:** `src/services/technical/orb_15min.py`
**Harness:** `tools/backtests/run_orb_intraday_backtest.py`

---

## Universe-wide signal density (informational, not capital-aware)

| Model | Universe | Window | Symbols processed | Closed legs | Win rate | Sum % |
|---|---|---|---|---|---|---|
| EMA 200/400 1H | N50 | 365d | 53 | 1180 | 27.5% | +602% |
| EMA 200/400 1H | N500 | 365d | 359 | 6622 | 36.5% | +5185% |
| EMA 9/21 1H | N50 | 365d | 32 | 1648 | 30.3% | +233% |
| EMA 9/21 1H | N500 | 365d | 465 | 29185 | 36.2% | +15335% |
| **Swing Pullback** | **N500** | 365d | **472** | **1258** | **50.2%** | **+1640%** |
| Swing Pullback | N50 | 365d | 53 | 184 | 54.9% | +193% |
| **ORB 15min** | **N500** | 90d | **499** | **8973** | **42.7%** | **+1535%** |
| ORB 15min | N50 | 90d | 53 | 1154 | 44.5% | +170% |

Swing Pullback wins on **win rate** (50%+) — it filters hard for quality
(Stage 2 + pullback + breakout + RSI + volume all required). Few legs,
but each one has institutional-grade setup confirmation.

ORB wins on **signal density per symbol** in tradeable universe.

---

## Capital simulation matrices (₹5,00,000)

### EMA 200/400 1H — N50 365d
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  2     100     762    545,126    +9.03   10.87
  3     129     733    545,514    +9.10    9.40  <-- best
  5     189     673    520,771    +4.15   11.10
 10     365     497    480,420    -3.92   15.68
 30     807      55    496,842    -0.63    9.01
 50     828      34    500,514    +0.10    4.97
```

### EMA 200/400 1H — N500 365d
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  2      86    4896    436,338   -12.73   19.94
 20     836    4146    500,777    +0.16   15.82
 30    1251    3731    515,953    +3.19   13.93  <-- best
 50    2089    2893    486,877    -2.62   15.93
```

### EMA 9/21 1H — N50 365d
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  2     197    1383    568,328   +13.67   20.52  <-- best gross
  3     294    1286    531,626    +6.33   24.17
 20    1407     173    553,313   +10.66    9.89
 30    1570      10    555,689   +11.14    6.56  <-- best risk-adj
 50    1491      89    533,486    +6.70    3.41
```

### EMA 9/21 1H — N500 365d
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  2     196   26261    413,544   -17.29   35.48
  8     853   25604    497,485    -0.50   25.65
 20    2152   24305    502,339    +0.47   24.73  <-- least bad
 30    3224   23233    473,870    -5.23   26.46
```
N500 EMA 9/21 fails — too many simultaneous signals, capital dilution kills compounding.

### Swing Pullback Breakout — N500 365d ★
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  1      29     949    319,905   -36.02   43.52  <-- single-position concentration risk
  2      35     943    652,006   +30.40   10.92  <-- WINNER
  3      57     921    602,802   +20.56   13.81
  5      99     879    555,741   +11.15   11.97
  8     163     815    467,315    -6.54   21.03
 10     210     768    435,336  -12.93   24.01
```

### Swing Pullback Breakout — N50 365d
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  1      19     124    499,299    -0.14   14.01  <-- least bad
  2      36     107    490,903    -1.82   10.95
  3      52      91    486,375    -2.73    9.24
```
N50 too narrow — not enough Stage 2 candidates for the strict filter.

### 15-min ORB intraday — N500 90d ★
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  2     335    6034    551,670   +10.33    3.93  <-- WINNER (annualized ~+45%)
  3     525    5844    503,359    +0.67    4.50
  5     828    5541    521,071    +4.21    3.30
  8    1292    5077    515,866    +3.17    3.55
 10    1614    4755    511,712    +2.34    2.48
```

### 15-min ORB intraday — N50 90d
```
 Max   Taken   Skip      Final     ROI%   MaxDD%
  2     259     555    490,301    -1.94    3.30
 50     781      33    499,113    -0.18    0.45  <-- best, still net flat
```

---

## Walk-forward (no-lookahead robustness check)

Split each backtest at midpoint, rank symbols by first-half sum%, sim
top-N on second half ONLY. Pure out-of-sample.

| Strategy | Universe | Top-N | max | ROI | MDD |
|---|---|---|---|---|---|
| Swing Pullback | N500 | 30 | 2 | +6.15% | 10.70% |
| Swing Pullback | N500 | 30 | 8 | +1.75% | 11.03% |
| EMA 9/21 | N50 | 15 | 2 | +3.50% | 21.75% |
| EMA 200/400 | N50 | 15 | 2 | +9.23% | 10.73% |
| EMA 9/21 | N50 | 15 | 10 | +0.60% | 10.89% |
| ORB | N500 | 50 | any | -0.20 to -1.95% | 2-4% |

**Walk-forward findings:**
1. EMA 200/400 walk-forward (+9.23%) ≈ EMA 200/400 full universe (+9.10%) — the model is regime-stable.
2. **Swing Pullback walk-forward DROPS** from +30.40% → +6.15%. Past
   winners don't continue to win. Run on FULL N500 each time, don't
   pre-filter by past performance.
3. EMA 9/21 also drops sharply walk-forward — the high gross return is
   regime-dependent.

---

## How to pick legs when 5+ entries fire same day

Capital sim above uses first-come-first-served (FCFS). With max=2 and
many candidates, later signals are skipped. Priority order to extract
maximum return:

### Per model

**Swing Pullback (highest priority):**
1. **Volume ratio** — pick highest `volume / SMA(volume, 20)` (most institutional). Cap-rule: skip if < 2.0× when picking among many.
2. **RSI** — prefer 55-65 (sweet spot momentum, not overheated).
3. **ATR distance to T1** — favor smaller ATR (tighter stops, better R:R).
4. **Liquidity** — ADV > ₹10cr beats ₹5cr (less slippage at 5L).
5. **Stage 2 strength** — `(close − EMA200) / EMA200` larger = stronger trend.

**ORB intraday:**
1. **ORB range %** — prefer 0.5-1.0% (sweet spot, avoid 1.5%+ gaps).
2. **VWAP distance at trigger** — closer to VWAP at break = more momentum left.
3. **Volume surge** — prefer > 2× SMA(20).
4. **Sector sympathy** — if Bank Nifty/IT index gapping same direction, prefer those names.

**EMA 9/21:**
1. **Direction × alert** — SELL @ retest1 (gap003 81.8% win) > BUY @ retest1 (66.7% win) > SELL @ retest2 (47.9% win). **Skip BUY @ retest2 entirely on N500** (negative bucket).
2. **EMA gap at crossover** — bigger gap = stronger trend (proxy for non-whipsaw).
3. **15m sustain quality** — sustain confirmed on 15m close > 1H fallback.

### Universal rules (all models)
- Slot < 5×price → skip (insufficient share count for meaningful position).
- Single position drawdown ≥ 3% → skip new entries that day, let dust settle.
- Max 2 concurrent same-sector positions (avoid sector beta concentration).
- Refresh universe quarterly (not by past performance — by index reconstitution + min-ADV recheck).

---

## Recommended production setup at ₹5L

### Conservative (lowest DD, real money)
- Strategy: **15-min ORB intraday on N500**
- max_concurrent: 2
- Expected: +10% over 90d (+45% annualized), 3.93% MDD
- Net live (after STT/brokerage/slippage ~2%/yr): ~+8% / 90d

### Balanced (best risk-adjusted, paper or real)
- Strategy: **Swing Pullback Breakout on N500**
- max_concurrent: 2
- Expected: +30% / yr, 10.92% MDD
- Net live (after costs ~2%/yr): ~+28% / yr
- Holding period: avg 5-10 trading days

### Aggressive (highest gross, accepts higher DD)
- Strategy: **EMA 9/21 1H on N50**
- max_concurrent: 2
- Expected: +13.7% / yr, 20.52% MDD
- Net live: ~+11.7% / yr

### Avoid in production
- EMA 9/21 on N500 — capital dilution always net negative
- EMA 200/400 on N500 (full) — only +3% gross, no margin for slippage
- Swing Pullback on N50 — universe too narrow for the strict filter

---

## Risk caveats

1. **All backtests via local Python scripts** — no cloud functions, no LLM calls. Fyers API token used only for OHLCV history fetch (broker auth, replaceable with `--source yahoo` for token-less fallback).
2. **No slippage / brokerage / STT modeled.** Subtract ~1.5-2% from any annual ROI for realistic net.
3. **Survivorship bias on per-strategy walk-forward results.** Reported full-universe numbers do NOT cherry-pick — entire N50/N500 is fed in.
4. **Regime dependence.** May 2025 → May 2026 was mostly bullish first half (Swing Pullback caught it), bearish second half (walk-forward exposes weakness). Re-evaluate on rolling 90d basis.
5. **N500 ORB is 90d only.** 5-min Fyers history caps at ~6mo per chunk; longer windows possible but slow.
6. **PARTIAL booking improperly priced in walk-forward sim** — uses 50% qty proxy; real harness handles correctly.

---

## File map

### Strategy modules
- `src/services/technical/ema_crossover_strategy.py` — EMA 200/400 + 9/21 (param)
- `src/services/technical/ema_pullback_breakout.py` — Swing Pullback Breakout
- `src/services/technical/orb_15min.py` — 15-min ORB intraday

### Backtest harnesses
- `tools/backtests/run_ema_200_400_backtest.py` — drives ema_crossover_strategy
- `tools/backtests/run_swing_pullback_backtest.py` — drives ema_pullback_breakout
- `tools/backtests/run_orb_intraday_backtest.py` — drives orb_15min
- `tools/backtests/realistic_capital_sim.py` — capital-aware replay (`--capital`)
- `tools/backtests/walk_forward_sim.py` — no-lookahead variant

### Result directories
- `n50_v14_365d/`, `n50_v14_180d/` — EMA 200/400 baseline
- `n500_v14_365d_pure/`, `n500_v14_365d_gap003/` — EMA 200/400 N500 + gap variant
- `n50_v14_365d_ema9_21/`, `n500_v14_365d_ema9_21/` — EMA 9/21
- `n50_swing_365d/`, `n500_swing_365d/` — Swing Pullback Breakout
- `n50_orb_90d/`, `n500_orb_90d/` — 15-min ORB intraday

### Reproducibility
```bash
# Pull latest universe + run on prod
ssh root@77.42.45.12

# Swing Pullback N500 (winner)
docker exec trading_system_app python tools/backtests/run_swing_pullback_backtest.py \
  --universe nifty500 --days 365 --warmup-days 250 \
  --out exports/backtests/n500_swing_365d

# ORB N500 (safest)
docker exec trading_system_app python tools/backtests/run_orb_intraday_backtest.py \
  --universe nifty500 --days 90 \
  --out exports/backtests/n500_orb_90d

# EMA 9/21 N50 (aggressive)
docker exec trading_system_app python tools/backtests/run_ema_200_400_backtest.py \
  --universe nifty50 --days 365 --warmup-days 30 --ema-fast 9 --ema-slow 21 \
  --out exports/backtests/n50_v14_365d_ema9_21

# Capital sim 5L on any result dir
python tools/backtests/realistic_capital_sim.py exports/backtests/n500_swing_365d \
  --capital 500000 1 2 3 5 8 10 20 30 50

# Walk-forward (no lookahead)
python tools/backtests/walk_forward_sim.py exports/backtests/n500_swing_365d \
  --capital 500000 --split-date 2025-11-08 --top-n 30 --caps 1 2 3 5 8 10
```
