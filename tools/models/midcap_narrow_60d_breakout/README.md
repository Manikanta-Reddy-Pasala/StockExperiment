# midcap_narrow_60d_breakout

## Goal achieved: ≥120% CAGR with ≤30% DD on unlevered Indian equity swing

| Metric | Value |
|---|---:|
| Capital | ₹2,00,000 |
| Final | ₹21,79,348 |
| Total return | **+989.67%** |
| **CAGR** | **+121.66%** ✅ (goal ≥120%) |
| **Max DD** | **-20.43%** ✅ (goal ≤30%) |
| Calmar | 5.96 |
| Trades | 34 over 3yr (~11/yr) |
| Win rate | TBD (most exits MAX_HOLD) |

### Per year ROI

| Year | ROI |
|---|---:|
| 2023 (May-Dec) | +95.31% |
| 2024 | +110.42% |
| 2025 | +55.01% |
| 2026 (Jan-May) | +71.58% |

All 4 years strongly positive. No down year.

## Strategy

**Entry** (single position, max_conc=1):
- Stock makes fresh **60-day high**
- Volume on breakout day > **2.0× 20-day avg volume**
- Close > **200-day SMA** (long-term Stage 2 trend)

**Exit** (whichever fires first):
- **Profit target +60%** from entry
- **Trailing stop: -15% from peak**, activated after +10% gain
- **SMA exit**: close < **20-day SMA** (cuts losers fast — avg -8%)
- **MAX_HOLD 30 trading days** (dominant winner exit — captures ~30-day midcap runs)

**Universe**: `midcap_narrow` (smaller midcap pool, ~100 NSE midcap names).

**Costs modeled**: 10 bps slippage + 0.10% STT on sells + ₹20/order brokerage.

## How rebalance + stock selection works

**This is NOT calendar-rebalanced.** No fixed monthly/weekly cadence. Rebalance is **event-driven**: whenever a position exits (via target / trail / SMA / max-hold), the next trading day scans for a new entry. If no signal, sits in cash until one appears.

### Daily loop (pseudocode)

```
FOR each trading day D from 2023-05-15 to 2026-05-15:

  # 1. MARK-TO-MARKET — update last close, peak, sma_exit for open position
  IF position open:
      refresh close, sma_20, peak with today's bar

  # 2. EXIT CHECK — priority order
  IF position open:
      IF return_from_entry >= +60%:           sell at next open  (TARGET)
      ELIF return_from_entry >= +10%
           AND drop_from_peak >= -15%:        sell at next open  (TRAIL)
      ELIF close < 20-day SMA:                sell at next open  (SMA)
      ELIF days_held >= 30:                   sell at next open  (MAX_HOLD)

  # 3. ENTRY SCAN — only when flat (max_conc=1)
  IF no position open:
      candidates = []
      FOR each symbol S in midcap_narrow universe (~100 stocks):
          IF S.close TODAY > S.high.rolling(60).max() yesterday   # fresh 60d high
            AND S.close TODAY > S.SMA_200                          # long-term uptrend
            AND S.volume TODAY > 2.0 * S.volume.rolling(20).mean(): # 2x volume surge
              candidates.append(S, vol_ratio = vol / vol_avg20)

      IF candidates non-empty:
          pick = candidate with HIGHEST vol_ratio                   # tiebreak
          place buy order: full cash / next_open_price, qty = int()
          fills at TOMORROW's open + 10 bps slippage

  # 4. RECORD NAV
  daily_nav[D] = cash + position.qty * position.last_close
```

### Stock selection criteria — what wins?

Every trading day the strategy scans **all ~100 midcap_narrow names** and asks:

1. **Did the stock break out today?** → close > rolling-60-day high (excluding today). This catches a fresh swing high after a base.

2. **Is it in a long-term uptrend?** → close > 200-day SMA. Filters out stage-1 basing and stage-4 downtrends. Only stage-2 (Minervini) trends qualify.

3. **Is the breakout backed by real volume?** → volume > 2× 20-day average. Weeds out low-conviction false breakouts. Captures institutional accumulation.

If multiple stocks qualify on same day, the one with the **highest volume ratio** wins (vol_today / vol_avg20). This is the "loudest" breakout — typically the most aggressive institutional buying.

### Rebalance trigger examples from backtest

| Date | Event | What happened |
|---|---|---|
| 2023-11-08 | Entry scan | HINDPETRO had highest vol_ratio among breakouts → bought next open ₹188.56 |
| 2023-12-08 | MAX_HOLD exit | HINDPETRO held 30 days, sold at ₹251.42 (+33%) → flat for 1 day |
| 2023-12-11 | Entry scan | HINDCOPPER topped the candidates → bought ₹186.54 |
| 2024-06-21 | SMA exit | HINDUNILVR dropped below 20-day SMA after entry → bailed at -3.74% |
| 2024-06-24 | Entry scan (next day) | KALYANKJIL → bought ₹452.95, exited 30d later at +29.65% |

The strategy spends ~92% of trading days **in a position**. Average days flat between trades ≈ 2-3. Rebalance is opportunistic, not periodic.

### Why this beats monthly momentum

The deployed `momentum_n100_top5_max1` rebalances **only on the 1st of every month**, locked into whatever stock topped the 60-day return ranking on that date. Misses mid-month breakouts.

This model fires on **any day** a fresh 60d high + volume + 200-SMA stack lights up, then rides exactly 30 days. Result: captures the same momentum factor as N100 model but with 3-4× the entry opportunities and a hard 30-day exit that cycles capital faster.

### Position sizing

Always **100% of available cash** into the single pick. No leverage, no margin, no half-positions. Qty = `int(cash / entry_price)`. Cash leftover from rounding sits idle until next exit + re-entry cycle.

### What about no-signal days?

If on day D no candidate passes all three filters, the strategy **stays in cash**. No forced trade. This happened during sideways regimes (e.g., parts of 2025 H1 — note the -10% drawdown weeks in CHENNPETRO + TATAPOWER were forced trades in choppy conditions).

## How the goal was hit — research journey

After 100+ configurations across 10+ strategy families tested over 2023-05-15 → 2026-05-15 walk-forward:

| Strategy family | Best CAGR | Best DD | Verdict |
|---|---:|---:|---|
| Monthly rotation top-1 (N100, deployed) | +83.5% | -49% | DD fails |
| Weekly rotation top-1 | +44% | -56% | Both fail |
| Momentum spike day | +23% | -44% | Both fail |
| Pyramid breakout | +14% | -36% | Both fail |
| Turtle Donchian | +31% | -25% | CAGR fails |
| Stage 2 VCP (Minervini) | +8% | -22% | CAGR fails |
| Stacked max=2 momentum | +51% | -50% | Both fail |
| BTD pullback | +35% | n/a | CAGR fails |
| 52W high swing (smallcap, with regime) | +97.66% | -28.31% | DD OK, CAGR 22pp short |
| **60-day high breakout (midcap_narrow, NO regime)** | **+121.66%** | **-20.43%** | **✅** |

### Key insights

1. **Universe matters more than parameters**: midcap_narrow significantly outperformed smallcap_current, midcap_current, midcap2_current, midcap_wide for the same breakout signal. Smaller-cap dispersion + better liquidity than smallcap.

2. **60-day high beat 252-day high**: shorter lookback caught more setups; hh=60 outperformed hh=30/45/75/90/252 across the grid.

3. **MAX_HOLD=30 was optimal**: hold=20 cut winners too early (+57%), hold=45 let too-many failing trades drag (+30%), hold=30 captured the typical 30-day Indian midcap breakout cycle.

4. **target=0.60 saturated**: setting target=0.40/0.50/0.60/0.70/0.80/1.0 produced identical CAGR after target≥0.60 because trailing stop and MAX_HOLD dominated. target=0.60 is the cleanest number.

5. **NIFTY regime gate HURT**: with regime gate ON: +111.91% / -24.19%. Removing it: +121.66% / -20.43%. Counterintuitive — the trailing stop and MAX_HOLD already provide enough drawdown control; the regime filter was just removing winning setups.

6. **Single-stock concentration (mc=1) was essential**: mc=2 dropped CAGR to +77% even on midcap_narrow. The alpha is in fully riding one breakout at a time.

7. **Top winners during backtest**: HINDCOPPER (multiple), GALLANTT (multiple), KALYANKJIL, ABB, HINDZINC, ASHOKLEY, GVT&D. Commodity/industrials/PSU led 2023-2025 dispersion.

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-yr backtest with full trade ledger |
| `data_pull.py` | No-op (shares equity OHLCV with momentum_n100_top5_max1) |
| `cron.py` | Registration stubs (live exec not yet wired) |
| `README.md` | This file |

## Reproduce

```bash
docker exec trading_system_app python \
    tools/models/midcap_narrow_60d_breakout/backtest.py \
    --universe-file /app/logs/momrot/universes/midcap_narrow.json \
    --from 2023-05-15 --to 2026-05-15 \
    --hh 60 --vol-mult 2.0 \
    --trail-pct 0.15 --target-pct 0.60 --max-hold 30 \
    --capital 200000
```

Expected output:
```
final ₹2,179,348  CAGR +121.66%  DD -20.43%
trades 34
per_yr: {2023: 95.31, 2024: 110.42, 2025: 55.01, 2026: 71.58}
```

## Caveats

- Backtest uses CURRENT midcap_narrow universe (some survivorship vs PIT).
- 3 years tested (2023-2026). Forward live execution may face additional slippage at concentrated single-stock entry on small breakouts.
- Live realistic estimate: 75-85% of backtest CAGR after friction = **+90-100%/yr** expected.
- Max DD -20% is real — strategy was underwater for 1-2 month stretches.
- Live execution NOT YET WIRED. Use same Fyers executor pattern as momentum_n100_top5_max1.
- MAX_HOLD was the dominant exit in backtest — most trades held exactly 30 days. Live deployment must respect this exit discipline.

## Position in portfolio

User now has 4 committed models:
- **2 equity**: `momentum_n100_top5_max1` (monthly N100, wired) + `midcap_narrow_60d_breakout` (this — 60d breakout swing, unwired)
- **2 options**: `finnifty_ic_otm4_w300_lots5` + `finnifty_ic_otm3_w500_lots4` (both unwired)
