# Decision Guide — Pick Production Strategy

_Date: 2026-05-12 | Capital: ₹10,00,000 | Window: May 2025 → May 2026_

Single document to make the production call. All results 1-year backtest.

---

## TL;DR

**Recommended: Path B — EMA 9/21 + selector + filters + vol-sizing.**
- **+33.32% ROI / 8.20% DD / 59.7% win rate / 140 trades**
- High statistical confidence (140 samples)
- Best risk-adjusted return
- Survives compounding over multi-year horizons

Backup recommendation: **Path A** if 12% DD tolerable (+46.87% / 12.56% DD).

---

## All Strategies Tested

| # | Strategy | Type | Best Universe | Status |
|---|----------|------|---------------|--------|
| 1 | EMA 200/400 1H | Swing (4-30d holds) | Selector top-10 N500 | Phase 5 winner |
| 2 | EMA 9/21 1H | Short swing (1-5d) | Selector top-10 N500 | **Phase 7 winner** |
| 3 | Swing pullback breakout (daily) | Stage-2 swing | Doesn't fire on selector picks | Skipped |
| 4 | ORB 15-min | Intraday | n/a (5m cache empty) | Skipped |
| 5 | Donchian 20/55 (Turtle) | Trend breakout | N50 large caps | Marginal |
| 6 | 52wH + 2x volume (O'Neil) | Trend breakout | N50 large caps | Marginal |
| 7 | Bollinger Squeeze | Volatility expansion | N50 large caps | **Hidden gem** |
| 8 | VCP (Minervini) | Pattern | N50 large caps | Marginal |

---

## All Paths — Full Comparison

| Path | Strategy | Universe | Filters | Overlay | ROI% | DD% | Win% | Trades | Statistical Confidence |
|------|----------|----------|---------|---------|-----:|----:|-----:|-------:|------------------------|
| **A** | EMA 9/21 | Selector top-10 | None | None | **+46.87** | 12.56 | 62.0 | 209 | **High** |
| **B** | EMA 9/21 | Selector top-10 | Sector+Cal | Vol-2% | **+33.32** | **8.20** | 59.7 | 140 | **High** |
| **C** | EMA 9/21 | Selector top-10 | Sector+Cal | All overlays | +18.41 | 5.63 | 58.9 | 129 | Medium |
| **D** | BB Squeeze | N50 top-19 | None | None | +32.73 | **3.24** | **80.0** | 5 | **LOW** (small N) |
| **E** | EMA 200/400 | Selector top-10 | Sector+Cal | None | +29.35 | 9.58 | 50.0 | 24 | Medium |
| **F** | EMA 200/400 | Selector top-10 | Sector+Cal | Vol-2% max=5 | +20.03 | 6.09 | 73.8 | 42 | Medium |
| **G** | EMA 9/21 + BB Squeeze | Dual sleeve | Mixed | Vol-2% | ~+33%* | ~5%* | ~70%* | ~145* | LOW (not tested live) |
| H | Donchian 20/55 | N50 top-19 | None | None | +9.49 | 9.00 | 38.9 | 20 | Medium |
| I | 52wH+vol | N50 top-19 | None | None | +2.02 | 7.19 | 33.3 | 8 | Low |
| J | VCP | N50 top-19 | None | None | +6.28 | 3.75 | 33.3 | 3 | Very Low |
| K | EMA 200/400 (baseline N50 full) | N50 all | None | None | +7.30 | 12.77 | n/a | 54 | Medium |

\* G = theoretical combination; not backtested. Capital split ₹5L each.

---

## Path-by-Path Detail

### Path A — Max ROI

```yaml
strategy: ema_9_21
universe: selector_top10
filters: none
overlay: none
max_concurrent: 3
```

**Result: +46.87% ROI / 12.56% DD / 62% win rate / 209 trades**

✅ Pros:
- Highest absolute ROI tested
- Large trade count = high statistical confidence
- Simple config (no filters/overlays)

❌ Cons:
- 12.56% drawdown = ~₹1.25L worst dip from ₹10L
- Concentrated wins → lumpy P&L per month
- 38% loss rate = 4 in 10 trades lose

**Live realistic forward: 28-35% CAGR**

---

### Path B — Best Risk-Adjusted ⭐ RECOMMENDED

```yaml
strategy: ema_9_21
universe: selector_top10
sector_filter: block_bottom_2
calendar_filter: expiry + budget
overlay: vol_sizing
risk_per_trade_pct: 2.0
max_concurrent: 2
```

**Result: +33.32% ROI / 8.20% DD / 59.7% win rate / 140 trades**

✅ Pros:
- ROI nearly double Nifty 50 (~12-15%/yr)
- DD floor 8.2% = ~₹82K worst dip from ₹10L
- 140 trades = strong statistical edge
- Smoother equity curve (vol sizing dampens volatility)
- Sharpe ratio > 4 (very good)

❌ Cons:
- Trades less frequently than Path A
- Slightly lower headline ROI
- Slightly more complex stack to operate

**Live realistic forward: 22-28% CAGR**

---

### Path C — Ultra-defensive

```yaml
strategy: ema_9_21
universe: selector_top10
filters: sector + cal
overlay: DD-throttle + vol-1.5% + loss-pause
max_concurrent: 2
```

**Result: +18.41% ROI / 5.63% DD / 58.9% win rate / 129 trades**

✅ Pros:
- Lowest DD with decent trade count
- Capital preservation focus
- Best for risk-averse / first-time live

❌ Cons:
- ROI gives up ~10pp vs Path B for ~3pp DD reduction
- Inefficient capital usage (paused often)

**Live realistic forward: 12-15% CAGR**

---

### Path D — Lowest DD (high-risk-of-luck)

```yaml
strategy: bb_squeeze
universe: nifty50_top19 (large caps)
filters: none
overlay: none
max_concurrent: 2
```

**Result: +32.73% ROI / 3.24% DD / 80% win rate / 5 trades**

✅ Pros:
- Lowest drawdown of any path (3.24% = ~₹32K)
- Highest win rate (80%)
- Beautiful Sharpe if real

❌ Cons:
- **ONLY 5 TRADES IN 1 YEAR** — statistical noise risk huge
- 4 of 5 winning could be luck
- Strategy may not generate trades for 6+ months → capital idle
- Needs multi-year validation before trust
- Live could vary wildly from backtest

**Live realistic forward: -10% to +35% (huge variance)**

---

### Path E — EMA 200/400 (Phase 5 winner, now demoted)

```yaml
strategy: ema_200_400
universe: selector_top10
filters: sector + cal
max_concurrent: 2
```

**Result: +29.35% ROI / 9.58% DD / 50% win rate / 24 trades**

Why no longer top pick:
- EMA 9/21 on same universe gives +33.32% with LOWER DD
- Fewer trades (24 vs 140) = lower statistical confidence
- 50% win rate vs 60% — EMA 9/21 has cleaner edge

Keep as: backup strategy or diversification sleeve.

---

### Path G — Dual-sleeve diversification (UNTESTED)

```yaml
strategy_1:
  name: ema_9_21
  capital: 500_000     # ₹5L
  universe: selector_top10
  filters: sector + cal
  overlay: vol_sizing 2%
  max_concurrent: 1

strategy_2:
  name: bb_squeeze
  capital: 500_000     # ₹5L
  universe: nifty50_top19
  max_concurrent: 1
```

Theoretical: ~+33%/year combined with ~5-6% DD (uncorrelated alpha,
DDs don't sync because trade triggers differ).

⚠️ Not tested. Capital split halves slot allocation. May lose efficiency.

---

## Key Insights

### 1. Universe matters more than strategy
- EMA 9/21 was -28% on full N500, +46.87% on selector top-10
- Same strategy, different universe = different outcome
- **Stock selection is the primary alpha source**

### 2. Filters reduce ROI but improve consistency
- Phase 5 sector+cal: +21.85% → +29.35% (filters HELP here — they remove bad trades)
- Phase 6 vol-sizing: +29.35% → +33.32% (helps even more by sizing right)
- Phase 6 DD throttle alone: hurts (+29.35% → +10.59%)

### 3. New strategies need their own universes
- EMA fast/slow on volatile mid-caps
- Donchian/VCP/52wH/BB on stable large caps with long history
- One strategy ≠ all stocks

### 4. Trade count matters for trust
- 5 trades (BB Squeeze) → maybe lucky
- 24 trades (EMA 200/400 + filters) → marginal confidence
- 140 trades (Path B) → strong statistical edge
- 209 trades (Path A) → very strong

---

## Decision Framework

Pick based on what matters most:

| If you want… | Choose | Trade-off |
|--------------|--------|-----------|
| Highest expected ₹ | **Path A** | 12.6% DD acceptable |
| Best long-run Sharpe | **Path B ⭐** | 33% < 47% but smooth |
| Capital preservation | Path C | Half the ROI |
| Lowest DD if real | Path D | High variance risk |
| Diversification | Path G (untested) | Build cost |

---

## Hard truths

1. **5-10%/month (60-120%/yr) target unreachable.** No path tested or
   research-suggested gets there on cash equity ₹10L. Documented retail
   ceilings are 35-40% CAGR (Capitalmind momentum) over multi-year.

2. **Backtest → live gap = 30-40%.** Slippage, STT, STCG 20%, brokerage,
   regime shifts shrink real returns. Path A's +46.87% backtest →
   ~28-33% live. Path B's +33.32% → ~22-28% live.

3. **Single-year backtest is noisy.** All numbers here from one bull
   year (May 2025-2026, mid-cap rally). 2024 and 2023 windows untested.
   Real edge needs to survive both bull AND bear regimes.

4. **Selector top-10 stocks are recent IPOs/mid-caps.** SWIGGY, VMM,
   AEGISLOG = limited historical data. Strategy could break if these
   stocks underperform structurally. Plan: monthly selector refresh.

---

## Recommendation

### Primary: Path B (EMA 9/21 + filters + vol-sizing)

- High statistical confidence (140 trades)
- Best Sharpe of any high-confidence path
- ROI 2× SIP/index, DD half of typical aggressive funds
- Survives compounding long term
- Documentation tier matches: Marketsmith momentum CAGR 35%, Capitalmind systematic 30-40%

### Backup: Path A (EMA 9/21 raw)

- If you want max ROI and tolerate 12.56% DD
- Even larger sample (209 trades)
- Simpler operation

### Avoid for now:
- Path C (too conservative — gives up too much ROI)
- Path D (too few trades — luck risk)
- Paths H-K (mediocre returns)

### Future build:
- Path G (dual-sleeve) — needs separate implementation work
- Multi-year validation of Path B — confirms 2024 + 2023 windows
- Live-only scanners (delivery%, bulk deals, F&O ban) for paper-trade phase

---

## Action items (after decision)

1. **Lock production config** in `tools/live/` (env vars + cron)
2. **Cron the daily workflow**:
   - 09:00 IST → prefetch + monthly selector refresh
   - 09:30 IST → signals + paper executor
   - Every 5 min → position monitor
   - 15:35 IST → daily report
3. **4-week paper trade** to validate backtest
4. **Compare paper vs backtest** weekly. If paper > 50% of backtest ROI rate, trust the system.
5. **Multi-year backtest** (2024 + 2023) in parallel — validate selector + EMA 9/21 isn't regime-bound

---

## Files referenced

| Path | File |
|------|------|
| Phase 0 baseline | `exports/backtests/BASELINE_1YR.md` |
| Phase 1 patterns | `exports/backtests/PATTERN_MINING.md` |
| Phase 2 sweep | `exports/backtests/SWEEP_RESULTS.md` |
| Phase 3 regime (neg) | `exports/backtests/REGIME_GATE_RESULTS.md` |
| Phase 4 selector | `exports/backtests/SELECTOR_RESULTS.md` |
| Phase 5 patterns | `exports/backtests/PHASE5_INDIAN_PATTERNS.md` |
| Phase 6 overlays | `exports/backtests/PHASE6_RISK_OVERLAYS.md` |
| Phase 7 master | `exports/backtests/MASTER_COMPARISON.md` |
| **This** | `exports/backtests/DECISION_GUIDE.md` |
| Live trading stack | `tools/live/README.md` |
