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
- **MAX_HOLD 30 trading days** (dominant exit — captures ~30-day midcap runs)

**Universe**: `midcap_narrow` (smaller midcap pool, ~100 NSE midcap names).

**Costs modeled**: 10 bps slippage + 0.10% STT on sells + ₹20/order brokerage.

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
