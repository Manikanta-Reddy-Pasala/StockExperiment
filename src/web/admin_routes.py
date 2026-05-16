"""
Admin Routes for Manual Triggers
Provides UI controls to manually trigger automated processes.
"""

from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
import subprocess
import threading
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Track running tasks (in-memory cache + database persistence)
running_tasks = {}


def save_task_to_db(task_id, task_data):
    """Save task state to database for persistence across page refreshes."""
    from src.models.database import get_database_manager
    from sqlalchemy import text
    import json

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        # Convert steps list to JSON
        steps_json = json.dumps(task_data.get('steps', []))

        # Upsert task
        query = text("""
            INSERT INTO admin_task_tracking
            (task_id, task_type, description, status, start_time, end_time, steps, output, error, updated_at)
            VALUES
            (:task_id, :task_type, :description, :status, :start_time, :end_time, CAST(:steps AS jsonb), :output, :error, NOW())
            ON CONFLICT (task_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                end_time = EXCLUDED.end_time,
                steps = EXCLUDED.steps,
                output = EXCLUDED.output,
                error = EXCLUDED.error,
                updated_at = NOW()
        """)

        session.execute(query, {
            'task_id': task_id,
            'task_type': task_data.get('type', 'unknown'),
            'description': task_data.get('description', ''),
            'status': task_data.get('status', 'pending'),
            'start_time': task_data.get('start_time'),
            'end_time': task_data.get('end_time'),
            'steps': steps_json,
            'output': task_data.get('output', ''),
            'error': task_data.get('error', '')
        })
        session.commit()


def get_task_from_db(task_id):
    """Get task state from database."""
    from src.models.database import get_database_manager
    from sqlalchemy import text
    import json

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        query = text("""
            SELECT task_id, task_type, description, status, start_time, end_time, steps, output, error
            FROM admin_task_tracking
            WHERE task_id = :task_id
        """)

        result = session.execute(query, {'task_id': task_id}).fetchone()

        if result:
            # PostgreSQL JSONB is already parsed as Python object, no need for json.loads()
            steps = result[6] if result[6] else []
            return {
                'type': result[1],
                'description': result[2],
                'status': result[3],
                'start_time': result[4].isoformat() if result[4] else None,
                'end_time': result[5].isoformat() if result[5] else None,
                'steps': steps if isinstance(steps, list) else json.loads(steps) if steps else [],
                'output': result[7] or '',
                'error': result[8] or ''
            }
    return None


def get_active_tasks_from_db():
    """Get all active (running) tasks from database."""
    from src.models.database import get_database_manager
    from sqlalchemy import text
    import json

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        query = text("""
            SELECT task_id, task_type, description, status, start_time, end_time, steps, output, error
            FROM admin_task_tracking
            WHERE status IN ('running', 'pending')
            OR (status IN ('failed', 'completed') AND updated_at > NOW() - INTERVAL '1 hour')
            ORDER BY created_at DESC
            LIMIT 10
        """)

        results = session.execute(query).fetchall()

        tasks = {}
        for row in results:
            # PostgreSQL JSONB is already parsed as Python object
            steps = row[6] if row[6] else []
            tasks[row[0]] = {
                'type': row[1],
                'description': row[2],
                'status': row[3],
                'start_time': row[4].isoformat() if row[4] else None,
                'end_time': row[5].isoformat() if row[5] else None,
                'steps': steps if isinstance(steps, list) else json.loads(steps) if steps else [],
                'output': row[7] or '',
                'error': row[8] or '',
                'failed': row[3] == 'failed',
                'completed': row[3] == 'completed'
            }

        return tasks


def run_command_async(task_id, command, description):
    """Run a command asynchronously and track its status."""
    try:
        running_tasks[task_id] = {
            'status': 'running',
            'description': description,
            'start_time': datetime.now().isoformat(),
            'output': '',
            'error': ''
        }

        # Get the project root directory (where run_pipeline.py lives)
        project_root = '/app' if os.path.exists('/app/run_pipeline.py') else os.getcwd()

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=project_root
        )
        
        running_tasks[task_id]['status'] = 'completed' if result.returncode == 0 else 'failed'
        running_tasks[task_id]['output'] = result.stdout
        running_tasks[task_id]['error'] = result.stderr
        running_tasks[task_id]['return_code'] = result.returncode
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
        
    except subprocess.TimeoutExpired:
        running_tasks[task_id]['status'] = 'timeout'
        running_tasks[task_id]['error'] = 'Task timeout after 1 hour'
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
    except Exception as e:
        running_tasks[task_id]['status'] = 'error'
        running_tasks[task_id]['error'] = str(e)
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()


@admin_bp.route('/')
def admin_dashboard():
    """Admin dashboard with manual trigger controls."""
    return render_template('admin/dashboard.html')


def run_function_async(task_id, func, description, task_type='generic', *args, **kwargs):
    """Run a Python function asynchronously and track its status."""
    try:
        task_data = {
            'type': task_type,
            'status': 'running',
            'description': description,
            'start_time': datetime.now().isoformat(),
            'output': '',
            'error': '',
            'steps': []
        }
        running_tasks[task_id] = task_data
        save_task_to_db(task_id, task_data)

        # Run the function
        result = func(*args, **kwargs)

        running_tasks[task_id]['status'] = 'completed'
        running_tasks[task_id]['output'] = str(result) if result else 'Completed successfully'
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
        save_task_to_db(task_id, running_tasks[task_id])

    except Exception as e:
        running_tasks[task_id]['status'] = 'failed'
        running_tasks[task_id]['error'] = str(e)
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
        save_task_to_db(task_id, running_tasks[task_id])
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)


@admin_bp.route('/trigger/pipeline', methods=['POST'])
def trigger_pipeline():
    """Trigger complete data pipeline."""
    task_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_pipeline():
        from src.services.data.pipeline_saga import PipelineSaga

        saga = PipelineSaga()
        result = saga.run_pipeline()
        return result

    thread = threading.Thread(
        target=run_function_async,
        args=(task_id, run_pipeline, 'Data Pipeline (4-step saga)')
    )
    thread.start()

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Data pipeline started'
    })


@admin_bp.route('/trigger/all', methods=['POST'])
def trigger_all():
    """Trigger all processes in sequence."""
    base_task_id = f"all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_all_tasks():
        """Run all tasks sequentially."""
        from src.services.data.pipeline_saga import PipelineSaga
        from src.models.database import get_database_manager

        overall_task_id = f"{base_task_id}_all"
        task_data = {
            'type': 'all',
            'status': 'running',
            'description': 'Data Pipeline',
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'output': '',
            'error': ''
        }
        running_tasks[overall_task_id] = task_data
        save_task_to_db(overall_task_id, task_data)

        db_manager = get_database_manager()

        # Data Pipeline
        try:
            running_tasks[overall_task_id]['steps'].append({
                'name': 'pipeline',
                'description': 'Data Pipeline',
                'status': 'running',
                'start_time': datetime.now().isoformat()
            })
            save_task_to_db(overall_task_id, running_tasks[overall_task_id])

            saga = PipelineSaga()
            saga.run_pipeline()

            running_tasks[overall_task_id]['steps'][-1]['status'] = 'completed'
            running_tasks[overall_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()
            save_task_to_db(overall_task_id, running_tasks[overall_task_id])

        except Exception as e:
            running_tasks[overall_task_id]['steps'][-1]['status'] = 'failed'
            running_tasks[overall_task_id]['steps'][-1]['error'] = str(e)
            running_tasks[overall_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()
            running_tasks[overall_task_id]['error'] += f"\nPipeline failed: {str(e)}"
            save_task_to_db(overall_task_id, running_tasks[overall_task_id])
            logger.error(f"Pipeline failed: {e}", exc_info=True)

        failed_count = len([s for s in running_tasks[overall_task_id]['steps'] if s['status'] == 'failed'])
        running_tasks[overall_task_id]['status'] = 'completed' if failed_count == 0 else 'failed'
        running_tasks[overall_task_id]['end_time'] = datetime.now().isoformat()
        save_task_to_db(overall_task_id, running_tasks[overall_task_id])

    thread = threading.Thread(target=run_all_tasks)
    thread.start()

    return jsonify({
        'success': True,
        'task_id': f"{base_task_id}_all",
        'message': 'All tasks started sequentially'
    })


@admin_bp.route('/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """Get status of a running task."""
    # Check in-memory first
    if task_id in running_tasks:
        return jsonify({
            'success': True,
            'task': running_tasks[task_id]
        })

    # Check database
    task_data = get_task_from_db(task_id)
    if task_data:
        # Add flags for UI
        task_data['failed'] = task_data['status'] == 'failed'
        task_data['completed'] = task_data['status'] == 'completed'
        if task_data['status'] == 'failed':
            task_data['failedSteps'] = [s for s in task_data.get('steps', []) if s.get('status') == 'failed']

        return jsonify({
            'success': True,
            'task': task_data
        })

    return jsonify({
        'success': False,
        'error': 'Task not found'
    }), 404


@admin_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """List all tasks."""
    # Get from database (includes completed/failed from last hour)
    db_tasks = get_active_tasks_from_db()

    # Merge with in-memory tasks
    all_tasks = {**db_tasks, **running_tasks}

    return jsonify({
        'success': True,
        'tasks': all_tasks
    })


@admin_bp.route('/tasks/active', methods=['GET'])
def get_active_tasks():
    """Get all active tasks (for page load)."""
    return list_tasks()


@admin_bp.route('/task/<task_id>/retry-failed', methods=['POST'])
def retry_failed_steps(task_id):
    """Retry only the failed steps of a task."""
    if task_id not in running_tasks:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404

    task = running_tasks[task_id]

    # Get failed steps
    failed_steps = [step for step in task.get('steps', []) if step.get('status') == 'failed']

    if not failed_steps:
        return jsonify({
            'success': False,
            'error': 'No failed steps to retry'
        }), 400

    # Create a new task for retrying failed steps
    new_task_id = f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def retry_steps():
        """Retry failed steps only."""
        running_tasks[new_task_id] = {
            'status': 'running',
            'description': 'Retry Failed Steps',
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'output': '',
            'error': ''
        }

        for step in failed_steps:
            step_name = step['name']
            command = None
            description = step['description']

            # Map step name to command
            if step_name == 'pipeline':
                command = ['python3', 'run_pipeline.py']
            elif step_name == 'csv_export':
                command = ['python3', '-c', 'from data_scheduler import export_daily_csv; export_daily_csv()']

            if not command:
                continue

            try:
                running_tasks[new_task_id]['steps'].append({
                    'name': step_name,
                    'description': description,
                    'status': 'running',
                    'start_time': datetime.now().isoformat()
                })

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )

                step_status = 'completed' if result.returncode == 0 else 'failed'
                running_tasks[new_task_id]['steps'][-1]['status'] = step_status
                running_tasks[new_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()
                running_tasks[new_task_id]['steps'][-1]['return_code'] = result.returncode

                if result.returncode != 0:
                    running_tasks[new_task_id]['error'] += f"\n{step_name} failed: {result.stderr}"

            except Exception as e:
                running_tasks[new_task_id]['steps'][-1]['status'] = 'error'
                running_tasks[new_task_id]['steps'][-1]['error'] = str(e)
                running_tasks[new_task_id]['error'] += f"\n{step_name} error: {str(e)}"

        running_tasks[new_task_id]['status'] = 'completed'
        running_tasks[new_task_id]['end_time'] = datetime.now().isoformat()

    thread = threading.Thread(target=retry_steps)
    thread.start()

    return jsonify({
        'success': True,
        'task_id': new_task_id,
        'message': f'Retrying {len(failed_steps)} failed steps',
        'failed_steps': [s['name'] for s in failed_steps]
    })


@admin_bp.route('/trigger/model/<model_name>', methods=['POST'])
def trigger_model_data_pull(model_name):
    """Trigger data pulls for a specific deployed model.

    model_name: momentum_n100_top5_max1 | midcap_narrow_60d_breakout |
                finnifty_ic_otm4_w300_lots5 | finnifty_ic_otm3_w500_lots4
    """
    allowed = {
        "momentum_n100_top5_max1": [
            ("Equity OHLCV (N50+N500)", "tools.models.momentum_n100_top5_max1.data_pull",
             "pull_daily_ohlcv"),
            ("N100 universe refresh", "tools.models.momentum_n100_top5_max1.data_pull",
             "refresh_universe"),
        ],
        "midcap_narrow_60d_breakout": [
            ("Equity OHLCV (N500 — covers midcap_narrow)",
             "tools.models.midcap_narrow_60d_breakout.data_pull", "pull_daily_ohlcv"),
            ("midcap_narrow universe refresh (skip top-30, take next 100)",
             "tools.models.midcap_narrow_60d_breakout.data_pull", "refresh_universe"),
        ],
        "finnifty_ic_otm4_w300_lots5": [
            ("Index spots (NIFTY/BN/FN)", "tools.models.finnifty_ic_otm4_w300_lots5.data_pull",
             "fetch_index_spots"),
            ("Option bhavcopy (NIFTY/BN/FN)", "tools.models.finnifty_ic_otm4_w300_lots5.data_pull",
             "fetch_option_bhav"),
        ],
        "finnifty_ic_otm3_w500_lots4": [
            ("Shares data with otm4_w300_lots5",
             "tools.models.finnifty_ic_otm3_w500_lots4.data_pull", "noop"),
        ],
    }
    if model_name not in allowed:
        return jsonify({"success": False, "error": f"Unknown model: {model_name}"}), 400

    task_id = f"model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = allowed[model_name]

    def run_model_jobs():
        import importlib
        for label, mod_name, func_name in jobs:
            try:
                mod = importlib.import_module(mod_name)
                getattr(mod, func_name)()
                logger.info(f"  ✅ {label}")
            except Exception as e:
                logger.error(f"  ❌ {label}: {e}", exc_info=True)

    thread = threading.Thread(
        target=run_function_async,
        args=(task_id, run_model_jobs, f"Model {model_name} data pull"),
    )
    thread.start()

    return jsonify({
        "success": True,
        "task_id": task_id,
        "message": f"Data pull started for {model_name}",
        "jobs": [j[0] for j in jobs],
    })


@admin_bp.route('/system/models-status', methods=['GET'])
def models_status():
    """Per-model data sufficiency audit.

    Validates trading days of data per symbol the model actually consumes,
    not generic row counts. NSE-holiday tolerance built in.

    momentum_n100_top5_max1:
      Loads N100 universe file. For each of 100 stocks, counts distinct
      trading-day bars in last 150 calendar days. Requires >= 90 trading
      days (≈ 100 trading days × 90% NSE holiday/listing tolerance).

    finnifty_ic_otm4_w300_lots5:
      - FINNIFTY spot: last 30 cal days requires >= 18 trading days.
      - FINNIFTY option chain: for the *current* monthly expiry, count
        unique strikes within ±5% of last spot. Require >= 5 strikes
        with at least 3 days of bars.

    midcap_narrow_60d_breakout:
      Loads midcap_narrow universe file (~100 NSE midcaps). For each
      symbol counts distinct trading-day bars in last 150 calendar days,
      requires >= 70 trading days (60d HH lookback + buffer).

    finnifty_ic_otm3_w500_lots4:
      - Shares data with otm4. Reports same FN spot + FN option checks.
    """
    try:
        from src.models.database import get_database_manager
        from sqlalchemy import text
        import json
        from pathlib import Path
        from datetime import date, timedelta

        today = date.today()
        db_manager = get_database_manager()

        models = []
        with db_manager.get_session() as session:

            # ============================================================
            # Model 1: momentum_n100_top5_max1 (equity)
            # ============================================================
            universe_file = Path("/app/logs/momrot/universes/n100_current.json")
            universe_symbols = []
            universe_age_days = None
            if universe_file.exists():
                try:
                    u = json.loads(universe_file.read_text())
                    universe_symbols = [s["symbol"] for s in u.get("stocks", [])]
                    mtime = date.fromtimestamp(universe_file.stat().st_mtime)
                    universe_age_days = (today - mtime).days
                except Exception:
                    pass

            min_trading_days = 90        # need at least 90 trading days/symbol
            window_calendar_days = 150   # look back 150 cal days (~100 trading)
            since = today - timedelta(days=window_calendar_days)

            # Count trading days per symbol in N100 universe
            fyers_symbols = [f"NSE:{s}-EQ" for s in universe_symbols]
            per_sym_days = {}
            latest_per_sym = {}
            if fyers_symbols:
                rows = session.execute(text("""
                    SELECT symbol,
                           COUNT(DISTINCT date) AS days,
                           MAX(date) AS latest
                    FROM historical_data
                    WHERE symbol = ANY(:syms)
                      AND date >= :since
                    GROUP BY symbol
                """), {"syms": fyers_symbols, "since": since}).fetchall()
                per_sym_days = {r.symbol: int(r.days) for r in rows}
                latest_per_sym = {r.symbol: r.latest for r in rows}

            # Symbols meeting threshold
            ok_syms = [s for s in fyers_symbols if per_sym_days.get(s, 0) >= min_trading_days]
            under_syms = [s for s in fyers_symbols if per_sym_days.get(s, 0) < min_trading_days]
            missing_syms = [s for s in fyers_symbols if s not in per_sym_days]

            latest_dates = [d for d in latest_per_sym.values() if d]
            overall_latest = max(latest_dates) if latest_dates else None
            stale_days = (today - overall_latest).days if overall_latest else 999

            # Sufficiency: at least 90% of universe has >=90 trading days
            cov_pct = (len(ok_syms) / len(fyers_symbols) * 100) if fyers_symbols else 0
            eq_ok = (
                len(universe_symbols) >= 50
                and cov_pct >= 90
                and stale_days <= 3
                and universe_age_days is not None and universe_age_days <= 14
            )

            # Sample worst symbols for diagnostics
            worst = sorted(per_sym_days.items(), key=lambda kv: kv[1])[:5]
            worst_str = ", ".join(
                f"{s.replace('NSE:', '').replace('-EQ', '')}={d}"
                for s, d in worst
            ) if worst else ""

            models.append({
                "name": "momentum_n100_top5_max1",
                "type": "equity",
                "wired": True,
                "data_sufficient": bool(eq_ok),
                "items": [
                    {"label": "N100 universe size",
                     "value": len(universe_symbols),
                     "required": ">= 50 syms",
                     "ok": len(universe_symbols) >= 50,
                     "extra": f"file age {universe_age_days}d" if universe_age_days is not None else "missing"},
                    {"label": f"Symbols w/ >= {min_trading_days} trading days "
                              f"(last {window_calendar_days}d)",
                     "value": f"{len(ok_syms)} / {len(fyers_symbols)}",
                     "required": ">= 90% coverage",
                     "ok": cov_pct >= 90,
                     "extra": f"{cov_pct:.1f}% coverage"},
                    {"label": "Symbols below threshold",
                     "value": len(under_syms),
                     "required": "0 (allow few)",
                     "ok": len(under_syms) <= 10,
                     "extra": worst_str},
                    {"label": "Symbols completely missing",
                     "value": len(missing_syms),
                     "required": "0",
                     "ok": len(missing_syms) == 0},
                    {"label": "Latest equity close",
                     "value": overall_latest.isoformat() if overall_latest else "—",
                     "required": "<= 3d old (holiday OK)",
                     "ok": stale_days <= 3,
                     "extra": f"{stale_days}d old"},
                ],
            })

            # ============================================================
            # Model 2: midcap_narrow_60d_breakout (equity swing)
            # ============================================================
            mc_uni_file = Path("/app/logs/momrot/universes/midcap_narrow.json")
            mc_syms_plain = []
            mc_uni_age_days = None
            if mc_uni_file.exists():
                try:
                    u = json.loads(mc_uni_file.read_text())
                    mc_syms_plain = [s["symbol"] for s in u.get("stocks", [])]
                    mtime = date.fromtimestamp(mc_uni_file.stat().st_mtime)
                    mc_uni_age_days = (today - mtime).days
                except Exception:
                    pass

            mc_fyers_syms = [f"NSE:{s}-EQ" for s in mc_syms_plain]
            mc_per_sym_days = {}
            mc_latest_per_sym = {}
            mc_min_trading_days = 70  # 60-day high needs ~60 + buffer
            if mc_fyers_syms:
                rows = session.execute(text("""
                    SELECT symbol,
                           COUNT(DISTINCT date) AS days,
                           MAX(date) AS latest
                    FROM historical_data
                    WHERE symbol = ANY(:syms)
                      AND date >= :since
                    GROUP BY symbol
                """), {"syms": mc_fyers_syms, "since": since}).fetchall()
                mc_per_sym_days = {r.symbol: int(r.days) for r in rows}
                mc_latest_per_sym = {r.symbol: r.latest for r in rows}

            mc_ok_syms = [s for s in mc_fyers_syms
                          if mc_per_sym_days.get(s, 0) >= mc_min_trading_days]
            mc_under_syms = [s for s in mc_fyers_syms
                             if mc_per_sym_days.get(s, 0) < mc_min_trading_days]
            mc_missing_syms = [s for s in mc_fyers_syms if s not in mc_per_sym_days]
            mc_latest_dates = [d for d in mc_latest_per_sym.values() if d]
            mc_overall_latest = max(mc_latest_dates) if mc_latest_dates else None
            mc_stale_days = (today - mc_overall_latest).days if mc_overall_latest else 999
            mc_cov_pct = (
                len(mc_ok_syms) / len(mc_fyers_syms) * 100
            ) if mc_fyers_syms else 0
            mc_ok = (
                len(mc_syms_plain) >= 80
                and mc_cov_pct >= 90
                and mc_stale_days <= 3
                and mc_uni_age_days is not None and mc_uni_age_days <= 35
            )

            mc_worst = sorted(mc_per_sym_days.items(), key=lambda kv: kv[1])[:5]
            mc_worst_str = ", ".join(
                f"{s.replace('NSE:', '').replace('-EQ', '')}={d}"
                for s, d in mc_worst
            ) if mc_worst else ""

            models.append({
                "name": "midcap_narrow_60d_breakout",
                "type": "equity",
                "wired": False,
                "data_sufficient": bool(mc_ok),
                "items": [
                    {"label": "midcap_narrow universe size",
                     "value": len(mc_syms_plain),
                     "required": ">= 80 syms",
                     "ok": len(mc_syms_plain) >= 80,
                     "extra": f"file age {mc_uni_age_days}d" if mc_uni_age_days is not None else "missing"},
                    {"label": f"Symbols w/ >= {mc_min_trading_days} trading days "
                              f"(last {window_calendar_days}d)",
                     "value": f"{len(mc_ok_syms)} / {len(mc_fyers_syms)}",
                     "required": ">= 90% coverage",
                     "ok": mc_cov_pct >= 90,
                     "extra": f"{mc_cov_pct:.1f}% coverage"},
                    {"label": "Symbols below threshold",
                     "value": len(mc_under_syms),
                     "required": "0 (allow few)",
                     "ok": len(mc_under_syms) <= 10,
                     "extra": mc_worst_str},
                    {"label": "Symbols completely missing",
                     "value": len(mc_missing_syms),
                     "required": "0",
                     "ok": len(mc_missing_syms) == 0},
                    {"label": "Latest equity close",
                     "value": mc_overall_latest.isoformat() if mc_overall_latest else "—",
                     "required": "<= 3d old (holiday OK)",
                     "ok": mc_stale_days <= 3,
                     "extra": f"{mc_stale_days}d old"},
                ],
            })

            # ============================================================
            # Model 3: finnifty_ic_otm4_w300_lots5 (options)
            # ============================================================
            # Spot sufficiency: last 30 calendar days >= 18 trading days
            spot_since = today - timedelta(days=30)
            spot_items = []
            spots_ok = True
            for sym in ("NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"):
                r = session.execute(text("""
                    SELECT COUNT(DISTINCT date) days, MAX(date) latest
                    FROM historical_data WHERE symbol = :s AND date >= :since
                """), {"s": sym, "since": spot_since}).fetchone()
                days = int(r.days) if r and r.days else 0
                latest = r.latest if r else None
                stale = (today - latest).days if latest else 999
                short = sym.replace("NSE:", "").replace("-INDEX", "")
                ok = days >= 18 and stale <= 3
                if not ok:
                    spots_ok = False
                spot_items.append({
                    "label": f"{short} spot trading days (30d)",
                    "value": days,
                    "required": ">= 18 (≈ 22 - holidays)",
                    "ok": ok,
                    "extra": (latest.isoformat() if latest else "—") + f" • {stale}d old",
                })

            # Current monthly expiry availability per underlying
            opt_items = []
            opts_ok = True
            current_finnifty_spot = session.execute(text("""
                SELECT close FROM historical_data
                WHERE symbol='NSE:FINNIFTY-INDEX' ORDER BY date DESC LIMIT 1
            """)).fetchone()
            fn_spot_val = float(current_finnifty_spot.close) if current_finnifty_spot else 0

            for u in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
                # Next monthly expiry from option_universe
                next_exp_row = session.execute(text("""
                    SELECT MIN(expiry) AS exp FROM option_universe
                    WHERE underlying = :u AND expiry_kind = 'monthly'
                      AND expiry >= :today
                """), {"u": u, "today": today}).fetchone()
                next_exp = next_exp_row.exp if next_exp_row else None

                if next_exp is None:
                    opt_items.append({
                        "label": f"{u} next monthly expiry",
                        "value": "—",
                        "required": "exists",
                        "ok": False,
                    })
                    opts_ok = False
                    continue

                # Strike coverage for that expiry: count distinct strikes that
                # have at least 1 bar in last 7 calendar days
                strike_cov = session.execute(text("""
                    SELECT COUNT(DISTINCT strike) AS strikes
                    FROM historical_options
                    WHERE underlying = :u AND expiry = :exp
                      AND candle_time::date >= :since
                """), {"u": u, "exp": next_exp,
                       "since": today - timedelta(days=7)}).fetchone()
                strikes = int(strike_cov.strikes) if strike_cov else 0
                ok = strikes >= 5
                if not ok:
                    opts_ok = False
                opt_items.append({
                    "label": f"{u} {next_exp.isoformat()} strikes (last 7d)",
                    "value": strikes,
                    "required": ">= 5 strikes",
                    "ok": ok,
                    "extra": f"expires {(next_exp - today).days}d",
                })

            fn_ok = spots_ok and opts_ok
            models.append({
                "name": "finnifty_ic_otm4_w300_lots5",
                "type": "options",
                "wired": False,
                "data_sufficient": bool(fn_ok),
                "items": spot_items + opt_items,
            })

            # ============================================================
            # Model 3: finnifty_ic_otm3_w500_lots4 (options) — shares data
            # ============================================================
            # Tighter strike-coverage requirement: needs wing strikes at
            # ±500pts on FINNIFTY which is wider than otm4 IC. Verify FN
            # option chain alone (NIFTY/BANKNIFTY not used by this model).
            fn_spot_item = next(
                (i for i in spot_items if i["label"].startswith("FINNIFTY ")), None
            )
            fn_opt_item = next(
                (i for i in opt_items if i["label"].startswith("FINNIFTY ")), None
            )
            otm3_items = []
            if fn_spot_item:
                otm3_items.append(fn_spot_item)
            if fn_opt_item:
                otm3_items.append(fn_opt_item)
            otm3_ok = bool(
                fn_spot_item and fn_spot_item.get("ok")
                and fn_opt_item and fn_opt_item.get("ok")
            )
            models.append({
                "name": "finnifty_ic_otm3_w500_lots4",
                "type": "options",
                "wired": False,
                "data_sufficient": otm3_ok,
                "items": otm3_items,
            })

        return jsonify({"success": True, "models": models, "as_of": today.isoformat()})

    except Exception as e:
        logger.error(f"Models status error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/system/status', methods=['GET'])
def system_status():
    """Get system status."""
    try:
        from src.models.database import get_database_manager
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Get database stats
            stocks_stats = session.execute(text("""
                SELECT
                    COUNT(*) as total_stocks,
                    COUNT(current_price) as with_price,
                    COUNT(market_cap) as with_market_cap,
                    MAX(last_updated) as last_updated
                FROM stocks
            """)).fetchone()

            history_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(*) as records,
                    MAX(date) as latest_date
                FROM historical_data
            """)).fetchone()

            tech_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols,
                    MAX(date) as latest_date
                FROM technical_indicators
            """)).fetchone()

            snapshot_stats = session.execute(text("""
                SELECT
                    COUNT(*) as total_snapshots,
                    COUNT(DISTINCT date) as unique_dates,
                    MAX(date) as latest_date
                FROM daily_suggested_stocks
            """)).fetchone()

            return jsonify({
                'success': True,
                'status': {
                    'stocks': {
                        'total': stocks_stats.total_stocks,
                        'with_price': stocks_stats.with_price,
                        'with_market_cap': stocks_stats.with_market_cap,
                        'last_updated': stocks_stats.last_updated.isoformat() if stocks_stats.last_updated else None
                    },
                    'historical_data': {
                        'symbols': history_stats.symbols,
                        'records': history_stats.records,
                        'latest_date': history_stats.latest_date.isoformat() if history_stats.latest_date else None
                    },
                    'technical_indicators': {
                        'symbols': tech_stats.symbols,
                        'latest_date': tech_stats.latest_date.isoformat() if tech_stats.latest_date else None
                    },
                    'daily_snapshots': {
                        'total': snapshot_stats.total_snapshots,
                        'unique_dates': snapshot_stats.unique_dates,
                        'latest_date': snapshot_stats.latest_date.isoformat() if snapshot_stats.latest_date else None
                    }
                }
            })

    except Exception as e:
        logger.error(f"System status error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
