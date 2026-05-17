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
                finnifty_ic_otm4_w300_lots5
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


# =============================================================================
# Per-model capital ledger (multi-model portfolio tracking)
# =============================================================================

@admin_bp.route('/models/portfolio', methods=['GET'])
def get_model_portfolio():
    """Per-model + aggregate stats: allocated, NAV, cash, position, PnL.

    Used by dashboard cards. Includes MTM if last close available.
    """
    try:
        from src.services.trading.model_ledger_service import (
            get_portfolio_stats, ensure_models_seeded,
        )
        from src.models.database import get_database_manager
        from sqlalchemy import text

        # Make sure rows exist for all known models
        ensure_models_seeded()

        # MTM price lookup using historical_data latest close
        db = get_database_manager()
        with db.get_session() as session:
            def lookup(sym):
                r = session.execute(text(
                    "SELECT close FROM historical_data "
                    "WHERE symbol = :s ORDER BY date DESC LIMIT 1"
                ), {"s": sym}).fetchone()
                return float(r.close) if r else None
            stats = get_portfolio_stats(price_lookup=lookup)
        return jsonify({"success": True, **stats})
    except Exception as e:
        logger.error(f"models portfolio error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/settings', methods=['GET'])
def list_model_settings():
    try:
        from src.services.trading.model_ledger_service import (
            get_all_settings, ensure_models_seeded,
        )
        ensure_models_seeded()
        return jsonify({"success": True, "settings": get_all_settings()})
    except Exception as e:
        logger.error(f"models settings error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/deposit', methods=['POST'])
def deposit_to_model(model_name):
    """Add cash to a model (monthly top-up). Increases allocated + cash."""
    try:
        from src.services.trading.model_ledger_service import deposit
        data = request.get_json() or {}
        return jsonify({
            "success": True,
            **deposit(model_name, float(data.get("amount", 0))),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"deposit error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/withdraw', methods=['POST'])
def withdraw_from_model(model_name):
    """Pull cash out of a model. Decreases allocated + cash."""
    try:
        from src.services.trading.model_ledger_service import withdraw
        data = request.get_json() or {}
        return jsonify({
            "success": True,
            **withdraw(model_name, float(data.get("amount", 0))),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"withdraw error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/bootstrap', methods=['POST'])
def bootstrap_model(model_name):
    """Auto-migrate legacy JSON ledger (momrot_ledger.json) into model_ledger.

    Body: {cash_buffer: optional float}  — extra liquid cash beyond
    position cost (e.g. unused balance from last buy).

    For momentum_n100_top5_max1: reads /app/logs/momrot/ledger/momrot_ledger.json
    """
    JSON_PATHS = {
        "momentum_n100_top5_max1": "/app/logs/momrot/ledger/momrot_ledger.json",
        # Add other model paths here as their legacy ledgers come online
    }
    try:
        from src.services.trading.model_ledger_service import (
            auto_bootstrap_from_json_ledger,
        )
        path = JSON_PATHS.get(model_name)
        if not path:
            return jsonify({
                "success": False,
                "error": f"No legacy JSON ledger known for {model_name}"
            }), 400
        data = request.get_json() or {}
        cash_buffer = float(data.get("cash_buffer", 0))
        return jsonify({
            "success": True,
            **auto_bootstrap_from_json_ledger(path, model_name, cash_buffer),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"bootstrap error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/reset', methods=['POST'])
def reset_model_route(model_name):
    """Hard reset model ledger to zero. Audit trail preserved."""
    try:
        from src.services.trading.model_ledger_service import reset_model
        return jsonify({"success": True, **reset_model(model_name)})
    except Exception as e:
        logger.error(f"reset model error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/enabled', methods=['POST'])
def toggle_model_enabled(model_name):
    try:
        from src.services.trading.model_ledger_service import set_enabled
        data = request.get_json() or {}
        return jsonify({
            "success": True,
            "settings": set_enabled(model_name, bool(data.get("enabled"))),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"toggle enabled error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/seed-position', methods=['POST'])
def seed_model_position(model_name):
    """Bootstrap a model's open position (no Fyers order).

    Body: {symbol, qty, entry_px, entry_date}
    """
    try:
        from src.services.trading.model_ledger_service import seed_position
        data = request.get_json() or {}
        ledger = seed_position(
            model_name,
            symbol=data["symbol"],
            qty=int(data["qty"]),
            entry_px=float(data["entry_px"]),
            entry_date_str=data["entry_date"],
        )
        return jsonify({"success": True, "ledger": ledger})
    except (KeyError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"seed position error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/reset-position', methods=['POST'])
def reset_model_position(model_name):
    """Mark model as flat. Reconciliation tool — no Fyers order placed."""
    try:
        from src.services.trading.model_ledger_service import reset_position
        return jsonify({"success": True, "ledger": reset_position(model_name)})
    except Exception as e:
        logger.error(f"reset position error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/models/<model_name>/trades', methods=['GET'])
def model_trade_history(model_name):
    try:
        from src.services.trading.model_ledger_service import get_trades
        limit = int(request.args.get("limit", 50))
        return jsonify({
            "success": True,
            "trades": get_trades(model_name, limit=limit),
        })
    except Exception as e:
        logger.error(f"trade history error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Today's signals across all wired models (T10 dashboard widget)
# =============================================================================

# Map model_name -> list of (relative_filename_templates) under /app/logs.
# We probe each candidate path; first existing file wins. Templates support
# {date} placeholder (ISO yyyy-mm-dd).
_SIGNAL_PATHS = {
    "momentum_n100_top5_max1": [
        "/app/logs/momrot/signals/{date}_momrot_n100.json",
    ],
    "momentum_pseudo_n100_adv": [
        "/app/logs/momrot_pseudo/signals/{date}_pseudo_n100.json",
        "/app/logs/momentum_pseudo_n100_adv/signals/{date}.json",
    ],
    "midcap_narrow_60d_breakout": [
        "/app/logs/midcap_narrow/signals/{date}.json",
        "/app/logs/midcap_narrow/signals/{date}_midcap_narrow.json",
    ],
    "n20_daily_large_only": [
        "/app/logs/n20_daily/signals/{date}_n20.json",
        "/app/logs/n20_daily_large_only/signals/{date}.json",
    ],
    "finnifty_ic_otm4_w300_lots5": [
        "/app/logs/finnifty_ic_otm4_w300_lots5/signals/{date}.json",
    ],
}


@admin_bp.route('/signals/today', methods=['GET'])
def signals_today():
    """Aggregate today's emitted signals across all wired models.

    Reads each model's signals/{today}.json file. Returns array per model
    with metadata so the dashboard can render an "what fired today" widget.

    ?date=YYYY-MM-DD optionally overrides today (debugging / history view).
    """
    try:
        import json
        from pathlib import Path
        from datetime import date as _date

        date_str = request.args.get("date") or _date.today().isoformat()

        results = []
        for model_name, candidates in _SIGNAL_PATHS.items():
            found_path = None
            signals = []
            err = None
            for tmpl in candidates:
                p = Path(tmpl.format(date=date_str))
                if p.exists():
                    found_path = str(p)
                    try:
                        data = json.loads(p.read_text() or "[]")
                        if isinstance(data, list):
                            signals = data
                        elif isinstance(data, dict) and "signals" in data:
                            signals = data["signals"]
                    except Exception as e:
                        err = f"parse: {e}"
                    break

            results.append({
                "model_name": model_name,
                "date": date_str,
                "path": found_path,
                "count": len(signals),
                "signals": signals,
                "error": err,
                "has_file": found_path is not None,
            })

        total = sum(r["count"] for r in results)
        return jsonify({
            "success": True,
            "date": date_str,
            "total_signals": total,
            "models": results,
        })
    except Exception as e:
        logger.error(f"signals/today error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# T7 + T9 — Per-model Admin Triggers + Today's Picks
# =============================================================================

# Per-equity-model wiring: where each model writes signals/rankings, and how
# to invoke its live_signal.py. Kept here so future models = single-row diff.
MODEL_PATHS = {
    "momentum_n100_top5_max1": {
        "signals_dir": "/app/logs/momrot/signals",
        "ranking_dir": "/app/logs/momrot/ranking",
        "live_signal": "tools/models/momentum_n100_top5_max1/live_signal.py",
        "extra_args": [
            "--universe-file", "/app/logs/momrot/universes/n100_current.json",
            "--top-n", "5",
        ],
        "label": "N100 monthly rotation top-5 (mc=1)",
        "universe_path": "/app/logs/momrot/universes/n100_current.json",
    },
    "momentum_pseudo_n100_adv": {
        "signals_dir": "/app/logs/momrot_pseudo/signals",
        "ranking_dir": "/app/logs/momrot_pseudo/ranking",
        "live_signal": "tools/models/momentum_pseudo_n100_adv/live_signal.py",
        "extra_args": [
            "--universes-file",
            "tools/models/momentum_pseudo_n100_adv/yearly_universes.json",
            "--top-n", "5",
        ],
        "label": "Pseudo-N100 monthly rotation top-5 (DISABLED — lookahead)",
        "universe_path":
            "tools/models/momentum_pseudo_n100_adv/yearly_universes.json",
    },
    "midcap_narrow_60d_breakout": {
        "signals_dir": "/app/logs/midcap_narrow/signals",
        "ranking_dir": "/app/logs/midcap_narrow/ranking",
        "live_signal": "tools/models/midcap_narrow_60d_breakout/live_signal.py",
        "extra_args": [
            "--universe-file", "/app/logs/momrot/universes/midcap_narrow.json",
        ],
        "label": "Midcap 60d-high breakout (event-driven)",
        "universe_path": "/app/logs/momrot/universes/midcap_narrow.json",
    },
    "n20_daily_large_only": {
        "signals_dir": "/app/logs/n20_daily/signals",
        "ranking_dir": "/app/logs/n20_daily/ranking",
        "live_signal": "tools/models/n20_daily_large_only/live_signal.py",
        "extra_args": ["--top-n", "1"],
        "label": "Top-20 ADV ∩ N100 daily rotation (mc=1)",
        "universe_path": None,  # PIT built each run from N500 OHLCV
    },
}


def _latest_signal_file(signals_dir: str):
    """Return (path, mtime_iso) of newest *.json signal file or (None, None)."""
    try:
        d = Path(signals_dir)
        if not d.exists():
            return None, None
        files = sorted(
            d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not files:
            return None, None
        f = files[0]
        return str(f), datetime.fromtimestamp(f.stat().st_mtime).isoformat()
    except Exception:
        return None, None


@admin_bp.route('/models/<model_name>/triggers/status', methods=['GET'])
def model_triggers_status(model_name):
    """Aggregate per-model status used by the Admin Triggers table row.

    Combines:
      - file-system newest signal file (mtime)
      - last execution row from model_trades table
      - current open position from model_ledger (via get_portfolio_stats)
      - settings: enabled flag, NAV, invested, win-rate, P&L %
    """
    try:
        from src.services.trading.model_ledger_service import (
            ensure_models_seeded, get_portfolio_stats,
        )
        from src.models.database import get_database_manager
        from sqlalchemy import text

        ensure_models_seeded()
        paths = MODEL_PATHS.get(model_name)
        if not paths:
            return jsonify({"success": False,
                            "error": f"Unknown model: {model_name}"}), 400

        # Newest signal file in this model's signals dir
        last_signal_file, last_signal_at = _latest_signal_file(
            paths["signals_dir"]
        )

        # Latest execution (BUY/SELL only) for this model from model_trades
        db = get_database_manager()
        last_execution_at = None
        last_order_id = None
        with db.get_session() as s:
            row = s.execute(text("""
                SELECT trade_at, fyers_order_id
                FROM model_trades
                WHERE model_name = :m
                  AND side IN ('BUY','SELL')
                ORDER BY trade_at DESC
                LIMIT 1
            """), {"m": model_name}).fetchone()
            if row:
                last_execution_at = (
                    row[0].isoformat() if row[0] else None
                )
                last_order_id = row[1]

            # Portfolio stats — find this model's slice
            def _mtm(sym):
                r = s.execute(text(
                    "SELECT close FROM historical_data "
                    "WHERE symbol=:s ORDER BY date DESC LIMIT 1"
                ), {"s": sym}).fetchone()
                return float(r.close) if r else None
            stats = get_portfolio_stats(price_lookup=_mtm)

        per_model = next(
            (m for m in stats["models"] if m["model_name"] == model_name), {}
        )

        current_position = None
        if per_model.get("open_symbol"):
            current_position = {
                "sym": per_model["open_symbol"],
                "qty": per_model["open_qty"],
                "entry_px": per_model["open_entry_px"],
                "entry_date": per_model.get("open_entry_date"),
                "mtm_price": per_model.get("open_mtm_price"),
                "position_value": per_model.get("position_value"),
            }

        return jsonify({
            "success": True,
            "model_name": model_name,
            "label": paths.get("label", model_name),
            "enabled": bool(per_model.get("enabled")),
            "last_signal_at": last_signal_at,
            "last_signal_file": last_signal_file,
            "last_execution_at": last_execution_at,
            "last_order_id": last_order_id,
            "current_position": current_position,
            "nav": per_model.get("nav", 0),
            "invested": per_model.get("invested_amount", 0),
            "cash": per_model.get("cash", 0),
            "pnl_total": per_model.get("pnl_total", 0),
            "pnl_pct": per_model.get("return_pct", 0),
            "realized_pnl": per_model.get("realized_pnl", 0),
            "total_trades": per_model.get("total_trades", 0),
            "win_rate_pct": per_model.get("win_rate_pct", 0),
        })
    except Exception as e:
        logger.error(f"triggers/status error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/<model_name>/run-signal', methods=['POST'])
def admin_run_signal(model_name):
    """Manually trigger live_signal.py for a model. Writes signal file to
    /tmp/manual_<model>_<ts>.json so it never overwrites scheduler output.
    Runs in background thread; returns task_id for polling.
    """
    paths = MODEL_PATHS.get(model_name)
    if not paths:
        return jsonify({"success": False,
                        "error": f"Unknown model: {model_name}"}), 400

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    signals_out = f"/tmp/manual_{model_name}_{ts}.json"
    task_id = f"signal_{model_name}_{ts}"

    project_root = '/app' if os.path.exists('/app/run_pipeline.py') else os.getcwd()
    cmd = [
        "python3", paths["live_signal"],
        *paths["extra_args"],
        "--signals-out", signals_out,
        "--force",
    ]

    def runner():
        run_command_async(task_id, cmd, f"Manual signal for {model_name}")
        # Stash output path on the task so the UI can pick it up for execute
        if task_id in running_tasks:
            running_tasks[task_id]["signals_out"] = signals_out

    threading.Thread(target=runner).start()

    return jsonify({
        "success": True,
        "task_id": task_id,
        "signals_out": signals_out,
        "message": f"Signal run started for {model_name}",
        "cmd": " ".join(cmd),
    })


@admin_bp.route('/<model_name>/run-execute', methods=['POST'])
def admin_run_execute(model_name):
    """Manually trigger fyers_executor.py against the latest signals file.

    Body (optional): {signals_file: <path>, dry_run: true}
      - If signals_file omitted, uses newest file in the model's signals_dir.
      - dry_run defaults to True for safety; pass false to actually place.
    """
    paths = MODEL_PATHS.get(model_name)
    if not paths:
        return jsonify({"success": False,
                        "error": f"Unknown model: {model_name}"}), 400

    data = request.get_json(silent=True) or {}
    signals_file = data.get("signals_file")
    if not signals_file:
        signals_file, _ = _latest_signal_file(paths["signals_dir"])
    if not signals_file:
        return jsonify({
            "success": False,
            "error": f"No signal file found in {paths['signals_dir']}; "
                     f"run /admin/{model_name}/run-signal first",
        }), 400

    dry_run = bool(data.get("dry_run", True))
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    task_id = f"execute_{model_name}_{ts}"
    user_id = os.environ.get("USER_ID", "1")

    cmd = [
        "python3", "tools/live/fyers_executor.py",
        "--signals", signals_file,
        "--user-id", user_id,
        "--model-name", model_name,
    ]
    if dry_run:
        cmd.append("--dry-run")

    threading.Thread(
        target=run_command_async,
        args=(task_id, cmd,
              f"Manual {'DRY-RUN' if dry_run else 'LIVE'} execute for {model_name}"),
    ).start()

    return jsonify({
        "success": True,
        "task_id": task_id,
        "signals_file": signals_file,
        "dry_run": dry_run,
        "message": f"Execute started for {model_name} (dry_run={dry_run})",
        "cmd": " ".join(cmd),
    })


@admin_bp.route('/<model_name>/toggle-enabled', methods=['POST'])
def admin_toggle_enabled(model_name):
    """Flip model_settings.enabled. Body optional: {enabled: bool}; if omitted
    we toggle the current value. Returns the new settings row.
    """
    try:
        from src.services.trading.model_ledger_service import (
            get_all_settings, set_enabled, ensure_models_seeded,
        )
        ensure_models_seeded()
        data = request.get_json(silent=True) or {}
        if "enabled" in data:
            new_val = bool(data["enabled"])
        else:
            current = next(
                (s for s in get_all_settings() if s["model_name"] == model_name),
                None,
            )
            if not current:
                return jsonify({"success": False,
                                "error": f"Unknown model: {model_name}"}), 400
            new_val = not bool(current.get("enabled"))
        return jsonify({
            "success": True,
            "settings": set_enabled(model_name, new_val),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"toggle-enabled error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route('/<model_name>/ranking', methods=['GET'])
def admin_model_ranking(model_name):
    """Read the per-model ranking JSON written by live_signal.py.

    Query: ?top=N (default 5; capped to the file's top_n)

    Each live_signal.py also writes today's top-N ranking to
    `/app/logs/<model>/ranking/<date>.json`. This endpoint returns the
    newest file in that dir, sliced to `top`.
    """
    paths = MODEL_PATHS.get(model_name)
    if not paths:
        return jsonify({"success": False,
                        "error": f"Unknown model: {model_name}"}), 400

    try:
        top = int(request.args.get("top", 5))
        d = Path(paths["ranking_dir"])
        if not d.exists():
            return jsonify({
                "success": True,
                "model": model_name,
                "label": paths.get("label", model_name),
                "ranking": [],
                "note": f"No ranking dir yet ({paths['ranking_dir']}) — "
                        "scheduler will create it on next live_signal run, "
                        "or hit /admin/<m>/run-signal.",
            })

        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        if not files:
            return jsonify({
                "success": True, "model": model_name,
                "label": paths.get("label", model_name),
                "ranking": [],
                "note": "No ranking files yet for this model.",
            })

        import json as _json
        payload = _json.loads(files[0].read_text())
        ranking = payload.get("top_n") or []
        return jsonify({
            "success": True,
            "model": model_name,
            "label": paths.get("label", model_name),
            "date": payload.get("date"),
            "universe_size": payload.get("universe_size"),
            "ranking": ranking[:top],
            "source": str(files[0]),
            "generated_at": datetime.fromtimestamp(
                files[0].stat().st_mtime).isoformat(),
        })
    except Exception as e:
        logger.error(f"model ranking error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
