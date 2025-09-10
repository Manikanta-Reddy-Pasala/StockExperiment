# Development Setup

This document explains how to set up the Trading System for development with auto-reloading.

## Quick Start

### Development Mode (Recommended for Development)

```bash
# Start in development mode with auto-reloading
./run.sh dev

# View logs
./run.sh logs

# Stop the system
./run.sh stop
```

### Production Mode

```bash
# Start in production mode
./run.sh start

# View logs
./run.sh logs

# Stop the system
./run.sh stop
```

## Development Features

When running in development mode (`./run.sh dev`), the following features are enabled:

### Auto-reloading
- **Python files**: Changes to any `.py` file in the `src/` directory will automatically restart the Flask application
- **HTML templates**: Changes to template files in `src/web/templates/` will be reflected immediately
- **Static files**: Changes to static files in `src/web/static/` will be reflected immediately

### Debug Mode
- Flask debug mode is enabled
- Detailed error messages are shown in the browser
- Interactive debugger is available for debugging

### Volume Mounts
The development setup mounts the following directories for live editing:
- `./src` → `/app/src` (Python source code)
- `./src/web/templates` → `/app/src/web/templates` (HTML templates)
- `./src/web/static` → `/app/src/web/static` (Static files)
- `./logs` → `/app/logs` (Log files)

## File Structure

```
StockExperiment/
├── docker-compose.yml          # Production Docker Compose
├── docker-compose.dev.yml      # Development Docker Compose override
├── Dockerfile                  # Docker image definition
├── run.sh                      # Main script for running the system
├── src/                        # Source code (mounted in dev mode)
│   ├── web/
│   │   ├── templates/          # HTML templates (auto-reload)
│   │   └── static/             # Static files (auto-reload)
│   └── ...                     # Python modules (auto-reload)
└── logs/                       # Log files
```

## Commands

| Command | Description |
|---------|-------------|
| `./run.sh dev` | Start in development mode with auto-reloading |
| `./run.sh start` | Start in production mode |
| `./run.sh stop` | Stop the system |
| `./run.sh logs` | View application logs |
| `./run.sh status` | Show service status |
| `./run.sh restart` | Restart the system |
| `./run.sh cleanup` | Remove all containers and data |

## Development Workflow

1. **Start development mode**: `./run.sh dev`
2. **Make changes** to Python files, HTML templates, or static files
3. **Changes are automatically applied** - no need to restart manually
4. **View logs** if needed: `./run.sh logs`
5. **Stop when done**: `./run.sh stop`

## Troubleshooting

### Auto-reloading not working
- Ensure you're using `./run.sh dev` (not `./run.sh start`)
- Check that the files are being mounted correctly: `docker compose -f docker-compose.yml -f docker-compose.dev.yml ps`
- View logs to see if there are any errors: `./run.sh logs`

### Port conflicts
- The system runs on port 5001 by default
- If you have port conflicts, you can modify the port in `docker-compose.dev.yml`

### Database issues
- The database is persistent across restarts
- To reset the database: `./run.sh cleanup` (⚠️ This will delete all data)

## Environment Variables

The system uses the following environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `FLASK_ENV`: Set to `development` in dev mode
- `FLASK_DEBUG`: Set to `1` in dev mode
- `PYTHONPATH`: Set to `/app` for proper imports

These are automatically configured in the Docker Compose files.
