# Coding Guidelines

This document outlines the coding style patterns and error handling rules for the trading application.

## Coding Style Patterns

-   **PEP 8**: All Python code should adhere to the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/).
-   **Modularity**: The application is organized into modules and services with specific responsibilities. When adding new features, follow the existing modular structure.
-   **Service-Oriented Architecture**: Business logic should be encapsulated within the service layer (`src/services`). The web routes in `src/web/routes` should be thin and act as a pass-through to the service layer.
-   **Database Access**: All database interactions should be performed through the `DatabaseManager` and the SQLAlchemy models defined in `src/models`. Avoid writing raw SQL queries in the application code.
-   **Configuration**: Application configuration should be stored in the `config.py` file. Access configuration values through the Flask application's config object.
-   **Logging**: Use the standard Python `logging` module for logging. Add meaningful log messages to help with debugging and monitoring.
-   **Docstrings**: All modules, classes, and functions should have docstrings that explain their purpose, arguments, and return values.

## Error Handling Rules

-   **Service Layer**: The service layer should raise exceptions when errors occur. Use specific exception types whenever possible.
-   **Web Layer**: The web layer is responsible for catching exceptions raised by the service layer and returning appropriate JSON error responses with a corresponding HTTP status code.
-   **Error Messages**: Error messages returned to the client should be clear, concise, and user-friendly. Avoid exposing sensitive information in error messages.
-   **Logging**: All exceptions should be logged with a stack trace to help with debugging. The `APILogger` utility can be used for this purpose.
-   **Broker API Errors**: Errors from broker APIs should be caught and handled gracefully. The application should not crash due to an error from an external service.
-   **Database Errors**: Errors from the database should be caught and handled. The application should be resilient to database connection errors.
