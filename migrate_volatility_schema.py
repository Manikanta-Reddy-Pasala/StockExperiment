#!/usr/bin/env python3
"""
Database Migration Script for Volatility Columns

This script adds the missing volatility columns to existing database installations.
Run this after updating the schema to ensure compatibility.
"""

import sys
import os
import psycopg2
from datetime import datetime

# Add project root to path
sys.path.append('/Users/manip/Documents/codeRepo/poc/StockExperiment')

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
    """, (table_name, column_name))
    return cursor.fetchone() is not None

def add_volatility_columns(cursor):
    """Add volatility columns to stocks table if they don't exist."""

    print("🔍 Checking existing volatility columns...")

    volatility_columns = [
        ('atr_14', 'DOUBLE PRECISION', 'Average True Range (14-day period)'),
        ('atr_percentage', 'DOUBLE PRECISION', 'ATR as percentage of current price'),
        ('historical_volatility_1y', 'DOUBLE PRECISION', 'Annualized historical volatility'),
        ('bid_ask_spread', 'DOUBLE PRECISION', 'Estimated bid-ask spread'),
        ('avg_daily_volume_20d', 'DOUBLE PRECISION', '20-day average daily volume'),
        ('updated_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP', 'For tracking volatility updates')
    ]

    columns_added = 0
    columns_existed = 0

    for column_name, column_type, description in volatility_columns:
        if check_column_exists(cursor, 'stocks', column_name):
            print(f"   ✅ {column_name}: Already exists")
            columns_existed += 1
        else:
            print(f"   ➕ {column_name}: Adding column...")
            cursor.execute(f"""
                ALTER TABLE stocks
                ADD COLUMN {column_name} {column_type}
            """)
            cursor.execute(f"""
                COMMENT ON COLUMN stocks.{column_name} IS '{description}'
            """)
            columns_added += 1
            print(f"      ✅ Added {column_name}")

    return columns_added, columns_existed

def add_volatility_indexes(cursor):
    """Add indexes for volatility columns if they don't exist."""

    print("🔍 Adding volatility indexes...")

    indexes = [
        ('idx_stocks_atr_percentage', 'atr_percentage'),
        ('idx_stocks_beta', 'beta'),
        ('idx_stocks_historical_volatility', 'historical_volatility_1y'),
        ('idx_stocks_avg_volume_20d', 'avg_daily_volume_20d'),
        ('idx_stocks_updated_at', 'updated_at')
    ]

    indexes_added = 0

    for index_name, column_name in indexes:
        try:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON stocks({column_name})
            """)
            print(f"   ✅ Index {index_name}: Created/verified")
            indexes_added += 1
        except Exception as e:
            print(f"   ⚠️  Index {index_name}: {e}")

    return indexes_added

def verify_schema_compatibility():
    """Verify that the database schema is compatible with volatility service."""

    database_url = os.getenv('DATABASE_URL', 'postgresql://trader:trader_password@localhost:5432/trading_system')

    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()

        print(f"📊 Connected to database: {database_url.split('@')[1] if '@' in database_url else database_url}")
        print()

        # Check if stocks table exists
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = 'stocks'
        """)

        if not cursor.fetchone():
            print("❌ Stocks table not found!")
            print("   Please run the full database initialization first:")
            print("   docker-compose exec trading_system python3 -c \"from src.models.database import init_db; init_db()\"")
            return False

        print("✅ Stocks table found")

        # Add missing volatility columns
        columns_added, columns_existed = add_volatility_columns(cursor)

        # Add volatility indexes
        indexes_added = add_volatility_indexes(cursor)

        # Commit changes
        conn.commit()

        print()
        print("🎉 SCHEMA MIGRATION COMPLETED")
        print("=" * 50)
        print(f"📊 Summary:")
        print(f"   • Volatility columns added: {columns_added}")
        print(f"   • Volatility columns existed: {columns_existed}")
        print(f"   • Indexes created/verified: {indexes_added}")

        if columns_added > 0:
            print()
            print("💡 Next steps:")
            print("   1. Run volatility calculation to populate data:")
            print("      python3 daily_volatility_update.py test")
            print("   2. Verify volatility data:")
            print("      python3 simple_volatility_check.py")

        # Verify column presence
        print()
        print("🔍 Final verification:")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'stocks'
            AND column_name IN ('atr_14', 'atr_percentage', 'beta', 'historical_volatility_1y', 'avg_daily_volume_20d', 'updated_at')
            ORDER BY column_name
        """)

        columns = cursor.fetchall()
        for column_name, data_type, is_nullable in columns:
            print(f"   ✅ {column_name}: {data_type} ({'nullable' if is_nullable == 'YES' else 'not null'})")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"❌ Migration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main migration function."""

    print("🚀 VOLATILITY SCHEMA MIGRATION")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    success = verify_schema_compatibility()

    if success:
        print()
        print("✅ Schema migration completed successfully!")
        print("   Database is now compatible with volatility calculations.")
        return 0
    else:
        print()
        print("❌ Schema migration failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())