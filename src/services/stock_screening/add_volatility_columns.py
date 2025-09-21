"""
Add Volatility Columns to Stocks Table

This script adds the missing volatility and risk metric columns to the stocks table
to support the new screening pipeline.
"""

import logging
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)

def add_volatility_columns():
    """Add missing volatility columns to stocks table."""

    # Database connection using environment variable
    database_url = os.getenv('DATABASE_URL', 'postgresql://trader:trader_password@localhost:5432/trading_system')

    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print("🔧 Adding missing volatility columns to stocks table...")

        # List of columns to add (with their types)
        volatility_columns = [
            ('atr_14', 'FLOAT'),
            ('atr_percentage', 'FLOAT'),
            ('historical_volatility_1y', 'FLOAT'),
            ('avg_daily_volume_20d', 'BIGINT'),
            ('avg_daily_turnover', 'FLOAT'),
            ('bid_ask_spread', 'FLOAT'),
            ('trades_per_day', 'INTEGER')
        ]

        columns_added = 0
        columns_existed = 0

        for column_name, column_type in volatility_columns:
            try:
                # Check if column already exists
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'stocks' AND column_name = %s
                """, (column_name,))

                if cursor.fetchone():
                    print(f"   ✓ Column '{column_name}' already exists")
                    columns_existed += 1
                else:
                    # Add the column
                    add_column_sql = f"ALTER TABLE stocks ADD COLUMN {column_name} {column_type}"
                    cursor.execute(add_column_sql)
                    print(f"   ✅ Added column '{column_name}' ({column_type})")
                    columns_added += 1

            except Exception as e:
                print(f"   ❌ Error adding column '{column_name}': {e}")
                continue

        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Columns added: {columns_added}")
        print(f"   ✓ Columns already existed: {columns_existed}")
        print(f"   📈 Total volatility columns: {len(volatility_columns)}")

        # Verify all columns now exist
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'stocks'
            AND column_name IN ('atr_14', 'atr_percentage', 'historical_volatility_1y',
                               'avg_daily_volume_20d', 'avg_daily_turnover',
                               'bid_ask_spread', 'trades_per_day')
            ORDER BY column_name
        """)

        existing_columns = cursor.fetchall()
        print(f"\n✅ VERIFICATION - Volatility columns in stocks table:")
        for col_name, col_type in existing_columns:
            print(f"   {col_name}: {col_type}")

        if len(existing_columns) == len(volatility_columns):
            print(f"\n🎉 SUCCESS: All {len(volatility_columns)} volatility columns are now available!")
            return True
        else:
            print(f"\n⚠️  WARNING: Only {len(existing_columns)}/{len(volatility_columns)} columns found")
            return False

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    success = add_volatility_columns()
    if success:
        print(f"\n✅ Database schema update completed successfully!")
        print(f"🚀 Stock screening pipeline is now ready to use!")
    else:
        print(f"\n❌ Database schema update failed!")
        exit(1)