#!/usr/bin/env python3
"""
Test if the orders page is working correctly
"""
import requests
import time
from bs4 import BeautifulSoup

def test_orders_page():
    """Test the orders page functionality."""
    print("🧪 Testing Orders Page Functionality\n")

    # Test 1: API endpoint
    print("1️⃣ Testing API endpoint...")
    try:
        response = requests.get('http://localhost:5001/api/orders')
        if response.status_code == 200:
            orders = response.json()
            print(f"   ✅ API working - {len(orders)} orders returned")
            if orders:
                order = orders[0]
                print(f"   📋 Sample order: {order['symbol']} - {order['status']} - ₹{order['price']}")
        else:
            print(f"   ❌ API failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ API error: {e}")

    print()

    # Test 2: Orders page HTML
    print("2️⃣ Testing Orders page HTML...")
    try:
        response = requests.get('http://localhost:5001/orders')
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Check for essential elements
            total_orders = soup.find(id='total-orders')
            orders_table = soup.find(id='orders-table')

            print(f"   ✅ Page loaded successfully")
            print(f"   📊 Total orders element: {'✅ Found' if total_orders else '❌ Missing'}")
            print(f"   📋 Orders table element: {'✅ Found' if orders_table else '❌ Missing'}")

            # Check if JavaScript is included
            scripts = soup.find_all('script')
            js_included = any('fetchOrders' in str(script) for script in scripts)
            print(f"   🔧 JavaScript included: {'✅ Yes' if js_included else '❌ No'}")

            # Check current values (before JS execution)
            if total_orders:
                current_value = total_orders.text.strip()
                print(f"   📈 Current total orders value: {current_value}")
        else:
            print(f"   ❌ Page failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Page error: {e}")

    print()

    # Test 3: Expected behavior
    print("3️⃣ Expected behavior analysis:")
    print("   📝 The page shows '0' initially (before JavaScript runs)")
    print("   ⚡ JavaScript should fetch data from /api/orders and update the display")
    print("   🔄 Auto-refresh should update data every 30 seconds")
    print("   💡 If you're seeing '0', try:")
    print("      - Hard refresh (Ctrl+F5 or Cmd+Shift+R)")
    print("      - Check browser console for JavaScript errors")
    print("      - Ensure JavaScript is enabled")

def main():
    try:
        test_orders_page()
        print(f"\n🌐 Visit http://localhost:5001/orders to see the page")
        print(f"📊 Open browser developer tools to see JavaScript logs")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app at http://localhost:5001")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == '__main__':
    main()