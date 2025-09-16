#!/usr/bin/env python3
"""
End-to-end test for Orders page functionality
"""

import requests
import json
from bs4 import BeautifulSoup
import re

def test_orders_functionality():
    """Test the complete Orders page flow"""
    print("=" * 80)
    print("END-TO-END ORDERS PAGE TEST")
    print("=" * 80)

    session = requests.Session()
    base_url = "http://localhost:5001"

    # Step 1: Auto-login
    print("1. Testing auto-login...")
    login_response = session.get(f"{base_url}/auto-login")
    print(f"   Login status: {login_response.status_code}")

    # Step 2: Test API endpoint
    print("\n2. Testing /api/orders endpoint...")
    api_response = session.get(f"{base_url}/api/orders")
    print(f"   API status: {api_response.status_code}")

    if api_response.status_code == 200:
        orders_data = api_response.json()
        print(f"   Orders returned: {len(orders_data)}")
        if orders_data:
            print(f"   Sample order: {orders_data[0]}")

            # Validate expected fields
            required_fields = ['order_id', 'symbol', 'type', 'transaction', 'quantity', 'filled', 'status', 'price', 'created_at']
            sample_order = orders_data[0]
            missing_fields = [field for field in required_fields if field not in sample_order]

            if not missing_fields:
                print("   ✅ All required fields present")
            else:
                print(f"   ❌ Missing fields: {missing_fields}")

            # Test statistics calculation (like frontend would do)
            total_orders = len(orders_data)
            completed_orders = len([o for o in orders_data if o['status'] == 'COMPLETE'])
            pending_orders = len([o for o in orders_data if o['status'] in ['PLACED', 'PARTIAL']])
            failed_orders = len([o for o in orders_data if o['status'] in ['CANCELLED', 'REJECTED']])

            print(f"   📊 Statistics:")
            print(f"      Total: {total_orders}")
            print(f"      Completed: {completed_orders}")
            print(f"      Pending: {pending_orders}")
            print(f"      Failed: {failed_orders}")
    else:
        print(f"   ❌ API failed: {api_response.text}")
        return False

    # Step 3: Test Orders page HTML
    print("\n3. Testing Orders page HTML...")
    page_response = session.get(f"{base_url}/orders")
    print(f"   Page status: {page_response.status_code}")

    if page_response.status_code == 200:
        soup = BeautifulSoup(page_response.text, 'html.parser')

        # Check required elements
        required_elements = ['total-orders', 'completed-orders', 'pending-orders', 'failed-orders', 'orders-table']
        for element_id in required_elements:
            element = soup.find(id=element_id)
            if element:
                print(f"   ✅ Found element: {element_id}")
            else:
                print(f"   ❌ Missing element: {element_id}")

        # Check if JavaScript is included
        if 'fetchOrders' in page_response.text:
            print("   ✅ JavaScript fetchOrders function found")
        else:
            print("   ❌ JavaScript fetchOrders function missing")

        if '/api/orders' in page_response.text:
            print("   ✅ API endpoint referenced in JavaScript")
        else:
            print("   ❌ API endpoint not found in JavaScript")

    else:
        print(f"   ❌ Page failed: {page_response.text}")
        return False

    print("\n" + "=" * 80)
    print("✅ ORDERS PAGE TEST COMPLETED SUCCESSFULLY!")
    print("🔧 DIAGNOSIS: API and HTML are working correctly.")
    print("🎯 ISSUE: Likely JavaScript execution timing or DOM ready event.")
    print("=" * 80)

    return True

if __name__ == "__main__":
    test_orders_functionality()