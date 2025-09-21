#!/usr/bin/env python3
"""
Script to test the live API deployment on Render.com
This script tests both GET and POST endpoints and prints results.
"""

import requests
import json
import sys

def test_get_endpoint(base_url):
    """Test the GET / endpoint"""
    print("Testing GET / endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing GET endpoint: {e}")
        return False

def test_post_endpoint(base_url):
    """Test the POST /predict endpoint"""
    print("\nTesting POST /predict endpoint...")
    
    # Test data for low income prediction
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 30,
        "native-country": "United-States"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Check if prediction is valid
        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                prediction = result["prediction"]
                print(f"Model Inference Result: {prediction}")
                return prediction in ["<=50K", ">50K"]
            else:
                print("No prediction found in response")
                return False
        return False
    except Exception as e:
        print(f"Error testing POST endpoint: {e}")
        return False

def main():
    """Main function to test the live API"""
    if len(sys.argv) != 2:
        print("Usage: python test_live_api.py <API_BASE_URL>")
        print("Example: python test_live_api.py https://census-income-api.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    print(f"Testing API at: {base_url}")
    print("=" * 50)
    
    # Test GET endpoint
    get_success = test_get_endpoint(base_url)
    
    # Test POST endpoint
    post_success = test_post_endpoint(base_url)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"GET / endpoint: {'PASS' if get_success else 'FAIL'}")
    print(f"POST /predict endpoint: {'PASS' if post_success else 'FAIL'}")
    
    if get_success and post_success:
        print("\n✅ All tests passed! API is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the API deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
