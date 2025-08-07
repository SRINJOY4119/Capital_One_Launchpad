import requests
import sys
import json

API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001a52cfa0cf9df446369ab0b90dbcd0df1"

def fetch_market_price(state_name):
    print(f"[INFO] Fetching market prices for state: {state_name}")
    params = {
        "api-key": API_KEY,
        "format": "json",
        "offset": "0",
        "limit": "4000",
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return json.dumps({"message": "Failed to fetch data"}, indent=4)
    
    records = data.get("records", [])
    filtered = []
    
    for record in records:
        if record.get("state", "").strip().lower() == state_name.strip().lower():
            filtered.append({
                "state": record.get("state"),
                "district": record.get("district"),
                "market": record.get("market"),
                "commodity": record.get("commodity"),
                "variety": record.get("variety"),
                "min_price": record.get("min_price"),
                "max_price": record.get("max_price"),
                "modal_price": record.get("modal_price"),
            })
    
    if filtered:
        print(f"[INFO] Found {len(filtered)} records for state: {state_name}\n")
        json_output = json.dumps(filtered, indent=4)
    else:
        message = {"message": "No records were found"}
        print(f"[INFO] {message['message']}")
        json_output = json.dumps(message, indent=4)

    print(json_output)
    return json_output

if __name__ == "__main__":
    state = 'Karnataka' if len(sys.argv) != 2 else sys.argv[1]
    fetch_market_price(state)
