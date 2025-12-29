import json

def get_mock_vendors():
    with open("mock_vendors.json", "r") as f:
        return json.load(f)

def search_vendors_tool(product_name: str) -> str:
    """
    Searches for vendors supplying a specific product.
    Returns a JSON string of matching vendors.
    """
    vendors = get_mock_vendors()
    matches = [v for v in vendors if product_name.lower() in v["product"].lower()]
    return json.dumps(matches, indent=2)

def check_approval_tool(cost: float, limit: float) -> dict:
    """
    Checks if the cost exceeds the limit.
    """
    if cost > limit:
        return {"status": "PAUSE", "message": f"Cost {cost} exceeds limit {limit}. Awaiting Manager Approval."}
    return {"status": "OK", "message": "Within limit."}
