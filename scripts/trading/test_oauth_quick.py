"""Quick OAuth2 test for Alpaca."""

from alpaca.trading.client import TradingClient

# OAuth2 credentials
api_key = "AKSF5FVOKXYKG4ATGCJQBA"
secret = "MFRGGZDFMZTWQYLCMNSGKZTHNBQWEY3EMVTGO2DBMJRWIZLGM5UGCYTDMRSWMZ3I"

print("Testing Alpaca connection with OAuth2 credentials...")
print(f"API Key: {api_key}")

try:
    # Create client
    client = TradingClient(
        api_key=api_key,
        secret_key=secret,
        paper=True,
    )

    # Get account
    account = client.get_account()

    print("\n[SUCCESS] Connected to Alpaca Paper Trading!")
    print(f"Account ID: {account.id}")
    print(f"Equity: ${float(account.equity):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Status: {account.status}")

except Exception as e:
    print(f"\n[ERROR] Connection failed: {e}")
    print(f"Error type: {type(e).__name__}")
