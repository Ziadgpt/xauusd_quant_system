import MetaTrader5 as mt5
import os

def initialize():
    path_to_terminal = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # or your actual path

    if not os.path.exists(path_to_terminal):
        print("❌ terminal64.exe not found. Check your path.")
        return False

    if not mt5.initialize(path=path_to_terminal):
        print("❌ Initialization failed:", mt5.last_error())
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Connected, but no account info. Are you logged in?")
        return False

    print("✅ Connected to MT5. Account:", account_info.login)
    return True
