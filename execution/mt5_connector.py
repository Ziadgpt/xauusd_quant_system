import MetaTrader5 as mt5
import os

def initialize():
    path_to_terminal = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # ✅ Update if your path is different

    if not os.path.exists(path_to_terminal):
        print("❌ terminal64.exe not found. Check your path.")
        return False

    if not mt5.initialize(path=path_to_terminal):
        print("❌ MT5 initialization failed:", mt5.last_error())
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ MT5 connected but no account info. Are you logged in?")
        return False

    print(f"✅ MT5 connected: {account_info.name} on {account_info.server}")
    return True

def shutdown():
    mt5.shutdown()
    print("🛑 MT5 shutdown complete.")
