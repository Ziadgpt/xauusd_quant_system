import MetaTrader5 as mt5
import time

def initialize():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    else:
        print("MT5 initialized successfully")
        return True

def shutdown():
    mt5.shutdown()

def check_connection():
    info = mt5.terminal_info()
    print("Connected to account:", info.login)
