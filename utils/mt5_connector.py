import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

def connect_mt5():
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not mt5.initialize():
        print(f"[❌] initialize() failed, error code: {mt5.last_error()}")
        return False

    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        print(f"[❌] login() failed, error code: {mt5.last_error()}")
        return False

    print("✅ MT5 connection successful")
    return True

def shutdown_mt5():
    mt5.shutdown()
