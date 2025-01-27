from dotenv import load_dotenv
import os
import logging
import datetime
import time
from typing import Optional, Tuple, Dict
from threading import Lock
import requests

from binance import Client
import pandas as pd
import pandas_ta as ta

load_dotenv()

# Configuration
CONFIG = {
    'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
    'API_KEY': os.getenv('API_KEY'),
    'API_SECRET': os.getenv('API_SECRET'),
    'SYMBOLS': ['BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'SOLUSDT', 'TRUMPUSDT',
                '1000PEPEUSDT', 'XRPUSDT', 'PENGUUSDT', 'BNBUSDT', 'ADAUSDT',
                'VINEUSDT', 'PNUTUSDT', 'RUNEUSDT', 'LTCUSDT', 'ANIMEUSDT',
                'WIFUSDT', 'LINKUSDT', '1000SHIBUSDT', 'HBARUSDT'],
    'INTERVAL': '1h',
    'LEVERAGE': 15,
    'CAPITAL': 1,
    'CAPITAL_PERCENTAGE_PER_TRADE': 0.6,
    'TAKE_PROFIT_PERCENTAGE': 4,
    'STOP_LOSS_PERCENTAGE': -2,
    'MIN_BALANCE': 0.6,
    'MAX_RUNTIME_HOURS': 24,
    'MONITORING_INTERVAL': 3600
}

# Logging Configuration
logging.basicConfig(
    filename='trading_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class BinanceTradingBot:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG
        self.active_positions = {}
        self.positions_lock = Lock()
        try:
            self.client = self._initialize_client()
            self._setup_leverage()
        except Exception as e:
            logging.critical(f"Bot initialization error: {e}")
            raise

    def _initialize_client(self) -> Client:
        """Initialize Binance client with error handling."""
        try:
            client = Client(self.config['API_KEY'], self.config['API_SECRET'])
            client.ping()
            return client
        except Exception as e:
            logging.critical(f"Failed to initialize Binance client: {e}")
            raise

    def _setup_leverage(self):
        """Set leverage for all trading symbols."""
        for symbol in self.config['SYMBOLS']:
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=self.config['LEVERAGE'])
            except Exception as e:
                logging.error(f"Leverage setup failed for {symbol}: {e}")

    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve and process historical price data."""
        try:
            candles = self.client.futures_klines(symbol=symbol, interval=self.config['INTERVAL'], limit=500)
            df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume', '', '', '', '', '', ''])
            df.to_csv(f"{symbol}_historical_data.csv", index=False)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')

            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df['macd'], df['signal'] = macd.iloc[:, 0], macd.iloc[:, 1]
            else:
                df['macd'], df['signal'] = None, None

            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx.iloc[:, 0] if adx is not None else None

            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None:
                df['upper_band'], df['middle_band'], df['lower_band'] = bbands.iloc[:, 0], bbands.iloc[:, 1], bbands.iloc[:, 2]
            else:
                df['upper_band'], df['middle_band'], df['lower_band'] = None, None, None

            df['sma_50'] = ta.sma(df['close'], length=50)

            return df
        except Exception as e:
            logging.error(f"Data retrieval failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_available_balance(self, asset: str = 'USDT') -> float:
        """Get available trading balance."""
        try:
            balance_info = self.client.futures_account_balance()
            for asset_info in balance_info:
                if asset_info['asset'] == asset:
                    return float(asset_info['availableBalance'])
            return 0.0
        except Exception as e:
            logging.error(f"Balance retrieval failed: {e}")
            return 0.0

    def send_telegram_message(self, message: str):
        """Send a message to Telegram."""
        url = f"https://api.telegram.org/bot{self.config['TELEGRAM_TOKEN']}/sendMessage"
        payload = {
            'chat_id': self.config['TELEGRAM_CHAT_ID'],
            'text': message
        }
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                logging.info("Mensaje enviado a Telegram con éxito.")
            else:
                logging.error(f"Error al enviar mensaje a Telegram: {response.text}")
        except Exception as e:
            logging.error(f"Excepción al enviar mensaje a Telegram: {e}")

    def check_and_close_positions(self) -> None:
        """Check all active positions and close them if conditions are met."""
        with self.positions_lock:
            positions_to_check = self.active_positions.copy()
        
        for symbol, position_data in positions_to_check.items():
            try:
                current_price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
                entry_price = float(position_data['entry_price'])
                position_size = float(position_data['position_size'])
                side = position_data['side']
                
                roi = ((current_price - entry_price) / entry_price) * 100 if side == 'BUY' else \
                      ((entry_price - current_price) / entry_price) * 100
                roi *= self.config['LEVERAGE']
                
                # Print ROI information
                print(f"ROI para {symbol}: {roi:.4f}%, a las {datetime.datetime.now()}")
                
                close_condition = (roi >= self.config['TAKE_PROFIT_PERCENTAGE'] or 
                                 roi <= self.config['STOP_LOSS_PERCENTAGE'])
                
                if close_condition:
                    close_side = 'SELL' if side == 'BUY' else 'BUY'
                    self.client.futures_create_order(
                        symbol=symbol,
                        side=close_side,
                        type='MARKET',
                        quantity=abs(position_size)
                    )
                    
                    profit_loss_value = (current_price - entry_price) * position_size if side == 'BUY' \
                                      else (entry_price - current_price) * position_size
                    
                    logging.info(f"Position closed for {symbol}. ROI: {roi:.2f}%")
                    self.send_telegram_message(
                        f"Operación cerrada:\nSímbolo: {symbol}\n"
                        f"Lado: {side}\nROI: {roi:.2f}%\n"
                        f"P/L: {profit_loss_value:.2f} USDT"
                    )
                    
                    with self.positions_lock:
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                            
            except Exception as e:
                logging.error(f"Error checking position for {symbol}: {e}")

    def place_futures_order(self, symbol: str, side: str) -> Optional[Tuple]:
        """Execute market order with dynamic capital allocation."""
        try:
            current_balance = self.get_available_balance()
            trade_amount = current_balance * self.config['CAPITAL_PERCENTAGE_PER_TRADE']
            
            price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == symbol)
            filters = {f['filterType']: f for f in symbol_info['filters']}
            quantity_precision = symbol_info['quantityPrecision']
            
            notional_value = trade_amount * self.config['LEVERAGE']
            quantity = notional_value / price
            min_qty = float(filters['LOT_SIZE']['minQty'])
            step_size = float(filters['LOT_SIZE']['stepSize'])
            
            if quantity < min_qty:
                logging.warning(f"Quantity too low for {symbol}")
                return None
            
            quantity = round((quantity // step_size) * step_size, quantity_precision)
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            logging.info(f"Order placed: {side} {quantity} {symbol} at price {price}")
            
            message = f"📊 Nueva Orden:\n" \
                     f"Crypto: {symbol}\n" \
                     f"Tipo: {side}\n" \
                     f"Cantidad: {quantity}\n" \
                     f"Precio: {price}"
            self.send_telegram_message(message)
            
            # Store position information
            with self.positions_lock:
                self.active_positions[symbol] = {
                    'side': side,
                    'entry_price': price,
                    'position_size': quantity,
                    'entry_time': datetime.datetime.now()
                }
            
            return order, price
            
        except Exception as e:
            logging.error(f"Order placement failed for {symbol}: {e}")
            return None

    def run(self):
        """Main trading loop with synchronized ROI monitoring."""
        start_time = datetime.datetime.now()
        last_check_time = datetime.datetime.now()
        
        print("Starting bot...")
        logging.info("Bot started")
        
        while datetime.datetime.now() - start_time < datetime.timedelta(hours=self.config['MAX_RUNTIME_HOURS']):
            try:
                current_time = datetime.datetime.now()
                
                # Check if it's time for a new cycle
                if (current_time - last_check_time).total_seconds() >= self.config['MONITORING_INTERVAL']:
                    print("-------------Starting New Cycle-------------")
                    
                    # First, check all active positions
                    self.check_and_close_positions()
                    
                    # Then check for new trading opportunities
                    available_balance = self.get_available_balance()
                    if available_balance >= self.config['MIN_BALANCE']:
                        for symbol in self.config['SYMBOLS']:
                            with self.positions_lock:
                                if symbol in self.active_positions:
                                    continue
                                    
                            df = self.get_historical_data(symbol)
                            if df.empty:
                                continue
                                
                            latest = df.iloc[-1]
                            if latest['adx'] > 30:
                                if (latest['rsi'] < 50 and
                                    latest['close'] < latest['lower_band'] and
                                    latest['macd'] > latest['signal'] and
                                    latest['close'] > latest['sma_50']) :
                                    logging.INFO(f"Nueva señal de compra para {symbol} a las {current_time}")
                                    self.place_futures_order(symbol, 'BUY')
                                    
                                elif (latest['rsi'] > 50 and
                                    latest['close'] > latest['upper_band'] and
                                    latest['macd'] < latest['signal'] and
                                    latest['close'] < latest['sma_50']):
                                    logging.INFO(f"Nueva señal de venta para {symbol} a las {current_time}")
                                    self.place_futures_order(symbol, 'SELL')
                    
                    last_check_time = current_time
                
                time.sleep(1)  # Short sleep to prevent CPU overuse
                
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                print(f"Critical Error: {e}")
                time.sleep(self.config['MONITORING_INTERVAL'])

def main():
    try:
        bot = BinanceTradingBot()
        bot.run()
    except Exception as e:
        logging.critical(f"Bot initialization failed: {e}")

if __name__ == "__main__":
    main()