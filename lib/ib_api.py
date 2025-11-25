import time
import datetime
import threading
import numpy as np
import json

from ibapi.client import EClient, TickAttribLast
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from ibapi.common import OrderId
from ibapi.order_state import OrderState
from ibapi.ticktype import TickTypeEnum
# from const import data_inf_prod_consts as DC
# from lib import prod_helper
# from lib.running_lists import RunningListPstBatch, RunningListAvg
from decimal import Decimal


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class IbApi(EWrapper, EClient):
    def __init__(
        self,
        host_ip_address,
        port_number,
        client_id
    ):
        EClient.__init__(self, self)
        self.nextorderId = None
        self.connect(host_ip_address, port_number, client_id)
        time.sleep(1)
        app_run_thread = threading.Thread(target=self.run)
        app_run_thread.start()
        time.sleep(1)

        self.key_symbol__val_id = {}
        self.market_data = {'data': {}, 'complete': False}
        self.position_info = {'data': {}, 'complete': False}
        self.empty_symbol_data = {'position_size': 0, 'entry_price_by_fill': 0.0}

    # **************************************************************************************************************
    # *** public methods start *************************************************************************************
    # **************************************************************************************************************

    def get_data_historical(self, symbol: str, number_of_months: int):
        if symbol in self.key_symbol__val_id.keys():
            request_id = self.key_symbol__val_id[symbol]
        else:
            request_id = 1 if len(self.key_symbol__val_id.values()) == 0 else max(self.key_symbol__val_id.values()) + 1

        self.reqHistoricalData(
            request_id,
            self.contract_creator(symbol),
            '',
            "{} M".format(number_of_months),  # "2 M",
            "1 day",
            "TRADES",
            1,
            1,
            False,
            []
        )
        while True:
            time.sleep(3)
            print('here!!!!!!!!!!!!!')
            # check here if you have the data
            if self.market_data['complete']:
                data_to_return = self.market_data['data'].copy()
                self.market_data = {'data': {}, 'complete': False}
                return data_to_return

    def get_position_info(self, symbol: str):
        self.reqPositions()
        while True:
            time.sleep(1)
            if self.position_info['complete']:
                if symbol not in self.position_info['data'].keys():
                    self.position_info['data'][symbol] = self.empty_symbol_data
                    position_info_to_return = self.position_info['data'][symbol].copy()
                else:
                    position_info_to_return = self.position_info['data'][symbol].copy()
                self.position_info = {'data': {}, 'complete': False}
                return position_info_to_return

    def place_order(self, symbol: str, quantity_pos_neg: int):
        """
        if placing the order succeeded, return True, if failed, return False
        """
        if quantity_pos_neg > 0:
            direction = 'BUY'
        elif quantity_pos_neg < 0:
            direction = 'SELL'
        else:
            return False

        self.submit_market_order(self.contract_creator(symbol), direction, abs(quantity_pos_neg))

        return True

    # **************************************************************************************************************
    # *** public methods end ***************************************************************************************
    # **************************************************************************************************************

    # **************************************************************************************************************
    # *** private methods start ************************************************************************************
    # **************************************************************************************************************

    def submit_market_order(self, contract, direction, qty):
        # Create order object
        order = Order()
        order.action = direction
        order.totalQuantity = qty
        order.orderType = 'MKT'
        order.transmit = True
        # submit order
        self.placeOrder(self.nextorderId, contract, order)
        self.nextorderId += 1

    def submit_limit_order(self, contract, direction, qty, lmt_price):
        # Create order object
        order = Order()
        order.action = direction
        order.totalQuantity = qty
        order.orderType = 'LMT'
        order.lmtPrice = lmt_price
        order.transmit = True
        # submit order
        self.placeOrder(self.nextorderId, contract, order)
        self.nextorderId += 1

    def contract_creator(self, symbol):
        my_contract = Contract()
        my_contract.symbol = symbol
        my_contract.secType = 'STK'
        my_contract.exchange = 'SMART'
        my_contract.currency = 'USD'
        return my_contract

    # **************************************************************************************************************
    # *** private methods end **************************************************************************************
    # **************************************************************************************************************

    # **************************************************************************************************************
    # *** override methods start ***********************************************************************************
    # **************************************************************************************************************

    def position(self, account: str, contract: Contract, position: Decimal, avgCost: float):
        super().position(account, contract, position, avgCost)
        print("Position.", "Account:", account, "Symbol:", contract.symbol, "SecType:", contract.secType, "Currency:",
              contract.currency, "Position:", decimalMaxString(position), "Avg cost:", floatMaxString(avgCost))
        # Position. Account: DU2870663 Symbol: AAPL SecType: STK Currency: USD Position: 10 Avg cost: 165.35
        # Position. Account: DU2870663 Symbol: AAPL SecType: STK Currency: USD Position: -20 Avg cost: 166.51775215
        self.position_info['data'][contract.symbol] = {
            'position_size': int(decimalMaxString(position)),
            'entry_price_by_fill': float(floatMaxString(avgCost))
        }

    def positionEnd(self):
        super().positionEnd()
        print("PositionEnd")
        self.position_info['complete'] = True

    def openOrder(self, orderId, contract, order, orderState):
        print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange,
              ':', order.action, order.orderType, order.totalQuantity, orderState.status)

    def error(self, reqId, errorCode:int, errorString:str, advancedOrderRejectJson = ""):
        print('Error! reqId {}, errorCode {}, errorString {}'.format(reqId, errorCode, errorString))
        # could check for abmibuous data error here

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def historicalData(self, reqId: int, bar):  # BarData):
        # print("HistoricalData. ReqId:", reqId, "BarData.", bar)
        self.market_data['data'][int(bar.date)] = {
            'd': int(bar.date),
            'o': float(bar.open),
            'h': float(bar.high),
            'l': float(bar.low),
            'c': float(bar.close),
            'v': float(bar.volume),
            'b': float(bar.barCount)
        }

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        # self.key_symbol__val_data[self.key_id__val_symbol[reqId]]['complete'] = True

        self.market_data['complete'] = True

    def historicalDataUpdate(self, reqId: int, bar):  # : BarData):
        print("HistoricalDataUpdate. ReqId:", reqId, "BarData.", bar)
