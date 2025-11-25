import os
import json
import numpy as np
from lib.ib_api import IbApi
from lib import const as CONST


class AccountSimIntraday:
    def __init__(self, simulation_mode=True, real_money_mode=False):
        self.simulation_mode = simulation_mode
        self.real_money_mode = real_money_mode
        if not self.simulation_mode:
            self.ib_api = IbApi(
                host_ip_address=CONST.UTIL_TWS_HOST_IP,
                port_number=CONST.UTIL_TWS_PORT_PAPER_MONEY if not self.real_money_mode else CONST.UTIL_TWS_PORT_REAL_MONEY,
                client_id=CONST.UTIL_TWS_CLIENT_ID
            )
        else:
            self.ib_api = None
        self.key_positionid__val_positions = {}
        # self.positions_list = []

    def get_position_data(self, symbol, market_data):
        position_ids = list(self.key_positionid__val_positions.keys())
        if len(position_ids) == 0:
            position_data = {
                'symbol': symbol,
                # 'position_id': None,
                'position_size_pos_neg': 0.0,
                'age_seconds': 0.0,
                'entry_price': 0.0,
                'entry_time': 0.0,
                'pl_unrealized': 0.0,
                'pl_realized': 0.0,
                'exit_price': 0.0,

                'pl_unrealized_change': 0.0,

                'close_signal_received': False,
            }
            return position_data
        elif len(position_ids) == 1:
            position_data = self.key_positionid__val_positions[position_ids[0]].update_status(market_data)
            if position_data['close_signal_received']:
                self.key_positionid__val_positions.pop(position_ids[0])
            position_data.pop('position_id')
            return position_data
        elif len(position_ids) == 2:
            position_data_0 = self.key_positionid__val_positions[position_ids[0]].update_status(market_data)
            position_data_1 = self.key_positionid__val_positions[position_ids[1]].update_status(market_data)
            assert position_data_0['close_signal_received'] and not position_data_1['close_signal_received']
            self.key_positionid__val_positions.pop(position_ids[0])
            position_data_1['pl_realized'] = position_data_0['pl_realized']
            position_data_1['close_signal_received'] = position_data_0['close_signal_received']
            position_data_1['exit_price'] = position_data_0['exit_price']
            position_data_1.pop('position_id')
            return position_data_1
        else:
            raise RuntimeError('Error: can only have two (2) positions')

    def execute_action(self, symbol, action, market_data, number_of_points_for_last_price=3):
        prices = market_data['prices'][-number_of_points_for_last_price:]
        price = sum(prices) / len(prices)
        proposed_position_size = int(CONST.POSITION_SIZE_USD / price)
        proposed_position_size = 1 if proposed_position_size == 0 else proposed_position_size
        position_ids = list(self.key_positionid__val_positions.keys())
        number_of_positions = len(position_ids)
        assert number_of_positions == 0 or number_of_positions == 1
        next_position_id = 0 if number_of_positions == 0 else position_ids[0] + 1
        if number_of_positions == 0:
            current_position_size = 0
        else:
            position_data = self.key_positionid__val_positions[position_ids[0]].get_position_data()
            assert not position_data['close_signal_received']
            current_position_size = position_data['position_size_pos_neg']
        self._manage_positions(
            symbol,
            action,
            current_position_size,
            proposed_position_size,
            position_ids,
            next_position_id,
            market_data
        )

    def _manage_positions(
            self,
            symbol,
            action,
            current_position_size,
            proposed_position_size,
            position_ids,
            next_position_id,
            market_data
    ):
        no_action = False
        if current_position_size == 0:
            if action == 0:
                no_action = True
            elif action == 1:
                self.key_positionid__val_positions[next_position_id] = Position(
                    symbol, next_position_id, proposed_position_size)
                self.key_positionid__val_positions[next_position_id].update_status(market_data)
            elif action == 2:
                self.key_positionid__val_positions[next_position_id] = Position(
                    symbol, next_position_id, proposed_position_size * -1)
                self.key_positionid__val_positions[next_position_id].update_status(market_data)
        elif current_position_size > 0:
            if action == 0:
                self.key_positionid__val_positions[position_ids[0]].close_position()
            elif action == 1:
                no_action = True
            elif action == 2:
                self.key_positionid__val_positions[position_ids[0]].close_position()
                self.key_positionid__val_positions[next_position_id] = Position(
                    symbol, next_position_id, proposed_position_size * -1)
                self.key_positionid__val_positions[next_position_id].update_status(market_data)

        elif current_position_size < 0:
            if action == 0:
                self.key_positionid__val_positions[position_ids[0]].close_position()
            elif action == 1:
                self.key_positionid__val_positions[position_ids[0]].close_position()
                self.key_positionid__val_positions[next_position_id] = Position(
                    symbol, next_position_id, proposed_position_size)
                self.key_positionid__val_positions[next_position_id].update_status(market_data)
            elif action == 2:
                no_action = True
        return no_action


class Position:
    def __init__(
            self,
            symbol: str,
            position_id: int,
            position_size_pos_neg: int,
            transaction_fee_usd=0.30
    ):
        self.symbol = symbol
        self.position_id = position_id
        self.position_size_pos_neg = position_size_pos_neg
        self.transaction_fee_usd = transaction_fee_usd
        # ....................................................
        self.entry_price = None
        self.entry_time = None
        self.age_seconds = None
        # self.entry_price, self.entry_time = self._get_position_info(self.market_data_at_open)
        # ....................................................
        self.pl_unrealized = None
        # ....................................................
        self.pl_realized = None
        self.exit_price = None
        # ....................................................
        self.close_signal_received = False
        # ....................................................
        self.pl_unrealized_list = []
        # ....................................................
        assert self.position_size_pos_neg != 0.0

    def _get_position_info(self, market_data, number_of_points_for_last_price=3):
        prices = market_data['prices'][-number_of_points_for_last_price:]
        price = sum(prices) / len(prices)
        # price = market_data['prices'][-1]
        timestamp = market_data['timestamps'][-1]
        return price, timestamp

    def update_status(self, market_data):
        assert self.pl_realized is None and self.exit_price is None
        # assert self.close_signal_received is False

        price_now, time_now = self._get_position_info(market_data)
        # ....................................................
        if self.entry_price is None and self.entry_time is None:
            self.entry_price = price_now
            self.entry_time = time_now
        # ....................................................
        self.age_seconds = time_now - self.entry_time
        if self.position_size_pos_neg > 0.0:
            self.pl_unrealized = (price_now - self.entry_price) * abs(float(self.position_size_pos_neg))
        elif self.position_size_pos_neg < 0.0:
            self.pl_unrealized = (self.entry_price - price_now) * abs(float(self.position_size_pos_neg))
        else:
            raise RuntimeError('bad')

        self.pl_unrealized_list.append(self.pl_unrealized)

        if self.close_signal_received:
            self.pl_realized = self.pl_unrealized - self.transaction_fee_usd
            self.exit_price = price_now

        return self.get_position_data()

    def close_position(self):
        assert self.pl_realized is None and self.exit_price is None
        assert self.close_signal_received is False
        self.close_signal_received = True

    def get_position_data(self):
        if len(self.pl_unrealized_list) == 0:
            pl_unrealized_change = 0.0
        elif len(self.pl_unrealized_list) == 1:
            pl_unrealized_change = self.pl_unrealized_list[-1]
        else:
            pl_unrealized_change = self.pl_unrealized_list[-1] - self.pl_unrealized_list[-2]


        position_data = {
            'symbol': self.symbol,
            'position_id': self.position_id,
            'position_size_pos_neg': self.position_size_pos_neg,
            'age_seconds': self.age_seconds,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'pl_unrealized': self.pl_unrealized,
            'pl_realized': 0.0 if self.pl_realized is None else self.pl_realized,
            'exit_price': self.exit_price,

            'pl_unrealized_change': pl_unrealized_change,

            'close_signal_received': self.close_signal_received,
        }
        return position_data.copy()
