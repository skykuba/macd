import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



class EMA:
    @staticmethod
    def generate_geometric_seq(a: float, q: float, n: int) -> list:
        seq = [a]
        for i in range(1, n):
            seq.append(seq[-1] * q)
        return seq

    @staticmethod
    def append_zeros(lst: list, n: int) -> list:
        return lst + [0] * n

    def calculate_ema(self, prices: list, n: int) -> list:
        ema = []
        alpha = 2 / (n + 1)
        dominator = self.generate_geometric_seq(1, 1 - alpha, n + 1)
        buffer=self.append_zeros([prices[0]],n)

        for i in range(len(prices)):
            if i == 0:
                ema.append(prices[i])
            else:
                if i < n:
                    numerator = sum(dominator[j] * buffer[j] for j in range(i + 1))
                    current_denominator = sum(dominator[:i + 1])
                else:
                    numerator = sum(dominator[j] * buffer[j] for j in range(n + 1))
                    current_denominator = sum(dominator)

                ema.append(numerator/current_denominator)

            if i + 1 < len(prices):
                buffer.insert(0, prices[i + 1])
                buffer.pop()

        return ema

class Trading:
    def __init__(self,capital: float,name_stock: str) ->None:
        self.capital=capital
        self.name_stock=name_stock
        self.reset_state()

    def reset_state(self) -> None:
        self.account=self.capital
        self.shares=0
        self.buy_points=[]
        self.sell_points=[]
        self.loss_deals=0
        self.total_deals=0
        self.max_loss=0
        self.account_history=[]

    def execute(self,macd: list, signal: list, prices: list, dates:pd.Series) ->None:
        self.macd=macd
        self.signal=signal
        self.prices=prices
        self.dates=dates
        for i in range(1,len(macd)):
            if self.is_cross(i):
                self.transaction2(i)
            self.account_history.append(self.shares*self.prices[i]+self.account)

    def is_cross(self, i: int) ->bool:
        return (self.macd[i-1]-self.signal[i-1])*(self.macd[i]-self.signal[i])<0

    def transaction(self, i: int) -> None:
        if self.macd[i-1]>self.signal[i-1] and i>36 :
            self.sell_points.append((i,self.macd[i]))
            if self.shares>0 :
                self.sell(i)
                self.total_deals+=1
        elif self.macd[i-1]<self.signal[i-1] and i>36:
            self.buy_points.append((i,self.macd[i]))
            if self.account>0:
                self.buy(i)
                self.total_deals+=1

    def transaction2(self, i: int) -> None:
        if self.macd[i-1]>self.signal[i-1] and i>36 :
            if self.buy_points and i -self.buy_points[-1][0]>10 and self.shares>0:
                self.sell_points.append((i,self.macd[i]))
                self.sell(i)
                self.total_deals+=1
        elif self.macd[i-1]<self.signal[i-1] and i>36:
            if self.account>0:
                self.buy_points.append((i,self.macd[i]))
                self.buy(i)
                self.total_deals+=1

    def buy(self,i:int)-> None:
        self.shares=self.account/self.prices[i]
        self.account=0
        print(f'Zakup dnia {self.dates[i]}')

    def sell(self,i: int) ->None:
        self.account=self.prices[i]*self.shares
        self.calculate_profits(i)
        self.shares=0

    def calculate_profits(self,i: int) ->None:
        last_buy_index=self.buy_points[-1][0]
        buy_price=self.prices[last_buy_index]
        sell_price=self.prices[i]
        if sell_price<buy_price:
            loss=(buy_price-sell_price)*self.shares
            self.max_loss=max(loss,self.max_loss)
            self.loss_deals+=1
            print(f'Strata dnia {self.dates[i-1]}: -{round(loss,2)}')
        else:
            profit=(sell_price-buy_price)*self.shares
            print(f'Zysk dnia {self.dates[i-1]}:   +{round(profit,2)}')

    def plot_wallet(self):
        plt.figure(figsize=(19, 7))
        plt.plot(self.dates[1:], self.account_history, label='Account Balance')
        plt.axhline(y=self.capital, color='gray', linestyle='--', label='Initial Capital')
        plt.title('Investment Wallet Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.show()

    def get_account_history_and_capital(self):
        return self.account_history, self.capital

    def get_max_loss(self):
        return self.max_loss



    def generate_transaction_table(self, file_path: str) -> pd.DataFrame:
            transactions = []
            for buy, sell in zip(self.buy_points, self.sell_points):
                buy_date = self.dates[buy[0]]
                buy_price = self.prices[buy[0]]
                sell_date = self.dates[sell[0]]
                sell_price = self.prices[sell[0]]
                profit_loss = round((sell_price - buy_price) * (self.capital / buy_price), 2)
                transactions.append({
                    'Buy Date': buy_date,
                    'Buy Price': round(buy_price, 2),
                    'Sell Date': sell_date,
                    'Sell Price': round(sell_price, 2),
                    'Profit/Loss': profit_loss
                })
            df = pd.DataFrame(transactions)
            df.to_csv(file_path, index=False)
            return df

class DataVisualizer:
    def __init__(self, data: pd.DataFrame, name_stock: str) -> None:
        self.data = data
        self.dates = pd.to_datetime(data['Date'])
        self.name_stock = name_stock

    def plot_chart(self) -> None:
        plt.figure(figsize=(19, 7))
        plt.plot(self.dates, self.data['Close'], label='Closing Prices')
        plt.title(self.name_stock + ' Closing Prices', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_macd(self, macd: list, signal: list, sell_points: list, buy_points: list) -> None:
        plt.figure(figsize=(19, 7))
        plt.plot(self.dates, macd, label='MACD', color='blue')
        plt.plot(self.dates, signal, label='Signal Line', color='red')
        self._plot_transaction_points(sell_points, buy_points)
        plt.title('MACD and Signal Line for ' + self.name_stock, fontsize=16)
        plt.xlabel('Date')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_price_chart(self, prices: list, sell_points: list, buy_points: list) -> None:
        plt.figure(figsize=(19, 7))
        plt.plot(self.dates, prices, label='Closing Prices', color='black')
        self._plot_transaction_points(sell_points, buy_points, prices)
        plt.title('Price Chart with Transactions of ' + self.name_stock, fontsize=16)
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.show()

    def _plot_transaction_points(self, sell_points: list, buy_points: list, prices: list = None) -> None:
        buy_dates = [self.dates[i] for i, _ in buy_points]
        sell_dates = [self.dates[i] for i, _ in sell_points]

        if prices is not None:
            buy_prices = [prices[i] for i, _ in buy_points]
            sell_prices = [prices[i] for i, _ in sell_points]
            plt.scatter(buy_dates, buy_prices, color='green', s=100, label='Buy', marker='^')
            plt.scatter(sell_dates, sell_prices, color='red', s=100, label='Sell', marker='v')
        else:
            plt.scatter(buy_dates, [y for _, y in buy_points], color='green', s=100, label='Buy', marker='^')
            plt.scatter(sell_dates, [y for _, y in sell_points], color='red', s=100, label='Sell', marker='v')

    def plot_price_and_wallet(self, prices: list, account_history: list, capital: float) -> None:
        fig, ax1 = plt.subplots(figsize=(19, 7))

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price', color='tab:blue')
        ax1.plot(self.dates[1:], prices[1:], label='Stock Price', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Account Balance', color='tab:green')
        ax2.plot(self.dates[1:], account_history, label='Account Balance', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        ax2.axhline(y=capital, color='gray', linestyle='--', label='Initial Capital')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        plt.title('Stock Price and Investment Wallet Over Time',fontsize=19)
        plt.show()

    def plot_transactions_in_period(self, start_date: str, end_date: str, prices: list, sell_points: list,
                                    buy_points: list) -> None:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        start_index = self.dates[self.dates >= start_date].index[0]
        end_index = self.dates[self.dates <= end_date].index[-1]

        chart_dates = self.dates[start_index:end_index + 1]
        chart_prices = prices[start_index:end_index + 1]

        filtered_sell_points = [(i, price) for i, price in sell_points if start_index <= i <= end_index]
        filtered_buy_points = [(i, price) for i, price in buy_points if start_index <= i <= end_index]

        plt.figure(figsize=(19, 7))
        plt.plot(chart_dates, chart_prices, label='Closing Prices', color='black')

        if filtered_buy_points:
            buy_dates = [self.dates[i] for i, _ in filtered_buy_points]
            buy_prices = [prices[i] for i, _ in filtered_buy_points]
            plt.scatter(buy_dates, buy_prices, color='green', s=100, label='Buy', marker='^')

        if filtered_sell_points:
            sell_dates = [self.dates[i] for i, _ in filtered_sell_points]
            sell_prices = [prices[i] for i, _ in filtered_sell_points]
            plt.scatter(sell_dates, sell_prices, color='red', s=100, label='Sell', marker='v')

        plt.title('Price Chart with Transactions of ' + self.name_stock,fontsize=16)
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.show()

    def plot_max_loss(self, max_loss_history: list) -> None:
        plt.figure(figsize=(19, 7))
        plt.plot(self.dates[1:], max_loss_history, label='Max Loss')
        plt.title('Max Loss Over Time')
        plt.xlabel('Date')
        plt.ylabel('Max Loss')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('data/spx_d.csv',delimiter=';')
    prices = data['Close'].tolist()
    print(prices[1])
    first_dat_traiding=36
    ema_calculator = EMA()
    macd = [a - b for a, b in zip(
        ema_calculator.calculate_ema(prices, 12),
        ema_calculator.calculate_ema(prices, 26)
    )]
    signal = ema_calculator.calculate_ema(macd, 9)
    capital=prices[first_dat_traiding]*1000

    strategy = Trading(capital=capital,name_stock="INTEL")
    strategy.execute(macd, signal, prices, data['Date'])
    net_worth = strategy.account + strategy.shares * prices[-1]
    print(f'\nKapital poczatkowy: {round(strategy.capital,2)}')
    print(f"Kapitał końcowy: {round(strategy.account, 2)}")
    print(f"Ilość akcji: {round(strategy.shares, 2)}")
    print(f"Wartość netto: {round(net_worth, 2)}")
    change = (prices[-1]-prices[first_dat_traiding]) / prices[first_dat_traiding] * 100
    profit_loss = (net_worth - strategy.capital) / strategy.capital * 100

    print(f"Wartość o jaką wzrósł/zmalał dany instrument finansowy {change:.2f}%")
    print(f"Zysk/Strata: {round(profit_loss, 2)}%")
    print(f"Procent stratnych transakcji: {round(strategy.loss_deals / strategy.total_deals * 100, 2)}%")
    print(f"Ilosc stratnych transakcji: {strategy.loss_deals}, ilośc trandukcji: {strategy.total_deals}")
    account_history,capital=strategy.get_account_history_and_capital()
    transaction_table = strategy.generate_transaction_table('transactions.csv')

    visualizer = DataVisualizer(data,"INTEL")
    visualizer.plot_chart()
    visualizer.plot_macd(macd, signal, strategy.sell_points, strategy.buy_points)
    visualizer.plot_price_chart(prices, strategy.sell_points, strategy.buy_points)
    strategy.plot_wallet()
    visualizer.plot_price_and_wallet(prices,account_history, capital)
    visualizer.plot_transactions_in_period('2022-07-10', '2022-11-18', prices, strategy.sell_points, strategy.buy_points)
