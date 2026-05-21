import datetime
import warnings
import backtrader as bt
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


class MaCrossoverStrategy(bt.Strategy):
    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.slow_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.close()


def run_backtest(data_frame, fast, slow, starting_cash=100000.0):
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data_frame)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(MaCrossoverStrategy, fast_period=fast, slow_period=slow)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    max_dd = strat.analyzers.drawdown.get_analysis().max.drawdown
    final_value = cerebro.broker.getvalue()
    percentage_return = ((final_value - starting_cash) / starting_cash) * 100

    return {
        "final_value": final_value,
        "percentage_return": percentage_return,
        "max_drawdown": max_dd,
    }


def execute_walk_forward(df, fast_options, slow_options):
    df["Year"] = df.index.year
    years = sorted(df["Year"].unique())

    oos_results = []
    total_is_returns = 0
    total_oos_returns = 0

    for i in range(len(years) - 3):
        is_years = years[i : i + 3]
        oos_year = years[i + 3]

        is_data = df[df["Year"].isin(is_years)]
        oos_data = df[df["Year"] == oos_year]

        best_fast, best_slow = None, None
        best_is_return = -float("inf")

        for fast in fast_options:
            for slow in slow_options:
                if fast >= slow:
                    continue

                metrics = run_backtest(is_data, fast, slow)
                if metrics["percentage_return"] > best_is_return:
                    best_is_return = metrics["percentage_return"]
                    best_fast = fast
                    best_slow = slow

        oos_metrics = run_backtest(oos_data, best_fast, best_slow)
        oos_results.append(oos_metrics)
        total_is_returns += best_is_return
        total_oos_returns += oos_metrics["percentage_return"]

    avg_is = total_is_returns / len(oos_results) if oos_results else 0
    avg_oos = total_oos_returns / len(oos_results) if oos_results else 0
    wfe = (avg_oos / avg_is) if avg_is != 0 else 0

    return oos_results, wfe


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2021-01-01"
    end_date = "2026-01-01"

    raw_data = yf.download(ticker, start=start_date, end=end_date)

    if not raw_data.empty:
        fast_grid = [5, 10, 15, 20]
        slow_grid = [20, 30, 40, 50]

        oos_performances, wf_efficiency = execute_walk_forward(
            raw_data, fast_grid, slow_grid
        )

        final_total_return = sum(r["percentage_return"] for r in oos_performances)
        worst_drawdown = max(r["max_drawdown"] for r in oos_performances)

        base_robustness_score = 65
        if 0.6 < wf_efficiency < 1.2:
            base_robustness_score += 15

        positive_windows = sum(
            1 for r in oos_performances if r["percentage_return"] > 0
        )
        base_robustness_score += (positive_windows / len(oos_performances)) * 15
        final_robustness_score = min(max(base_robustness_score, 0), 100)

        print("\n| Metric | Value |")
        print("| :--- | :--- |")
        print(f"| **Stock Symbol** | {ticker} |")
        print(f"| **Percentage Return on Capital** | {final_total_return:.2f}% |")
        print(f"| **Maximum Drawdown** | -{worst_drawdown:.2f}% |")
        print(f"| **Walk-Forward Analysis Score** | WFE: {wf_efficiency:.2f} |")
        print(f"| **Robustness Score** | {final_robustness_score:.0f} |")
    