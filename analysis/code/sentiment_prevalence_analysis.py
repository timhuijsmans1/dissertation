import json
import os
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path
from datetime import datetime

def load_file_names(file_path):
    ticker_folders = [folder for folder in os.listdir(file_path) if folder[0] != '.']
    daily_file_paths = {}
    for folder in ticker_folders:
        prediction_folder = os.path.join(file_path, folder, "predictions")
        daily_file_paths[folder] = sorted([
                os.path.join(prediction_folder, file) 
                for file in os.listdir(prediction_folder) if file[0] != '.'
        ])
    return(daily_file_paths)

def prevalence_extractor(file_path):
    with open(file_path) as f:
        data = json.load(f)
    emoticon_prevalence = data['emoticon']['prevalence']
    union_prevalence = data['union']['prevalence']
    return union_prevalence, emoticon_prevalence

def prevalence_df_builder(all_file_paths, ticker):
    all_prevalence = []
    for file_path in all_file_paths[ticker]:
        union_prevalence, emoticon_prevalence = prevalence_extractor(file_path)
        date_string = Path(file_path).stem
        date_object = datetime.strptime(date_string, '%Y-%m-%d')
        all_prevalence.append(
            (ticker, 
            union_prevalence['negative'],
            union_prevalence['positive'], 
            emoticon_prevalence['negative'],
            emoticon_prevalence['positive'],
            date_object)
        )
    prevalence_df = pd.DataFrame(
                all_prevalence, 
                columns=[
                'ticker', 
                'union_prevalence_neg',
                'union_prevalence_pos', 
                'emoticon_prevalence_neg',
                'emoticon_prevalence_pos', 
                'date'
                ]
    )
    return prevalence_df

def add_close_prices(df, ticker):
    start_date = datetime.strftime(df['date'].min(), '%Y-%m-%d')
    end_date = datetime.strftime(df['date'].max(), '%Y-%m-%d')
    ticker_data = yf.download(ticker, start_date, end_date)
    daily_returns = ticker_data['Close'].pct_change(1)
    df.set_index('date', inplace=True, drop=False)
    df['close_price'] = ticker_data['Close']
    df['daily_return'] = daily_returns
    return df

def df_builder(daily_file_paths):
    ticker_list = list(daily_file_paths.keys())
    total_df = pd.DataFrame()
    for ticker in ticker_list:
        prevalence_df = prevalence_df_builder(daily_file_paths, ticker)
        df_with_close = add_close_prices(prevalence_df, ticker)
        total_df = total_df.append(df_with_close)
    return total_df

def plot_multiple_cols(df):
    colums_to_plot = df.loc[:, ['date', 'daily_return', 'union_prevalence_pos', 'emoticon_prevalence_pos']]
    dfm = colums_to_plot.melt('date', var_name='cols', value_name='vals')
    print(dfm.head())
    return dfm

def plot_prev_returns(df):
    for stock_ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == stock_ticker]
        melt_df = plot_multiple_cols(ticker_df)
        start_date = datetime.strftime(melt_df['date'].min(), '%Y-%m-%d')
        end_date = datetime.strftime(melt_df['date'].max(), '%Y-%m-%d')

        plt.figure(figsize=(20,2.5), dpi=100)
        sns.set(style="whitegrid", rc={"lines.linewidth": 0.7})
        sns.catplot(x='date', y="vals", hue='cols', data=melt_df, kind='point', s=1)
        plt.title(f'{stock_ticker} for {start_date} to {end_date}')
        plt.savefig(f'{stock_ticker}_prevalence_returns.png', dpi=100)

if __name__ == "__main__":
    DATA_FOLDER = "../../unseen_tweet_prediction/data/pipeline_output"
    daily_file_paths = load_file_names(DATA_FOLDER)
    total_df = df_builder(daily_file_paths)
    trading_days_df = total_df.dropna()
    plot_prev_returns(trading_days_df)


    