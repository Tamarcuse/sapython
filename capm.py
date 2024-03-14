import csv
import yahoofinancials as yf
import pandas
from statsmodels.formula.api import ols

COLUMNS = {
  "firm": 0,
  "ticker": 1,
  "start_date": 2,
  "end_date": 3
}

SNP_TICKER = "SPY"
US_BONDS_TICKER = "^IRX"
UNUSED_COLUMNS = ['date','high','low','open','close','volume']

def get_daily_price_data(ticker, start, end):
  return yf.YahooFinancials(ticker).get_historical_price_data(
    start_date=start,
    end_date=end,
    time_interval='daily'
  )

def create_dataframe(price_data, ticker, price_column):
  df = pandas.DataFrame(price_data[ticker]['prices'])
  df = df.dropna()
  df = df.drop(UNUSED_COLUMNS, axis=1)
  df["formatted_date"]=pandas.to_datetime(df["formatted_date"])
  df = df.rename({"formatted_date":"date"},axis=1)
  df = df.rename({"adjclose":price_column},axis=1)
  return df

def create_price_data_frame(price_data, ticker, price_column):
  df = create_dataframe(price_data, ticker, price_column)
  df["r_{}".format(price_column)]=df[price_column].pct_change()
  return df

def create_return_data_frame(price_data, ticker, price_column):
  df = create_dataframe(price_data, ticker, price_column)
  df[price_column]=df[price_column]/100
  df["r_{}".format(price_column)]=((1+df[price_column]) ** (1/365)) - 1
  return df

def calc_sharpe(price_data):
  r_firm_avg = price_data['r_firm'].mean()
  r_rf_avg = price_data['r_rf'].mean()
  r_firm_std_dev = price_data['r_firm'].std()
  return (r_firm_avg - r_rf_avg) / r_firm_std_dev

def calc_treynor(price_data, beta):
  r_firm_avg = price_data['r_firm'].mean()
  r_rf_avg = price_data['r_rf'].mean()
  return (r_firm_avg - r_rf_avg) / beta

def calc_yearly_return(df):
  first_value = df.iloc[0]['firm']
  last_value = df.iloc[-1]['firm']
  r_total = (last_value - first_value) - 1

  first_date = df.iloc[0]['date']
  last_date = df.iloc[-1]['date']
  num_of_days = (last_date-first_date).days

  return ((1 + r_total) ** (365 / num_of_days)) - 1


with open('playground.csv', 'r', newline='') as csvfile:
  csv_reader = csv.reader(csvfile)
  # skip first row (headers)
  next(csv_reader)

  for row in csv_reader:
    # skip empty lines
    if not row:
      continue

    ticker = row[COLUMNS["ticker"]]
    start_date = row[COLUMNS["start_date"]]
    end_date = row[COLUMNS["end_date"]]

    firm_df = create_price_data_frame(get_daily_price_data(ticker, start_date, end_date), ticker, 'firm')
    snp_df = create_price_data_frame(get_daily_price_data(SNP_TICKER, start_date, end_date), SNP_TICKER, 'market')
    us_bonds_df = create_return_data_frame(get_daily_price_data(US_BONDS_TICKER, start_date, end_date), US_BONDS_TICKER, 'rf')

    df = pandas.merge(pandas.merge(firm_df, snp_df, on='date'), us_bonds_df, on='date')
    df = df.dropna()
    df = df[['date','firm','market','rf','r_firm','r_market', 'r_rf']]

    model = ols(formula='r_firm - r_rf ~ r_market - r_rf', data=df).fit()
    alpha = model.params['Intercept']
    beta = model.params['r_market']

    sharpe_ratio = calc_sharpe(df)
    treynor_ratio = calc_treynor(df, beta)
    yearly_return = calc_yearly_return(df)

    print(yearly_return)
