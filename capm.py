import csv
import yahoofinancials as yf
import pandas

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
  df = df.drop(UNUSED_COLUMNS, axis=1)
  df["formatted_date"]=pandas.to_datetime(df["formatted_date"])
  df = df.rename({"formatted_date":"date"},axis=1)
  df = df.rename({"adjclose":price_column},axis=1)
  df["r_{}".format(price_column)]=df[price_column].pct_change()
  return df


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

    firm_df = create_dataframe(get_daily_price_data(ticker, start_date, end_date), ticker, 'firm')
    snp_df = create_dataframe(get_daily_price_data(SNP_TICKER, start_date, end_date), SNP_TICKER, 'market')
    us_bonds_df = create_dataframe(get_daily_price_data(US_BONDS_TICKER, start_date, end_date), US_BONDS_TICKER, 'rf')

    df = pandas.merge(pandas.merge(firm_df, snp_df, on='date'), us_bonds_df, on='date')
    df = df.dropna()
    df = df[['date','firm','market','rf','r_firm','r_market','r_rf']]
    print(df)
