import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from scipy.stats import zscore
from datetime import datetime, timedelta 
from io import BytesIO
from textblob import TextBlob
import base64
import plotly.graph_objs as go
import plotly.io as pio
import requests
import plotly.express as px
import csv

def name_to_symbol_csv(company_name, csv_path=os.path.join(os.path.dirname(__file__), 'symbols.csv')):
    company_name = company_name.strip().lower()
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Exact match (case-insensitive)
            for row in reader:
                if row['name'].strip().lower() == company_name:
                    symbol = row['symbol'].replace('$', '').strip()
                    return symbol
            csvfile.seek(0)
            next(reader)  # skip header
            # Partial match (case-insensitive)
            for row in reader:
                if company_name in row['name'].strip().lower():
                    symbol = row['symbol'].replace('$', '').strip()
                    return symbol
    except Exception as e:
        print("CSV symbol lookup error:", e)
    return None

def get_valid_yf_symbol(symbol):
    for sym_try in [symbol, symbol + '.NS', symbol + '.BO']:
        try:
            ticker = yf.Ticker(sym_try)
            info = ticker.info
            # If info dict has 'regularMarketPrice', it's a valid symbol
            if info and info.get('regularMarketPrice', None) is not None:
                return sym_try, ticker
        except Exception:
            continue
    return None, None

def generate_chart(df, period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='#5c6bc0', width=2),
        name='Close',
        customdata=np.stack([df['Open'], df['High'], df['Low'], df['Volume']], axis=-1),
        hovertemplate=
            'Date: %{x|%m/%d/%Y}<br>' +
            'Close: %{y:.2f}<br>' +
            'Open: %{customdata[0]:.2f}<br>' +
            'High: %{customdata[1]:.2f}<br>' +
            'Low: %{customdata[2]:.2f}<br>' +
            'Volume: %{customdata[3]:,}<extra></extra>'
    ))
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            showgrid=False,
            tickformat='%b %d' if period in ['1m'] else '%b %Y',
            tickfont=dict(size=12),
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            tickfont=dict(size=12),
            title=None
        ),
        showlegend=False,
        height=300
    )
    return fig.to_html(full_html=False)

def fetch_sentiment(symbol):
    news_items = fetch_news(symbol)
    if not news_items:
        return 0.0  # Neutral sentiment if no news
    sentiments = []
    for item in news_items:
        headline = item.get('title', '') or ''
        if headline:
            blob = TextBlob(headline)
            sentiments.append(blob.sentiment.polarity)
    if sentiments:
        return float(np.mean(sentiments))
    return 0.0

NEWSAPI_KEY = '0dcd7ffa3ba64dc6880f58e229a2b9e5'  # <-- Replace with your NewsAPI key

def fetch_news(symbol):
    news = []
    try:
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
        company_name = info.get('shortName', '').strip()
    except Exception:
        company_name = ''

    if company_name and company_name.lower() != symbol.lower():
        query = f'"{company_name}" OR "{symbol}"'
    else:
        query = symbol

    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    url = (
        f'https://newsapi.org/v2/everything?'
        f'q={query}&'
        f'from={from_date}&'
        f'to={to_date}&'
        f'sortBy=publishedAt&'
        f'language=en&'
        f'apiKey={NEWSAPI_KEY}'
    )
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get('articles', [])
        if not articles and query != symbol:
            url2 = (
                f'https://newsapi.org/v2/everything?'
                f'q={symbol}&'
                f'from={from_date}&'
                f'to={to_date}&'
                f'sortBy=publishedAt&'
                f'language=en&'
                f'apiKey={NEWSAPI_KEY}'
            )
            response2 = requests.get(url2)
            data2 = response2.json()
            articles = data2.get('articles', [])
        for item in articles[:6]:
            news.append({
                'title': item.get('title'),
                'link': item.get('url'),
                'publisher': item.get('source', {}).get('name', ''),
                'summary': item.get('description', ''),
                'providerPublishTime': item.get('publishedAt', '')
            })
    except Exception as e:
        print("NewsAPI error:", e)
        news = []
    return news

BASE_DIR = os.path.join(os.path.dirname(__file__), 'stock market project', 'stock')
os.makedirs(BASE_DIR, exist_ok=True)  # Ensure the folder exists before DB_PATH is used

DB_PATH = os.path.join(BASE_DIR, 'users.db')

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'supersecretkey'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_user(username):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username, email, password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    if user:
        return {'username': user[0], 'email': user[1], 'password': user[2]}
    return None

def add_user(username, email, password):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if get_user(username):
            flash('Username already exists!')
            return redirect(url_for('register'))

        if add_user(username, email, password):
            # Log registration
            log_path = os.path.join(BASE_DIR, 'logs.txt')
            with open(log_path, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Registered user: {username}\n")
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Try again.')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            # Log login
            log_path = os.path.join(BASE_DIR, 'logs.txt')
            with open(log_path, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - User login: {username}\n")
            if username == 'admin':
                return redirect(url_for('admin_dashboard'))  # Redirect admin to admin dashboard
            else:
                return redirect(url_for('dashboard1'))  # Redirect normal users to dashboard1
        else:
            flash('Invalid credentials!')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/dashboard1', methods=['GET', 'POST'])
def dashboard1():
    if 'user' not in session:
        return redirect(url_for('login'))

    period = request.args.get('period', '1d')
    raw_input = request.args.get('symbol', '').strip()

    if request.method == 'POST':
        raw_input = request.form.get('symbol', '').strip()
        period = request.form.get('period', period)

    # Only process if a symbol is provided
    if not raw_input:
        return render_template('dashboard1.html', stock={}, period=period, news=[], chart_html="")

    # Try to get symbol from CSV, else treat input as symbol
    symbol = None
    symbol_csv = name_to_symbol_csv(raw_input)
    if symbol_csv:
        symbol = symbol_csv.replace('$', '').strip()
    else:
        symbol = raw_input.replace('$', '').strip()

    if not symbol:
        flash("No matching stock symbol found for the entered company name or symbol.", "warning")
        return render_template('dashboard1.html', stock={}, period=period, news=[], chart_html="")

    # Try all possible Yahoo Finance symbol formats
    possible_symbols = [symbol]
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        possible_symbols.append(symbol + '.NS')
        possible_symbols.append(symbol + '.BO')

    valid_symbol = None
    ticker = None
    for sym_try in possible_symbols:
        try:
            ticker_try = yf.Ticker(sym_try)
            info = ticker_try.info
            # If info dict has 'regularMarketPrice', it's a valid symbol
            if info and info.get('regularMarketPrice', None) is not None:
                valid_symbol = sym_try
                ticker = ticker_try
                break
        except Exception:
            continue

    if not valid_symbol or not ticker:
        flash("No data available for the selected symbol. The stock may be delisted or not supported on Yahoo Finance.", "warning")
        return render_template('dashboard1.html', stock={}, period=period, news=[], chart_html="")

    news = fetch_news(valid_symbol)
    chart_html = ""
    stock_data = {}
    period_map = {
        '1d': {'days': 1, 'interval': '1m'},
        '5d': {'days': 5, 'interval': '5m'},
        '1m': {'days': 30, 'interval': '1d'},
        '6m': {'days': 182, 'interval': '1d'},
        'ytd': {'days': 'ytd', 'interval': '1d'},
        '1y': {'days': 365, 'interval': '1d'},
        '2y': {'days': 730, 'interval': '1d'}
    }

    try:
        # For 1d and 5d, try intraday intervals
        if period in ['1d', '5d']:
            yf_args = period_map[period]
            df = ticker.history(period=period, interval=yf_args['interval'])
            if df.empty or 'Close' not in df.columns:
                flash(f"No intraday data available for {period} period.", "warning")
                return render_template('dashboard1.html', stock={}, period=period, news=news, chart_html="")
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if period == '1d':
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[t.strftime('%I:%M %p') for t in df.index],
                    y=df[price_col],
                    mode='lines',
                    name=valid_symbol,
                    customdata=np.stack([df['Open'], df['High'], df['Low'], df['Volume']], axis=-1),
                    hovertemplate=
                        'Time: %{x}<br>' +
                        'Close: %{y:.2f}<br>' +
                        'Open: %{customdata[0]:.2f}<br>' +
                        'High: %{customdata[1]:.2f}<br>' +
                        'Low: %{customdata[2]:.2f}<br>' +
                        'Volume: %{customdata[3]:,}<extra></extra>'
                ))
                fig.update_layout(
                    title=f'{valid_symbol.upper()} Intraday Chart (1D)',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    xaxis=dict(
                        showgrid=False,
                        tickformat='%I:%M %p',
                        tickfont=dict(size=12),
                        title=None,
                        tickmode='array',
                        tickvals=[df.index[i].strftime('%I:%M %p') for i in np.linspace(0, len(df.index)-1, 10, dtype=int)],
                        tickangle=0
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#f0f0f0',
                        tickfont=dict(size=12),
                        title=None
                    ),
                    template='plotly_white',
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=450
                )
                chart_html = fig.to_html(full_html=False)
            else:  # 5d
                df = df.between_time('09:30', '16:30')
                df = df[df.index.dayofweek < 5]
                fig = go.Figure()
                tickvals = []
                ticktext = []
                for date, group in df.groupby(df.index.date):
                    group = group.copy()
                    group.index = pd.to_datetime(group.index)
                    if len(group) > 1:
                        fig.add_trace(go.Scatter(
                            x=group.index,
                            y=group['Close'],
                            mode='lines',
                            line=dict(width=2, color='#5c6bc0'),
                            connectgaps=False,
                            customdata=np.stack([group['Open'], group['High'], group['Low'], group['Volume']], axis=-1),
                            hovertemplate=
                                'Date & Time: %{x|%b %d %I:%M %p}<br>' +
                                'Close: %{y:.2f}<br>' +
                                'Open: %{customdata[0]:.2f}<br>' +
                                'High: %{customdata[1]:.2f}<br>' +
                                'Low: %{customdata[2]:.2f}<br>' +
                                'Volume: %{customdata[3]:,}<extra></extra>'
                        ))
                        tickvals.append(group.index[0])
                        ticktext.append(date.strftime('%b %d'))
                fig.update_layout(
                    title=f'{valid_symbol.upper()} 5-Day Intraday Chart',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis=dict(
                        showgrid=False,
                        tickmode='array',
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickfont=dict(size=12),
                        title=None,
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),
                            dict(bounds=[16.5, 9.5], pattern="hour")
                        ]
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#f0f0f0',
                        tickfont=dict(size=12),
                        title=None
                    ),
                    template='plotly_white',
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=450,
                    showlegend=False
                )
                chart_html = fig.to_html(full_html=False)
            last_row = df.iloc[-1]
        else:
            # For longer periods, always use daily data for 2 years and filter
            df = ticker.history(period='2y', interval='1d')
            if df.empty or 'Close' not in df.columns:
                flash("No data available for the selected period.", "warning")
                return render_template('dashboard1.html', stock={}, period=period, news=news, chart_html="")
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if period == 'ytd':
                # Ensure start_of_year matches df.index timezone
                tz = df.index.tz
                start_of_year = pd.Timestamp(datetime.now().year, 1, 1, tz=tz)
                df_period = df[df.index >= start_of_year]
            else:
                days = period_map[period]['days']
                df_period = df[df.index >= (df.index.max() - pd.Timedelta(days=days))]
            if df_period.empty or 'Close' not in df_period.columns:
                flash("No data available for the selected period.", "warning")
                return render_template('dashboard1.html', stock={}, period=period, news=news, chart_html="")
            chart_html = generate_chart(df_period, period)
            last_row = df_period.iloc[-1]
            df = df_period

        info = ticker.info
        previous_close = round(float(df['Close'].iloc[-2]), 2) if len(df) > 1 else round(float(df['Close'].iloc[-1]), 2)

        stock_data = {
            'ticker': valid_symbol,
            'name': info.get('shortName', valid_symbol),
            'date': last_row.name.strftime('%m/%d %I:%M %p') if hasattr(last_row.name, 'strftime') else str(last_row.name),
            'close': round(float(last_row['Close']), 2),
            'open': round(float(last_row['Open']), 2),
            'high': round(float(last_row['High']), 2),
            'low': round(float(last_row['Low']), 2),
            'volume': int(last_row['Volume']),
            'previous_close': previous_close,
            'days_range': f"{round(float(last_row['Low']), 2)} - {round(float(last_row['High']), 2)}",
            'market_cap': f"{round(info.get('marketCap', 0)/1e12, 3)}T" if info.get('marketCap', 0) > 1e12 else f"{round(info.get('marketCap', 0)/1e9, 3)}B" if info.get('marketCap', 0) > 1e9 else str(info.get('marketCap', 'N/A')),
            'earnings_date': info.get('earningsDate', 'N/A'),
            'week_range': f"{round(float(df['Low'][-252:].min()), 2)} - {round(float(df['High'][-252:].max()), 2)}",
            'beta': info.get('beta', 'N/A'),
            'dividend_yield': f"{info.get('dividendRate', '--')} ({info.get('dividendYield', '--')})",
            'bid': info.get('bid', '--'),
            'ask': info.get('ask', '--'),
            'avg_volume': info.get('averageVolume', '--'),
            'pe_ratio': info.get('trailingPE', '--'),
            'eps': info.get('trailingEps', '--'),
            'ex_dividend_date': info.get('exDividendDate', '--'),
            'target_est': info.get('targetMeanPrice', '--')
        }
        return render_template('dashboard1.html', stock=stock_data, period=period, news=news, chart_html=chart_html)
    except Exception as e:
        flash(f"Error fetching data: {str(e)}", "error")
        return render_template('dashboard1.html', stock={}, period=period, news=news, chart_html="")

    return render_template('dashboard1.html', stock=stock_data, period=period, news=news, chart_html=chart_html)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbols = []
    results = []
    error = None
    adj_close_table = {
        'header': [
            {'date': '', 'label': '1 Day Out'},
            {'date': '', 'label': '7 Days Out'},
            {'date': '', 'label': '14 Days Out'}
        ],
        'rows': []
    }

    symbol = request.args.get('symbol')
    symbols = []
    if request.method == 'POST':
        raw_inputs = [s.strip() for s in request.form.get('symbol', '').split(',') if s.strip()]
        symbols = []
        for name in raw_inputs:
            sym = name_to_symbol_csv(name)
            if sym:
                symbols.append(sym)
            else:
                flash(f"No matching stock symbol found for '{name}'.", "warning")
        if not symbols:
            return render_template('dashboard.html', results=[], symbols=[], adj_close_table=adj_close_table, error="No valid symbols found.")
    elif symbol:
        symbols = [symbol.strip().upper()]

    if symbols:
        for symbol in symbols:
            try:
                df = yf.download(symbol, period='2y')
                if df.empty or 'Close' not in df.columns:
                    results.append({'symbol': symbol, 'error': 'No data found or Close missing.'})
                    continue

                # Feature Engineering
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['STD20'] = df['Close'].rolling(window=20).std()
                df['Upper'] = df['MA20'] + 2 * df['STD20']
                df['Lower'] = df['MA20'] - 2 * df['STD20']
                df['Return'] = df['Close'].pct_change()
                df['Volatility'] = df['Return'].rolling(window=10).std()
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                delta = df['Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                roll_up = up.rolling(14).mean()
                roll_down = down.rolling(14).mean()
                RS = roll_up / roll_down
                df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
                # Momentum
                df['Momentum'] = df['Close'] - df['Close'].shift(10)
                # Lagged returns/prices
                df['Lag1'] = df['Close'].shift(1)
                df['Lag7'] = df['Close'].shift(7)
                df['Lag14'] = df['Close'].shift(14)
                # Fill missing values
                df = df.fillna(method='bfill').fillna(method='ffill')
                # Remove outliers (optional, using z-score)
                df = df[(np.abs(zscore(df['Close'])) < 3)]
                df = df.dropna()

                # Optionally, add sentiment from news headlines (stub)
                
                sentiment_score = fetch_sentiment(symbol)
                df['Sentiment'] = sentiment_score  # Same value for all rows (stub)
                
                feature_cols = [
                    'MA20', 'MA50', 'STD20', 'Upper', 'Lower', 'Return', 'Volatility',
                    'EMA12', 'EMA26', 'MACD', 'RSI', 'Momentum', 'Lag1', 'Lag7', 'Lag14', 'Sentiment'
                ]

                feature_cols = [
                    'MA20', 'MA50', 'STD20', 'Upper', 'Lower', 'Return', 'Volatility',
                    'EMA12', 'EMA26', 'MACD', 'RSI', 'Momentum', 'Lag1', 'Lag7', 'Lag14'
                ]
                
                X = df[feature_cols].values
                y = df['Close'].values

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Multi-step prediction targets
                y_1 = df['Close'].shift(-1).dropna().values
                y_7 = df['Close'].shift(-7).dropna().values
                y_14 = df['Close'].shift(-14).dropna().values

                # Align X for multi-step
                X_1 = X_scaled[:-1]
                X_7 = X_scaled[:-7]
                X_14 = X_scaled[:-14]

                # Model selection and ensemble
                tscv = TimeSeriesSplit(n_splits=5)
                param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
                param_grid_gb = {'n_estimators': [100, 200], 'max_depth': [3, 5]}

                rf = RandomForestRegressor(random_state=42)
                gb = GradientBoostingRegressor(random_state=42)

                grid_rf_1 = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_gb_1 = GridSearchCV(gb, param_grid_gb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_rf_1.fit(X_1, y_1)
                grid_gb_1.fit(X_1, y_1)

                grid_rf_7 = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_gb_7 = GridSearchCV(gb, param_grid_gb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_rf_7.fit(X_7, y_7)
                grid_gb_7.fit(X_7, y_7)

                grid_rf_14 = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_gb_14 = GridSearchCV(gb, param_grid_gb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_rf_14.fit(X_14, y_14)
                grid_gb_14.fit(X_14, y_14)

                # Ensemble averaging
                def ensemble_predict(models, X):
                    return np.mean([m.predict(X) for m in models], axis=0)

                # Prepare last row for prediction
                last_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
                last_features_scaled = scaler.transform(last_features)

                pred_1 = ensemble_predict([grid_rf_1.best_estimator_, grid_gb_1.best_estimator_], last_features_scaled)[0]
                pred_7 = ensemble_predict([grid_rf_7.best_estimator_, grid_gb_7.best_estimator_], last_features_scaled)[0]
                pred_14 = ensemble_predict([grid_rf_14.best_estimator_, grid_gb_14.best_estimator_], last_features_scaled)[0]

                symbol_upper = symbol.upper()
                currency = '₹' if symbol_upper.endswith('.NS') or symbol_upper.endswith('.BSE') or symbol_upper in ['INFY', 'TCS', 'RELIANCE'] else '$'

                preds = {
                    '1 Day Out': f"{currency}{round(pred_1, 2)}",
                    '7 Days Out': f"{currency}{round(pred_7, 2)}",
                    '14 Days Out': f"{currency}{round(pred_14, 2)}"
                }

                last_close = round(df['Close'].iloc[-1], 2)
                current_price = round(float(df['Close'].iloc[-1]), 2)
                high_52week = round(float(df['High'][-252:].max()), 2)
                low_52week = round(float(df['Low'][-252:].min()), 2)
                last_closing_price = round(float(df['Close'].iloc[-2]), 2) if len(df) > 1 else current_price

                eval_table = {
                    'RF Training Score': f"{round(grid_rf_1.best_estimator_.score(X_1, y_1)*100, 2)}%",
                    'GB Training Score': f"{round(grid_gb_1.best_estimator_.score(X_1, y_1)*100, 2)}%"
                }
                norm_table = {
                    'Mean': f"{round(np.mean(df['Close']), 2)}",
                    'Median': f"{round(np.median(df['Close']), 2)}",
                    'STD': f"{round(float(np.std(df['Close'])), 2)}"
                }

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Upper'],
                    mode='lines', name=f'{symbol} Upper Bands',
                    line=dict(color='deepskyblue')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Lower'],
                    mode='lines', name=f'{symbol} Lower Bands',
                    line=dict(color='orange')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name=f'{symbol} Closing Price',
                    line=dict(color='yellow'),
                    customdata=np.stack([df['Open'], df['High'], df['Low'], df['Volume']], axis=-1),
                    hovertemplate=
                        'Date & Time: %{x|%b %d, %Y %I:%M %p}<br>' +
                        'Close: %{y:.2f}<br>' +
                        'Open: %{customdata[0]:.2f}<br>' +
                        'High: %{customdata[1]:.2f}<br>' +
                        'Low: %{customdata[2]:.2f}<br>' +
                        'Volume: %{customdata[3]:,}<extra></extra>'
                ))
                future_dates = pd.bdate_range(df.index[-1], periods=15)[[1,7,14]]
                fig.add_trace(go.Scatter(
                    x=future_dates, y=[pred_1, pred_7, pred_14],
                    mode='lines+markers', name='Predicted',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title=f'Bollinger Bands® - {symbol}',
                    xaxis_title='Dates',
                    yaxis_title='Adjusted Close',
                    plot_bgcolor='#222',
                    paper_bgcolor='#222',
                    font=dict(color='deepskyblue'),
                    legend=dict(bgcolor='#181818')
                )
                plot_html = pio.to_html(fig, full_html=False)

                results.append({
                    'symbol': symbol,
                    'last_close': last_close,
                    'current_price': current_price,
                    'high_52week': high_52week,
                    'low_52week': low_52week,
                    'last_closing_price': last_closing_price,
                    'preds': preds,
                    'eval_table': eval_table,
                    'norm_table': norm_table,
                    'plot_html': plot_html,
                    'error': None
                })

            except Exception as e:
                results.append({'symbol': symbol, 'error': str(e)})

        today = datetime.today()
        date_1 = (today + timedelta(days=1)).strftime('%d-%m-%Y')
        date_7 = (today + timedelta(days=7)).strftime('%d-%m-%Y')
        date_14 = (today + timedelta(days=14)).strftime('%d-%m-%Y')

        adj_close_table = {
            'header': [
                {'date': date_1, 'label': '1 Day Out'},
                {'date': date_7, 'label': '7 Days Out'},
                {'date': date_14, 'label': '14 Days Out'}
            ],
            'rows': []
        }

        for idx, res in enumerate(results):
            if res.get('error'):
                continue

            try:
                score_str = res['eval_table']['RF Training Score'].replace('%', '')
                score = float(score_str)
            except Exception:
                score = 0

            if score >= 97:
                confidence_emoji = '🟢'
                action = 'Buy'
            elif score >= 94:
                confidence_emoji = '🟡'
                action = 'Consider'
            else:
                confidence_emoji = '🔴'
                action = 'Not Buy'

            adj_row = {
                'symbol': res['symbol'],
                'preds': [
                    {'value': res['preds']['1 Day Out'], 'action': action},
                    {'value': res['preds']['7 Days Out'], 'action': action},
                    {'value': res['preds']['14 Days Out'], 'action': action},
                ],
                'confidence': confidence_emoji
            }
            adj_close_table['rows'].append(adj_row)

        return render_template(
            'dashboard.html',
            results=results,
            symbols=symbols,
            adj_close_table=adj_close_table,
            error=error
        )

    return render_template('dashboard.html', results=results, symbols=symbols, adj_close_table=adj_close_table, error=error)

@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if 'user' not in session or session['user'] != 'admin':
        flash('Admin access required.')
        return redirect(url_for('login'))

    conn = get_db_connection()
    c = conn.cursor()
    users = c.execute('SELECT username, email, created_at FROM users').fetchall()
    conn.close()

    logs = []
    log_path = os.path.join(BASE_DIR, 'logs.txt')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = f.readlines()

    return render_template('admin_dashboard.html', users=users, logs=logs)

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    if 'user' not in session or session['user'] != 'admin':
        flash('Admin access required.')
        return redirect(url_for('login'))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()

    # Log the deletion
    log_path = os.path.join(BASE_DIR, 'logs.txt')
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Deleted user: {username}\n")

    flash(f'User {username} deleted.')
    return redirect(url_for('admin_dashboard'))

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %Y'):
    try:
        import datetime
        return datetime.datetime.fromtimestamp(value).strftime(format)
    except Exception:
        return value
    
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)