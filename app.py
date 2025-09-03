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
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta 
from io import BytesIO
import base64
import plotly.graph_objs as go
import plotly.io as pio
import requests
import plotly.express as px

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

    symbol = request.args.get('symbol')
    period = request.args.get('period', '1d')

    if request.method == 'POST':
        symbol = request.form.get('symbol', symbol)
        period = request.form.get('period', period)

    chart_html = ""
    stock_data = {}
    news = []
    period_map = {
        '1d': {'period': '1d', 'interval': '1m'},
        '5d': {'period': '5d', 'interval': '5m'},
        '1m': {'period': '1mo', 'interval': '1d'},
        '6m': {'period': '6mo', 'interval': '1d'},
        'ytd': {'period': 'ytd', 'interval': '1d'},
        '1y': {'period': '1y', 'interval': '1d'},
        '2y': {'period': '2y', 'interval': '1d'}
    }

    if symbol:
        news = fetch_news(symbol)
        yf_args = period_map.get(period, period_map['1d'])
        try:
            if period == '1d':
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=yf_args['period'], interval=yf_args['interval'])
            elif period == '5d':
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='5d', interval='5m')
                df = df.copy()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            
                # Filter for market hours and weekdays only
                df = df.between_time('09:30', '16:30')
                df = df[df.index.dayofweek < 5]  # Monday=0, Sunday=6
            
                fig = go.Figure()
            
                # Plot each day as a separate trace to avoid lines between days
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
                        # For x-axis ticks: use market open time for each day
                        tickvals.append(group.index[0])
                        ticktext.append(date.strftime('%b %d'))  # e.g., "Jul 19"
            
                fig.update_layout(
                    title=f'{symbol.upper()} 5-Day Intraday Chart',
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
                            dict(bounds=["sat", "mon"]),  # hide weekends
                            dict(bounds=[16.5, 9.5], pattern="hour")  # hide non-market hours (4:30pm to 9:30am)
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
            else:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=yf_args['period'], interval=yf_args['interval'])

            if df.empty or 'Close' not in df.columns:
                flash("No data available for the selected period or symbol.", "warning")
                return render_template('dashboard1.html', stock={}, period=period, news=news, chart_html="")

            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if period == '1d':
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                fig = go.Figure()
                df['Time'] = df.index.time
                fig.add_trace(go.Scatter(
                    x=[t.strftime('%I:%M %p') for t in df.index],  # Format as "HH:MM AM/PM"
                    y=df[price_col],
                    mode='lines',
                    name=symbol,
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
                    title=f'{symbol.upper()} Intraday Chart (1D)',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    xaxis=dict(
                        showgrid=False,
                        tickformat='%I:%M %p',  # AM/PM format
                        tickfont=dict(size=12),
                        title=None,
                        tickmode='array',
                        tickvals=[df.index[i].strftime('%I:%M %p') for i in np.linspace(0, len(df.index)-1, 10, dtype=int)],
                        tickangle=0  # Horizontal labels
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
            else:
                chart_html = generate_chart(df, period)
            
            last_row = df.iloc[-1]
            info = ticker.info
            previous_close = round(float(df['Close'].iloc[-2]), 2) if len(df) > 1 else round(float(df['Close'].iloc[-1]), 2)

            stock_data = {
                'ticker': symbol,
                'name': info.get('shortName', symbol),
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
        symbols = [s.strip().upper() for s in request.form.get('symbol', '').split(',') if s.strip()]
    elif symbol:
        symbols = [symbol.strip().upper()]

    if symbols:
        for symbol in symbols:
            try:
                df = yf.download(symbol, period='2y')
                if df.empty or 'Close' not in df.columns:
                    results.append({'symbol': symbol, 'error': 'No data found or Close missing.'})
                    continue
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['STD20'] = df['Close'].rolling(window=20).std()
                df['Upper'] = df['MA20'] + 2 * df['STD20']
                df['Lower'] = df['MA20'] - 2 * df['STD20']
                df['Return'] = df['Close'].pct_change()
                df['Volatility'] = df['Return'].rolling(window=10).std()
                df = df.dropna()

                feature_cols = ['MA20', 'MA50', 'STD20', 'Upper', 'Lower', 'Return', 'Volatility']
                X = df[feature_cols].values
                y = df['Close'].values

                train_size = int(len(df) * 0.85)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
                rf.fit(X_train, y_train)

                symbol_upper = symbol.upper()
                if symbol_upper.endswith('.NS') or symbol_upper.endswith('.BSE') or symbol_upper in ['INFY', 'TCS', 'RELIANCE']:
                    currency = '₹'
                else:
                    currency = '$'

                preds = {}
                future_closes = list(map(float, df['Close'].values[-20:]))
                future_returns = list(map(float, df['Return'].values[-10:]))

                prediction_days = [1, 7, 14]
                prediction_labels = ['1 Day Out', '7 Days Out', '14 Days Out']
                current_index = 0
                predicted_prices = []

                for days, label in zip(prediction_days, prediction_labels):
                    for _ in range(days - current_index):
                        if len(future_closes) < 20:
                            ma20 = np.mean(future_closes)
                            std20 = np.std(future_closes)
                        else:
                            ma20 = np.mean(future_closes[-20:])
                            std20 = np.std(future_closes[-20:])
                        if len(future_closes) < 50:
                            ma50 = np.mean(future_closes)
                        else:
                            ma50 = np.mean(future_closes[-50:])
                        upper = ma20 + 2 * std20
                        lower = ma20 - 2 * std20
                        ret = (future_closes[-1] - future_closes[-2]) / future_closes[-2] if len(future_closes) > 1 else 0
                        if len(future_returns) < 10:
                            vol = np.std(future_returns)
                        else:
                            vol = np.std(future_returns[-10:])
                        features = np.array([[float(ma20), float(ma50), float(std20), float(upper), float(lower), float(ret), float(vol)]])
                        pred_price = float(rf.predict(features)[0])
                        future_returns.append(float(ret))
                        future_closes.append(float(pred_price))
                    preds[label] = f"{currency}{round(pred_price, 2)}"
                    predicted_prices.append(pred_price)
                    current_index = days

                last_close = round(df['Close'].iloc[-1], 2)
                current_price = round(float(df['Close'].iloc[-1]), 2)
                high_52week = round(float(df['High'][-252:].max()), 2)
                low_52week = round(float(df['Low'][-252:].min()), 2)
                last_closing_price = round(float(df['Close'].iloc[-2]), 2) if len(df) > 1 else current_price

                eval_table = {
                    'Training Score Mean': f"{round(rf.score(X_train, y_train)*100, 2)}%",
                    'Testing Score Mean': f"{round(rf.score(X_test, y_test)*100, 2)}%"
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
                last_date = df.index[-1]
                future_dates = pd.bdate_range(last_date, periods=prediction_days[-1]+1)[prediction_days]
                fig.add_trace(go.Scatter(
                    x=future_dates, y=predicted_prices,
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
                score_str = res['eval_table']['Training Score Mean'].replace('%', '')
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