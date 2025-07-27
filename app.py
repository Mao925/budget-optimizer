import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pulp
import jpholiday
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from pytrends.request import TrendReq # Googleトレンド取得ライブラリ
import time # APIリクエストのための待機時間

# --- 0. ヘルパー関数 ---
def to_excel(df):
    """データフレームをExcel形式のバイトデータに変換する"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# --- Googleトレンド関連の関数 (修正) ---
# トレンド取得対象のキーワードを定義 (ユーザーの指定に基づき更新)
TREND_KEYWORDS = {
    'general_trend': ['塾講師 バイト', '塾 バイト'],
    'brand_trend': ['塾講師ステーション'],
    'school_trend': [
        '早稲田アカデミー バイト', '武田塾 バイト', 'ITTO個別指導学院 バイト',
        'ビザビ バイト', '早稲アカ バイト', '早稲アカ 求人', '個別指導キャンパス バイト',
        '武田塾 応募', 'トーマス バイト', 'ナビ個別指導学院 アルバイト',
        'アクシス 求人', 'ナビ個別指導学院 バイト', 'ITTO バイト', 'アクシス バイト',
        '代々木個別指導学院 バイト', '臨海セミナー バイト', '個別指導Axis バイト'
    ],
    'station_trend': [
        '東京 塾 求人', '神奈川 塾 求人', '千葉 塾 求人', '名古屋 塾 求人',
        '京都 塾 求人', '東京 塾 バイト', '福島 塾 求人', '大阪 塾 求人',
        '浜松 塾 求人', '相模原 塾 求人', 'つくば 塾 求人', '八王子 塾 求人',
        '大宮 塾 バイト', '千葉 塾 バイト', '東大宮 塾 バイト', '福島 塾 バイト',
        '吉祥寺 塾 バイト', '湘南台 塾 バイト', '武蔵小杉 塾 バイト', '津田沼 塾 バイト'
    ]
}

# --- モデルが使用する特徴量リストをグローバル定数として定義 (修正) ---
FEATURES = ['cost', 'log_cost', 'weekday', 'month', 'week', 'is_holiday'] + list(TREND_KEYWORDS.keys())


@st.cache_data(ttl=3600) # 1時間キャッシュ
def fetch_and_prepare_trends_data(start_date, end_date):
    """指定された期間のGoogleトレンドデータを取得し、日別データフレームを返す"""
    pytrends = TrendReq(hl='ja-JP', tz=360) # 日本語、日本時間で設定
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    trends_df = pd.DataFrame(index=all_dates)

    for category, kw_list in TREND_KEYWORDS.items():
        try:
            # pytrendsは内部で5キーワードごとに分割してリクエストを送信する
            pytrends.build_payload(kw_list, cat=0, timeframe=f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}', geo='JP')
            time.sleep(1) # APIへの負荷を軽減
            
            interest_over_time_df = pytrends.interest_over_time()

            if not interest_over_time_df.empty and 'isPartial' in interest_over_time_df.columns:
                trends_df[category] = interest_over_time_df.drop(columns='isPartial').mean(axis=1)
        except Exception as e:
            # st.warning(f"トレンドデータの取得中にエラーが発生しました ({category}): {e}")
            trends_df[category] = 0

    trends_df = trends_df.resample('D').ffill()
    trends_df = trends_df.bfill()
    trends_df = trends_df.fillna(0)
    
    # カラム名を 'date' から '日' に変更
    return trends_df.reset_index().rename(columns={'index': '日'})


# --- 1. データ処理関数 (修正) ---
@st.cache_data
def preprocess_data(uploaded_file, column_mapping, training_start_date, training_end_date):
    """アップロードされたファイルを前処理し、特徴量を作成する"""
    if uploaded_file is None: return None, "ファイルがアップロードされていません。", None
    
    df = None
    for encoding in ['utf-8-sig', 'cp932', 'utf-8', 'sjis']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, header=None, dtype=str)
            break
        except Exception:
            continue
    
    if df is None: return None, "ファイルの読み込みに失敗しました。対応する文字コードが見つかりません。", None

    header_row_index = -1
    for i, row in df.iterrows():
        if row.astype(str).str.contains('キャンペーン名').any():
            header_row_index = i
            break
            
    if header_row_index == -1:
        st.session_state.column_mapping_required = True
        st.session_state.raw_df_columns = df.iloc[0].tolist()
        return None, "ヘッダー行 ('キャンペーン名') が見つかりません。列のマッピングを行ってください。", None

    new_header = df.iloc[header_row_index]
    df = df[header_row_index + 1:]
    df.columns = new_header
    
    total_row_index = df[df.iloc[:, 0].astype(str).str.contains('合計', na=False)].index
    if not total_row_index.empty:
        df = df.loc[:total_row_index[0]-1]

    df = df.rename(columns=column_mapping)
    
    # 必須列を 'date' から '日' に変更
    required_cols = ['日', 'campaign_name', 'cost', 'conversions']
    if not all(col in df.columns for col in required_cols):
        return None, f"必要な列 {required_cols} が見つかりません。列のマッピングを確認してください。", None
        
    df_selected = df[required_cols].copy()
    
    try:
        # データ型変換の対象列を '日' に変更
        df_selected['日'] = pd.to_datetime(df_selected['日'])
        df_selected['cost'] = pd.to_numeric(df_selected['cost'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        df_selected['conversions'] = pd.to_numeric(df_selected['conversions'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    except Exception as e:
        return None, f"データ型変換中にエラーが発生しました: {e}。CSVのフォーマットを確認してください。", None

    # 特徴量エンジニアリングの参照列を '日' に変更
    df_selected['weekday'] = df_selected['日'].dt.weekday
    df_selected['month'] = df_selected['日'].dt.month
    df_selected['week'] = df_selected['日'].dt.isocalendar().week.astype(int)
    df_selected['is_holiday'] = df_selected['日'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    df_selected['log_cost'] = np.log1p(df_selected['cost'])

    # --- Googleトレンドデータを取得してマージ (結合キーを '日' に変更) ---
    try:
        trends_df = fetch_and_prepare_trends_data(training_start_date, training_end_date)
        df_selected = pd.merge(df_selected, trends_df, on='日', how='left')
        df_selected[list(TREND_KEYWORDS.keys())] = df_selected[list(TREND_KEYWORDS.keys())].fillna(0)
    except Exception as e:
        return None, f"Googleトレンドデータの取得またはマージに失敗しました: {e}", None
    # --- ここまで修正 ---

    # 学習期間でのフィルタリング (参照列を '日' に変更)
    training_df = df_selected[(df_selected['日'] >= pd.to_datetime(training_start_date)) & (df_selected['日'] <= pd.to_datetime(training_end_date))]
    
    if training_df.empty:
        return None, "指定された学習期間に有効なデータがありませんでした。", None

    return training_df.sort_values(by='日').reset_index(drop=True), None, trends_df

# --- 2. モデル学習関数 (修正) ---
@st.cache_data
def train_models(_df):
    """キャンペーンごとにXGBoostモデルを学習する"""
    models = {}
    # st.session_stateへの依存を削除
    
    campaign_names = _df['campaign_name'].unique()
    # 日付の参照列を '日' に変更
    latest_date = _df['日'].max()
    
    for campaign in campaign_names:
        campaign_df = _df[_df['campaign_name'] == campaign].copy()
        if len(campaign_df) < 10: continue

        # 日付の参照列を '日' に変更
        recency_in_days = (latest_date - campaign_df['日']).dt.days
        sample_weights = np.where(recency_in_days <= 30, 3.0, 1.0)
        
        # グローバル定数 FEATURES を使用
        X = campaign_df[FEATURES]
        y = campaign_df['conversions']
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y, sample_weight=sample_weights)
        models[campaign] = model
        
    return models

# --- 3. 最適化関数 (修正) ---
def optimize_budget_allocation(total_budget, models, features_today, campaign_max_budgets):
    """数理最適化を用いて、CVを最大化する予算配分を計算する"""
    campaign_names = list(models.keys())
    problem = pulp.LpProblem("Budget_Allocation_Problem", pulp.LpMaximize)
    
    step = max(1000, int(total_budget / 100))
    budget_steps = list(range(step, total_budget + 1, step))
    if not budget_steps: return {}, None, {}
    
    choices = pulp.LpVariable.dicts("Choice", (campaign_names, budget_steps), cat='Binary')
    
    predicted_cvs = {}
    # st.session_stateへの依存を削除し、グローバル定数 FEATURES を使用
    for campaign in campaign_names:
        predicted_cvs[campaign] = {}
        pred_data_list = []
        for budget in budget_steps:
            pred_data_point = {
                'cost': budget, 
                'log_cost': np.log1p(budget),
                **features_today
            }
            pred_data_list.append(pred_data_point)

        pred_df = pd.DataFrame(pred_data_list)[FEATURES]
        predictions = models[campaign].predict(pred_df)
        for i, budget in enumerate(budget_steps):
            predicted_cvs[campaign][budget] = max(0, predictions[i])

    problem += pulp.lpSum(predicted_cvs[c][b] * choices[c][b] for c in campaign_names for b in budget_steps)
    
    problem += pulp.lpSum(b * choices[c][b] for c in campaign_names for b in budget_steps) <= total_budget
    for c in campaign_names:
        problem += pulp.lpSum(choices[c][b] for b in budget_steps) == 1
    for c in campaign_names:
        problem += pulp.lpSum(b * choices[c][b] for b in budget_steps) <= campaign_max_budgets[c]

    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    optimal_allocation, total_predicted_cv, cv_per_campaign = {}, None, {}
    if pulp.LpStatus[problem.status] == "Optimal":
        total_predicted_cv = pulp.value(problem.objective)
        for c in campaign_names:
            for b in budget_steps:
                if choices[c][b].varValue == 1:
                    optimal_allocation[c] = b
                    cv_per_campaign[c] = predicted_cvs[c][b]
                    break
    return optimal_allocation, total_predicted_cv, cv_per_campaign

# --- 4. Streamlit UIの構築 ---
st.set_page_config(layout="wide", page_title="広告予算配分 最適化シミュレーター")
st.title('🚀 広告予算配分 最適化シミュレーター (Googleトレンド対応版)')

# --- サイドバー ---
with st.sidebar:
    st.header('⚙️ 基本設定')
    uploaded_file = st.file_uploader("① パフォーマンスレポートをアップロード", type=['csv'])

    if uploaded_file:
        default_start_date = datetime.now().date() - timedelta(days=90)
        default_end_date = datetime.now().date()
        date_detection_info = "学習期間を指定してください。"

        try:
            uploaded_file.seek(0)
            temp_df_for_dates = None
            possible_date_cols = ['日', '日付', 'Date', 'Day']
            date_col_name = None
            header_row_index = -1
            found_encoding = None

            for encoding in ['utf-8-sig', 'cp932', 'utf-8', 'sjis']:
                try:
                    uploaded_file.seek(0)
                    header_peek = pd.read_csv(uploaded_file, encoding=encoding, header=None, dtype=str, nrows=20)
                    
                    for i, row in header_peek.iterrows():
                        if 'キャンペーン名' in row.astype(str).values:
                            header_row_index = i
                            for col in possible_date_cols:
                                if col in row.values:
                                    date_col_name = col
                                    break
                            break
                    if date_col_name:
                        found_encoding = encoding
                        break
                except Exception:
                    continue
            
            if date_col_name and found_encoding:
                uploaded_file.seek(0)
                date_series = pd.read_csv(uploaded_file, encoding=found_encoding, header=header_row_index, usecols=[date_col_name], dtype=str).iloc[:, 0]
                dates = pd.to_datetime(date_series, errors='coerce').dropna()
                if not dates.empty:
                    default_start_date = dates.min().date()
                    default_end_date = dates.max().date()
                    date_detection_info = "ファイルから期間を自動設定しました。変更も可能です。"
        except Exception:
            pass
        finally:
            uploaded_file.seek(0)

        st.subheader("② 学習データ期間")
        st.info(date_detection_info)
        training_start_date = st.date_input('開始日', value=default_start_date)
        training_end_date = st.date_input('終了日', value=default_end_date)
        
        process_button = st.button('データを処理し、モデルを学習する', type="primary", use_container_width=True)
    else:
        process_button = False

# --- メイン画面 ---
# --- session_state の初期化 (修正) ---
if 'column_mapping_required' not in st.session_state:
    st.session_state.column_mapping_required = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'daily_results' not in st.session_state:
    st.session_state.daily_results = None
if 'optim_period' not in st.session_state:
    st.session_state.optim_period = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'raw_df_columns' not in st.session_state:
    st.session_state.raw_df_columns = []


if uploaded_file is None:
    st.info("サイドバーからCSVファイルをアップロードして開始してください。")
else:
    # 列名マッピングから '日': 'date' を削除
    column_mapping = {
        'キャンペーン名': 'campaign_name', 'コスト': 'cost', 'コンバージョン数': 'conversions'
    }
    if st.session_state.column_mapping_required:
        with st.expander("⚠️ 列名のマッピングが必要です", expanded=True):
            st.warning("CSVの列名を特定できませんでした。以下のドロップダウンから対応する列名を選択してください。")
            # マッピング辞書から '日' を削除
            jp_to_en = {'日': '日', 'キャンペーン名': 'campaign_name', 'コスト': 'cost', 'コンバージョン数': 'conversions'}
            raw_cols = st.session_state.raw_df_columns
            user_mapping = {}
            for jp_col, en_col in jp_to_en.items():
                # ユーザーが選択したCSVの列名をキー、プログラム内部で使う列名(en_col)をバリューとする
                selected_col = st.selectbox(f"「{jp_col}」に対応する列", [None] + raw_cols)
                if selected_col:
                    user_mapping[selected_col] = en_col
            
            column_mapping = {k: v for k, v in user_mapping.items() if k is not None}


    if process_button:
        with st.spinner('データ前処理とGoogleトレンドデータ取得、モデル学習を実行中... (少し時間がかかります)'):
            data, error_message, trends_for_training = preprocess_data(uploaded_file, column_mapping, training_start_date, training_end_date)
            if error_message:
                st.error(f"エラー: {error_message}")
                st.session_state.data_processed = False
            else:
                st.session_state.original_data = data
                st.session_state.trained_models = train_models(data)
                st.session_state.data_processed = True
                st.success("データ処理とモデル学習が完了しました。")
    
    if st.session_state.data_processed:
        st.header("③ 最適化シミュレーション")
        
        col1, col2 = st.columns(2)
        with col1:
            optim_start_date = st.date_input('最適化期間（開始日）', value=datetime.now().date() + timedelta(days=1))
        with col2:
            optim_end_date = st.date_input('最適化期間（終了日）', value=datetime.now().date() + timedelta(days=7))

        if optim_start_date > optim_end_date:
            st.error('エラー: 終了日は開始日以降に設定してください。')
        else:
            st.subheader("インタラクティブ設定")
            total_daily_budget = st.slider('1日あたりの総予算（円）', min_value=10000, max_value=500000, step=1000, value=100000)
            
            with st.expander("キャンペーン別の上限予算を設定する（推奨）"):
                st.info("キャンペーン毎に消化可能な上限予算をスライダーで設定すると、より現実的な配分になります。")
                campaign_max_budgets_input = {}
                for campaign in st.session_state.trained_models.keys():
                    campaign_max_budgets_input[campaign] = st.slider(f"【{campaign}】の上限予算（円/日）", 0, total_daily_budget, total_daily_budget, 500)

            run_optimization_button = st.button('🚀 この設定で最適化を実行する', type="primary", use_container_width=True)

            if run_optimization_button:
                with st.spinner('最適化計算を実行中... (トレンドデータを考慮します)'):
                    trends_for_optim = fetch_and_prepare_trends_data(optim_start_date, optim_end_date)
                    # 結合キーを '日' にするため、インデックスを '日' に設定
                    trends_for_optim.set_index('日', inplace=True)

                    date_range = pd.date_range(optim_start_date, optim_end_date)
                    daily_results = []
                    progress_bar = st.progress(0, text="最適化計算を実行中...")

                    for i, target_date in enumerate(date_range):
                        target_date_ts = pd.Timestamp(target_date)
                        
                        trend_date_to_use = min(target_date_ts, trends_for_optim.index.max())
                        
                        features_for_today = {
                            'weekday': target_date.weekday(),
                            'month': target_date.month, 
                            'week': target_date.isocalendar().week,
                            'is_holiday': 1 if jpholiday.is_holiday(target_date) else 0,
                            **trends_for_optim.loc[trend_date_to_use].to_dict()
                        }

                        optimal_budgets, total_cv, cv_breakdown = optimize_budget_allocation(total_daily_budget, st.session_state.trained_models, features_for_today, campaign_max_budgets_input)
                        daily_results.append({'date': target_date, 'allocation': optimal_budgets, 'cv': total_cv, 'cv_breakdown': cv_breakdown})
                        progress_bar.progress((i + 1) / len(date_range), text=f"最適化計算: {target_date.strftime('%m/%d')}")
                    
                    st.session_state.daily_results = daily_results
                    st.session_state.optim_period = (optim_start_date, optim_end_date)
                    st.success("最適化が完了しました。")

    # 結果表示
    if st.session_state.daily_results:
        st.header(f'📊 最適化結果（{st.session_state.optim_period[0].strftime("%Y/%m/%d")} 〜 {st.session_state.optim_period[1].strftime("%Y/%m/%d")}）')
        
        daily_results = st.session_state.daily_results
        if not daily_results or not any(res['allocation'] for res in daily_results):
            st.warning("最適化に失敗しました。解決可能な配分が見つかりませんでした。総予算や上限予算の設定を確認してください。")
        else:
            with st.container(border=True):
                st.subheader("📈 パフォーマンスサマリー")
                
                total_allocated_sum = sum(sum(res['allocation'].values()) for res in daily_results if res['allocation'])
                total_predicted_cv_sum = sum(res['cv'] for res in daily_results if res['cv'] is not None)
                total_predicted_cpa = total_allocated_sum / total_predicted_cv_sum if total_predicted_cv_sum > 0 else 0

                past_data = st.session_state.original_data
                past_period_start = st.session_state.optim_period[0] - timedelta(days=len(daily_results))
                past_period_end = st.session_state.optim_period[0] - timedelta(days=1)
                # 過去データの日付列も '日' を参照
                past_perf_df = past_data[(past_data['日'] >= pd.to_datetime(past_period_start)) & (past_data['日'] <= pd.to_datetime(past_period_end))]
                
                past_cost = past_perf_df['cost'].sum()
                past_cv = past_perf_df['conversions'].sum()
                past_cpa = past_cost / past_cv if past_cv > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="予測 配分合計金額", value=f"{total_allocated_sum:,.0f} 円", help="シミュレーション期間中の合計予算")
                    st.metric(label="過去実績 コスト", value=f"{past_cost:,.0f} 円", delta=f"{total_allocated_sum - past_cost:,.0f} 円", delta_color="inverse", help=f"過去の同期間 ({past_period_start.strftime('%m/%d')}~{past_period_end.strftime('%m/%d')}) の実績")
                with col2:
                    st.metric(label="予測 総CV数", value=f"{total_predicted_cv_sum:.2f} 件", help="シミュレーション期間中の合計予測CV数")
                    st.metric(label="過去実績 CV数", value=f"{past_cv:,.0f} 件", delta=f"{total_predicted_cv_sum - past_cv:.2f} 件", help="過去の同期間の実績")
                with col3:
                    st.metric(label="予測 CPA", value=f"{total_predicted_cpa:,.0f} 円", help="期間平均の予測CPA")
                    st.metric(label="過去実績 CPA", value=f"{past_cpa:,.0f} 円", delta=f"{total_predicted_cpa - past_cpa:,.0f} 円", delta_color="inverse", help="過去の同期間の実績")

            avg_allocations = pd.DataFrame([res['allocation'] for res in daily_results if res['allocation']]).mean()
            avg_cvs = pd.DataFrame([res['cv_breakdown'] for res in daily_results if res['cv_breakdown']]).mean()
            result_df = pd.DataFrame({'1日あたりの平均推奨予算': avg_allocations, '1日あたりの平均予測CV数': avg_cvs}).fillna(0)
            result_df['予測CPA'] = (result_df['1日あたりの平均推奨予算'] / result_df['1日あたりの平均予測CV数']).replace([np.inf, -np.inf], 0).fillna(0)
            result_df = result_df.round({'1日あたりの平均推奨予算': 0, '1日あたりの平均予測CV数': 2, '予測CPA': 0}).astype({'1日あたりの平均推奨予算': int, '予測CPA': int})
            
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                with st.container(border=True):
                    st.subheader("📋 キャンペーン別パフォーマンス（日平均）")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        label="詳細結果をダウンロード (Excel)",
                        data=to_excel(result_df),
                        file_name=f"optimization_result_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        use_container_width=True
                    )
            with col2:
                with st.container(border=True):
                    st.subheader("🍰 予算配分比率（日平均）")
                    plot_df = result_df[result_df['1日あたりの平均推奨予算'] > 0]
                    if not plot_df.empty:
                        fig = px.pie(plot_df, values='1日あたりの平均推奨予算', names=plot_df.index, hole=.3)
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("配分されたキャンペーンがありません。")

            with st.container(border=True):
                st.subheader("📅 日別パフォーマンス推移")
                
                tab1, tab2, tab3 = st.tabs(["予測CV数 推移", "日別 予算配分", "キャンペーン別 予測CPA比較"])

                with tab1:
                    daily_cv_df = pd.DataFrame([{'date': r['date'], 'cv': r['cv']} for r in daily_results if r['cv'] is not None]).set_index('date')
                    if not daily_cv_df.empty:
                        fig = px.line(daily_cv_df, x=daily_cv_df.index, y='cv', title='日別 予測CV数推移', markers=True)
                        fig.update_layout(xaxis_title='日付', yaxis_title='予測CV数')
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    alloc_df = pd.DataFrame([res['allocation'] for res in daily_results], index=[res['date'] for res in daily_results]).fillna(0)
                    if not alloc_df.empty:
                        fig = px.bar(alloc_df, x=alloc_df.index, y=alloc_df.columns, title='日別 予算配分推移',
                                     labels={'value': '予算（円）', 'index': '日付', 'variable': 'キャンペーン'})
                        st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    cpa_df = result_df[result_df['予測CPA'] > 0].sort_values('予測CPA', ascending=True)
                    if not cpa_df.empty:
                        fig = px.bar(cpa_df, x=cpa_df.index, y='予測CPA', title='キャンペーン別 予測CPA比較',
                                     labels={'x': 'キャンペーン', 'y': '予測CPA（円）'}, text_auto=True)
                        st.plotly_chart(fig, use_container_width=True)
