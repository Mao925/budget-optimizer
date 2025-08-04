import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pulp
import jpholiday
from datetime import datetime, timedelta
import plotly.express as px
from io import BytesIO
from pytrends.request import TrendReq
import time
import collections

# --- 0. アプリケーション設定とヘルパー関数 ---
st.set_page_config(layout="wide", page_title="広告予算配分 最適化シミュレーター")

def to_excel(df):
    """データフレームをExcel形式のバイトデータに変換する"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def get_features(trend_keywords_dict):
    """トレンドキーワード辞書から特徴量リストを生成する"""
    return ['cost', 'log_cost', 'weekday', 'month', 'week', 'is_holiday'] + list(trend_keywords_dict.keys())

# --- Googleトレンド関連の関数 (修正) ---
@st.cache_data(ttl=3600) # 1時間キャッシュ
def fetch_and_prepare_trends_data(start_date, end_date, trend_keywords_dict):
    """
    指定された期間とキーワードでGoogleトレンドデータを取得し、日別データフレームを返す。
    長期間のリクエストはエラーになるため、180日単位のチャンクでデータを取得する。
    """
    pytrends = TrendReq(hl='ja-JP', tz=360)
    
    all_trends_data = {}
    for category, kw_list in trend_keywords_dict.items():
        if not kw_list:
            all_trends_data[category] = pd.Series(dtype=float)
            continue

        # チャンクでデータを取得
        total_df = pd.DataFrame()
        current_start = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        while current_start <= end_date_dt:
            current_end = current_start + timedelta(days=180)
            if current_end > end_date_dt:
                current_end = end_date_dt
            
            timeframe = f'{current_start.strftime("%Y-%m-%d")} {current_end.strftime("%Y-%m-%d")}'
            
            try:
                pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='JP')
                time.sleep(1) # APIへの負荷を軽減
                interest_over_time_df = pytrends.interest_over_time()
                
                if not interest_over_time_df.empty:
                    total_df = pd.concat([total_df, interest_over_time_df])
                
            except Exception as e:
                st.warning(f"トレンドデータの一部取得中にエラーが発生しました ({category}, {timeframe}): {e}")
                pass

            current_start = current_end + timedelta(days=1)

        if not total_df.empty and 'isPartial' in total_df.columns:
            # チャンクの境界で発生する可能性のある重複を削除
            total_df = total_df[~total_df.index.duplicated(keep='first')]
            all_trends_data[category] = total_df.drop(columns='isPartial').mean(axis=1)
        else:
            all_trends_data[category] = pd.Series(dtype=float)

    # 全カテゴリを一つのデータフレームに結合
    trends_df = pd.DataFrame(all_trends_data)
    
    # 全ての日付が含まれるようにインデックスを再設定
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    trends_df = trends_df.reindex(all_dates)

    # 欠損値を前方・後方で埋め、それでも残る場合は0で埋める
    trends_df = trends_df.ffill().bfill().fillna(0)
    
    return trends_df.reset_index().rename(columns={'index': '日'})


# --- 1. データ処理関数 ---
@st.cache_data
def preprocess_data(uploaded_file, column_mapping, training_start_date, training_end_date, trend_keywords_dict):
    """アップロードされたファイルを前処理し、特徴量を作成する"""
    if uploaded_file is None: return None, "ファイルがアップロードされていません。", None

    df = None
    # さまざまなエンコーディングを試す
    for encoding in ['utf-8-sig', 'cp932', 'utf-8', 'sjis']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, header=None, dtype=str)
            break
        except Exception:
            continue
    if df is None: return None, "ファイルの読み込みに失敗しました。対応する文字コードが見つかりません。", None

    # ヘッダー行を自動検出
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
    
    # 合計行を削除
    total_row_index = df[df.iloc[:, 0].astype(str).str.contains('合計', na=False)].index
    if not total_row_index.empty:
        df = df.loc[:total_row_index[0]-1]

    df = df.rename(columns=column_mapping)
    
    required_cols = ['日', 'campaign_name', 'cost', 'conversions']
    if not all(col in df.columns for col in required_cols):
        return None, f"必要な列 {required_cols} が見つかりません。列のマッピングを確認してください。", None
        
    df_selected = df[required_cols].copy()
    
    try:
        df_selected['日'] = pd.to_datetime(df_selected['日'])
        df_selected['cost'] = pd.to_numeric(df_selected['cost'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        df_selected['conversions'] = pd.to_numeric(df_selected['conversions'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    except Exception as e:
        return None, f"データ型変換中にエラーが発生しました: {e}。CSVのフォーマットを確認してください。", None

    # 特徴量エンジニアリング
    df_selected['weekday'] = df_selected['日'].dt.weekday
    df_selected['month'] = df_selected['日'].dt.month
    df_selected['week'] = df_selected['日'].dt.isocalendar().week.astype(int)
    df_selected['is_holiday'] = df_selected['日'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    df_selected['log_cost'] = np.log1p(df_selected['cost'])

    # Googleトレンドデータを取得してマージ
    try:
        trends_df = fetch_and_prepare_trends_data(training_start_date, training_end_date, trend_keywords_dict)
        df_selected = pd.merge(df_selected, trends_df, on='日', how='left')
        df_selected[list(trend_keywords_dict.keys())] = df_selected[list(trend_keywords_dict.keys())].fillna(0)
    except Exception as e:
        return None, f"Googleトレンドデータの取得またはマージに失敗しました: {e}", None

    # 学習期間でフィルタリング
    training_df = df_selected[(df_selected['日'] >= pd.to_datetime(training_start_date)) & (df_selected['日'] <= pd.to_datetime(training_end_date))]
    
    if training_df.empty:
        return None, "指定された学習期間に有効なデータがありませんでした。", None

    return training_df.sort_values(by='日').reset_index(drop=True), None

# --- 2. モデル学習関数 ---
@st.cache_data
def train_models(_df, trend_keywords_dict):
    """キャンペーンごとにXGBoostモデルを学習し、特徴量の重要度も返す"""
    models = {}
    feature_importances = pd.DataFrame()
    features = get_features(trend_keywords_dict)
    
    campaign_names = _df['campaign_name'].unique()
    latest_date = _df['日'].max()
    
    for campaign in campaign_names:
        campaign_df = _df[_df['campaign_name'] == campaign].copy()
        if len(campaign_df) < 10: continue

        # 直近のデータに重み付け
        recency_in_days = (latest_date - campaign_df['日']).dt.days
        sample_weights = np.where(recency_in_days <= 30, 3.0, 1.0)
        
        X = campaign_df[features]
        y = campaign_df['conversions']
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
            max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model.fit(X, y, sample_weight=sample_weights)
        models[campaign] = model

        # 特徴量の重要度を格納
        temp_importance = pd.DataFrame(
            data={'feature': features, 'importance': model.feature_importances_, 'campaign': campaign}
        )
        feature_importances = pd.concat([feature_importances, temp_importance], ignore_index=True)
        
    return models, feature_importances

# --- 3. 最適化関数 ---
def optimize_budget_allocation(total_budget, models, features_today, campaign_max_budgets, trend_keywords_dict):
    """数理最適化を用いて、CVを最大化する予算配分を計算する"""
    campaign_names = list(models.keys())
    features = get_features(trend_keywords_dict)
    problem = pulp.LpProblem("Budget_Allocation_Problem", pulp.LpMaximize)
    
    # 予算の刻み幅を設定
    step = max(1000, int(total_budget / 100))
    budget_steps = list(range(0, total_budget + 1, step))
    if not budget_steps: budget_steps = [0, total_budget]
    
    choices = pulp.LpVariable.dicts("Choice", (campaign_names, budget_steps), cat='Binary')
    
    # 各キャンペーン・各予算ステップでのCV数を予測
    predicted_cvs = {}
    for campaign in campaign_names:
        predicted_cvs[campaign] = {}
        pred_data_list = []
        for budget in budget_steps:
            pred_data_point = {'cost': budget, 'log_cost': np.log1p(budget), **features_today}
            pred_data_list.append(pred_data_point)

        pred_df = pd.DataFrame(pred_data_list)[features]
        predictions = models[campaign].predict(pred_df)
        for i, budget in enumerate(budget_steps):
            predicted_cvs[campaign][budget] = max(0, predictions[i])

    # 目的関数：総CV数を最大化
    problem += pulp.lpSum(predicted_cvs[c][b] * choices[c][b] for c in campaign_names for b in budget_steps)
    
    # 制約条件
    # 1. 総予算を超えない
    problem += pulp.lpSum(b * choices[c][b] for c in campaign_names for b in budget_steps) <= total_budget
    # 2. 各キャンペーンは1つの予算しか選べない
    for c in campaign_names:
        problem += pulp.lpSum(choices[c][b] for b in budget_steps) == 1
    # 3. 各キャンペーンの上限予算を超えない
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
    
    return optimal_allocation, total_predicted_cv, cv_per_campaign, pulp.LpStatus[problem.status]

# --- 4. Streamlit UI ---
st.title('🚀 広告予算配分 最適化シミュレーター (v2)')

# --- session_state の初期化 ---
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'column_mapping_required' not in st.session_state:
    st.session_state.column_mapping_required = False
if 'trend_keywords' not in st.session_state:
    # デフォルトのキーワードを設定
    st.session_state.trend_keywords = collections.OrderedDict({
        'general_trend': ['塾講師 バイト', '塾 バイト'],
        'brand_trend': ['塾講師ステーション'],
        'school_trend': ['早稲田アカデミー バイト', '武田塾 バイト', 'ITTO個別指導学院 バイト', '臨海セミナー バイト'],
        'station_trend': ['東京 塾 求人', '大阪 塾 求人', '名古屋 塾 求人', '京都 塾 求人']
    })

# --- サイドバー ---
with st.sidebar:
    st.header('⚙️ 基本設定')
    uploaded_file = st.file_uploader("① パフォーマンスレポートをアップロード", type=['csv'])

    if uploaded_file:
        # 学習期間の自動設定
        default_start_date = datetime.now().date() - timedelta(days=90)
        default_end_date = datetime.now().date()
        try:
            uploaded_file.seek(0)
            temp_df = pd.read_csv(uploaded_file, encoding='cp932', on_bad_lines='skip')
            date_col = next((col for col in temp_df.columns if temp_df[col].astype(str).str.match(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}').any()), None)
            if date_col:
                dates = pd.to_datetime(temp_df[date_col], errors='coerce').dropna()
                if not dates.empty:
                    default_start_date, default_end_date = dates.min().date(), dates.max().date()
        except Exception:
            pass # エラーでもデフォルト値で続行
        finally:
            uploaded_file.seek(0)

        st.subheader("② 学習データ期間")
        training_start_date = st.date_input('開始日', value=default_start_date)
        training_end_date = st.date_input('終了日', value=default_end_date)
    
    # トレンドキーワード設定
    with st.expander("③ Googleトレンドキーワード設定 (推奨)", expanded=False):
        st.info("予測精度向上のため、関連するキーワードをカテゴリごとに入力してください。カンマ区切りで複数指定可能です。")
        
        temp_keywords = {}
        for category, keywords in st.session_state.trend_keywords.items():
            input_str = st.text_area(f"カテゴリ: {category}", ", ".join(keywords), height=50)
            temp_keywords[category] = [kw.strip() for kw in input_str.split(',') if kw.strip()]
        
        if st.button("キーワードを更新"):
            st.session_state.trend_keywords = collections.OrderedDict(temp_keywords)
            st.success("キーワードを更新しました。")

    if uploaded_file:
        process_button = st.button('データを処理し、モデルを学習する', type="primary", use_container_width=True)
    else:
        process_button = False

# --- メイン画面 ---
if uploaded_file is None:
    st.info("サイドバーからCSVファイルをアップロードして開始してください。")
else:
    # 列名マッピング処理
    column_mapping = {'日': '日', 'キャンペーン名': 'campaign_name', 'コスト': 'cost', 'コンバージョン数': 'conversions'}
    if st.session_state.column_mapping_required:
        with st.expander("⚠️ 列名のマッピングが必要です", expanded=True):
            st.warning("CSVの列名を特定できませんでした。以下のドロップダウンから対応する列名を選択してください。")
            raw_cols = st.session_state.raw_df_columns
            user_mapping = {}
            for jp_col, en_col in column_mapping.items():
                selected_col = st.selectbox(f"「{jp_col}」に対応する列", [None] + raw_cols)
                if selected_col:
                    user_mapping[selected_col] = en_col
            column_mapping = {k: v for k, v in user_mapping.items() if k is not None}

    if process_button:
        with st.spinner('データ前処理とGoogleトレンドデータ取得、モデル学習を実行中...'):
            data, error_message = preprocess_data(uploaded_file, column_mapping, training_start_date, training_end_date, st.session_state.trend_keywords)
            if error_message:
                st.error(f"エラー: {error_message}")
                st.session_state.data_processed = False
            else:
                st.session_state.original_data = data
                models, importances = train_models(data, st.session_state.trend_keywords)
                st.session_state.trained_models = models
                st.session_state.feature_importances = importances
                st.session_state.data_processed = True
                st.success("データ処理とモデル学習が完了しました。")
    
    if st.session_state.data_processed:
        st.header("④ 最適化シミュレーション")
        
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
                with st.spinner('最適化計算を実行中...'):
                    # 最適化期間のトレンドデータを取得
                    trends_for_optim = fetch_and_prepare_trends_data(optim_start_date, optim_end_date, st.session_state.trend_keywords)
                    trends_for_optim.set_index('日', inplace=True)

                    date_range = pd.date_range(optim_start_date, optim_end_date)
                    daily_results = []
                    progress_bar = st.progress(0, text="最適化計算を実行中...")

                    for i, target_date in enumerate(date_range):
                        target_date_ts = pd.Timestamp(target_date)
                        trend_date_to_use = min(target_date_ts, trends_for_optim.index.max())
                        
                        features_for_today = {
                            'weekday': target_date.weekday(), 'month': target_date.month,
                            'week': target_date.isocalendar().week,
                            'is_holiday': 1 if jpholiday.is_holiday(target_date) else 0,
                            **trends_for_optim.loc[trend_date_to_use].to_dict()
                        }

                        optimal_budgets, total_cv, cv_breakdown, status = optimize_budget_allocation(
                            total_daily_budget, st.session_state.trained_models, features_for_today, 
                            campaign_max_budgets_input, st.session_state.trend_keywords
                        )
                        daily_results.append({'date': target_date, 'allocation': optimal_budgets, 'cv': total_cv, 'cv_breakdown': cv_breakdown, 'status': status})
                        progress_bar.progress((i + 1) / len(date_range), text=f"最適化計算: {target_date.strftime('%m/%d')}")
                    
                    st.session_state.daily_results = daily_results
                    st.session_state.optim_period = (optim_start_date, optim_end_date)
                    st.success("最適化が完了しました。")

        # --- 結果表示 ---
        if 'daily_results' in st.session_state and st.session_state.daily_results:
            st.header(f'📊 最適化結果（{st.session_state.optim_period[0].strftime("%Y/%m/%d")} 〜 {st.session_state.optim_period[1].strftime("%Y/%m/%d")}）')
            
            daily_results = st.session_state.daily_results
            # 1日でも最適化が失敗したかチェック
            if any(res['status'] != "Optimal" for res in daily_results):
                failed_dates = [res['date'].strftime('%Y-%m-%d') for res in daily_results if res['status'] != "Optimal"]
                st.error(f"以下の日付で最適化に失敗しました: {', '.join(failed_dates)}\n\n"
                         f"**考えられる原因と対策:**\n"
                         f"- 設定された「1日あたりの総予算」が低すぎる可能性があります。\n"
                         f"- 「キャンペーン別の上限予算」の制約が厳しすぎる可能性があります。\n\n"
                         f"予算設定を見直して、再度最適化を実行してください。")
            
            # パフォーマンスサマリー
            with st.container(border=True):
                st.subheader("📈 パフォーマンスサマリー")
                total_allocated_sum = sum(sum(res['allocation'].values()) for res in daily_results if res['allocation'])
                total_predicted_cv_sum = sum(res['cv'] for res in daily_results if res['cv'] is not None)
                total_predicted_cpa = total_allocated_sum / total_predicted_cv_sum if total_predicted_cv_sum > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label="予測 配分合計金額", value=f"{total_allocated_sum:,.0f} 円")
                col2.metric(label="予測 総CV数", value=f"{total_predicted_cv_sum:.2f} 件")
                col3.metric(label="予測 CPA", value=f"{total_predicted_cpa:,.0f} 円")

            # 結果詳細
            avg_allocations = pd.DataFrame([res['allocation'] for res in daily_results if res['allocation']]).mean()
            avg_cvs = pd.DataFrame([res['cv_breakdown'] for res in daily_results if res['cv_breakdown']]).mean()
            result_df = pd.DataFrame({'1日あたりの平均推奨予算': avg_allocations, '1日あたりの平均予測CV数': avg_cvs}).fillna(0)
            result_df['予測CPA'] = (result_df['1日あたりの平均推奨予算'] / result_df['1日あたりの平均予測CV数']).replace([np.inf, -np.inf], 0).fillna(0)
            result_df = result_df.round(2).astype({'1日あたりの平均推奨予算': int, '予測CPA': int})
            
            tab1, tab2, tab3 = st.tabs(["サマリー & 推移", "日別詳細データ", "予測モデルの診断情報"])

            with tab1:
                col1, col2 = st.columns([0.6, 0.4])
                with col1:
                    st.subheader("📋 キャンペーン別パフォーマンス（日平均）")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        label="詳細結果をダウンロード (Excel)", data=to_excel(result_df),
                        file_name=f"optimization_result_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel", use_container_width=True
                    )
                with col2:
                    st.subheader("🍰 予算配分比率（日平均）")
                    plot_df = result_df[result_df['1日あたりの平均推奨予算'] > 0]
                    if not plot_df.empty:
                        fig_pie = px.pie(plot_df, values='1日あたりの平均推奨予算', names=plot_df.index, hole=.3)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("📅 日別パフォーマンス推移")
                daily_cv_df = pd.DataFrame([{'date': r['date'], 'cv': r['cv']} for r in daily_results if r['cv'] is not None]).set_index('date')
                if not daily_cv_df.empty:
                    fig_line = px.line(daily_cv_df, x=daily_cv_df.index, y='cv', title='日別 予測CV数推移', markers=True)
                    fig_line.update_layout(xaxis_title='日付', yaxis_title='予測CV数')
                    st.plotly_chart(fig_line, use_container_width=True)

            with tab2:
                st.subheader("📋 日別 予算配分詳細")
                alloc_df = pd.DataFrame([res['allocation'] for res in daily_results], index=[res['date'].strftime('%Y-%m-%d') for res in daily_results]).fillna(0).astype(int)
                st.dataframe(alloc_df, use_container_width=True)

            with tab3:
                st.subheader("🩺 特徴量の重要度")
                st.info("これは、どの要素（曜日、月、祝日、トレンドなど）がCV数予測に影響を与えたかを示す指標です。最適化結果の「健全性」を確認するための診断機能としてご活用ください。")
                
                importances = st.session_state.get('feature_importances')
                if importances is not None and not importances.empty:
                    # キャンペーンごとに平均を取る
                    avg_importances = importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
                    fig_imp = px.bar(avg_importances, x=avg_importances.values, y=avg_importances.index, orientation='h', title='特徴量の重要度（全キャンペーン平均）')
                    fig_imp.update_layout(xaxis_title='重要度', yaxis_title='特徴量')
                    st.plotly_chart(fig_imp, use_container_width=True)
