# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager
import warnings
import io

# 日本語フォントの設定
try:
    font_path = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family='IPAexGothic'))
    if font_path:
        plt.rcParams['font.family'] = 'IPAexGothic'
except:
    st.warning("日本語フォント（IPAexGothic）が見つかりませんでした。グラフの日本語が文字化けする可能性があります。")

# 警告を非表示にする
warnings.filterwarnings('ignore')

# Matplotlibのスタイルを設定
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Streamlit アプリケーションのUI設定 ==============
st.set_page_config(
    page_title="CPA予測モデル v3.0",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- カスタムCSSで見た目を調整 ---
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-1kyxreq {
        border-radius: 10px;
    }
    .st-emotion-cache-1kyxreq {
        background-color: #e8f0fe; /* 戦略プランの背景色 */
        border: 1px solid #1967d2;
    }
</style>
""", unsafe_allow_html=True)


# ============== 関数の定義 ==============

@st.cache_data
def load_and_preprocess_data(file, recency_days, recency_weight):
    """
    単一のCSVファイルを読み込み、前処理を行う関数。
    - 日付カラムの自動認識
    - 特徴量エンジニアリング (CTR, CVR, CPC)
    - 直近データへの重み付け
    - データクレンジングと外れ値除去
    """
    if file is None:
        return None, None, "ファイルがアップロードされていません。サイドバーからCSVファイルをアップロードしてください。"

    log_messages = []

    def read_csv_robust(f):
        encodings = ['utf-8-sig', 'shift_jis', 'cp932', 'utf-8']
        f.seek(0)
        for encoding in encodings:
            try:
                return pd.read_csv(f, encoding=encoding)
            except UnicodeDecodeError:
                f.seek(0)
                continue
        raise UnicodeDecodeError(f"ファイル '{f.name}' をどのエンコーディングでも読み込めませんでした。")

    try:
        df = read_csv_robust(file)
        log_messages.append("✅ CSVファイルの読み込みに成功しました。")
    except Exception as e:
        return None, None, f"❌ ファイル読み込みエラー: {e}"

    # --- 日付カラムの自動認識 ---
    date_col_candidates = ['日', '日付', 'Date', 'Day']
    date_col = None
    for col in date_col_candidates:
        if col in df.columns:
            date_col = col
            break
    if not date_col:
        return None, None, f"❌ 日付カラムが見つかりません。カラム名を {', '.join(date_col_candidates)} のいずれかに変更してください。"
    
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        log_messages.append(f"✅ 日付カラム '{date_col}' を認識しました。")
    except Exception as e:
        return None, None, f"❌ 日付カラム '{date_col}' を日付形式に変換できませんでした: {e}"

    # --- データクレンジングと型変換 ---
    df.columns = df.columns.str.replace(' ', '').str.replace('　', '')
    required_columns = ['インプレッション数', 'クリック数', 'コスト', 'コンバージョン数']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, None, f"❌ 必要なカラムが見つかりません: {missing_cols}"

    for col in required_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=required_columns, inplace=True)
    log_messages.append("✅ データ型のクレンジングと欠損値の除去を行いました。")

    # --- 特徴量エンジニアリング ---
    df['CTR'] = np.divide(df['クリック数'], df['インプレッション数'], out=np.zeros_like(df['クリック数'], dtype=float), where=df['インプレッション数']!=0)
    df['CVR'] = np.divide(df['コンバージョン数'], df['クリック数'], out=np.zeros_like(df['コンバージョン数'], dtype=float), where=df['クリック数']!=0)
    df['CPC'] = np.divide(df['コスト'], df['クリック数'], out=np.zeros_like(df['コスト'], dtype=float), where=df['クリック数']!=0)
    log_messages.append("✅ 特徴量（CTR, CVR, CPC）を生成しました。")

    # --- 重み付け ---
    latest_date = df[date_col].max()
    df['weight'] = np.where(df[date_col] >= latest_date - pd.Timedelta(days=recency_days), recency_weight, 1.0)
    log_messages.append(f"✅ 直近{recency_days}日間のデータに{recency_weight}倍の重みを付けました。")

    # --- 外れ値除去 ---
    Q1, Q3 = df['コスト'].quantile(0.25), df['コスト'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    original_rows = len(df)
    df_cleaned = df[(df['コスト'] >= lower_bound) & (df['コスト'] <= upper_bound)].copy()
    log_messages.append(f"✅ 外れ値を除去しました。(元のデータ数: {original_rows}, 除去後: {len(df_cleaned)})")

    if len(df_cleaned) < 30:
        return None, None, "❌ 前処理後のデータが少なすぎます（30件未満）。十分なデータ量のあるCSVをアップロードしてください。"
    
    log_messages.append("✅ データの前処理が完了しました。")
    return df_cleaned, log_messages, None

@st.cache_resource
def train_and_evaluate_model(_df_cleaned):
    """
    単一のLightGBMモデルを学習させ、クロスバリデーションで評価する関数。
    """
    features = ['コスト', 'インプレッション数', 'クリック数', 'CTR', 'CVR', 'CPC']
    # 'インプレッションシェア' など、シミュレーションで変動させられないものは一旦除外
    # 必要であれば、シミュレーションモデルに追加する
    target = 'コンバージョン数'

    X, y, weights = _df_cleaned[features], _df_cleaned[target], _df_cleaned['weight']

    model_params = {
        'objective': 'poisson',
        'random_state': 42,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 10,
        'max_depth': 7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'n_jobs': -1
    }
    
    # --- クロスバリデーションによるモデル評価 ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(_df_cleaned))
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        weights_train = weights.iloc[train_idx]

        model_cv = lgb.LGBMRegressor(**model_params)
        model_cv.fit(X_train, y_train, sample_weight=weights_train)
        
        val_preds = model_cv.predict(X_val)
        maes.append(mean_absolute_error(y_val, val_preds))

    avg_mae = np.mean(maes)

    # --- 全データで最終モデルを学習 ---
    final_model = lgb.LGBMRegressor(**model_params)
    final_model.fit(X, y, sample_weight=weights)

    feature_importance = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': final_model.feature_importances_
    }).sort_values('重要度', ascending=False)

    return final_model, feature_importance, avg_mae

@st.cache_resource
def create_simulation_models(_df_cleaned):
    """
    シミュレーションでコストに連動して変動する指標を予測するための、
    単純な線形回帰モデル群を作成する。
    """
    sim_models = {}
    # コストが上がると変動する可能性のある指標
    sim_target_features = ['インプレッション数', 'クリック数', 'CPC'] 
    
    for feature in sim_target_features:
        X = _df_cleaned[['コスト']]
        y = _df_cleaned[feature]
        model = LinearRegression()
        model.fit(X, y)
        sim_models[feature] = model
        
    return sim_models

def run_simulation(df_cleaned, main_model, sim_models, input_budget):
    """
    指定された予算範囲でシミュレーションを実行し、結果をDataFrameで返す。
    """
    min_cost_data = df_cleaned['コスト'].min()
    max_cost_data = df_cleaned['コスト'].max()
    
    graph_max_cost = max(max_cost_data, input_budget) * 1.2
    num_steps = 200
    cost_range = np.linspace(min_cost_data, graph_max_cost, num_steps)
    
    sim_results = []

    for cost in cost_range:
        sim_data = {'コスト': cost}
        
        # シミュレーションモデルで各指標を予測
        for feature, model in sim_models.items():
            sim_data[feature] = model.predict(pd.DataFrame({'コスト': [cost]}))[0]

        # 予測した指標から、さらに特徴量を計算
        sim_data['CTR'] = np.divide(sim_data['クリック数'], sim_data['インプレッション数'], where=sim_data['インプレッション数']!=0)
        # CVRは学習データの平均値を仮定（コスト変動でCVR自体は大きく変わらないと仮定）
        sim_data['CVR'] = df_cleaned['CVR'].mean() 

        sim_df_row = pd.DataFrame([sim_data])
        
        # メインモデルが学習した特徴量の順番に合わせる
        features_for_prediction = main_model.feature_name_
        
        # メインモデルでCV数を予測
        predicted_cv = main_model.predict(sim_df_row[features_for_prediction])[0]
        
        sim_results.append({'コスト': cost, '予測コンバージョン数': predicted_cv})

    sim_df = pd.DataFrame(sim_results)
    sim_df['予測コンバージョン数'] = sim_df['予測コンバージョン数'].clip(lower=0)
    sim_df['予測CPA'] = sim_df.apply(lambda r: r['コスト'] / r['予測コンバージョン数'] if r['予測コンバージョン数'] > 0.01 else np.nan, axis=1)
    
    sim_df.dropna(subset=['予測CPA'], inplace=True)
    return sim_df

def create_plot(sim_df, input_budget, predicted_cv, predicted_cpa, cpa_best_point, cv_max_point):
    """
    シミュレーション結果を可視化するグラフを生成する。
    戦略的ポイントもハイライトする。
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 1軸目: 予測コンバージョン数
    color = 'royalblue'
    ax1.set_xlabel('予算（コスト）[円]', fontsize=14)
    ax1.set_ylabel('予測コンバージョン数', fontsize=14, color=color)
    ax1.plot(sim_df['コスト'], sim_df['予測コンバージョン数'], linestyle='-', color=color, label='予測コンバージョン数', linewidth=2.5, zorder=10)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 2軸目: 予測CPA
    ax2 = ax1.twinx()
    color = 'mediumseagreen'
    ax2.set_ylabel('予測CPA [円]', fontsize=14, color=color)
    ax2.plot(sim_df['コスト'], sim_df['予測CPA'], linestyle='--', color=color, label='予測CPA', linewidth=2.5, zorder=10)
    ax2.tick_params(axis='y', labelcolor=color)

    # 入力された予算のプロット
    if predicted_cv is not None and predicted_cpa is not None:
        ax1.axvline(x=input_budget, color='tomato', linestyle=':', linewidth=2, label=f"入力予算: {input_budget:,.0f}円", zorder=5)
        ax1.plot(input_budget, predicted_cv, 'o', color='tomato', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=20)
        ax2.plot(input_budget, predicted_cpa, 'o', color='tomato', markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=20)

    # 戦略的ポイントのプロット
    if cpa_best_point is not None:
        ax1.plot(cpa_best_point['コスト'], cpa_best_point['予測コンバージョン数'], 'D', color='gold', markersize=12, markeredgecolor='black', label='CPA最良化点', zorder=21)
        ax2.plot(cpa_best_point['コスト'], cpa_best_point['予測CPA'], 'D', color='gold', markersize=12, markeredgecolor='black', zorder=21)

    if cv_max_point is not None:
        ax1.plot(cv_max_point['コスト'], cv_max_point['予測コンバージョン数'], 's', color='darkviolet', markersize=12, markeredgecolor='white', label='CV最大化点', zorder=21)
        ax2.plot(cv_max_point['コスト'], cv_max_point['予測CPA'], 's', color='darkviolet', markersize=12, markeredgecolor='white', zorder=21)

    # 凡例をまとめる
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 重複ラベルを削除
    from collections import OrderedDict
    all_labels = labels + labels2
    all_lines = lines + lines2
    by_label = OrderedDict(zip(all_labels, all_lines))
    ax2.legend(by_label.values(), by_label.keys(), loc='best', fontsize=12, frameon=True, shadow=True)
    
    fig.tight_layout()
    return fig


# ============== UI: サイドバー ==============
with st.sidebar:
    st.markdown("## ⚙️ 設定パネル")
    
    with st.expander("📁 データアップロード", expanded=True):
        uploaded_file = st.file_uploader(
            "パフォーマンスレポート (CSV)", 
            type="csv",
            help="日付カラム（'日'または'日付'）を含むCSVファイルをアップロードしてください。"
        )

    st.markdown("---")
    st.markdown("## 🛠️ モデル詳細設定")
    recency_days = st.number_input("直近と見なす期間（日）", min_value=1, max_value=90, value=30, step=1)
    recency_weight = st.slider("直近データの重み", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

    st.markdown("---")
    st.markdown("## 予算シミュレーション")
    input_budget = st.number_input(
        "予測したい予算（円）を入力", 
        min_value=0, 
        value=None, 
        placeholder="例: 50000", 
        step=1000, 
        help="任意の予算額を円単位で入力してください。"
    )


# ============== UI: メインページ ==============
st.title('💡 CPA予測モデル v3.0')
st.markdown("過去の広告パフォーマンスデータから、最適な広告予算と目標CPAをデータドリブンで導き出します。")

with st.expander("🚀 このツールの特徴"):
    st.markdown("""
    - **単一ファイルで分析**: 日付カラムを含むCSVを1つアップロードするだけで、すぐに分析を開始できます。
    - **戦略プランの自動提案**: 「効率重視」と「獲得重視」の2つの戦略プランを自動で算出し、意思決定をサポートします。
    - **高精度な予測モデル**: LightGBMを採用し、特徴量エンジニアリングを導入することで、より現実に即した予測を提供します。
    - **信頼性の可視化**: モデルの予測誤差（MAE）を提示し、予測結果の信頼性を客観的に判断できます。
    """)

st.markdown("---")

# ============== メイン処理 ==============
if uploaded_file:
    with st.spinner('データを読み込み、モデルを準備中です...'):
        df_cleaned, log_messages, error_message = load_and_preprocess_data(uploaded_file, recency_days, recency_weight)

    if error_message:
        st.error(error_message)
    else:
        with st.spinner('予測モデルを学習・評価中です...'):
            main_model, feature_importance, avg_mae = train_and_evaluate_model(df_cleaned)
            sim_models = create_simulation_models(df_cleaned)

        st.success("モデルの準備が完了しました。")

        with st.container(border=True):
            st.markdown("#### 🧠 モデルの信頼性")
            st.metric(label="予測誤差の平均（MAE）", value=f"{avg_mae:.2f} 件",
                      help="このモデルの予測には、平均してこれくらいの件数の誤差が見込まれることを示します。値が小さいほど信頼性が高いと判断できます。")

        # シミュレーションと戦略プランの計算
        # input_budgetがNoneでもグラフ描画のために一度実行
        sim_budget = input_budget if input_budget is not None else df_cleaned['コスト'].median()
        sim_df = run_simulation(df_cleaned, main_model, sim_models, sim_budget)

        if not sim_df.empty:
            cpa_best_point = sim_df.loc[sim_df['予測CPA'].idxmin()]
            cv_max_point = sim_df.loc[sim_df['予測コンバージョン数'].idxmax()]

            st.markdown("---")
            st.markdown("### 📈 戦略プランのご提案")
            st.markdown("データに基づき、2つの戦略的な選択肢を算出しました。")

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("##### 🥇 効率重視プラン (CPA最良化)")
                    st.metric("推奨予算", f"{cpa_best_point['コスト']:,.0f} 円")
                    st.metric("予測CPA", f"{cpa_best_point['予測CPA']:,.0f} 円")
                    st.metric("予測CV数", f"{cpa_best_point['予測コンバージョン数']:.2f} 件")
            with col2:
                with st.container(border=True):
                    st.markdown("##### 🚀 獲得重視プラン (CV最大化)")
                    st.metric("推奨予算", f"{cv_max_point['コスト']:,.0f} 円")
                    st.metric("予測CPA", f"{cv_max_point['予測CPA']:,.0f} 円")
                    st.metric("予測CV数", f"{cv_max_point['予測コンバージョン数']:.2f} 件")
        else:
            cpa_best_point, cv_max_point = None, None
            st.warning("シミュレーション結果が空のため、戦略プランを提案できませんでした。")

        # ユーザー入力予算での予測
        predicted_cv_user, predicted_cpa_user = None, None
        if input_budget is not None and input_budget > 0:
            sim_df_user = run_simulation(df_cleaned, main_model, sim_models, input_budget)
            if not sim_df_user.empty:
                # 入力予算に最も近い行を取得
                user_point = sim_df_user.iloc[(sim_df_user['コスト']-input_budget).abs().argsort()[:1]]
                predicted_cv_user = user_point['予測コンバージョン数'].iloc[0]
                predicted_cpa_user = user_point['予測CPA'].iloc[0]

                st.markdown("---")
                st.markdown(f"### 予算 **{input_budget:,.0f}円** のシミュレーション結果")
                
                max_cost_data = df_cleaned['コスト'].max()
                if input_budget > max_cost_data * 1.1:
                    st.warning(f"⚠️ **警告:** 入力された予算は、学習データの最大コスト ({max_cost_data:,.0f}円) を大きく超えています。予測は外挿であり、精度が低い可能性があります。")

                col1_user, col2_user = st.columns(2)
                with col1_user:
                    st.metric(label="🎯 予測コンバージョン数", value=f"{predicted_cv_user:.2f} 件")
                with col2_user:
                    cpa_display = f"{predicted_cpa_user:,.0f} 円" if pd.notna(predicted_cpa_user) else "算出不可"
                    st.metric(label="💰 予測CPA", value=cpa_display)

        # グラフと詳細情報の表示
        tab1, tab2, tab3 = st.tabs(["📊 **予測結果の全体像グラフ**", "🧠 **モデルの分析情報**", "📄 **学習データ詳細**"])

        with tab1:
            if not sim_df.empty:
                with st.spinner('グラフを生成中です...'):
                    fig = create_plot(sim_df, input_budget, predicted_cv_user, predicted_cpa_user, cpa_best_point, cv_max_point)
                    st.pyplot(fig)
            else:
                st.warning("シミュレーション結果が空のため、グラフを描画できませんでした。")
        with tab2:
            st.markdown("#### モデルの特徴量重要度")
            st.markdown("コンバージョン数予測モデルが、どの指標を重視して予測を行ったかを示します。")
            st.dataframe(feature_importance.style.background_gradient(cmap='viridis', subset=['重要度']))
            st.markdown("---")
            st.markdown("#### データ処理ログ")
            st.code("\n".join(log_messages))
        with tab3:
            st.markdown("モデルの学習に使用されたデータ（特徴量生成・外れ値除去後）です。")
            st.dataframe(df_cleaned.head(100))
            
else:
    st.info('サイドバーの「データアップロード」から、分析用のCSVファイルをアップロードしてください。')

