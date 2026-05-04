import hashlib
import json
import random
import re
import string
import time
from datetime import datetime
from pathlib import Path

import requests as _requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── Supabase REST helpers (no client library needed) ───────────────────────────
def _sb_available() -> bool:
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        return bool(url and key and "your-project-id" not in url)
    except Exception:
        return False

def _sb_headers() -> dict:
    key = st.secrets.get("SUPABASE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

def _sb_url(path: str) -> str:
    base = st.secrets.get("SUPABASE_URL", "").rstrip("/")
    return f"{base}/rest/v1/{path}"

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    VotingClassifier, StackingClassifier, HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, LocalOutlierFactor
from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.utils.multiclass import type_of_target

HAS_XGB = HAS_LGBM = HAS_CATBOOST = HAS_IMBLEARN = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    pass
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    pass
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    pass
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    pass

# ── Model constants ────────────────────────────────────────────────────────────
ANOMALY_MODELS = {"Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope"}
MINMAX_MODELS  = {"Bernoulli Naive Bayes", "Complement Naive Bayes"}
ESTIMATOR_MODELS = {
    "Random Forest", "Extra Trees", "Gradient Boosting", "Histogram Gradient Boosting",
    "AdaBoost", "Bagging", "XGBoost", "LightGBM", "CatBoost",
    "Isolation Forest", "Balanced Random Forest",
}
MODEL_CATEGORIES = {
    "Ensemble": [
        "Random Forest", "Extra Trees", "Gradient Boosting",
        "Histogram Gradient Boosting", "AdaBoost", "Bagging",
        "Voting Classifier", "Stacking Classifier",
    ] + (["XGBoost"] if HAS_XGB else [])
      + (["LightGBM"] if HAS_LGBM else [])
      + (["CatBoost"] if HAS_CATBOOST else [])
      + (["Balanced Random Forest"] if HAS_IMBLEARN else []),
    "Linear & Discriminant": [
        "Logistic Regression", "Ridge Classifier", "SGD Classifier",
        "Passive Aggressive", "Perceptron",
        "Linear Discriminant Analysis", "Quadratic Discriminant Analysis",
    ],
    "Tree":           ["Decision Tree", "Extra Tree (Single)"],
    "Naive Bayes":    ["Gaussian Naive Bayes", "Bernoulli Naive Bayes", "Complement Naive Bayes"],
    "Neighbors":      ["K-Nearest Neighbors", "Nearest Centroid"],
    "SVM":            ["SVM (RBF)", "SVM (Linear)", "SVM (Poly)", "Nu-SVC", "Linear SVC"],
    "Neural Network": ["MLP Neural Network"],
    "Anomaly Detection": [
        "Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope",
    ],
}
TOTAL_MODELS = sum(len(v) for v in MODEL_CATEGORIES.values())

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI — Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body { font-family: 'Inter', sans-serif !important; }

/* ── Global background ───────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at 0% 0%,   rgba(34,211,238,0.10) 0%, transparent 42%),
        radial-gradient(ellipse at 100% 0%,  rgba(139,92,246,0.10) 0%, transparent 42%),
        radial-gradient(ellipse at 50% 100%, rgba(59,130,246,0.08) 0%, transparent 40%),
        linear-gradient(180deg, #04101f 0%, #070d1c 40%, #090e20 100%);
    color: #dde7ff;
    min-height: 100vh;
}
.block-container { max-width: 1440px; padding-top: 1rem; padding-bottom: 2.5rem; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(4,9,22,0.99) 0%, rgba(7,13,28,0.99) 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] * { color: #dde7ff !important; }

/* ── Glass panels ────────────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 1.5rem 1.6rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.28);
    margin-bottom: 1rem;
}
.glass-sm {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.20);
    margin-bottom: 0.75rem;
}

/* ── Auth card ───────────────────────────────────────── */
.auth-card {
    background: rgba(10,16,36,0.85);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 28px;
    padding: 2.6rem 3rem;
    backdrop-filter: blur(28px);
    box-shadow: 0 40px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(34,211,238,0.04);
    position: relative;
    overflow: hidden;
}
.auth-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #22d3ee 0%, #3b82f6 50%, #8b5cf6 100%);
    border-radius: 28px 28px 0 0;
}
.auth-glow {
    position: absolute; top: -80px; right: -80px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(34,211,238,0.06), transparent 70%);
    pointer-events: none;
}
.auth-logo {
    font-size: 1.6rem; font-weight: 900; text-align: center; margin-bottom: 0.25rem;
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.auth-title {
    font-size: 1.55rem; font-weight: 800; color: white;
    text-align: center; margin-bottom: 0.25rem; line-height: 1.2;
}
.auth-subtitle {
    font-size: 0.88rem; color: #6b7fa8;
    text-align: center; margin-bottom: 1.6rem; line-height: 1.5;
}
.auth-divider {
    display: flex; align-items: center; gap: 0.8rem; margin: 1rem 0;
}
.auth-divider hr { flex: 1; border-color: rgba(255,255,255,0.08); margin: 0; }
.auth-divider span { font-size: 0.75rem; color: #3d5070; font-weight: 500; }
.demo-creds {
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.78rem;
    color: #5a8099;
    line-height: 1.6;
}

/* ── OTP card ────────────────────────────────────────── */
.otp-card {
    background: rgba(10,16,36,0.85);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 28px;
    padding: 2.8rem 3rem;
    backdrop-filter: blur(28px);
    box-shadow: 0 40px 80px rgba(0,0,0,0.55);
    position: relative; overflow: hidden;
}
.otp-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #8b5cf6, #22d3ee, #3b82f6);
    border-radius: 28px 28px 0 0;
}
.otp-shield {
    text-align: center; font-size: 3.5rem;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 18px rgba(34,211,238,0.35));
}
.otp-display {
    background: linear-gradient(135deg, rgba(34,211,238,0.07), rgba(59,130,246,0.06));
    border: 1px dashed rgba(34,211,238,0.28);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    text-align: center;
    margin: 1.2rem 0;
    position: relative;
}
.otp-dev-badge {
    position: absolute; top: -10px; left: 50%; transform: translateX(-50%);
    background: rgba(34,211,238,0.15);
    border: 1px solid rgba(34,211,238,0.25);
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.65rem; color: #22d3ee; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    white-space: nowrap;
}
.otp-code {
    font-size: 2.8rem; font-weight: 900;
    letter-spacing: 0.5em;
    color: #22d3ee;
    font-family: 'Courier New', monospace !important;
    text-shadow: 0 0 20px rgba(34,211,238,0.4);
    padding-left: 0.5em;
}
.otp-expiry { font-size: 0.78rem; color: #5a7099; margin-top: 0.35rem; }
.otp-expired { color: #f87171; }

/* ── Hero section ────────────────────────────────────── */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg,
        rgba(34,211,238,0.09) 0%,
        rgba(59,130,246,0.07) 50%,
        rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 26px;
    padding: 2.4rem 2.8rem;
    box-shadow: 0 24px 60px rgba(0,0,0,0.30);
    backdrop-filter: blur(16px);
    margin-bottom: 1.4rem;
}
.hero::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #22d3ee, #3b82f6, #8b5cf6);
}
.hero::after {
    content: "";
    position: absolute; width: 480px; height: 480px;
    right: -120px; top: -120px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.10), transparent 65%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.22);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem; color: #22d3ee; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem; font-weight: 900; color: white;
    margin: 0; line-height: 1.06; letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(90deg, #22d3ee, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-subtitle {
    color: #8aa0c8; margin-top: 0.85rem;
    max-width: 60ch; font-size: 1.05rem; line-height: 1.65;
}
.badge-row { display: flex; gap: 0.65rem; flex-wrap: wrap; margin-top: 1.3rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 999px;
    padding: 0.5rem 0.95rem;
    font-size: 0.84rem; color: #b8cef0; font-weight: 600;
    transition: all 0.2s ease; cursor: default;
}
.badge:hover {
    background: rgba(34,211,238,0.09);
    border-color: rgba(34,211,238,0.26);
    color: #22d3ee; transform: translateY(-1px);
}

/* ── Metric cards ────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1.4rem 1.4rem 1.2rem;
    box-shadow: 0 16px 38px rgba(0,0,0,0.22);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden;
    min-height: 160px;
}
.metric-card::after {
    content: "";
    position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: var(--card-accent, linear-gradient(90deg, #22d3ee, #3b82f6));
    opacity: 0.65;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 24px 48px rgba(0,0,0,0.30); }
.metric-icon { font-size: 1.7rem; margin-bottom: 0.55rem; }
.metric-title {
    color: #6b7fa8; font-size: 0.78rem; margin-bottom: 0.35rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
}
.metric-value { color: white; font-size: 2.1rem; font-weight: 900; line-height: 1.1; }
.metric-sub { color: #4a5d80; font-size: 0.78rem; margin-top: 0.4rem; }

/* ── Small stat ──────────────────────────────────────── */
.small-stat {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    transition: all 0.2s ease;
}
.small-stat:hover { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.11); }
.small-stat-title { color: #5a7099; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
.small-stat-value { color: white; font-weight: 800; font-size: 1.55rem; margin-top: 0.2rem; }

/* ── Section headers ─────────────────────────────────── */
.section-title { color: white; font-size: 1.3rem; font-weight: 800; margin-bottom: 0.15rem; }
.section-sub   { color: #5a7099; font-size: 0.86rem; margin-bottom: 0.9rem; line-height: 1.55; }

/* ── Logo & nav (sidebar) ────────────────────────────── */
.logo-box {
    background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(59,130,246,0.07));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    margin-bottom: 1.1rem;
}
.logo-name {
    font-size: 1.15rem; font-weight: 900;
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.logo-sub { font-size: 0.7rem; color: #3d5070; margin-top: 0.1rem; }
.user-box {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
}
.user-name { font-size: 0.92rem; font-weight: 700; color: white; }
.user-email { font-size: 0.72rem; color: #3d5070; margin-top: 0.1rem; word-break: break-all; }
.role-badge {
    display: inline-block;
    background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.20);
    border-radius: 999px;
    padding: 0.18rem 0.65rem;
    font-size: 0.68rem; color: #22d3ee; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-top: 0.4rem;
}
.nav-section-label {
    font-size: 0.65rem; color: #2d3f5a; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0.5rem 0.5rem 0.35rem;
}

/* ── Activity feed ───────────────────────────────────── */
.activity-item {
    display: flex; align-items: flex-start; gap: 0.75rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.activity-item:last-child { border-bottom: none; }
.activity-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22d3ee; margin-top: 0.38rem; flex-shrink: 0;
    box-shadow: 0 0 7px rgba(34,211,238,0.55);
}
.activity-dot.warn { background: #f59e0b; box-shadow: 0 0 7px rgba(245,158,11,0.55); }
.activity-dot.error { background: #f87171; box-shadow: 0 0 7px rgba(248,113,113,0.55); }
.activity-text { font-size: 0.84rem; color: #b8cef0; line-height: 1.4; }
.activity-time { font-size: 0.72rem; color: #3d5070; margin-top: 0.12rem; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #22d3ee 0%, #3b82f6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 13px !important; font-weight: 700 !important;
    font-size: 0.92rem !important; padding: 0.7rem 1.2rem !important;
    min-height: 48px !important;
    box-shadow: 0 8px 22px rgba(34,211,238,0.18) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px rgba(34,211,238,0.28) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Inputs ──────────────────────────────────────────── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 13px !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-baseweb="input"] > div:focus-within {
    border-color: rgba(34,211,238,0.38) !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.07) !important;
}
input, textarea { font-size: 0.92rem !important; color: #dde7ff !important; }
label { font-size: 0.82rem !important; color: #6b7fa8 !important; font-weight: 600 !important; letter-spacing: 0.02em !important; }

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(255,255,255,0.03);
    border-radius: 14px; padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none; border-radius: 10px;
    color: #6b7fa8; padding: 0.55rem 1.15rem;
    font-size: 0.88rem; font-weight: 600;
    transition: all 0.18s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #aabfdf; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(34,211,238,0.16), rgba(59,130,246,0.12)) !important;
    color: #22d3ee !important;
    border: 1px solid rgba(34,211,238,0.20) !important;
}

/* ── Dataframe ───────────────────────────────────────── */
div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }

/* ── Progress ────────────────────────────────────────── */
.stProgress > div > div { background: linear-gradient(90deg, #22d3ee, #3b82f6) !important; border-radius: 999px !important; }

/* ── Alerts ──────────────────────────────────────────── */
.stAlert { border-radius: 13px !important; }

/* ── Radio ───────────────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio > div { gap: 0.1rem; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.65rem 0.9rem !important;
    border-radius: 11px !important;
    transition: all 0.18s ease !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    border: 1px solid transparent !important;
    color: #7b8db0 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(34,211,238,0.07) !important;
    color: #22d3ee !important;
}

/* ── HR ──────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06); margin: 0.75rem 0; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* ── Caption ─────────────────────────────────────────── */
.stCaption { font-size: 0.78rem !important; color: #3d5070 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
USERS_FILE      = Path("users.json")
OTP_VALIDITY    = 300   # seconds (5 min)


def hash_password(pw: str) -> str:
    return hashlib.sha256(f"fraud-detect-salt::{pw}".encode()).hexdigest()


def validate_email(email: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email.strip()))


def validate_phone(phone: str) -> bool:
    digits = re.sub(r"[\s\-\+\(\)]", "", phone)
    return digits.isdigit() and 7 <= len(digits) <= 15


def generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


_USER_COLS = ["username", "full_name", "email", "phone", "password", "role"]

def _default_users() -> list[dict]:
    return [
        {"username": "admin",      "full_name": "System Admin",    "email": "admin@fraudguard.ai",     "phone": "", "password": hash_password("admin123"),    "role": "Admin"},
        {"username": "researcher", "full_name": "Research Analyst", "email": "research@fraudguard.ai", "phone": "", "password": hash_password("research123"), "role": "Researcher"},
        {"username": "user1",      "full_name": "End User",         "email": "user1@fraudguard.ai",    "phone": "", "password": hash_password("user123"),     "role": "End User"},
    ]


def load_users() -> pd.DataFrame:
    # ── Try Supabase via direct REST call ─────────────────────────────────────
    if _sb_available():
        try:
            cols = ",".join(_USER_COLS)
            r = _requests.get(
                _sb_url(f"users?select={cols}"),
                headers=_sb_headers(),
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if data:
                return pd.DataFrame(data)[_USER_COLS]
            # Empty table on first deploy — seed defaults
            defaults = _default_users()
            _requests.post(
                _sb_url("users"),
                headers={**_sb_headers(), "Prefer": "return=minimal"},
                json=defaults,
                timeout=10,
            ).raise_for_status()
            return pd.DataFrame(defaults)
        except Exception as e:
            st.warning(f"Database error: {e}")

    # ── Fallback: local users.json ────────────────────────────────────────────
    if USERS_FILE.exists():
        try:
            data = json.loads(USERS_FILE.read_text())
            df = pd.DataFrame(data)
            for col in ["email", "full_name", "phone"]:
                if col not in df.columns:
                    df[col] = ""
            mask = df["full_name"].astype(str).str.strip() == ""
            df.loc[mask, "full_name"] = df.loc[mask, "username"]
            return df
        except Exception:
            pass
    defaults = _default_users()
    df = pd.DataFrame(defaults)
    _save_local(df)
    return df


def save_users(df: pd.DataFrame) -> None:
    # ── Try Supabase via direct REST call ─────────────────────────────────────
    if _sb_available():
        try:
            records = df[_USER_COLS].to_dict(orient="records")
            _requests.post(
                _sb_url("users"),
                headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
                json=records,
                timeout=10,
            ).raise_for_status()
            return
        except Exception as e:
            st.warning(f"Database write error: {e}")

    # ── Fallback: local users.json ────────────────────────────────────────────
    _save_local(df)


def _save_local(df: pd.DataFrame) -> None:
    try:
        USERS_FILE.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    except OSError:
        pass


def log_activity(action: str, status: str = "success") -> None:
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    st.session_state.activity_log.insert(0, {
        "action": action,
        "status": status,
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    if len(st.session_state.activity_log) > 60:
        st.session_state.activity_log = st.session_state.activity_log[:60]


# ── UI helpers ─────────────────────────────────────────────────────────────────
def metric_card(title, value, icon="📊", subtext="", accent="linear-gradient(90deg,#22d3ee,#3b82f6)"):
    st.markdown(
        f'<div class="metric-card" style="--card-accent:{accent}">'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{subtext}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def small_stat(title, value):
    st.markdown(
        f'<div class="small-stat"><div class="small-stat-title">{title}</div>'
        f'<div class="small-stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ── ML helpers (preserved exactly) ────────────────────────────────────────────
def make_preprocessor(X: pd.DataFrame, model_name: str = "") -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if X[c].nunique() <= 50]
    scaler = MinMaxScaler() if model_name in MINMAX_MODELS else StandardScaler()
    transformers = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", scaler)]), num_cols)]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=10)),
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers)


def build_model(name, n=150):
    if name == "Random Forest":               return RandomForestClassifier(n_estimators=n, random_state=42, class_weight="balanced")
    if name == "Extra Trees":                 return ExtraTreesClassifier(n_estimators=n, random_state=42, class_weight="balanced")
    if name == "Gradient Boosting":           return GradientBoostingClassifier(n_estimators=n, random_state=42)
    if name == "Histogram Gradient Boosting": return HistGradientBoostingClassifier(max_iter=n, random_state=42)
    if name == "AdaBoost":                    return AdaBoostClassifier(n_estimators=n, random_state=42, algorithm="SAMME")
    if name == "Bagging":                     return BaggingClassifier(n_estimators=n, random_state=42)
    if name == "Voting Classifier":
        return VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ], voting="soft")
    if name == "Stacking Classifier":
        return StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ],
            final_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42), cv=3,
        )
    if name == "XGBoost" and HAS_XGB:          return XGBClassifier(n_estimators=n, random_state=42, eval_metric="logloss", verbosity=0)
    if name == "LightGBM" and HAS_LGBM:        return LGBMClassifier(n_estimators=n, random_state=42, verbose=-1)
    if name == "CatBoost" and HAS_CATBOOST:    return CatBoostClassifier(iterations=n, random_seed=42, verbose=0)
    if name == "Balanced Random Forest" and HAS_IMBLEARN: return BalancedRandomForestClassifier(n_estimators=n, random_state=42)
    if name == "Logistic Regression":          return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "Ridge Classifier":             return RidgeClassifier(class_weight="balanced")
    if name == "SGD Classifier":               return SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42, class_weight="balanced")
    if name == "Passive Aggressive":           return PassiveAggressiveClassifier(max_iter=1000, random_state=42, class_weight="balanced")
    if name == "Perceptron":                   return Perceptron(max_iter=1000, random_state=42, class_weight="balanced")
    if name == "Linear Discriminant Analysis": return LinearDiscriminantAnalysis()
    if name == "Quadratic Discriminant Analysis": return QuadraticDiscriminantAnalysis()
    if name == "Decision Tree":                return DecisionTreeClassifier(random_state=42, class_weight="balanced")
    if name == "Extra Tree (Single)":          return ExtraTreeClassifier(random_state=42, class_weight="balanced")
    if name == "Gaussian Naive Bayes":         return GaussianNB()
    if name == "Bernoulli Naive Bayes":        return BernoulliNB()
    if name == "Complement Naive Bayes":       return ComplementNB()
    if name == "K-Nearest Neighbors":          return KNeighborsClassifier(n_neighbors=5)
    if name == "Nearest Centroid":             return NearestCentroid()
    if name == "SVM (RBF)":    return SVC(kernel="rbf",    probability=True, class_weight="balanced", random_state=42)
    if name == "SVM (Linear)": return SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
    if name == "SVM (Poly)":   return SVC(kernel="poly",   probability=True, class_weight="balanced", random_state=42)
    if name == "Nu-SVC":       return NuSVC(probability=True, random_state=42)
    if name == "Linear SVC":   return LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)
    if name == "MLP Neural Network": return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    if name == "Isolation Forest":   return IsolationForest(n_estimators=n, random_state=42, contamination="auto")
    if name == "One-Class SVM":      return OneClassSVM(nu=0.1, kernel="rbf")
    if name == "Local Outlier Factor": return LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    if name == "Elliptic Envelope":  return EllipticEnvelope(contamination=0.1, random_state=42)
    return RandomForestClassifier(n_estimators=100, random_state=42)


def generate_synthetic_fraud(fraud_df: pd.DataFrame, n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = len(fraud_df)
    fraud_reset = fraud_df.reset_index(drop=True)
    rows = []
    for _ in range(n_samples):
        tmpl = fraud_reset.iloc[rng.integers(0, n_fraud)].copy()
        row = {}
        for col in fraud_reset.columns:
            if pd.api.types.is_numeric_dtype(fraud_reset[col]):
                std = fraud_reset[col].std()
                row[col] = tmpl[col] + (rng.normal(0, std * 0.05) if std > 0 else 0)
            else:
                row[col] = fraud_reset[col].iloc[rng.integers(0, n_fraud)]
        rows.append(row)
    synth = pd.DataFrame(rows)
    for col in fraud_df.columns:
        try:
            synth[col] = synth[col].astype(fraud_df[col].dtype)
        except Exception:
            pass
    return synth


def encode_y(y: pd.Series) -> pd.Series:
    if y.dtype == object or str(y.dtype) == "bool":
        return pd.Series(LabelEncoder().fit_transform(y.astype(str)))
    if pd.api.types.is_float_dtype(y):
        non_null = y.dropna()
        if len(non_null) > 0 and (non_null == non_null.round()).all():
            return pd.Series(y.astype(int).values).reset_index(drop=True)
    return pd.Series(y.values).reset_index(drop=True)


def check_target(y: pd.Series, col_name: str, df: pd.DataFrame) -> str | None:
    tt = type_of_target(y.dropna())
    if tt in ("continuous", "continuous-multioutput"):
        candidates = [c for c in df.columns if c != col_name and df[c].nunique() <= 20]
        hint = (f" Columns that look like class labels: **{', '.join(candidates[:6])}**" if candidates else "")
        return (
            f"**'{col_name}'** contains continuous numeric values — classifiers need "
            f"discrete labels (e.g. 0/1, fraud/legitimate).{hint}"
        )
    return None


def do_logout():
    for k in ["login_user", "pending_user", "otp_code", "otp_expiry",
              "train_results", "ensemble_results", "synth_state", "activity_log",
              "reset_otp_code", "reset_otp_expiry", "reset_target_user"]:
        st.session_state[k] = None if k != "activity_log" else []
    st.session_state.otp_verified   = False
    st.session_state.reset_step     = 1
    st.session_state.forgot_mode    = False
    st.session_state.page           = "Dashboard"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "users":             None,
    "login_user":        None,
    "pending_user":      None,
    "otp_code":          None,
    "otp_expiry":        None,
    "otp_verified":      False,
    "page":              "Dashboard",
    "train_results":     None,
    "ensemble_results":  None,
    "synth_state":       None,
    "activity_log":      [],
    "remembered_user":   "",
    "forgot_mode":       False,
    "reset_step":        1,
    "reset_otp_code":    None,
    "reset_otp_expiry":  None,
    "reset_target_user": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.users is None:
    st.session_state.users = load_users()


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — NOT LOGGED IN  →  AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.login_user is None and st.session_state.pending_user is None:

    # Hero on auth page
    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">🛡️ Enterprise Fraud Intelligence</div>'
        '<div class="hero-title">AI-Powered <span>Fraud Intelligence</span> Platform</div>'
        '<div class="hero-subtitle">'
        'Detect fraud in real-time, train 35+ ML models, generate synthetic fraud data, '
        'compare ensemble methods, and get explainable insights.'
        '</div>'
        '<div class="badge-row">'
        '<div class="badge">🔐 Role-Based Access</div>'
        f'<div class="badge">🤖 {TOTAL_MODELS}+ ML Models</div>'
        '<div class="badge">🧬 Synthetic Data Generator</div>'
        '<div class="badge">📊 Ensemble Comparison</div>'
        '<div class="badge">📋 Reports & Export</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    _, auth_col, _ = st.columns([1, 1.5, 1])
    with auth_col:
        st.markdown('<div class="auth-card"><div class="auth-glow"></div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-logo">🛡️ FraudGuard AI</div>', unsafe_allow_html=True)

        # ── FORGOT PASSWORD MODE ───────────────────────────────────────────────
        if st.session_state.forgot_mode:
            st.markdown('<div class="auth-title">Reset Password</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">We\'ll send a verification code to your email.</div>', unsafe_allow_html=True)

            if st.session_state.reset_step == 1:
                reset_email = st.text_input("Registered Email", placeholder="you@example.com", key="reset_email_input")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Send Reset Code", use_container_width=True, key="btn_send_reset"):
                        if not reset_email.strip() or not validate_email(reset_email):
                            st.error("Enter a valid email address.")
                        else:
                            df_u = st.session_state.users
                            match = df_u[df_u["email"].str.lower() == reset_email.strip().lower()]
                            if match.empty:
                                # Don't reveal whether email exists
                                st.info("If that email is registered, a code has been sent.")
                            else:
                                otp = generate_otp()
                                st.session_state.reset_otp_code    = otp
                                st.session_state.reset_otp_expiry  = time.time() + OTP_VALIDITY
                                st.session_state.reset_target_user = match.iloc[0]["username"]
                                st.session_state.reset_step = 2
                                st.rerun()
                with c2:
                    if st.button("← Back to Login", use_container_width=True, key="btn_back_from_forgot"):
                        st.session_state.forgot_mode = False
                        st.session_state.reset_step  = 1
                        st.rerun()

            elif st.session_state.reset_step == 2:
                rem = max(0, int((st.session_state.reset_otp_expiry or 0) - time.time()))
                m, s = divmod(rem, 60)
                st.markdown(
                    f'<div class="otp-display">'
                    f'<div class="otp-dev-badge">DEV MODE</div>'
                    f'<div class="otp-code" style="font-size:2rem;letter-spacing:0.4em">'
                    f'{st.session_state.reset_otp_code}</div>'
                    f'<div class="otp-expiry">Expires in {m:02d}:{s:02d}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                reset_code  = st.text_input("Verification Code", placeholder="6-digit code", max_chars=6, key="reset_code_input")
                new_pwd     = st.text_input("New Password", type="password", placeholder="Min. 8 characters", key="reset_new_pwd")
                confirm_pwd = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reset_confirm_pwd")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Update Password", use_container_width=True, key="btn_update_pwd"):
                        if not reset_code:
                            st.error("Enter the verification code.")
                        elif time.time() > st.session_state.reset_otp_expiry:
                            st.error("Code expired. Go back and request a new one.")
                        elif reset_code.strip() != st.session_state.reset_otp_code:
                            st.error("Invalid code.")
                        elif len(new_pwd) < 8:
                            st.error("Password must be at least 8 characters.")
                        elif new_pwd != confirm_pwd:
                            st.error("Passwords do not match.")
                        else:
                            df_u = st.session_state.users
                            idx  = df_u[df_u["username"] == st.session_state.reset_target_user].index
                            df_u.loc[idx, "password"] = hash_password(new_pwd)
                            st.session_state.users = df_u
                            save_users(df_u)
                            st.success("Password updated! You can now log in.")
                            time.sleep(1)
                            st.session_state.forgot_mode  = False
                            st.session_state.reset_step   = 1
                            st.session_state.reset_otp_code  = None
                            st.session_state.reset_target_user = None
                            st.rerun()
                with c2:
                    if st.button("← Back", use_container_width=True, key="btn_back_step2"):
                        st.session_state.reset_step = 1
                        st.rerun()

        # ── NORMAL AUTH TABS ───────────────────────────────────────────────────
        else:
            tab_login, tab_signup = st.tabs(["🔑  Sign In", "✨  Create Account"])

            # ── LOGIN TAB ─────────────────────────────────────────────────────
            with tab_login:
                st.write("")
                identifier = st.text_input(
                    "Username or Email",
                    placeholder="admin  or  admin@fraudguard.ai",
                    key="login_id",
                    value=st.session_state.remembered_user,
                )
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd")

                col_rem, col_forg = st.columns([1, 1])
                with col_rem:
                    remember_me = st.checkbox("Remember me", key="cb_remember")
                with col_forg:
                    st.markdown(
                        '<div style="text-align:right;padding-top:0.2rem">'
                        '<span style="color:#22d3ee;font-size:0.82rem;cursor:pointer;font-weight:600" '
                        'id="forgot_link">Forgot password?</span></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Forgot password?", key="btn_forgot_link", help="Reset your password"):
                        st.session_state.forgot_mode = True
                        st.rerun()

                st.write("")
                if st.button("Sign In", use_container_width=True, key="btn_login"):
                    ident = identifier.strip()
                    if not ident:
                        st.error("Username or email is required.")
                    elif not password:
                        st.error("Password is required.")
                    else:
                        h    = hash_password(password)
                        df_u = st.session_state.users
                        row  = df_u[
                            ((df_u["username"] == ident) | (df_u["email"].str.lower() == ident.lower())) &
                            (df_u["password"] == h)
                        ]
                        if row.empty:
                            st.error("Invalid credentials. Please try again.")
                        else:
                            if remember_me:
                                st.session_state.remembered_user = ident
                            user_dict = row.iloc[0].to_dict()
                            otp = generate_otp()
                            st.session_state.pending_user = user_dict
                            st.session_state.otp_code     = otp
                            st.session_state.otp_expiry   = time.time() + OTP_VALIDITY
                            st.session_state.otp_verified = False
                            st.rerun()

                st.markdown(
                    '<div class="demo-creds">'
                    '🔑 <b>Demo accounts</b><br>'
                    'admin / admin123 &nbsp;·&nbsp; researcher / research123 &nbsp;·&nbsp; user1 / user123'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── SIGN UP TAB ───────────────────────────────────────────────────
            with tab_signup:
                st.write("")
                with st.form("form_signup", clear_on_submit=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        sf_name  = st.text_input("Full Name *",           placeholder="Jane Smith")
                        sf_email = st.text_input("Email *",               placeholder="jane@example.com")
                        sf_phone = st.text_input("Phone Number",          placeholder="+1 234 567 8900")
                    with c2:
                        sf_user  = st.text_input("Username *",            placeholder="janesmith")
                        sf_pwd   = st.text_input("Password *",            type="password", placeholder="Min. 8 characters")
                        sf_cpwd  = st.text_input("Confirm Password *",    type="password", placeholder="Re-enter password")
                    sf_role = st.selectbox("Role", ["End User", "Researcher", "Admin"])
                    st.write("")
                    submitted = st.form_submit_button("Create Account", use_container_width=True)

                if submitted:
                    errors = []
                    if not sf_name.strip():
                        errors.append("Full Name is required.")
                    if not sf_email.strip() or not validate_email(sf_email):
                        errors.append("A valid email address is required.")
                    if not sf_user.strip():
                        errors.append("Username is required.")
                    if len(sf_pwd) < 8:
                        errors.append("Password must be at least 8 characters.")
                    if sf_pwd != sf_cpwd:
                        errors.append("Passwords do not match.")
                    if sf_phone.strip() and not validate_phone(sf_phone):
                        errors.append("Phone number format is invalid.")
                    df_u = st.session_state.users
                    if (df_u["username"] == sf_user.strip()).any():
                        errors.append(f"Username '{sf_user.strip()}' is already taken.")
                    if sf_email.strip() and (df_u["email"].str.lower() == sf_email.strip().lower()).any():
                        errors.append("An account with this email already exists.")

                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        new_user = {
                            "username":  sf_user.strip(),
                            "full_name": sf_name.strip(),
                            "email":     sf_email.strip(),
                            "phone":     sf_phone.strip(),
                            "password":  hash_password(sf_pwd),
                            "role":      sf_role,
                        }
                        st.session_state.users = pd.concat(
                            [df_u, pd.DataFrame([new_user])], ignore_index=True
                        )
                        save_users(st.session_state.users)
                        otp = generate_otp()
                        st.session_state.pending_user = new_user
                        st.session_state.otp_code     = otp
                        st.session_state.otp_expiry   = time.time() + OTP_VALIDITY
                        st.session_state.otp_verified = False
                        st.success("Account created! Proceeding to 2FA verification…")
                        time.sleep(0.6)
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)   # close auth-card
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# GATE 2 — LOGGED IN BUT NOT 2FA VERIFIED  →  OTP PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.otp_verified:
    pending = st.session_state.pending_user or {}

    with st.sidebar:
        st.markdown(
            '<div class="logo-box"><div class="logo-name">🛡️ FraudGuard AI</div>'
            '<div class="logo-sub">Two-Factor Verification</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Verify your identity to continue.")

    _, otp_col, _ = st.columns([1, 1.4, 1])
    with otp_col:
        st.markdown('<div class="otp-card">', unsafe_allow_html=True)
        st.markdown('<div class="otp-shield">🔐</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Two-Factor Authentication</div>', unsafe_allow_html=True)

        email_raw = pending.get("email", "")
        if email_raw:
            at_pos = email_raw.find("@")
            masked  = email_raw[:2] + "••••" + (email_raw[at_pos:] if at_pos > 0 else "")
            dest_str = f"<b>{masked}</b>"
        else:
            dest_str = "your registered contact"
        st.markdown(
            f'<div class="auth-subtitle">'
            f'A 6-digit verification code was sent to {dest_str}.'
            f'</div>',
            unsafe_allow_html=True,
        )

        remaining = max(0, int((st.session_state.otp_expiry or 0) - time.time()))
        mins, secs = divmod(remaining, 60)
        expiry_cls = "otp-expiry otp-expired" if remaining == 0 else "otp-expiry"

        st.markdown(
            f'<div class="otp-display">'
            f'<div class="otp-dev-badge">DEV MODE — Simulated OTP</div>'
            f'<div class="otp-code">{st.session_state.otp_code or "------"}</div>'
            f'<div class="{expiry_cls}">{"Expired" if remaining == 0 else f"Expires in {mins:02d}:{secs:02d}"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        otp_method = st.radio(
            "Delivery method",
            ["📧 Email OTP", "📱 Phone OTP (simulated)"],
            horizontal=True, key="otp_method",
        )
        if "Phone" in otp_method and not pending.get("phone"):
            st.caption("No phone number on file — using email simulation.")

        otp_input = st.text_input(
            "Enter 6-digit code",
            placeholder="_ _ _ _ _ _",
            max_chars=6,
            key="otp_input_field",
        )

        col_v, col_r = st.columns(2)
        with col_v:
            if st.button("Verify & Continue", use_container_width=True, key="btn_verify_otp"):
                if not otp_input.strip():
                    st.error("Please enter the verification code.")
                elif time.time() > (st.session_state.otp_expiry or 0):
                    st.error("Code expired. Click Resend to get a new one.")
                elif otp_input.strip() != st.session_state.otp_code:
                    st.error("Incorrect code. Please try again.")
                else:
                    st.session_state.login_user   = st.session_state.pending_user
                    st.session_state.otp_verified = True
                    st.session_state.page         = "Dashboard"
                    name = pending.get("full_name") or pending.get("username", "User")
                    log_activity(f"Signed in — welcome back, {name}")
                    st.success("Verified! Loading your dashboard…")
                    time.sleep(0.5)
                    st.rerun()

        with col_r:
            if st.button(
                f"Resend ({mins:02d}:{secs:02d})" if remaining > 0 else "Resend Code",
                use_container_width=True,
                key="btn_resend_otp",
                disabled=remaining > 0,
            ):
                new_otp = generate_otp()
                st.session_state.otp_code   = new_otp
                st.session_state.otp_expiry = time.time() + OTP_VALIDITY
                st.info("A new code has been generated.")
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("← Back to Login", use_container_width=True, key="btn_otp_back"):
            st.session_state.pending_user = None
            st.session_state.otp_code     = None
            st.session_state.otp_expiry   = None
            st.session_state.login_user   = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP  — fully authenticated
# ══════════════════════════════════════════════════════════════════════════════
role      = st.session_state.login_user["role"]
user_info = st.session_state.login_user

NAV_PAGES = [
    ("Dashboard",        "🏠"),
    ("Synthetic Data",   "🧬"),
    ("Train & Predict",  "🎯"),
    ("Ensemble Compare", "📊"),
    ("Reports",          "📋"),
]
if role == "Admin":
    NAV_PAGES.insert(5, ("User Management", "👥"))
NAV_PAGES.append(("Settings", "⚙️"))

nav_labels = [f"{icon}  {name}" for name, icon in NAV_PAGES]
nav_names  = [name for name, icon in NAV_PAGES]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    display_name = user_info.get("full_name") or user_info.get("username", "User")
    email_disp   = user_info.get("email", "")

    st.markdown(
        '<div class="logo-box">'
        '<div class="logo-name">🛡️ FraudGuard AI</div>'
        '<div class="logo-sub">Fraud Intelligence Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="user-box">'
        f'<div class="user-name">👤 {display_name}</div>'
        f'<div class="user-email">{email_disp}</div>'
        f'<div class="role-badge">{role}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    try:
        curr_idx = nav_names.index(st.session_state.page)
    except ValueError:
        curr_idx = 0

    selected_label = st.radio(
        "",
        nav_labels,
        index=curr_idx,
        key="nav_radio",
        label_visibility="collapsed",
    )
    st.session_state.page = nav_names[nav_labels.index(selected_label)]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.button("🚪  Sign Out", use_container_width=True, key="sidebar_logout", on_click=do_logout)

    opt_status = (
        f"XGB {'✅' if HAS_XGB else '❌'} · "
        f"LGB {'✅' if HAS_LGBM else '❌'} · "
        f"CBC {'✅' if HAS_CATBOOST else '❌'} · "
        f"IMB {'✅' if HAS_IMBLEARN else '❌'}"
    )
    st.markdown(f'<div style="font-size:0.65rem;color:#2d3f5a;margin-top:0.4rem;line-height:1.6">{opt_status}</div>', unsafe_allow_html=True)

page = st.session_state.page

# ── HERO (shown on all pages) ─────────────────────────────────────────────────
st.markdown(
    f'<div class="hero">'
    f'<div class="hero-tag">🛡️ AI-Powered Fraud Intelligence</div>'
    f'<div class="hero-title">AI-Powered <span>Fraud Intelligence</span> Platform</div>'
    f'<div class="hero-subtitle">'
    f'Detect fraud in real-time, train ML models, generate synthetic fraud data, '
    f'compare ensemble methods, and get explainable insights.'
    f'</div>'
    f'<div class="badge-row">'
    f'<div class="badge">🤖 {TOTAL_MODELS}+ Models</div>'
    f'<div class="badge">🧬 Synthetic Generator</div>'
    f'<div class="badge">📊 Ensemble Compare</div>'
    f'<div class="badge">🔐 {role}</div>'
    f'<div class="badge">👤 {display_name}</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    # Summary metric cards
    tr = st.session_state.train_results
    last_acc = f"{tr['acc']:.3f}" if tr else "—"
    last_model = tr["model_name"] if tr else "None yet"
    sessions = sum(1 for a in st.session_state.activity_log if "Trained" in a.get("action", ""))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Available Models", str(TOTAL_MODELS), "🤖", "8 algorithm categories",
                    "linear-gradient(90deg,#22d3ee,#3b82f6)")
    with c2:
        metric_card("Training Sessions", str(sessions), "🎯", "This session",
                    "linear-gradient(90deg,#3b82f6,#8b5cf6)")
    with c3:
        metric_card("Last Accuracy", last_acc, "📈", last_model,
                    "linear-gradient(90deg,#10b981,#22d3ee)")
    with c4:
        active_pkg = sum([HAS_XGB, HAS_LGBM, HAS_CATBOOST, HAS_IMBLEARN])
        metric_card("Optional Packages", f"{active_pkg}/4", "📦", "XGB·LGB·CBC·IMB",
                    "linear-gradient(90deg,#f59e0b,#ef4444)")

    st.write("")
    left_col, right_col = st.columns([1.4, 1])

    # Model category chart
    with left_col:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        section_header("Model Categories", "Distribution of available ML algorithms")
        cat_df = pd.DataFrame([
            {"Category": cat, "Count": len(models)}
            for cat, models in MODEL_CATEGORIES.items()
        ])
        fig_cat = px.bar(
            cat_df, x="Category", y="Count", text="Count",
            template="plotly_dark",
            color="Category",
            color_discrete_sequence=["#22d3ee","#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4","#6366f1"],
            title="",
        )
        fig_cat.update_traces(textposition="outside", marker_line_width=0)
        fig_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_tickangle=-25, height=320,
            margin=dict(t=10, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Activity feed
    with right_col:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        section_header("Recent Activity", "Session event log")
        log = st.session_state.activity_log
        if not log:
            st.markdown(
                '<div style="color:#3d5070;font-size:0.88rem;padding:1rem 0;text-align:center">'
                '🌙 No activity yet — start by training a model or generating synthetic data.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            items_html = ""
            for entry in log[:12]:
                dot_cls = "activity-dot warn" if entry["status"] == "warn" else (
                    "activity-dot error" if entry["status"] == "error" else "activity-dot"
                )
                items_html += (
                    f'<div class="activity-item">'
                    f'<div class="{dot_cls}"></div>'
                    f'<div><div class="activity-text">{entry["action"]}</div>'
                    f'<div class="activity-time">{entry["time"]}</div></div>'
                    f'</div>'
                )
            st.markdown(items_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Last model performance + quick start
    perf_col, qs_col = st.columns(2)
    with perf_col:
        st.markdown('<div class="glass-sm">', unsafe_allow_html=True)
        section_header("Last Model Performance")
        if tr:
            metrics = {"Accuracy": tr["acc"], "Precision": tr["prec"], "Recall": tr["rec"], "F1 Score": tr["f1"]}
            if tr["auc"]:
                metrics["ROC AUC"] = tr["auc"]
            mdf = pd.DataFrame({"Metric": list(metrics.keys()), "Score": list(metrics.values())})
            fig_m = px.bar(
                mdf, x="Metric", y="Score", text="Score",
                template="plotly_dark", color="Metric",
                color_discrete_sequence=["#22d3ee","#3b82f6","#10b981","#f59e0b","#8b5cf6"],
                title=tr["model_name"],
            )
            fig_m.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
            fig_m.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=260, margin=dict(t=30, b=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 1.12]),
            )
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.markdown(
                '<div style="color:#3d5070;font-size:0.88rem;padding:1.2rem 0;text-align:center">'
                '📊 Train a model to see performance here.'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with qs_col:
        st.markdown('<div class="glass-sm">', unsafe_allow_html=True)
        section_header("Quick Start", "Your workflow in 4 steps")
        steps = [
            ("1", "Go to **Synthetic Data** → upload CSV → generate 500+ fraud rows → download."),
            ("2", "Go to **Train & Predict** → upload CSV → pick a model → train & export results."),
            ("3", "Go to **Ensemble Compare** → compare Bagging, Boosting, and Stacking side-by-side."),
            ("4", "Go to **Reports** → review all session results → download CSVs."),
        ]
        for num, desc in steps:
            st.markdown(
                f'<div style="display:flex;gap:0.8rem;align-items:flex-start;padding:0.5rem 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.04)">'
                f'<div style="background:linear-gradient(135deg,#22d3ee,#3b82f6);color:white;'
                f'border-radius:8px;width:24px;height:24px;display:flex;align-items:center;'
                f'justify-content:center;font-size:0.72rem;font-weight:900;flex-shrink:0">{num}</div>'
                f'<div style="font-size:0.85rem;color:#8aa0c8;line-height:1.5">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Synthetic Data":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    section_header(
        "Synthetic Fraud Data Generator",
        "Generate AI-style synthetic fraud transactions from your dataset's own statistical distribution.",
    )
    st.write(
        "Upload your original fraud dataset. The generator identifies existing fraud rows, "
        "learns their numeric distributions and categorical frequencies, then synthesises new "
        "realistic fraud transactions using Gaussian perturbation (5% noise). "
        "The result is your original data **plus** the synthetic fraud rows — ready to re-train on."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload original dataset CSV (up to 1 GB)", type=["csv"], key="synth_up")

    if up is not None:
        with st.spinner("Reading CSV…"):
            df_orig = pd.read_csv(up)
            df_orig.columns = [c.strip() for c in df_orig.columns]

        c1, c2, c3 = st.columns(3)
        with c1: small_stat("Rows",    f"{df_orig.shape[0]:,}")
        with c2: small_stat("Columns", f"{df_orig.shape[1]:,}")
        with c3: small_stat("Missing", f"{int(df_orig.isna().sum().sum()):,}")

        st.write("")
        col1, col2, col3 = st.columns(3)
        with col1:
            target_col  = st.selectbox("Fraud label column", df_orig.columns.tolist(), key="sc_target")
        with col2:
            fraud_vals  = df_orig[target_col].unique().tolist()
            fraud_value = st.selectbox("Value that means FRAUD", fraud_vals, key="sc_fval")
        with col3:
            n_synth = st.slider("Synthetic fraud rows to generate", 100, 5000, 500, 100)

        if st.button("Generate Synthetic Fraud Transactions", use_container_width=True):
            fraud_rows = df_orig[df_orig[target_col] == fraud_value]
            if len(fraud_rows) == 0:
                st.error("No rows found with that fraud value. Check your column/value selection.")
            else:
                with st.spinner(f"Generating {n_synth} synthetic fraud rows…"):
                    synth_df  = generate_synthetic_fraud(fraud_rows, n_samples=n_synth)
                    augmented = pd.concat([df_orig, synth_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

                st.session_state.synth_state = {
                    "orig_rows":  df_orig.shape[0],
                    "orig_cols":  df_orig.shape[1],
                    "n_synth":    n_synth,
                    "fraud_col":  target_col,
                    "fraud_val":  fraud_value,
                    "augmented":  augmented,
                    "orig_fraud": len(fraud_rows),
                }
                log_activity(f"Generated {n_synth} synthetic fraud rows from {df_orig.shape[0]:,} original rows")
                st.success(f"Done! Generated {n_synth} synthetic fraud rows.")

    ss = st.session_state.synth_state
    if ss is not None:
        aug = ss["augmented"]
        st.markdown("---")
        section_header("Augmented Dataset Summary", "Original + synthetic rows merged and shuffled")

        c1, c2, c3, c4 = st.columns(4)
        with c1: small_stat("Original Rows",       f"{ss['orig_rows']:,}")
        with c2: small_stat("Original Fraud Rows", f"{ss['orig_fraud']:,}")
        with c3: small_stat("Synthetic Added",      f"{ss['n_synth']:,}")
        with c4: small_stat("Augmented Total",      f"{aug.shape[0]:,}")

        st.write("")
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        section_header("Download Augmented Dataset")
        st.write(
            f"Original **{ss['orig_rows']:,}** rows + **{ss['n_synth']:,}** synthetic fraud rows "
            f"→ **{aug.shape[0]:,}** total rows."
        )
        csv_aug = aug.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️  Download Augmented Dataset CSV ({aug.shape[0]:,} rows)",
            data=csv_aug,
            file_name="augmented_fraud_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        section_header("Preview — last 20 rows (includes synthetic)", "Synthetic rows were shuffled into the dataset")
        st.dataframe(aug.tail(20), use_container_width=True, hide_index=True)

        fraud_counts = aug[ss["fraud_col"]].value_counts().reset_index()
        fraud_counts.columns = ["Label", "Count"]
        fig = px.bar(
            fraud_counts, x="Label", y="Count", text="Count",
            template="plotly_dark", color="Label",
            color_discrete_sequence=["#22d3ee","#f59e0b","#10b981","#ef4444"],
            title="Class Distribution in Augmented Dataset",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Train & Predict":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    section_header("Train Model from CSV", "Upload your dataset and generate real ML predictions")
    uploaded_file = st.file_uploader("Upload CSV file (up to 1 GB)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip() for c in df.columns]

        prev_col, conf_col = st.columns([1.2, 1])
        with prev_col:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            section_header("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            a, b, c = st.columns(3)
            with a: small_stat("Rows",    f"{df.shape[0]:,}")
            with b: small_stat("Columns", f"{df.shape[1]:,}")
            with c: small_stat("Missing", f"{int(df.isna().sum().sum()):,}")
            st.markdown('</div>', unsafe_allow_html=True)

        with conf_col:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            section_header("Training Configuration")
            c1, c2 = st.columns(2)
            with c1:
                target_col = st.selectbox("Target Column",   df.columns.tolist())
                sel_cat    = st.selectbox("Model Category",  list(MODEL_CATEGORIES.keys()))
                model_name = st.selectbox("Model",           MODEL_CATEGORIES[sel_cat])
            with c2:
                test_size  = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)
                n_est = 150
                if model_name in ESTIMATOR_MODELS:
                    n_est = st.slider("Estimators / Trees", 50, 400, 150, 50)
                if model_name in ANOMALY_MODELS:
                    st.info("Anomaly model: trains without labels, maps outliers → fraud.")
                max_rows_tp = len(df)
                if len(df) > 100_000:
                    st.warning(f"Large dataset ({len(df):,} rows).")
                    max_rows_tp = st.slider("Max training rows", 10_000, len(df), min(150_000, len(df)), 10_000, key="tp_maxrows")
            run_train = st.button("Train Model and Generate Results", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if run_train:
            st.session_state.train_results = None
            is_anomaly = model_name in ANOMALY_MODELS
            y = encode_y(df[target_col])
            X = df.drop(columns=[target_col]).reset_index(drop=True)

            if X.shape[1] == 0:
                st.error("Need at least one feature column.")
                st.stop()
            if not is_anomaly:
                err = check_target(y, target_col, df)
                if err:
                    st.error(err)
                    st.stop()
            if not is_anomaly and y.nunique() < 2:
                st.error("Target column needs at least 2 classes.")
                st.stop()

            if max_rows_tp < len(X):
                idx = np.random.RandomState(42).choice(len(X), max_rows_tp, replace=False)
                X = X.iloc[idx].reset_index(drop=True)
                y = y.iloc[idx].reset_index(drop=True)
                st.info(f"Training on {max_rows_tp:,} sampled rows out of {len(df):,} total.")

            pre  = make_preprocessor(X, model_name)
            clf  = build_model(model_name, n=n_est)
            pipe = Pipeline([("pre", pre), ("clf", clf)])

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=42,
                    stratify=None if is_anomaly else y)
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

            try:
                with st.spinner(f"Training {model_name}…"):
                    pipe.fit(X_tr, y_tr)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

            try:
                y_pred = pipe.predict(X_te)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            if is_anomaly:
                y_pred = np.where(y_pred == -1, 1, 0)

            y_prob = None
            if not is_anomaly:
                try:
                    if y_te.nunique() == 2:
                        y_prob = pipe.predict_proba(X_te)[:, 1]
                except Exception:
                    pass

            acc  = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
            f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            auc  = None
            if y_prob is not None:
                try:
                    auc = roc_auc_score(y_te, y_prob)
                except Exception:
                    pass

            res_df = X_te.copy().reset_index(drop=True)
            res_df["Actual"]    = pd.Series(y_te).reset_index(drop=True)
            res_df["Predicted"] = pd.Series(y_pred).reset_index(drop=True)
            if y_prob is not None:
                res_df["Fraud Probability"] = pd.Series(y_prob).reset_index(drop=True)

            st.session_state.train_results = dict(
                model_name=model_name, is_anomaly=is_anomaly,
                acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                y_test=y_te.tolist(), y_pred=y_pred.tolist(), results_df=res_df,
            )
            log_activity(f"Trained {model_name} — Accuracy {acc:.3f}  F1 {f1:.3f}")

        tr = st.session_state.train_results
        if tr is not None:
            if tr["is_anomaly"]:
                st.info(f"**{tr['model_name']}** trained without labels. −1 → fraud=1, +1 → normal=0.")
            st.markdown(f"### Results — {tr['model_name']}")
            k1, k2, k3, k4 = st.columns(4)
            with k1: metric_card("Accuracy",  f"{tr['acc']:.3f}",  "🎯", "Overall accuracy",  "linear-gradient(90deg,#22d3ee,#3b82f6)")
            with k2: metric_card("Precision", f"{tr['prec']:.3f}", "📌", "Weighted precision","linear-gradient(90deg,#3b82f6,#8b5cf6)")
            with k3: metric_card("Recall",    f"{tr['rec']:.3f}",  "🔁", "Weighted recall",   "linear-gradient(90deg,#10b981,#22d3ee)")
            with k4: metric_card("F1 Score",  f"{tr['f1']:.3f}",   "📈", "Weighted F1",       "linear-gradient(90deg,#f59e0b,#ef4444)")
            if tr["auc"] is not None:
                st.info(f"ROC AUC: **{tr['auc']:.3f}**")

            tabs = st.tabs(["📊 Visual Analytics", "🔍 Predictions Table", "📋 Classification Report"])
            with tabs[0]:
                mdf = pd.DataFrame({
                    "Metric": ["Accuracy","Precision","Recall","F1"] + (["ROC AUC"] if tr["auc"] else []),
                    "Score":  [tr["acc"],tr["prec"],tr["rec"],tr["f1"]] + ([tr["auc"]] if tr["auc"] else []),
                })
                fig = px.bar(
                    mdf, x="Metric", y="Score", text="Score", template="plotly_dark",
                    color="Metric",
                    color_discrete_sequence=["#22d3ee","#3b82f6","#10b981","#f59e0b","#8b5cf6"],
                    title=f"{tr['model_name']} — Performance Metrics",
                )
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    yaxis=dict(range=[0, 1.12], gridcolor="rgba(255,255,255,0.04)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                )
                st.plotly_chart(fig, use_container_width=True)
                cm = confusion_matrix(tr["y_test"], tr["y_pred"])
                fig_cm = px.imshow(
                    pd.DataFrame(cm), text_auto=True,
                    color_continuous_scale="Blues", aspect="auto",
                    title="Confusion Matrix",
                )
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with tabs[1]:
                st.dataframe(tr["results_df"].head(100), use_container_width=True)
                st.download_button(
                    "⬇️  Download Predictions CSV",
                    data=tr["results_df"].to_csv(index=False).encode(),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with tabs[2]:
                rpt = classification_report(tr["y_test"], tr["y_pred"], output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(rpt).transpose(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ENSEMBLE COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ensemble Compare":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    section_header(
        "Ensemble Method Comparison",
        "Trains Bagging, Boosting, and Stacking simultaneously and compares them side-by-side on the same split.",
    )
    st.write("All models train on the **same** train/test split so the comparison is perfectly fair.")
    st.markdown('</div>', unsafe_allow_html=True)

    ens_file = st.file_uploader("Upload CSV dataset (up to 1 GB)", type=["csv"], key="ens_up")

    if ens_file is not None:
        df_ens = pd.read_csv(ens_file)
        df_ens.columns = [c.strip() for c in df_ens.columns]

        c1, c2, c3 = st.columns(3)
        with c1: small_stat("Rows",    f"{df_ens.shape[0]:,}")
        with c2: small_stat("Columns", f"{df_ens.shape[1]:,}")
        with c3: small_stat("Missing", f"{int(df_ens.isna().sum().sum()):,}")

        st.write("")
        col1, col2, col3 = st.columns(3)
        with col1: ens_target = st.selectbox("Target Column", df_ens.columns.tolist(), key="ens_tgt")
        with col2: ens_test   = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05, key="ens_ts")
        with col3: ens_n      = st.slider("Estimators (all models)", 50, 300, 100, 50, key="ens_n")

        ens_max_rows = len(df_ens)
        if len(df_ens) > 100_000:
            st.warning(f"Large dataset ({len(df_ens):,} rows).")
            ens_max_rows = st.slider("Max training rows", 10_000, len(df_ens), min(150_000, len(df_ens)), 10_000, key="ens_maxrows")

        if st.button("Compare All Ensemble Methods", use_container_width=True):
            st.session_state.ensemble_results = None
            y_ens = encode_y(df_ens[ens_target])
            X_ens = df_ens.drop(columns=[ens_target]).reset_index(drop=True)

            err_ens = check_target(y_ens, ens_target, df_ens)
            if err_ens:
                st.error(err_ens)
                st.stop()
            if y_ens.nunique() < 2:
                st.error("Target column needs at least 2 classes.")
                st.stop()

            if ens_max_rows < len(X_ens):
                idx = np.random.RandomState(42).choice(len(X_ens), ens_max_rows, replace=False)
                X_ens = X_ens.iloc[idx].reset_index(drop=True)
                y_ens = y_ens.iloc[idx].reset_index(drop=True)
                st.info(f"Training on {ens_max_rows:,} sampled rows out of {len(df_ens):,} total.")

            pre_ens = make_preprocessor(X_ens)
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X_ens, y_ens, test_size=ens_test, random_state=42, stratify=y_ens)
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(X_ens, y_ens, test_size=ens_test, random_state=42)

            ENSEMBLE_SUITE = {
                "🎒 Bagging":                   BaggingClassifier(n_estimators=ens_n, random_state=42),
                "🌲 Random Forest (Bagging)":   RandomForestClassifier(n_estimators=ens_n, random_state=42, class_weight="balanced"),
                "🚀 AdaBoost (Boosting)":        AdaBoostClassifier(n_estimators=ens_n, random_state=42, algorithm="SAMME"),
                "📈 Gradient Boosting":          GradientBoostingClassifier(n_estimators=ens_n, random_state=42),
                "⚡ Histogram GBM":              HistGradientBoostingClassifier(max_iter=ens_n, random_state=42),
                "🧩 Stacking (RF+LR → GBM)":   StackingClassifier(
                    estimators=[
                        ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                        ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
                    ],
                    final_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
                    cv=3,
                ),
            }
            if HAS_XGB:
                ENSEMBLE_SUITE["💥 XGBoost (Boosting)"] = XGBClassifier(n_estimators=ens_n, random_state=42, eval_metric="logloss", verbosity=0)
            if HAS_LGBM:
                ENSEMBLE_SUITE["🔥 LightGBM (Boosting)"] = LGBMClassifier(n_estimators=ens_n, random_state=42, verbose=-1)

            results = []
            prog  = st.progress(0, text="Starting…")
            total = len(ENSEMBLE_SUITE)
            for i, (mname, clf) in enumerate(ENSEMBLE_SUITE.items()):
                prog.progress(i / total, text=f"Training {mname}…")
                t0 = time.time()
                try:
                    p = Pipeline([("pre", pre_ens), ("clf", clf)])
                    p.fit(Xtr, ytr)
                    yp = p.predict(Xte)
                    elapsed = round(time.time() - t0, 1)
                    results.append({
                        "Model":     mname,
                        "Category":  ("Boosting" if "Boosting" in mname else
                                      "Stacking" if "Stacking" in mname else "Bagging"),
                        "Accuracy":  round(accuracy_score(yte, yp), 4),
                        "Precision": round(precision_score(yte, yp, average="weighted", zero_division=0), 4),
                        "Recall":    round(recall_score(yte, yp, average="weighted", zero_division=0), 4),
                        "F1 Score":  round(f1_score(yte, yp, average="weighted", zero_division=0), 4),
                        "Time (s)":  elapsed,
                    })
                except Exception as e:
                    results.append({
                        "Model": mname, "Category": "Error",
                        "Accuracy": "—", "Precision": str(e)[:50],
                        "Recall": "—", "F1 Score": "—", "Time (s)": "—",
                    })
            prog.progress(1.0, text="All models trained!")
            st.session_state.ensemble_results = results
            log_activity(f"Ensemble comparison complete — {len(results)} models evaluated")

    er = st.session_state.ensemble_results
    if er:
        res_df = pd.DataFrame(er)
        st.markdown("### Comparison Results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        num_df = res_df[res_df["F1 Score"] != "—"].copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce")

        if not num_df.empty:
            melted = num_df.melt(
                id_vars="Model",
                value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                var_name="Metric", value_name="Score",
            )
            fig = px.bar(
                melted, x="Model", y="Score", color="Metric", barmode="group",
                template="plotly_dark",
                color_discrete_sequence=["#22d3ee","#3b82f6","#10b981","#f59e0b"],
                title="Bagging vs Boosting vs Stacking — Side-by-Side Comparison",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-30, height=480,
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            best = num_df.loc[num_df["F1 Score"].idxmax()]
            st.success(f"Best model by F1 Score: **{best['Model']}** — F1 = {best['F1 Score']:.4f}")

        st.download_button(
            "⬇️  Download Comparison CSV",
            data=res_df.to_csv(index=False).encode(),
            file_name="ensemble_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    section_header("Reports & Export", "Summary of all results generated in this session")

    tr = st.session_state.train_results
    er = st.session_state.ensemble_results
    ss = st.session_state.synth_state

    if not tr and not er and not ss:
        st.markdown(
            '<div class="glass" style="text-align:center;padding:2.5rem">'
            '<div style="font-size:3rem;margin-bottom:1rem">📋</div>'
            '<div style="font-size:1.1rem;color:#5a7099;font-weight:600">No results yet.</div>'
            '<div style="font-size:0.88rem;color:#3d5070;margin-top:0.5rem">'
            'Run a model in <b>Train &amp; Predict</b>, compare ensembles, or generate synthetic data — '
            'then return here to download everything.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        if tr:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            section_header("Last Training Run", f"Model: {tr['model_name']}")
            c1, c2, c3, c4 = st.columns(4)
            with c1: small_stat("Accuracy",  f"{tr['acc']:.4f}")
            with c2: small_stat("Precision", f"{tr['prec']:.4f}")
            with c3: small_stat("Recall",    f"{tr['rec']:.4f}")
            with c4: small_stat("F1 Score",  f"{tr['f1']:.4f}")
            if tr["auc"]:
                st.info(f"ROC AUC: **{tr['auc']:.4f}**")
            st.write("")
            st.download_button(
                f"⬇️  Download {tr['model_name']} Predictions CSV",
                data=tr["results_df"].to_csv(index=False).encode(),
                file_name=f"{tr['model_name'].replace(' ', '_')}_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if er:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            section_header("Ensemble Comparison Results")
            er_df = pd.DataFrame(er)
            st.dataframe(er_df, use_container_width=True, hide_index=True)
            st.write("")
            st.download_button(
                "⬇️  Download Ensemble Comparison CSV",
                data=er_df.to_csv(index=False).encode(),
                file_name="ensemble_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if ss:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            section_header("Synthetic Data Generation")
            c1, c2, c3 = st.columns(3)
            with c1: small_stat("Original Rows",  f"{ss['orig_rows']:,}")
            with c2: small_stat("Synthetic Added", f"{ss['n_synth']:,}")
            with c3: small_stat("Augmented Total", f"{ss['augmented'].shape[0]:,}")
            st.write("")
            st.download_button(
                "⬇️  Download Augmented Dataset CSV",
                data=ss["augmented"].to_csv(index=False).encode("utf-8"),
                file_name="augmented_fraud_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.activity_log:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        section_header("Session Activity Log", "Complete event history for this session")
        log_df = pd.DataFrame(st.session_state.activity_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️  Download Activity Log CSV",
            data=log_df.to_csv(index=False).encode(),
            file_name="session_activity_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: USER MANAGEMENT  (Admin only)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "User Management":
    if role != "Admin":
        st.warning("This page is restricted to Administrators.")
    else:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        section_header("User Management", "Create new accounts and view all registered users")

        with st.form("form_add_user"):
            c1, c2, c3 = st.columns(3)
            with c1:
                nu_name  = st.text_input("Full Name")
                nu_user  = st.text_input("Username")
            with c2:
                nu_email = st.text_input("Email")
                nu_phone = st.text_input("Phone (optional)")
            with c3:
                nu_pwd  = st.text_input("Password", type="password")
                nu_role = st.selectbox("Role", ["End User", "Researcher", "Admin"])
            if st.form_submit_button("Create User", use_container_width=True):
                errs = []
                if not nu_user.strip():
                    errs.append("Username is required.")
                if not nu_pwd:
                    errs.append("Password is required.")
                if nu_email.strip() and not validate_email(nu_email):
                    errs.append("Invalid email format.")
                if (st.session_state.users["username"] == nu_user.strip()).any():
                    errs.append(f"Username '{nu_user.strip()}' already exists.")
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    new_u = {
                        "username":  nu_user.strip(),
                        "full_name": nu_name.strip() or nu_user.strip(),
                        "email":     nu_email.strip(),
                        "phone":     nu_phone.strip(),
                        "password":  hash_password(nu_pwd),
                        "role":      nu_role,
                    }
                    st.session_state.users = pd.concat(
                        [st.session_state.users, pd.DataFrame([new_u])], ignore_index=True
                    )
                    save_users(st.session_state.users)
                    log_activity(f"Admin created new user '{nu_user.strip()}' ({nu_role})")
                    st.success(f"User '{nu_user.strip()}' created successfully.")

        st.write("")
        section_header("All Users", f"{len(st.session_state.users)} registered accounts")
        disp = st.session_state.users.copy()
        disp["password"] = "••••••••"
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    section_header("Account & Platform Settings", "Your session details and platform configuration")

    c1, c2, c3, c4 = st.columns(4)
    with c1: small_stat("Username", user_info.get("username", "—"))
    with c2: small_stat("Role",     role)
    with c3: small_stat("Models",   str(TOTAL_MODELS))
    with c4: small_stat("Upload Limit", "1 GB")

    st.write("")
    section_header("Account Details")
    info_c1, info_c2 = st.columns(2)
    with info_c1:
        small_stat("Full Name", user_info.get("full_name", "—"))
        st.write("")
        small_stat("Email",     user_info.get("email", "—") or "—")
    with info_c2:
        small_stat("Phone",     user_info.get("phone", "—") or "—")
        st.write("")
        small_stat("Session 2FA", "✅ Verified")

    st.write("")
    section_header("Optional Packages")
    pkg_c1, pkg_c2, pkg_c3, pkg_c4 = st.columns(4)
    with pkg_c1: small_stat("XGBoost",          "✅ Active" if HAS_XGB      else "❌ Not installed")
    with pkg_c2: small_stat("LightGBM",         "✅ Active" if HAS_LGBM     else "❌ Not installed")
    with pkg_c3: small_stat("CatBoost",         "✅ Active" if HAS_CATBOOST else "❌ Not installed")
    with pkg_c4: small_stat("imbalanced-learn", "✅ Active" if HAS_IMBLEARN else "❌ Not installed")

    st.write("")
    section_header("Model Categories")
    for cat, models in MODEL_CATEGORIES.items():
        with st.expander(f"{cat} — {len(models)} models"):
            st.write("  ·  ".join(models))

    st.markdown('</div>', unsafe_allow_html=True)
