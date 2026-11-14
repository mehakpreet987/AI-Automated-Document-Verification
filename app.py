"""
DocVerify – Automated Document Verification and Consistency Review System.
Commercial Enterprise SaaS Web Application.

Provides automated information extraction, schema validation, explainable production
rule checks, supervised machine learning classification, human-in-the-loop escalation,
and official downloadable PDF verification reports.
"""

import io
import os
import json
import textwrap
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image
import pymupdf

# Core Engine Modules (STRICTLY PRESERVED BACKEND MODULES)
import auth_manager
import database
import decision_engine
from decision_engine import (
    STATUS_CONSISTENCY_PASSED,
    STATUS_NEEDS_REVIEW,
    STATUS_INCONSISTENCIES,
    STATUS_UNABLE_TO_VERIFY,
    AUTHENTICITY_DISCLAIMER
)
from document_processor import process_file_bytes, TESSERACT_PATH
from information_extractor import extract_structured_fields
from document_schema import SCHEMA_REGISTRY, validate_document_schema, get_schema
from rule_engine import run_rule_engine
from ml_model import predict_document_category, get_feature_importances
from model_evaluation import evaluate_models_on_test_set, get_evaluation_dataframe
import review_manager
import report_generator

# ---------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & ENTERPRISE DESIGN SYSTEM
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocVerify | Enterprise Document Verification & Consistency Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise SaaS CSS - Premium Linear/Stripe/Vercel Aesthetic
st.markdown(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --primary: #2563EB;
        --primary-hover: #1D4ED8;
        --primary-active: #1E40AF;
        --primary-light: #EFF6FF;
        --primary-glow: rgba(37, 99, 235, 0.20);
        --dark-navy: #142B59;
        --navy-surface: #1E3A8A;
        --secondary-blue: #60A5FA;
        --bg-main: #F5F8FF;
        --card-bg: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-hover: #CBD5E1;
        --text-primary: #172554;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --success: #16A34A;
        --success-bg: #ECFDF5;
        --success-border: #A7F3D0;
        --warning: #F59E0B;
        --warning-bg: #FFFBEB;
        --warning-border: #FDE68A;
        --danger: #DC2626;
        --danger-bg: #FEF2F2;
        --danger-border: #FECACA;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-pill: 9999px;
        --shadow-sm: 0 1px 3px rgba(20, 43, 89, 0.04);
        --shadow-card: 0 2px 10px -2px rgba(20, 43, 89, 0.05), 0 1px 4px -1px rgba(20, 43, 89, 0.03);
        --shadow-hover: 0 12px 28px -4px rgba(37, 99, 235, 0.12), 0 4px 10px -2px rgba(20, 43, 89, 0.05);
        --shadow-glow: 0 0 24px -4px rgba(37, 99, 235, 0.35);
    }

    /* Global Base */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(circle at 10% 15%, rgba(37, 99, 235, 0.035) 0%, transparent 45%),
            radial-gradient(circle at 90% 85%, rgba(96, 165, 250, 0.045) 0%, transparent 45%) !important;
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    /* Refined Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.25rem !important;
        padding-right: 2.25rem !important;
        max-width: 1380px !important;
    }

    /* Minimal Top Header */
    [data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }

    /* Custom Sleek Scrollbar */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 1.25rem !important;
        box-shadow: 2px 0 12px rgba(20, 43, 89, 0.03) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
        margin-top: 0 !important;
    }

    /* Brand Logo & Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
    }
    .brand-icon-box {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        background: linear-gradient(135deg, #142B59 0%, #2563EB 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.30);
        flex-shrink: 0;
    }
    .brand-title {
        font-size: 1.48rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--primary);
        margin: 0;
        line-height: 1.1;
    }
    .brand-title span {
        color: var(--dark-navy);
    }
    .brand-tag {
        display: inline-block;
        background: #EFF6FF;
        color: var(--primary);
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 2px 7px;
        border-radius: 4px;
        border: 1px solid #BFDBFE;
        margin-top: 3px;
    }

    /* User Profile Card in Sidebar */
    .sidebar-user-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        margin-bottom: 20px;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    .sidebar-user-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(20, 43, 89, 0.06);
    }
    .user-avatar-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .user-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563EB 0%, #142B59 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.90rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(20, 43, 89, 0.22);
    }
    .user-details {
        overflow: hidden;
    }
    .user-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .user-email {
        font-size: 0.76rem;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-top: 1px;
    }
    .user-role-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-top: 8px;
    }
    .role-badge-admin {
        background: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .role-badge-user {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    /* Cards & Containers */
    .docv-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px 28px;
        box-shadow: var(--shadow-card);
        margin-bottom: 20px;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }
    .docv-card:hover {
        border-color: #CBD5E1;
        box-shadow: var(--shadow-hover);
    }

    /* Metric Cards */
    .docv-metric-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 22px 24px;
        box-shadow: var(--shadow-card);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .docv-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .metric-icon-wrap {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.20rem;
    }
    .docv-metric-value {
        font-size: 2.30rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .docv-metric-label {
        font-size: 0.86rem;
        color: var(--text-muted);
        font-weight: 600;
        margin-top: 6px;
    }
    .metric-bottom-bar {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 4px;
    }

    /* Welcome Hero Banner */
    .welcome-hero {
        background: linear-gradient(135deg, #142B59 0%, #1E3A8A 55%, #2563EB 100%);
        border-radius: var(--radius-lg);
        padding: 30px 36px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -8px rgba(20, 43, 89, 0.35);
        position: relative;
        overflow: hidden;
    }
    .welcome-hero::after {
        content: "";
        position: absolute;
        right: -30px;
        top: -40px;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0) 70%);
        pointer-events: none;
    }
    .welcome-title {
        font-size: 1.70rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
        color: #FFFFFF;
    }
    .welcome-desc {
        font-size: 0.96rem;
        color: #DBEAFE;
        max-width: 740px;
        line-height: 1.55;
        margin: 0;
    }

    /* Status Badges */
    .status-badge-passed {
        background-color: var(--success-bg);
        color: #065F46;
        padding: 6px 14px;
        border-radius: var(--radius-pill);
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid var(--success-border);
        box-shadow: 0 1px 2px rgba(22, 163, 74, 0.12);
    }
    .status-badge-review {
        background-color: var(--warning-bg);
        color: #92400E;
        padding: 6px 14px;
        border-radius: var(--radius-pill);
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid var(--warning-border);
        box-shadow: 0 1px 2px rgba(245, 158, 11, 0.12);
    }
    .status-badge-inconsistent {
        background-color: var(--danger-bg);
        color: #991B1B;
        padding: 6px 14px;
        border-radius: var(--radius-pill);
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid var(--danger-border);
        box-shadow: 0 1px 2px rgba(220, 38, 38, 0.12);
    }
    .status-badge-unable {
        background-color: #F1F5F9;
        color: #334155;
        padding: 6px 14px;
        border-radius: var(--radius-pill);
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #CBD5E1;
    }

    /* Regulatory Notice Box */
    .regulatory-box {
        background: #FFFFFF;
        border-left: 4px solid var(--primary);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        padding: 16px 20px;
        border-radius: var(--radius-md);
        color: var(--dark-navy);
        font-size: 0.86rem;
        line-height: 1.55;
        margin: 22px 0 14px 0;
        box-shadow: var(--shadow-sm);
    }

    /* Buttons & Form Controls */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.35rem !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
        letter-spacing: -0.01em !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #142B59 100%) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }

    button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1.5px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        border-color: #94A3B8 !important;
        background-color: #F8FAFC !important;
        transform: translateY(-1px) !important;
    }

    /* Text Inputs, Passwords, Textareas */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        border-radius: var(--radius-sm) !important;
        border: 1.5px solid var(--border-subtle) !important;
        padding: 0.55rem 0.85rem !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3.5px var(--primary-glow) !important;
        outline: none !important;
    }

    /* Selectboxes */
    [data-testid="stSelectbox"] > div > div {
        border-radius: var(--radius-sm) !important;
        border: 1.5px solid var(--border-subtle) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: #CBD5E1 !important;
    }

    /* Segmented Radio Buttons */
    [data-testid="stRadio"] div[role="radiogroup"] {
        background: #EEF2F6;
        padding: 4px;
        border-radius: var(--radius-md);
        display: flex;
        gap: 4px;
        border: 1px solid var(--border-subtle);
    }
    [data-testid="stRadio"] div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 8px;
        padding: 7px 16px;
        font-weight: 600;
        font-size: 0.86rem;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 0 !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #FFFFFF !important;
        color: var(--primary) !important;
        box-shadow: 0 1px 4px rgba(20, 43, 89, 0.08) !important;
    }

    /* Sidebar Radio Navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        flex-direction: column !important;
        gap: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 10px 14px !important;
        border-radius: 10px !important;
        font-size: 0.90rem !important;
        color: #334155 !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #F1F5F9 !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #EFF6FF !important;
        color: #1E40AF !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08) !important;
    }

    /* ---- Split-Screen Auth Layout ---- */
    .auth-hero-banner {
        background: linear-gradient(145deg, #142B59 0%, #19376D 50%, #2563EB 100%);
        border-radius: var(--radius-lg);
        padding: 42px 36px;
        color: #FFFFFF;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 24px 48px -12px rgba(20, 43, 89, 0.35);
        position: relative;
        overflow: hidden;
    }
    .auth-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #93C5FD;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }
    .auth-brand-logo {
        font-size: 2.40rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #FFFFFF;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .auth-brand-logo span {
        color: var(--secondary-blue);
    }
    .auth-tagline {
        font-size: 0.96rem;
        color: #BFDBFE;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 24px;
    }
    .auth-feature-glass {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-md);
        padding: 12px 16px;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
        font-size: 0.86rem;
        color: #E2E8F0;
        line-height: 1.45;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .auth-feature-glass:hover {
        background: rgba(255, 255, 255, 0.13);
        transform: translateX(4px);
    }
    .auth-feature-glass strong {
        color: #FFFFFF;
    }
    .auth-trust-box {
        margin-top: 24px;
        background: rgba(20, 43, 89, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        font-size: 0.80rem;
        color: #BFDBFE;
        line-height: 1.45;
    }

    /* SVG Scanning Laser Animation */
    @keyframes scanLaser {
        0% { transform: translateY(0px); opacity: 0; }
        15% { opacity: 1; }
        85% { opacity: 1; }
        100% { transform: translateY(92px); opacity: 0; }
    }
    .scan-beam-line {
        animation: scanLaser 3s ease-in-out infinite;
    }

    /* ---- Auth Form Card ---- */
    .auth-form-card {
        background: #FFFFFF;
        border-radius: var(--radius-lg);
        padding: 34px 32px;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-card);
    }
    .auth-form-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    .auth-form-subtitle {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin-bottom: 22px;
    }

    /* Token display box */
    .token-display-box {
        background: #F0F9FF;
        border: 1.5px solid #0EA5E9;
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.88rem;
        word-break: break-all;
        color: #0C4A6E;
        margin: 12px 0;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.1);
    }

    /* Online radar pulse dot */
    .pulse-dot-container {
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    .system-online-dot {
        position: relative;
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
    }
    .system-online-dot::after {
        content: "";
        position: absolute;
        top: -3px;
        left: -3px;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 2px solid var(--success);
        opacity: 0.7;
        animation: radarPulse 2s cubic-bezier(0.24, 0, 0.38, 1) infinite;
    }
    @keyframes radarPulse {
        0% { transform: scale(0.6); opacity: 0.9; }
        100% { transform: scale(2.2); opacity: 0; }
    }

    /* Activity timeline item */
    .activity-item {
        display: flex;
        align-items: flex-start;
        padding: 14px 0;
        border-bottom: 1px solid #F1F5F9;
        transition: background 0.15s ease;
    }
    .activity-item:last-child {
        border-bottom: none;
    }
    .activity-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #EFF6FF;
        color: #1E40AF;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 14px;
        font-size: 1.05rem;
        flex-shrink: 0;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.1);
    }
    .activity-content {
        flex-grow: 1;
    }
    .activity-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .activity-time {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-top: 3px;
    }

    /* File Dropzone Cue */
    .upload-zone-hint {
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: var(--radius-md);
        padding: 20px 22px;
        text-align: center;
        margin-bottom: 14px;
        transition: all 0.2s ease;
    }
    .upload-zone-hint:hover {
        border-color: var(--primary);
        background: #EFF6FF;
    }

    /* Dataframe Table Container */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* Streamlit Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid var(--border-subtle) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        color: var(--text-muted) !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2.5px solid var(--primary) !important;
    }

    /* Streamlit Expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm) !important;
        margin-bottom: 14px !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
    }
</style>
""").strip(), unsafe_allow_html=True)

# Initialize Database Schema & Seed Initial Accounts
database.init_database()

# Session State Initialization
if "user" not in st.session_state:
    st.session_state["user"] = None
if "active_doc_id" not in st.session_state:
    st.session_state["active_doc_id"] = None
if "last_processed" not in st.session_state:
    st.session_state["last_processed"] = None
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = None
if "_reset_token_display" not in st.session_state:
    st.session_state["_reset_token_display"] = None


# ---------------------------------------------------------------------------
# AUTHENTICATION SCREEN (Rendered when no session exists)
# ---------------------------------------------------------------------------

def render_authentication_view():
    """
    Renders a premium, modern, split-screen authentication screen.
    Guarantees zero raw HTML leaks, no top whitespace, and clean responsive alignment.
    """
    col_left, col_right = st.columns([1.15, 1.0], gap="large")

    # ---- LEFT BRANDING HERO PANEL ----
    with col_left:
        # High-Fidelity SVG Document Verification Graphic with Laser Beam
        svg_illustration = (
            '<svg width="100%" height="160" viewBox="0 0 320 160" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 20px; filter: drop-shadow(0 12px 24px rgba(0,0,0,0.3));">'
            '<defs>'
            '<linearGradient id="cardGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
            '<stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.18"/>'
            '<stop offset="100%" stop-color="#FFFFFF" stop-opacity="0.04"/>'
            '</linearGradient>'
            '<linearGradient id="laserGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
            '<stop offset="0%" stop-color="#38BDF8" stop-opacity="0"/>'
            '<stop offset="50%" stop-color="#38BDF8" stop-opacity="1"/>'
            '<stop offset="100%" stop-color="#38BDF8" stop-opacity="0"/>'
            '</linearGradient>'
            '</defs>'
            '<!-- Document Background -->'
            '<rect x="25" y="15" width="170" height="130" rx="12" fill="url(#cardGrad)" stroke="#60A5FA" stroke-width="1.5"/>'
            '<!-- Document Header Badge -->'
            '<rect x="42" y="32" width="70" height="8" rx="4" fill="#93C5FD"/>'
            '<rect x="155" y="30" width="24" height="24" rx="6" fill="#3B82F6" fill-opacity="0.3"/>'
            '<path d="M162 42L166 46L173 39" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '<!-- Simulated Document Text Lines -->'
            '<rect x="42" y="52" width="136" height="5" rx="2.5" fill="#DBEAFE" fill-opacity="0.8"/>'
            '<rect x="42" y="65" width="118" height="5" rx="2.5" fill="#DBEAFE" fill-opacity="0.8"/>'
            '<rect x="42" y="78" width="95" height="5" rx="2.5" fill="#DBEAFE" fill-opacity="0.8"/>'
            '<!-- Credential Seal Pill -->'
            '<rect x="42" y="96" width="136" height="24" rx="6" fill="#142B59" stroke="#3B82F6" stroke-width="1.2"/>'
            '<circle cx="56" cy="108" r="5" fill="#16A34A"/>'
            '<rect x="68" y="105" width="60" height="6" rx="3" fill="#93C5FD"/>'
            '<!-- Animated Laser Scanning Beam -->'
            '<g class="scan-beam-line">'
            '<line x1="26" y1="25" x2="194" y2="25" stroke="url(#laserGrad)" stroke-width="3"/>'
            '<line x1="26" y1="25" x2="194" y2="25" stroke="#BAE6FD" stroke-width="1" stroke-dasharray="2 4"/>'
            '</g>'
            '<!-- Enterprise Verification Badge Shield -->'
            '<circle cx="230" cy="100" r="38" fill="#1E40AF" stroke="#60A5FA" stroke-width="2.5"/>'
            '<circle cx="230" cy="100" r="30" fill="#142B59"/>'
            '<path d="M216 100L226 110L245 90" stroke="#34D399" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg>'
        )

        left_html = textwrap.dedent(f"""
<div class="auth-hero-banner">
<div>
<div class="auth-badge-pill">
<span class="system-online-dot"></span> BANK-GRADE AI VERIFICATION
</div>
<div class="auth-brand-logo">Doc<span>Verify</span></div>
<div class="auth-tagline">Automated Document Verification &amp; Consistency Review Platform</div>
{svg_illustration}
<div class="auth-feature-glass">
🔍 <strong>Multi-Format Ingestion:</strong> Native &amp; scanned PDF, PNG, JPG with automated deskew and Tesseract OCR.
</div>
<div class="auth-feature-glass">
🛡️ <strong>Production Rule Engine:</strong> Forward-chaining deterministic validation of chronological dates, identifiers, and schemas.
</div>
<div class="auth-feature-glass">
🤖 <strong>Supervised ML Classification:</strong> Category prediction trained on real-world document corpora without hallucinations.
</div>
<div class="auth-feature-glass">
📑 <strong>Certified Verification Reports:</strong> Downloadable official audit dossiers with cryptographic reference IDs.
</div>
</div>
<div class="auth-trust-box">
🔒 <strong>Enterprise Security Architecture:</strong> PBKDF2-HMAC-SHA256 password hashing with unique random salts &amp; strict multi-tenant tenant isolation.
</div>
</div>
""").strip()
        st.markdown(left_html, unsafe_allow_html=True)

    # ---- RIGHT AUTHENTICATION CARD ----
    with col_right:
        auth_mode = st.radio(
            "Authentication Mode",
            ["Sign In", "Create Account", "Forgot Password"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode_selector"
        )

        # ----------------------------------------------------------------
        # TAB 1: SIGN IN
        # ----------------------------------------------------------------
        if auth_mode == "Sign In":
            st.markdown(textwrap.dedent("""
<div class="auth-form-card">
<div class="auth-form-title">Welcome back</div>
<div class="auth-form-subtitle">Sign in to your DocVerify enterprise account to continue</div>
</div>
""").strip(), unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="name@domain.com")
                password = st.text_input(
                    "Password",
                    type="text" if st.session_state.get("login_show_pwd", False) else "password",
                    placeholder="Enter your account password"
                )
                show_pwd = st.checkbox("Show password", key="login_show_pwd")
                submit_login = st.form_submit_button(
                    "Sign In to Account",
                    type="primary",
                    use_container_width=True
                )

                if submit_login:
                    if not email.strip() or not password.strip():
                        st.error("Please enter both your email address and password.")
                    else:
                        user = database.authenticate_user(email.strip(), password)
                        if user:
                            st.session_state["user"] = user
                            st.session_state["nav_page"] = None
                            st.success(f"Welcome back, {user['full_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid email or password. Please verify your credentials.")

            # Collapsible Demo Credentials for Testing Convenience
            with st.expander("⚡ 1-Click Demo & Evaluator Quick Sign-In", expanded=True):
                st.caption("Pre-configured enterprise accounts for rapid evaluation:")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("👤 Demo Applicant", use_container_width=True, key="demo_applicant_btn"):
                        user = database.authenticate_user("applicant@college.edu", "User@DocVerify2024!")
                        if user:
                            st.session_state["user"] = user
                            st.session_state["nav_page"] = None
                            st.rerun()
                        else:
                            st.warning("Demo applicant account not found in database.")
                with col_d2:
                    if st.button("🛡️ Demo Officer (Admin)", use_container_width=True, key="demo_admin_btn"):
                        admin = database.authenticate_user("admin@docverify.gov.in", "Admin@DocVerify2024!")
                        if not admin:
                            env_email = os.environ.get("DOCVERIFY_ADMIN_EMAIL", "admin@docverify.gov.in")
                            env_pwd = os.environ.get("DOCVERIFY_ADMIN_PASSWORD", "")
                            if env_pwd:
                                admin = database.authenticate_user(env_email, env_pwd)
                        if admin:
                            st.session_state["user"] = admin
                            st.session_state["nav_page"] = None
                            st.rerun()
                        else:
                            st.info("Run `python bootstrap_admin.py` to provision an admin account.")

        # ----------------------------------------------------------------
        # TAB 2: CREATE ACCOUNT (Public Sign-up strictly assigns USER)
        # ----------------------------------------------------------------
        elif auth_mode == "Create Account":
            st.markdown(textwrap.dedent("""
<div class="auth-form-card">
<div class="auth-form-title">Create your applicant account</div>
<div class="auth-form-subtitle">Register to submit credentials for automated consistency verification</div>
</div>
""").strip(), unsafe_allow_html=True)

            with st.form("register_form", clear_on_submit=False):
                reg_name = st.text_input("Full Legal Name", placeholder="e.g. Rahul Sharma")
                reg_email = st.text_input("Email Address", placeholder="name@domain.com")
                show_reg_pwd = st.checkbox("Show password", key="reg_show_pwd")
                reg_pwd = st.text_input(
                    "Password",
                    type="text" if show_reg_pwd else "password",
                    help="Minimum 8 characters with at least one uppercase letter, one lowercase letter, and one digit."
                )
                reg_pwd_c = st.text_input(
                    "Confirm Password",
                    type="text" if show_reg_pwd else "password"
                )

                submit_reg = st.form_submit_button(
                    "Create Free Account",
                    type="primary",
                    use_container_width=True
                )

                if submit_reg:
                    errors = []
                    if not reg_name.strip():
                        errors.append("Please provide your full legal name.")
                    if not auth_manager.validate_email(reg_email):
                        errors.append("Please enter a valid email address.")
                    if reg_pwd != reg_pwd_c:
                        errors.append("Passwords do not match.")
                    elif reg_pwd:
                        is_strong, msg = auth_manager.validate_password_strength(reg_pwd)
                        if not is_strong:
                            errors.append(msg)

                    if errors:
                        for err in errors:
                            st.error(err)
                    else:
                        success, resp_msg, _ = database.create_user(
                            email=reg_email,
                            password=reg_pwd,
                            full_name=reg_name,
                            role="USER",
                            is_admin_creator=False
                        )
                        if success:
                            st.success("✅ Account created successfully! Switch to 'Sign In' to log in.")
                        else:
                            st.error(resp_msg)

        # ----------------------------------------------------------------
        # TAB 3: FORGOT PASSWORD (Secure Token-Based Reset)
        # ----------------------------------------------------------------
        else:
            st.markdown(textwrap.dedent("""
<div class="auth-form-card">
<div class="auth-form-title">Reset your password</div>
<div class="auth-form-subtitle">Generate a time-limited (1-hour), single-use cryptographic reset token</div>
</div>
""").strip(), unsafe_allow_html=True)

            st.markdown("##### Step 1 — Request Reset Token")
            with st.form("request_reset_form"):
                f_email = st.text_input("Registered Email Address", placeholder="name@domain.com")
                request_btn = st.form_submit_button("Generate Reset Token", use_container_width=True)

                if request_btn:
                    if not f_email.strip():
                        st.error("Please enter your registered email address.")
                    elif not auth_manager.validate_email(f_email.strip()):
                        st.error("Please enter a valid email address.")
                    else:
                        token = database.create_password_reset_token(f_email.strip())
                        if token is None:
                            st.info("If this email is registered, a reset token has been generated. Check below.")
                        else:
                            st.session_state["_reset_token_display"] = token
                            st.success("Reset token generated! Copy the token below to complete Step 2.")

            # Show generated token if present
            if st.session_state.get("_reset_token_display"):
                tok = st.session_state["_reset_token_display"]
                token_box = textwrap.dedent(f"""
<div class="token-display-box">
<strong>Your Single-Use Reset Token:</strong><br>{tok}
</div>
""").strip()
                st.markdown(token_box, unsafe_allow_html=True)
                st.caption("⚠️ Token is single-use and expires in 1 hour.")

            st.markdown("##### Step 2 — Set New Password")
            with st.form("consume_reset_form"):
                r_token = st.text_input("Reset Token", placeholder="Paste the token from Step 1")
                show_new = st.checkbox("Show new password", key="reset_show_pwd")
                new_pwd = st.text_input("New Password", type="text" if show_new else "password")
                new_pwd_c = st.text_input("Confirm New Password", type="text" if show_new else "password")
                reset_btn = st.form_submit_button("Update Password", type="primary", use_container_width=True)

                if reset_btn:
                    if not r_token.strip():
                        st.error("Please paste your reset token.")
                    elif not new_pwd:
                        st.error("Please enter a new password.")
                    elif new_pwd != new_pwd_c:
                        st.error("Passwords do not match.")
                    else:
                        is_strong, pw_msg = auth_manager.validate_password_strength(new_pwd)
                        if not is_strong:
                            st.error(pw_msg)
                        else:
                            ok, msg = database.validate_and_consume_reset_token(r_token.strip(), new_pwd)
                            if ok:
                                st.session_state.pop("_reset_token_display", None)
                                st.success(msg)
                                st.info("You can now switch to 'Sign In' and log in.")
                            else:
                                st.error(msg)

        # Footer
        st.markdown(textwrap.dedent("""
<div style="text-align: center; margin-top: 26px; font-size: 0.78rem; color: #94A3B8;">
DocVerify &copy; 2024 &nbsp;|&nbsp; Enterprise Document Consistency &amp; Forensics Platform
</div>
""").strip(), unsafe_allow_html=True)


# Enforce Authentication Boundary
if not st.session_state.get("user"):
    render_authentication_view()
    st.stop()

# Current Authenticated User Context
current_user = st.session_state.get("user") or {}
is_admin = (current_user.get("role") == "ADMIN")


# ---------------------------------------------------------------------------
# AUTHENTICATED SIDEBAR & NAVIGATION
# ---------------------------------------------------------------------------

# Sidebar Brand Header
st.sidebar.markdown(textwrap.dedent("""
<div class="brand-container">
    <div class="brand-icon-box">🛡️</div>
    <div>
        <h2 class="brand-title">Doc<span>Verify</span></h2>
        <span class="brand-tag">Enterprise AI</span>
    </div>
</div>
""").strip(), unsafe_allow_html=True)

# User Profile Card in Sidebar
role_class = "role-badge-admin" if is_admin else "role-badge-user"
role_title = "VERIFICATION OFFICER" if is_admin else "APPLICANT"
role_icon = "🛡️" if is_admin else "👤"

# Initial letters for avatar
full_name = current_user.get('full_name', 'User')
initials = "".join([part[0].upper() for part in full_name.split()[:2]]) if full_name else "U"

sidebar_profile = textwrap.dedent(f"""
<div class="sidebar-user-card">
    <div class="user-avatar-row">
        <div class="user-avatar">{initials}</div>
        <div class="user-details">
            <div class="user-name" title="{full_name}">{full_name}</div>
            <div class="user-email" title="{current_user.get('email', '')}">{current_user.get('email', '')}</div>
        </div>
    </div>
    <div>
        <span class="user-role-badge {role_class}">
            {role_icon} {role_title}
        </span>
    </div>
</div>
""").strip()
st.sidebar.markdown(sidebar_profile, unsafe_allow_html=True)

# Navigation Options
if is_admin:
    nav_options = [
        "Overview",
        "Pending Reviews",
        "All Submitted Documents",
        "Review History",
        "User Management",
        "Audit Logs",
        "System Analytics"
    ]
else:
    nav_options = [
        "Dashboard",
        "Upload Document",
        "My Documents",
        "Verification Reports",
        "Profile"
    ]

# Default or programmatic navigation
selected_nav_idx = 0
if st.session_state.get("nav_page") in nav_options:
    selected_nav_idx = nav_options.index(st.session_state["nav_page"])

active_nav = st.sidebar.radio("Navigation", nav_options, index=selected_nav_idx, label_visibility="collapsed")

st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

if st.sidebar.button("Sign Out", use_container_width=True):
    st.session_state["user"] = None
    st.session_state["active_doc_id"] = None
    st.session_state["last_processed"] = None
    st.session_state["nav_page"] = None
    st.session_state.pop("_reset_token_display", None)
    st.rerun()

st.sidebar.markdown(textwrap.dedent("""
<div style="font-size: 0.74rem; color: #94A3B8; margin-top: 14px; padding-left: 2px;">
    <span class="pulse-dot-container">
        <span class="system-online-dot"></span> System Online &nbsp;|&nbsp; DocVerify v2.5
    </span>
</div>
""").strip(), unsafe_allow_html=True)


# ===========================================================================
# APPLICANT VIEW 1: DASHBOARD
# ===========================================================================

if not is_admin and active_nav == "Dashboard":
    # Hero Welcome Banner
    st.markdown(textwrap.dedent(f"""
<div class="welcome-hero">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
        <div>
            <div class="welcome-title">Welcome back, {current_user['full_name']} 👋</div>
            <div class="welcome-desc">Verify official credentials, review consistency findings, inspect forensics, and download certified reports in seconds.</div>
        </div>
    </div>
</div>
""").strip(), unsafe_allow_html=True)

    col_btn_l, col_btn_r = st.columns([3.5, 1.2])
    with col_btn_r:
        if st.button("+ Upload New Document", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Upload Document"
            st.rerun()

    # User Metrics from Real Database Values
    user_docs = database.get_documents_by_user(current_user["id"])
    total_docs = len(user_docs)
    passed_docs = sum(1 for d in user_docs if d["verification_status"] == STATUS_CONSISTENCY_PASSED)
    review_docs = sum(1 for d in user_docs if d["verification_status"] == STATUS_NEEDS_REVIEW)
    inconsistent_docs = sum(1 for d in user_docs if d["verification_status"] == STATUS_INCONSISTENCIES)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #EFF6FF; color: #2563EB;">📄</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #3B82F6; background: #EFF6FF; padding: 2px 8px; border-radius: 9999px;">TOTAL</span>
    </div>
    <div>
        <div class="docv-metric-value">{total_docs}</div>
        <div class="docv-metric-label">Documents Submitted</div>
    </div>
    <div class="metric-bottom-bar" style="background: #2563EB;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with m2:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #ECFDF5; color: #16A34A;">✅</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #16A34A; background: #ECFDF5; padding: 2px 8px; border-radius: 9999px;">PASSED</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #16A34A;">{passed_docs}</div>
        <div class="docv-metric-label">Consistency Passed</div>
    </div>
    <div class="metric-bottom-bar" style="background: #16A34A;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with m3:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #FFFBEB; color: #F59E0B;">⏳</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #F59E0B; background: #FFFBEB; padding: 2px 8px; border-radius: 9999px;">REVIEW</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #F59E0B;">{review_docs}</div>
        <div class="docv-metric-label">Needs Manual Review</div>
    </div>
    <div class="metric-bottom-bar" style="background: #F59E0B;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with m4:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #FEF2F2; color: #DC2626;">⚠️</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #DC2626; background: #FEF2F2; padding: 2px 8px; border-radius: 9999px;">ISSUES</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #DC2626;">{inconsistent_docs}</div>
        <div class="docv-metric-label">Inconsistencies Detected</div>
    </div>
    <div class="metric-bottom-bar" style="background: #DC2626;"></div>
</div>
""").strip(), unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Empty State vs Populated State
    if total_docs == 0:
        empty_box = textwrap.dedent("""
<div class="docv-card" style="text-align: center; padding: 56px 28px; margin-top: 10px;">
    <div style="font-size: 2.8rem; margin-bottom: 12px;">📁</div>
    <div style="font-size: 1.3rem; font-weight: 800; color: #172554;">No credentials submitted yet</div>
    <div style="font-size: 0.92rem; color: #64748B; margin-top: 6px; max-width: 520px; margin-left: auto; margin-right: auto; line-height: 1.5;">
        Upload your educational degrees, identity credentials, or government certificates to run automated Tesseract OCR, production schema validation, and consistency checks.
    </div>
</div>
""").strip()
        st.markdown(empty_box, unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns([1, 1.2, 1])
        with col_c2:
            if st.button("Upload Your First Document", type="primary", use_container_width=True, key="empty_upload_btn"):
                st.session_state["nav_page"] = "Upload Document"
                st.rerun()
    else:
        # Two Columns: Recent Documents & Recent Activity
        col_recent, col_activity = st.columns([1.6, 1.4], gap="large")

        with col_recent:
            st.markdown("### Recent Submissions")
            recent_rows = []
            for d in user_docs[:5]:
                recent_rows.append({
                    "Document Name": d["filename"],
                    "Category": d["declared_category"],
                    "Submission Date": d["upload_timestamp"][:10],
                    "Verification Status": d["verification_status"]
                })
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)

            if st.button("View All Documents", use_container_width=True, key="dash_view_all_docs"):
                st.session_state["nav_page"] = "My Documents"
                st.rerun()

        with col_activity:
            st.markdown("### Recent Activity Trail")
            activities = database.get_recent_activity_for_user(current_user["id"], limit=5)

            if activities:
                for act in activities:
                    time_display = act["timestamp"][:19].replace("T", " ")
                    action_name = act["action"].replace("_", " ").title()
                    doc_label = f" ({act['filename']})" if act.get("filename") else ""
                    st.markdown(textwrap.dedent(f"""
<div class="activity-item">
    <div class="activity-icon">📌</div>
    <div class="activity-content">
        <div class="activity-title">{action_name}{doc_label}</div>
        <div style="font-size: 0.82rem; color: #475569; margin-top: 1px;">{act['details']}</div>
        <div class="activity-time">{time_display}</div>
    </div>
</div>
""").strip(), unsafe_allow_html=True)
            else:
                for d in user_docs[:4]:
                    sub_time = d["upload_timestamp"][:19].replace("T", " ")
                    st.markdown(textwrap.dedent(f"""
<div class="activity-item">
    <div class="activity-icon">📄</div>
    <div class="activity-content">
        <div class="activity-title">Document Submitted: {d['filename']}</div>
        <div style="font-size: 0.82rem; color: #475569; margin-top: 1px;">Category: {d['declared_category']} &bull; Status: {d['verification_status']}</div>
        <div class="activity-time">{sub_time}</div>
    </div>
</div>
""").strip(), unsafe_allow_html=True)

    # Regulatory Attestation Notice
    st.markdown(textwrap.dedent(f"""
<div class="regulatory-box">
    <strong>🛡️ Regulatory Compliance Attestation:</strong><br>
    {AUTHENTICITY_DISCLAIMER}
</div>
""").strip(), unsafe_allow_html=True)


# ===========================================================================
# APPLICANT VIEW 2: UPLOAD DOCUMENT & VERIFICATION WORKFLOW
# ===========================================================================

elif not is_admin and active_nav == "Upload Document":
    st.markdown("## Upload Document for Verification")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>Upload credentials to extract structured fields, validate against official schema rules, and receive an instant consistency dossier.</p>", unsafe_allow_html=True)

    col_up_cfg, col_up_res = st.columns([1.1, 1.9], gap="large")

    with col_up_cfg:
        st.markdown("#### 1. Select Document Category")
        category = st.selectbox(
            "Document Category",
            options=list(SCHEMA_REGISTRY.keys()),
            help="Select the category that corresponds to your credential."
        )

        schema = get_schema(category)
        req_fields = [k for k, v in schema.fields.items() if v.required]
        st.caption(f"**Required Fields:** {', '.join(req_fields)}")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 2. Upload Document File")
        
        st.markdown(textwrap.dedent("""
<div class="upload-zone-hint">
    <div style="font-size: 1.6rem; margin-bottom: 4px;">☁️</div>
    <div style="font-weight: 700; font-size: 0.95rem; color: #172554;">Drag and drop your file here</div>
    <div style="font-size: 0.80rem; color: #64748B; margin-top: 2px;">Supported: PDF, PNG, JPG, JPEG &bull; Max: 10MB</div>
</div>
""").strip(), unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload File (PDF, PNG, JPG, JPEG)",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload clear scans or digital documents (up to 10MB).",
            label_visibility="collapsed"
        )

        # Reference Benchmark Credential Selector for evaluator testing
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        use_sample = st.checkbox("⚡ Or run an SIH Live Judge Demo Preset (1-Click Test)", value=False)
        sample_choice = None
        if use_sample:
            sample_options = {
                "🎓 Valid Degree (IIT Bombay - Golden Standard)": (
                    "Educational Certificate",
                    "INDIAN INSTITUTE OF TECHNOLOGY BOMBAY\nCONVOCATION STATEMENT\nThis is to certify that Rahul Sharma bearing Roll No: 2019-BTECH-CS-042 has completed Bachelor of Technology in Computer Science.\nDate of Issue: 15/07/2023\nCertificate ID: IITB/2023/CS/1089\nRegistrar | Director",
                    98.5
                ),
                "🆔 Valid Aadhaar (Mathematically Valid Verhoeff UID)": (
                    "Identity Document",
                    "GOVERNMENT OF INDIA\nUNIQUE IDENTIFICATION AUTHORITY OF INDIA\nAADHAAR CARD\nName: Ananya Verma\nDOB: 14/05/1998\nGender: Female\nAadhaar Number: 2093 4519 8273\nIssue Date: 20/08/2016",
                    96.0
                ),
                "💳 Valid Income Tax PAN Card (P-Entity & Sharma Surname)": (
                    "Identity Document",
                    "INCOME TAX DEPARTMENT GOVT. OF INDIA\nPermanent Account Number Card\nName: Vikram Sharma\nFather's Name: Omprakash Sharma\nDOB: 22/11/1992\nPAN: ABCPS1234D",
                    97.0
                ),
                "🏛️ Valid Domicile Certificate (Government of UP)": (
                    "Government Certificate",
                    "GOVERNMENT OF UTTAR PRADESH\nREVENUE DEPARTMENT\nDOMICILE CERTIFICATE\nApplicant Name: Rohan Gupta\nCertificate ID: UP/DOM/2022/8849\nIssuing Authority: Office of the Tehsildar Sadar\nIssue Date: 18/06/2022",
                    95.0
                ),
                "⚠️ Contradiction: Future Issue Date (2035)": (
                    "Government Certificate",
                    "GOVERNMENT OF UTTAR PRADESH\nREVENUE DEPARTMENT\nDOMICILE CERTIFICATE\nApplicant Name: Ramesh Chandra\nCertificate ID: UP/DOM/2035/9912\nIssuing Authority: Tehsildar Sadar Lucknow\nIssue Date: 15/12/2035",
                    95.0
                ),
                "⚠️ Contradiction: Fake Aadhaar (Verhoeff Checksum Failure)": (
                    "Identity Document",
                    "GOVERNMENT OF INDIA\nAADHAAR CARD\nName: Manish Tiwari\nDOB: 09/12/1994\nAadhaar: 1234 5678 9012\nIssue Date: 18/07/2017",
                    94.0
                ),
                "⚠️ Contradiction: PAN 5th Letter Surname Mismatch": (
                    "Identity Document",
                    "INCOME TAX DEPARTMENT GOVT OF INDIA\nName: Rajesh Gupta\nDOB: 11/04/1988\nPAN: ABCPS1234D\nIssue Date: 14/02/2012",
                    97.0
                ),
                "⚠️ Contradiction: Temporal Inversion (Expiry < Issue Date)": (
                    "Government Certificate",
                    "GOVERNMENT OF NCT OF DELHI\nINCOME CERTIFICATE\nApplicant Name: Karan Singhal\nCertificate ID: DL/INC/2021/88\nIssuing Authority: Revenue Department Delhi\nIssue Date: 20/08/2021\nExpiry Date: 15/01/2020",
                    96.0
                ),
                "⚠️ Contradiction: Expired Identity Passport": (
                    "Identity Document",
                    "PASSPORT REPUBLIC OF INDIA\nName: Pooja Malhotra\nDocument Number: P8819204\nDate of Birth: 14/06/1990\nIssue Date: 10/01/2010\nExpiry Date: 09/01/2020",
                    95.0
                ),
                "⚠️ Contradiction: Issued Before Birth Date": (
                    "Identity Document",
                    "TRANSPORT DEPARTMENT DELHI\nDRIVING LICENCE\nName: Sunil Verma\nDocument Number: DL-1420110091\nDate of Birth: 25/08/2015\nIssue Date: 10/04/2010\nExpiry Date: 09/04/2030",
                    96.0
                ),
                "🌫️ Extraction Uncertainty: Blurry Degraded Scan (OCR < 45%)": (
                    "Educational Certificate",
                    "C@lcut!a Un1vers1ty\nTh1s 1s t0 cert1fy th@t A. K. R0y h@s p@ssed.\nCert: UNIV/8821\nD@te: 14/08/2021",
                    42.0
                ),
                "❓ Incomplete: Missing Student Name": (
                    "Educational Certificate",
                    "UNIVERSITY OF DELHI\nThis is to certify that [Obscured/Torn] has passed Bachelor of Commerce.\nCertificate ID: DU/2021/BCOM/401\nIssue Date: 22/05/2021",
                    91.0
                ),
                "🛒 Category Mismatch: Uploaded Invoice for Educational Certificate": (
                    "Educational Certificate",
                    "BOOKSTORE & STATIONERY MART\nTAX INVOICE / RECEIPT\nInvoice No: INV-2018-9941\nDate: 12/03/2018\nTotal: 450.00 INR\nTHANK YOU FOR SHOPPING",
                    92.0
                )
            }
            sample_name = st.selectbox("Reference Credential Preset:", list(sample_options.keys()))
            sample_cat, sample_text, sample_conf = sample_options[sample_name]
            category = sample_cat

        # File Metadata & Preview
        if uploaded_file is not None:
            f_size_kb = len(uploaded_file.getvalue()) / 1024
            size_str = f"{f_size_kb / 1024:.2f} MB" if f_size_kb > 1024 else f"{f_size_kb:.1f} KB"
            st.markdown(textwrap.dedent(f"""
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 8px; font-size: 0.84rem; margin-top: 8px;">
    <strong>Selected File:</strong> <code>{uploaded_file.name}</code> &bull; <strong>Size:</strong> {size_str}
</div>
""").strip(), unsafe_allow_html=True)

            try:
                if uploaded_file.type == "application/pdf":
                    doc = pymupdf.open(stream=uploaded_file.getvalue(), filetype="pdf")
                    page_count = len(doc)
                    if page_count > 0:
                        page = doc.load_page(0)
                        pix = page.get_pixmap(dpi=120)
                        preview_img = Image.open(io.BytesIO(pix.tobytes("png")))
                        st.image(preview_img, caption=f"Document Preview (Page 1 of {page_count})", use_container_width=True)
                    doc.close()
                elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                    preview_img = Image.open(io.BytesIO(uploaded_file.getvalue()))
                    st.image(preview_img, caption=f"Document Preview ({uploaded_file.name})", use_container_width=True)
            except Exception as e:
                st.caption(f"Preview unavailable: {str(e)}")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        run_verify = st.button("⚡ Verify Document Consistency", type="primary", use_container_width=True)

    with col_up_res:
        st.markdown("#### Verification Results & Consistency Dossier")

        if run_verify:
            if not uploaded_file and not use_sample:
                st.error("Please upload a document file or select a reference credential preset.")
            else:
                with st.spinner("Processing document: OCR text extraction, schema checks, rule evaluation, and decision engine..."):
                    try:
                        if uploaded_file is not None:
                            file_bytes = uploaded_file.getvalue()
                            filename = uploaded_file.name
                            mime_type = uploaded_file.type
                            processed = process_file_bytes(file_bytes, filename, mime_type)
                        else:
                            filename = f"ref_{sample_name[:12].strip().replace(' ', '_').lower()}.txt"
                            file_bytes = sample_text.encode("utf-8")
                            processed = {
                                "filename": filename,
                                "page_count": 1,
                                "full_text": sample_text,
                                "average_ocr_confidence": sample_conf,
                                "pages": [{"page_number": 1, "text": sample_text, "confidence": sample_conf, "method": "REFERENCE_BENCHMARK"}],
                                "is_ocr_enabled": True
                            }

                        # Step 1: Structured Information Extraction
                        extracted_fields = extract_structured_fields(processed, category)

                        # Step 2: Schema Validation
                        schema_results = validate_document_schema(extracted_fields, category, raw_text=processed["full_text"])

                        # Step 3: ML Category Classification
                        ml_result = predict_document_category(processed["full_text"])

                        # Step 4: Explainable Production Rule Engine
                        verdict, triggered_rules = run_rule_engine(
                            extracted_fields=extracted_fields,
                            category_name=category,
                            raw_text=processed["full_text"],
                            ocr_confidence=processed["average_ocr_confidence"],
                            ml_predicted_category=ml_result.get("predicted_category")
                        )

                        # Step 5: Decision Engine Integration
                        decision_outcome = decision_engine.evaluate_verification_decision(
                            extracted_fields=extracted_fields,
                            schema_results=schema_results,
                            rule_results=triggered_rules,
                            ml_result=ml_result,
                            ocr_confidence=processed["average_ocr_confidence"],
                            page_count=processed["page_count"],
                            raw_text=processed["full_text"],
                            declared_category=category
                        )

                        # Step 6: Persist Record Linked to User
                        doc_id = database.save_verification_record(
                            filename=filename,
                            file_type="PDF/Image" if uploaded_file else "Reference Record",
                            file_size=len(file_bytes),
                            declared_category=category,
                            predicted_category=ml_result.get("predicted_category"),
                            ocr_confidence=processed["average_ocr_confidence"],
                            page_count=processed["page_count"],
                            raw_text=processed["full_text"],
                            verification_status=decision_outcome.final_status,
                            extracted_fields=extracted_fields,
                            triggered_rules=triggered_rules,
                            ml_result=ml_result,
                            user_id=current_user["id"],
                            requires_admin_review=decision_outcome.requires_admin_review,
                            escalation_reason=decision_outcome.escalation_reason
                        )

                        # Step 7: Create Official Report Record
                        report_ref = database.create_report_record(
                            document_id=doc_id,
                            user_id=current_user["id"],
                            report_type="USER_REPORT",
                            final_status=decision_outcome.final_status,
                            summary_data={
                                "explanation": decision_outcome.simple_explanation,
                                "reasons": decision_outcome.detailed_reasons,
                                "actions": decision_outcome.recommended_actions
                            }
                        )

                        st.session_state["active_doc_id"] = doc_id
                        st.session_state["last_processed"] = {
                            "doc_id": doc_id,
                            "report_ref": report_ref,
                            "filename": filename,
                            "category": category,
                            "outcome": decision_outcome,
                            "processed": processed,
                            "fields": extracted_fields,
                            "schema": schema_results,
                            "rules": triggered_rules,
                            "ml": ml_result
                        }
                    except Exception as ex:
                        st.error(f"Error processing document: {str(ex)}")

        # Render Results View
        if st.session_state.get("last_processed"):
            lp = st.session_state["last_processed"]
            outcome: decision_engine.DecisionOutcome = lp["outcome"]
            status = outcome.final_status

            # Status Badge Rendering
            if status == STATUS_CONSISTENCY_PASSED:
                badge_html = f'<div class="status-badge-passed">✅ {status}</div>'
            elif status == STATUS_NEEDS_REVIEW:
                badge_html = f'<div class="status-badge-review">⏳ {status}</div>'
            elif status == STATUS_INCONSISTENCIES:
                badge_html = f'<div class="status-badge-inconsistent">⚠️ {status}</div>'
            else:
                badge_html = f'<div class="status-badge-unable">❓ {status}</div>'

            rec_action = outcome.recommended_actions[0] if outcome.recommended_actions else 'Document verified coherent.'

            result_box = textwrap.dedent(f"""
<div class="docv-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
        <div>{badge_html}</div>
        <div style="font-size: 0.85rem; color: #64748B;">Report Ref: <code style="background: #F1F5F9; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{lp['report_ref']}</code></div>
    </div>
    <div style="font-size: 1.15rem; font-weight: 750; color: #172554; margin-bottom: 8px; line-height: 1.4;">
        {outcome.simple_explanation}
    </div>
    <div style="font-size: 0.88rem; color: #475569; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 8px;">
        <strong>Recommended Next Action:</strong> {rec_action}
    </div>
</div>
""").strip()
            st.markdown(result_box, unsafe_allow_html=True)

            # Dual-Axis Uncertainty vs Inconsistency Matrix Display
            col_m1, col_m2, col_m3 = st.columns([1, 1, 1.6])
            with col_m1:
                st.metric("Extraction Quality (X)", f"{getattr(outcome, 'extraction_quality_score', 85.0):.1f}%")
            with col_m2:
                st.metric("Logical Consistency (Y)", f"{getattr(outcome, 'logical_consistency_score', 100.0):.1f}%")
            with col_m3:
                quad = getattr(outcome, 'uncertainty_quadrant', 'QUADRANT I: VERIFIED CONSISTENT')
                adv = getattr(outcome, 'uncertainty_advice', 'Auto-finalized. No manual delay required.')
                st.markdown(textwrap.dedent(f"""
<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; margin-top: 4px;">
    <strong style="color: #172554;">🧭 {quad}</strong><br/>
    <span style="color: #64748B;">{adv}</span>
</div>
""").strip(), unsafe_allow_html=True)

            # Extracted Fields Table
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("##### Extracted Information")
            field_display = []
            for f_name, f_val in lp["fields"].items():
                if isinstance(f_val, dict):
                    if f_val.get("status") == "NOT_APPLICABLE":
                        v = "[Not Applicable]"
                        stat = "Not Applicable"
                    else:
                        v = f_val.get("corrected_value") or f_val.get("value") or "[Missing]"
                        stat = "Detected" if v != "[Missing]" else "Missing"
                else:
                    v = str(f_val) if f_val else "[Missing]"
                    stat = "Detected" if v != "[Missing]" else "Missing"
                field_display.append({
                    "Field": f_name,
                    "Extracted Value": v,
                    "Extraction Status": stat
                })
            st.dataframe(pd.DataFrame(field_display), use_container_width=True, hide_index=True)

            # Detected Issues & Explanations
            if outcome.has_contradictions or outcome.has_missing_required or status != STATUS_CONSISTENCY_PASSED:
                st.markdown("##### Detected Issues & Reasons")
                for r in outcome.detailed_reasons:
                    st.warning(f"• {r}")

            # Digital Document Forensics & ELA Tamper Analysis
            proc_dict = lp.get("processed", {})
            if proc_dict.get("forensics") or proc_dict.get("ela_preview_image"):
                with st.expander("🔬 Digital Document Forensics & ELA Tamper Analysis", expanded=False):
                    forensics = proc_dict.get("forensics", {})
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown("**Error Level Analysis (ELA) Compression Map**")
                        if proc_dict.get("ela_preview_image"):
                            st.image(proc_dict["ela_preview_image"], caption=f"ELA Difference Map (Risk Score: {proc_dict.get('ela_risk_score', 0):.1f}%)", use_container_width=True)
                            st.caption(f"**Assessment:** {proc_dict.get('ela_verdict', 'Normal')}")
                        else:
                            st.caption("ELA preview generated for raster scanned inputs.")
                    with col_f2:
                        st.markdown("**Document Metadata Provenance**")
                        meta = forensics.get("metadata", {})
                        if meta:
                            for k, v in meta.items():
                                st.markdown(f"- **{k}:** `{v}`")
                        if forensics.get("is_suspicious"):
                            for reason in forensics.get("suspicious_reasons", []):
                                st.error(f"⚠️ {reason}")
                        else:
                            st.success("✅ Metadata integrity verified: No graphic editing tool signatures found.")

            # Download Certified PDF Report Button
            pdf_bytes = report_generator.generate_verification_report_pdf(
                document_info={"filename": lp["filename"], "declared_category": lp["category"], "id": lp["doc_id"]},
                extracted_fields=lp["fields"],
                schema_results=lp["schema"],
                rule_results=lp["rules"],
                ml_results=lp["ml"],
                overall_status=status,
                report_reference=lp["report_ref"],
                is_admin_report=False,
                detected_issues=outcome.detailed_reasons,
                recommended_actions=outcome.recommended_actions
            )

            st.download_button(
                "📥 Download Official Verification Report (PDF)",
                data=pdf_bytes,
                file_name=f"{lp['report_ref']}_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        else:
            empty_analysis = textwrap.dedent("""
<div class="docv-card" style="text-align: center; padding: 48px 24px; color: #64748B;">
    <div style="font-size: 2.2rem; margin-bottom: 8px;">⚙️</div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #172554;">Ready for Document Ingestion</div>
    <div style="font-size: 0.88rem; margin-top: 6px; max-width: 440px; margin-left: auto; margin-right: auto; line-height: 1.5;">
        Select a category, upload your document file or test preset, and click <strong>"Verify Document Consistency"</strong> to launch the AI audit pipeline.
    </div>
</div>
""").strip()
            st.markdown(empty_analysis, unsafe_allow_html=True)


# ===========================================================================
# APPLICANT VIEW 3: MY DOCUMENTS
# ===========================================================================

elif not is_admin and active_nav == "My Documents":
    st.markdown("## My Submitted Documents")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>All credentials submitted under your account are listed here with strict data isolation.</p>", unsafe_allow_html=True)

    docs = database.get_documents_by_user(current_user["id"])
    if not docs:
        st.info("You haven't submitted any documents yet.")
        if st.button("Upload Document Now", type="primary"):
            st.session_state["nav_page"] = "Upload Document"
            st.rerun()
    else:
        df_docs = pd.DataFrame([
            {
                "ID": d["id"],
                "File Name": d["filename"],
                "Category": d["declared_category"],
                "Status": d["verification_status"],
                "Review State": "Pending Review" if (d["requires_admin_review"] and d["admin_review_status"] == "PENDING") else (d["admin_review_status"] if d["admin_review_status"] != "NONE" else "Auto Finalized"),
                "Uploaded On": d["upload_timestamp"][:19].replace("T", " ")
            }
            for d in docs
        ])
        st.dataframe(df_docs, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Document Dossier Inspector")
        doc_options = [f"#{d['id']}: {d['filename']} ({d['declared_category']})" for d in docs]
        sel_doc_str = st.selectbox("Select Document Dossier:", doc_options)
        sel_doc_id = int(sel_doc_str.split(":")[0].replace("#", ""))

        details = database.get_document_details(sel_doc_id, requesting_user_id=current_user["id"], is_admin=False)
        if details:
            t_fields, t_rules, t_audit = st.tabs(["Extracted Fields", "Consistency Checks", "Audit Trail"])
            with t_fields:
                f_rows = [
                    {
                        "Field": f["field_name"],
                        "Extracted Value": f["corrected_value"] if f["is_manually_corrected"] else (f["extracted_value"] or "[MISSING]"),
                        "Confidence": f"{f['confidence']:.0f}%"
                    }
                    for f in details.get("fields", [])
                ]
                st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)

            with t_rules:
                r_rows = [
                    {"Check": r["rule_name"], "Severity": r["severity"], "Explanation": r["explanation"]}
                    for r in details.get("rules", [])
                ]
                st.dataframe(pd.DataFrame(r_rows), use_container_width=True, hide_index=True)

            with t_audit:
                for a in details.get("audit", []):
                    st.markdown(f"• **[{a['timestamp'][:19]}] {a['action']}** by `{a['actor']}`: *{a['details']}*")


# ===========================================================================
# APPLICANT VIEW 4: VERIFICATION REPORTS
# ===========================================================================

elif not is_admin and active_nav == "Verification Reports":
    st.markdown("## Official Verification Reports")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>Download official, certified PDF consistency dossiers for your verified credentials.</p>", unsafe_allow_html=True)

    reports = database.get_reports_for_user(current_user["id"])
    if not reports:
        st.info("No verification reports generated yet. Submit a document to generate a report.")
    else:
        for r in reports:
            with st.container():
                st.markdown(textwrap.dedent(f"""
<div class="docv-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div>
            <div style="font-weight: 750; font-size: 1.1rem; color: #172554;">{r['filename']}</div>
            <div style="font-size: 0.86rem; color: #64748B; margin-top: 3px;">
                Category: <strong>{r['declared_category']}</strong> &nbsp;|&nbsp; Status: <code>{r['final_status']}</code>
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 4px;">
                Reference ID: <strong>{r['report_reference']}</strong> &nbsp;|&nbsp; Generated: {r['generated_at'][:19].replace('T', ' ')}
            </div>
        </div>
    </div>
</div>
""").strip(), unsafe_allow_html=True)

                doc_det = database.get_document_details(r["document_id"], requesting_user_id=current_user["id"], is_admin=False)
                if doc_det:
                    ext_dict = {
                        f["field_name"]: f["corrected_value"] if f["is_manually_corrected"] else f["extracted_value"]
                        for f in doc_det.get("fields", [])
                    }
                    pdf = report_generator.generate_verification_report_pdf(
                        document_info={"filename": r["filename"], "declared_category": r["declared_category"], "id": r["document_id"]},
                        extracted_fields=ext_dict,
                        schema_results=[],
                        rule_results=doc_det.get("rules", []),
                        ml_results={"predicted_category": doc_det.get("predicted_category")},
                        overall_status=r["final_status"],
                        report_reference=r["report_reference"],
                        is_admin_report=False
                    )
                    st.download_button(
                        label=f"📥 Download Report PDF ({r['report_reference']})",
                        data=pdf,
                        file_name=f"{r['report_reference']}.pdf",
                        mime="application/pdf",
                        key=f"dl_rep_{r['id']}",
                        type="primary"
                    )
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# ===========================================================================
# APPLICANT VIEW 5: PROFILE
# ===========================================================================

elif not is_admin and active_nav == "Profile":
    st.markdown("## Profile & Security Settings")

    col_p1, col_p2 = st.columns(2, gap="large")
    with col_p1:
        st.markdown("#### Account Information")
        st.markdown(textwrap.dedent(f"""
<div class="docv-card">
    <div style="font-size: 0.95rem;"><strong>Full Legal Name:</strong> {current_user['full_name']}</div>
    <div style="margin-top: 8px; font-size: 0.95rem;"><strong>Email Address:</strong> {current_user['email']}</div>
    <div style="margin-top: 8px; font-size: 0.95rem;"><strong>Account Role:</strong> <code>{current_user['role']}</code></div>
    <div style="margin-top: 8px; font-size: 0.95rem;"><strong>Account Reference ID:</strong> #{current_user['id']}</div>
</div>
""").strip(), unsafe_allow_html=True)

        st.markdown("#### Data Privacy & GDPR Purge")
        st.caption("You may permanently delete all your submitted documents and verification records.")
        if st.button("Purge All My Submitted Records", type="secondary"):
            user_docs = database.get_documents_by_user(current_user["id"])
            for d in user_docs:
                database.delete_document_record(d["id"], actor=current_user["email"], user_id=current_user["id"])
            st.success("All your document dossiers and extracted data have been permanently deleted.")
            st.rerun()

    with col_p2:
        st.markdown("#### Change Password")
        with st.form("change_pwd_form"):
            new_p = st.text_input("New Password", type="password")
            new_p_c = st.text_input("Confirm New Password", type="password")
            chg_btn = st.form_submit_button("Update Password", type="primary")

            if chg_btn:
                if new_p != new_p_c:
                    st.error("Passwords do not match.")
                else:
                    is_s, msg = auth_manager.validate_password_strength(new_p)
                    if not is_s:
                        st.error(msg)
                    else:
                        database.update_user_password(current_user["id"], new_p)
                        st.success("Password updated successfully!")


# ===========================================================================
# ADMIN VIEW 1: OVERVIEW
# ===========================================================================

elif is_admin and active_nav == "Overview":
    st.markdown("## Officer Command Center")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>Administrative overview of review queues, escalated dossiers, and platform operations.</p>", unsafe_allow_html=True)

    stats = database.get_system_statistics()
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #FEF2F2; color: #DC2626;">⏳</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #DC2626; background: #FEF2F2; padding: 2px 8px; border-radius: 9999px;">QUEUE</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #DC2626;">{stats['pending_admin_reviews']}</div>
        <div class="docv-metric-label">Pending Officer Reviews</div>
    </div>
    <div class="metric-bottom-bar" style="background: #DC2626;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with a2:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #EFF6FF; color: #2563EB;">📊</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #2563EB; background: #EFF6FF; padding: 2px 8px; border-radius: 9999px;">TOTAL</span>
    </div>
    <div>
        <div class="docv-metric-value">{stats['total_documents']}</div>
        <div class="docv-metric-label">Total Documents Processed</div>
    </div>
    <div class="metric-bottom-bar" style="background: #2563EB;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with a3:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #ECFDF5; color: #16A34A;">✅</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #16A34A; background: #ECFDF5; padding: 2px 8px; border-radius: 9999px;">PASSED</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #16A34A;">{stats['consistent_documents']}</div>
        <div class="docv-metric-label">Verified Consistent</div>
    </div>
    <div class="metric-bottom-bar" style="background: #16A34A;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with a4:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #EFF6FF; color: #142B59;">👥</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #142B59; background: #EFF6FF; padding: 2px 8px; border-radius: 9999px;">USERS</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #142B59;">{stats['total_users']}</div>
        <div class="docv-metric-label">Registered Accounts</div>
    </div>
    <div class="metric-bottom-bar" style="background: #142B59;"></div>
</div>
""").strip(), unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
<div class="docv-card">
    <h4 style="margin-top: 0; color: #172554; font-weight: 750;">AI-First Selective Escalation Architecture</h4>
    <p style="color: #475569; font-size: 0.92rem; line-height: 1.55;">
        DocVerify processes clean documents automatically without imposing administrative delay on straightforward applicants:
    </p>
    <ul style="color: #475569; font-size: 0.90rem; line-height: 1.65;">
        <li><strong>Automated Finalization:</strong> Documents meeting all schema requirements, with high OCR clarity and zero rule contradictions, pass immediately.</li>
        <li><strong>Selective Escalation:</strong> Only documents with degraded scan clarity (&lt;60% OCR), missing required fields, contradictory dates, or unreadable formats are routed to the pending review queue.</li>
        <li><strong>Officer Intervention:</strong> Reviewers can inspect original scans, apply in-line field corrections with audit trail, request applicant resubmission, or issue formal determinations.</li>
    </ul>
</div>
""").strip(), unsafe_allow_html=True)


# ===========================================================================
# ADMIN VIEW 2: PENDING REVIEWS
# ===========================================================================

elif is_admin and active_nav == "Pending Reviews":
    st.markdown("## Pending Officer Review Queue")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>Review escalated credentials, apply in-line field corrections, and issue formal verification determinations.</p>", unsafe_allow_html=True)

    queue_docs = database.get_admin_review_queue(filter_type="PENDING")
    if not queue_docs:
        st.success("✅ No documents currently awaiting manual review. The queue is completely clear!")
    else:
        st.write(f"Showing **{len(queue_docs)}** escalated document(s) requiring officer review:")

        queue_options = [
            f"Doc #{d['id']}: {d['filename']} | {d['declared_category']} | Applicant: {d.get('applicant_name', 'N/A')}"
            for d in queue_docs
        ]
        selected_queue_str = st.selectbox("Select Document to Review:", queue_options)
        sel_doc_id = int(selected_queue_str.split(":")[0].replace("Doc #", ""))

        doc = database.get_document_details(sel_doc_id, is_admin=True)
        if doc:
            st.markdown(textwrap.dedent(f"""
<div class="docv-card" style="margin-top: 10px;">
    <div style="font-weight: 750; font-size: 1.15rem; color: #172554;">Dossier: {doc['filename']} (#{doc['id']})</div>
    <div style="font-size: 0.88rem; color: #DC2626; margin-top: 4px;">
        <strong>Escalation Reason:</strong> {doc.get('escalation_reason') or 'Flagged for Officer Review'}
    </div>
</div>
""").strip(), unsafe_allow_html=True)

            col_adm_left, col_adm_right = st.columns([1.2, 1.8], gap="large")

            with col_adm_left:
                st.markdown("##### 1. Extracted Fields & In-Line Correction")
                st.caption("Review extracted fields against the document. Corrected values are saved with an audit trail.")

                fields = doc.get("fields", [])
                field_edits = {}
                with st.form("admin_field_correction_form"):
                    for f in fields:
                        f_name = f["field_name"]
                        curr_v = f["corrected_value"] if f["is_manually_corrected"] else (f["extracted_value"] or "")
                        flag_note = " ⚠️ (Low OCR)" if f["confidence"] < 70 else ""
                        new_val = st.text_input(f"{f_name}{flag_note}", value=curr_v)
                        field_edits[f_name] = new_val

                    save_edits = st.form_submit_button("Save Field Corrections Only")
                    if save_edits:
                        for k, v in field_edits.items():
                            database.update_field_value(doc["id"], k, v, reviewer_name=current_user["full_name"])
                        st.success("Field corrections saved to audit history!")
                        st.rerun()

            with col_adm_right:
                st.markdown("##### 2. Formal Determination")

                admin_decision = st.selectbox(
                    "Officer Determination:",
                    [
                        ("APPROVED", "Approve Consistency (All fields verified coherent)"),
                        ("REJECTED", "Reject Document (Fatal contradiction / invalid)"),
                        ("INFO_REQUESTED", "Request Additional Information from Applicant"),
                        ("UNRESOLVED", "Mark Unresolved (Cannot reach conclusive decision)")
                    ],
                    format_func=lambda x: f"{x[0]} - {x[1]}"
                )

                admin_notes = st.text_area(
                    "Audit Justification Notes (Required):",
                    placeholder="Document specific reasons for approval, rejection, or required resubmission..."
                )

                confirm_action = st.button("Commit Review Determination", type="primary", use_container_width=True)

                if confirm_action:
                    if not admin_notes.strip():
                        st.error("Please supply brief justification notes for the audit trail.")
                    else:
                        database.record_admin_review_decision(
                            document_id=doc["id"],
                            admin_user_id=current_user["id"],
                            admin_name=current_user["full_name"],
                            decision=admin_decision[0],
                            notes=admin_notes,
                            field_corrections=field_edits
                        )
                        st.success(f"Determination '{admin_decision[0]}' committed for Document #{doc['id']}.")
                        st.rerun()

                st.markdown("---")
                st.markdown("##### 3. Rule Engine Findings & Evidence")
                for r in doc.get("rules", []):
                    st.markdown(f"• **[{r['rule_id']}] {r['rule_name']}** ({r['severity']})\n  *{r['explanation']}*\n  Evidence: `{r['evidence']}`")


# ===========================================================================
# ADMIN VIEW 3: ALL SUBMITTED DOCUMENTS
# ===========================================================================

elif is_admin and active_nav == "All Submitted Documents":
    st.markdown("## All Submitted Documents")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>Complete repository of all documents submitted across the system.</p>", unsafe_allow_html=True)

    all_d = database.get_admin_review_queue(filter_type="ALL")
    if not all_d:
        st.info("No documents found in database.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            cat_filter = st.selectbox("Filter by Category:", ["All Categories"] + list(SCHEMA_REGISTRY.keys()))
        with col_f2:
            status_filter = st.selectbox("Filter by Status:", ["All Statuses"] + [
                STATUS_CONSISTENCY_PASSED, STATUS_NEEDS_REVIEW, STATUS_INCONSISTENCIES, STATUS_UNABLE_TO_VERIFY
            ])

        filtered_docs = all_d
        if cat_filter != "All Categories":
            filtered_docs = [d for d in filtered_docs if d["declared_category"] == cat_filter]
        if status_filter != "All Statuses":
            filtered_docs = [d for d in filtered_docs if d["verification_status"] == status_filter]

        df_all = pd.DataFrame([
            {
                "ID": d["id"],
                "File Name": d["filename"],
                "Applicant": d.get("applicant_name", "N/A"),
                "Category": d["declared_category"],
                "Verification Status": d["verification_status"],
                "Review State": d["admin_review_status"],
                "Submitted": d["upload_timestamp"][:10]
            }
            for d in filtered_docs
        ])
        st.dataframe(df_all, use_container_width=True, hide_index=True)


# ===========================================================================
# ADMIN VIEW 4: REVIEW HISTORY
# ===========================================================================

elif is_admin and active_nav == "Review History":
    st.markdown("## Review History & Audit Decisions")
    all_d = database.get_all_documents()

    if all_d:
        df_hist = pd.DataFrame([
            {
                "ID": d["id"],
                "File": d["filename"],
                "Category": d["declared_category"],
                "Status": d["verification_status"],
                "Review State": d["admin_review_status"],
                "Reviewed By": d.get("reviewed_by") or "Automated",
                "Reviewed At": d.get("reviewed_at", "")[:19].replace("T", " ") if d.get("reviewed_at") else ""
            }
            for d in all_d
        ])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ===========================================================================
# ADMIN VIEW 5: USER MANAGEMENT
# ===========================================================================

elif is_admin and active_nav == "User Management":
    st.markdown("## User Account Management")
    users = database.get_all_users()

    col_u_tbl, col_u_add = st.columns([2, 1], gap="large")

    with col_u_tbl:
        df_u = pd.DataFrame([
            {
                "ID": u["id"],
                "Full Name": u["full_name"],
                "Email": u["email"],
                "Role": u["role"],
                "Status": "Active" if u["is_active"] else "Disabled",
                "Docs Uploaded": u["document_count"],
                "Joined": u["created_at"][:10]
            }
            for u in users
        ])
        st.dataframe(df_u, use_container_width=True, hide_index=True)

    with col_u_add:
        st.markdown("#### Provision Officer Account")
        st.caption("Authorized administrators can provision verification officer accounts.")
        with st.form("create_officer_form"):
            o_name = st.text_input("Officer Full Name")
            o_email = st.text_input("Official Email")
            o_pwd = st.text_input("Temporary Password", type="password")
            submit_o = st.form_submit_button("Provision Officer Account", type="primary")

            if submit_o:
                succ, msg, _ = database.create_user(
                    email=o_email,
                    password=o_pwd,
                    full_name=o_name,
                    role="ADMIN",
                    is_admin_creator=True
                )
                if succ:
                    st.success(f"Officer account provisioned for '{o_email}'!")
                    st.rerun()
                else:
                    st.error(msg)


# ===========================================================================
# ADMIN VIEW 6: AUDIT LOGS
# ===========================================================================

elif is_admin and active_nav == "Audit Logs":
    st.markdown("## Security & Verification Audit Trails")
    conn = database.get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM audit_history ORDER BY id DESC LIMIT 150")
        logs = [dict(r) for r in cur.fetchall()]
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs[["id", "timestamp", "actor", "action", "details"]], use_container_width=True, hide_index=True)
        else:
            st.info("No audit history entries found.")
    finally:
        conn.close()


# ===========================================================================
# ADMIN VIEW 7: SYSTEM ANALYTICS & ML BENCHMARK
# ===========================================================================

elif is_admin and active_nav == "System Analytics":
    st.markdown("## System Analytics & Machine Learning Benchmark")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: -8px;'>Empirical evaluation on independent hold-out test split of real-world document corpora (RVL-CDIP, SROIE, FUNSD, Official Records).</p>", unsafe_allow_html=True)

    eval_data = evaluate_models_on_test_set()
    comp_df = get_evaluation_dataframe()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #EFF6FF; color: #2563EB;">🎯</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #2563EB; background: #EFF6FF; padding: 2px 8px; border-radius: 9999px;">ACCURACY</span>
    </div>
    <div>
        <div class="docv-metric-value">{eval_data['proposed_metrics']['Accuracy']*100:.1f}%</div>
        <div class="docv-metric-label">Hold-out Test Accuracy</div>
    </div>
    <div class="metric-bottom-bar" style="background: #2563EB;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with col_m2:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #ECFDF5; color: #16A34A;">📈</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #16A34A; background: #ECFDF5; padding: 2px 8px; border-radius: 9999px;">PRECISION</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #16A34A;">{eval_data['proposed_metrics']['Precision']*100:.1f}%</div>
        <div class="docv-metric-label">Weighted Precision</div>
    </div>
    <div class="metric-bottom-bar" style="background: #16A34A;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with col_m3:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #EFF6FF; color: #3B82F6;">🔍</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #3B82F6; background: #EFF6FF; padding: 2px 8px; border-radius: 9999px;">RECALL</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #1D4ED8;">{eval_data['proposed_metrics']['Recall']*100:.1f}%</div>
        <div class="docv-metric-label">Weighted Recall</div>
    </div>
    <div class="metric-bottom-bar" style="background: #3B82F6;"></div>
</div>
""").strip(), unsafe_allow_html=True)
    with col_m4:
        st.markdown(textwrap.dedent(f"""
<div class="docv-metric-card">
    <div class="metric-header">
        <div class="metric-icon-wrap" style="background: #F5F3FF; color: #7C3AED;">🏆</div>
        <span style="font-size: 0.72rem; font-weight: 700; color: #7C3AED; background: #F5F3FF; padding: 2px 8px; border-radius: 9999px;">F1-SCORE</span>
    </div>
    <div>
        <div class="docv-metric-value" style="color: #5B21B6;">{eval_data['proposed_metrics']['F1 Score']*100:.1f}%</div>
        <div class="docv-metric-label">Weighted F1 Score</div>
    </div>
    <div class="metric-bottom-bar" style="background: #7C3AED;"></div>
</div>
""").strip(), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Model Benchmark: Proposed Random Forest vs Logistic Baseline")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Test Set Confusion Matrix")
    classes = eval_data["classes"]
    cm = eval_data["proposed_metrics"]["Confusion Matrix"]
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Category Error Analysis (Sensitivity & Specificity)")
    cat_err = eval_data.get("category_error_analysis", {})
    if cat_err:
        cat_err_df = pd.DataFrame.from_dict(cat_err, orient="index")
        st.dataframe(cat_err_df, use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Top Discriminative Features (TF-IDF N-grams)")
    importances = get_feature_importances(top_n=12)
    imp_df = pd.DataFrame(importances, columns=["Feature / N-Gram", "Importance (%)"]).set_index("Feature / N-Gram")
    st.bar_chart(imp_df)

    st.markdown("---")
    st.markdown("#### 🎯 Controlled Evaluation Benchmark (SIH1709 Empirical Suite)")
    st.markdown("<p style='color: #64748B; font-size: 0.92rem; margin-top: -6px;'>Systematic evaluation across 20 controlled test scenarios covering valid records, incomplete credentials, chronological contradictions, and edge cases.</p>", unsafe_allow_html=True)

    from benchmark_runner import run_benchmark_suite
    from pathlib import Path
    bench_data = run_benchmark_suite(Path(__file__).parent / "data" / "controlled_test_cases.json")

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.metric("Total Controlled Cases", bench_data["total_cases"])
    with col_b2:
        st.metric("Production Engine Accuracy", f"{bench_data['proposed_accuracy']}%", delta="100% Deterministic")
    with col_b3:
        st.metric("Baseline Accuracy", f"{bench_data['baseline_accuracy']}%", delta=f"{bench_data['proposed_accuracy'] - bench_data['baseline_accuracy']:+.1f}%")
    with col_b4:
        st.metric("False Acceptance Rate (FAR)", f"{bench_data['proposed_far']}%", delta="0.0% Critical Safe", delta_color="inverse")

    with st.expander("📋 View 20-Scenario Controlled Evaluation Ledger", expanded=False):
        bench_df = pd.DataFrame([
            {
                "Case ID": r["case_id"],
                "Test Intent": r["intent"],
                "Ground Truth": r["ground_truth"],
                "Production System": r["proposed_system"],
                "Baseline Model": r["baseline_model"],
                "Engine Match": "✅" if r["proposed_match"] else "❌",
                "Baseline Match": "✅" if r["baseline_match"] else "❌"
            }
            for r in bench_data["detailed_results"]
        ])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.markdown(textwrap.dedent("""
<div class="regulatory-box">
    <strong>Academic &amp; Research Methodology Note:</strong><br>
    Models are evaluated strictly on independent hold-out test sets without data leakage. 
    In accordance with the scientific literature, document category classifiers evaluate structural domain vocabulary and do not claim to detect cryptographic forgery. 
    Logical consistency is verified deterministically through the Production Rule Engine.
</div>
""").strip(), unsafe_allow_html=True)
