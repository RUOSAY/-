import streamlit as st
import jieba
import jieba.posseg as pseg
import jieba.analyse
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ---------------------- 日志配置 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app_runtime.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import os
import imageio.v2 as imageio
import wordcloud
import platform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
from matplotlib.patches import Patch
import io
import streamlit.components.v1 as components
import docx
import math
import json
import hashlib
import random
from snownlp import SnowNLP

# ---------------------- 页面配置 ----------------------
st.set_page_config(page_title="中文评论文本综合分析工具", layout="wide", page_icon="📊")

# ---------------------- 统一参数配置区 ----------------------
# 1. 文件路径配置
STOPWORDS_FILE = "000停用词库.txt"
SENTIMENT_DICT_FILE = "001情感词典.txt"
POS_WORDS_FILE = "001正面情绪词.txt"
NEG_WORDS_FILE = "001负面情绪词.txt"
NEGATION_WORDS_FILE = "002否定词.txt"
DEGREE_WORDS_FILE = "003程度副词.txt"

# 跨系统中文字体搜索路径
SYSTEM_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simhei.ttf",
    "msyh.ttc", "simsun.ttc", "simhei.ttf",
    "/Library/Fonts/SimSun.ttc", "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
]

# ---------------------- 工具函数 ----------------------
def get_available_font():
    """检测系统可用中文字体，优先返回物理路径以支持 wordcloud"""
    for font_path in SYSTEM_FONT_PATHS:
        if os.path.exists(font_path):
            return font_path
    try:
        from matplotlib import font_manager
        target_keys = ['YaHei', 'SimSun', 'SimHei', 'PingFang', 'Heiti', 'Micro Hei', 'STHeiti']
        for f in font_manager.fontManager.ttflist:
            if any(key.lower() in f.name.lower() for key in target_keys):
                if f.fname and os.path.exists(f.fname):
                    return f.fname
    except Exception as e:
        logger.warning(f"获取系统可用中文字体时发生异常: {e}")
    return None

def setup_matplotlib_font():
    """全局配置 matplotlib 字体"""
    font_path = get_available_font()
    if font_path and os.path.exists(font_path):
        from matplotlib import font_manager
        try:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [prop.get_name(), 'sans-serif', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logger.error(f"全局配置 matplotlib 字体失败: {e}", exc_info=True)
            st.toast(f"⚠️ 字体配置失败，图表可能出现乱码: {e}")
    else:
        logger.warning("未找到可用的中文字体，图表可能出现乱码")
        # st.toast("⚠️ 未找到合适的中文字体，图表可能出现乱码")
    return font_path

def load_data_from_file(uploaded_file):
    """从不同格式的文件中提取文本数据（优化内存使用）"""
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if uploaded_file.name.endswith('.txt'):
            # 对于大文本文件，逐行读取避免内存溢出
            if file_size_mb > 10:
                st.info(f"📄 检测到较大的文本文件（{file_size_mb:.1f} MB），正在逐行读取...")
            
            content = uploaded_file.read().decode("utf-8")
            lines = []
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
            return lines, None
            
        elif uploaded_file.name.endswith('.docx'):
            # Word文档通常不会太大，直接读取
            doc = docx.Document(uploaded_file)
            return [p.text.strip() for p in doc.paragraphs if p.text.strip()], None
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # 对于大Excel文件，使用分批读取或限制行数
            if file_size_mb > 20:
                st.info(f"📊 检测到较大的Excel文件（{file_size_mb:.1f} MB），正在优化读取...")
                # 对于大文件，先读取一部分查看结构
                preview_df = pd.read_excel(uploaded_file, nrows=1000)
                total_rows = preview_df.shape[0]
                
                # 询问用户是否读取全部数据
                if total_rows >= 1000:
                    col1, col2 = st.columns(2)
                    with col1:
                        read_all = st.checkbox("读取所有行（可能消耗大量内存）", value=False)
                    with col2:
                        if not read_all:
                            max_rows = st.number_input("读取行数", min_value=1000, max_value=total_rows, value=min(10000, total_rows), step=1000)
                    
                    if read_all:
                        df = pd.read_excel(uploaded_file)
                        st.warning(f"⚠️ 已加载全部 {len(df)} 行数据，内存使用可能较高")
                    else:
                        df = pd.read_excel(uploaded_file, nrows=max_rows)
                        st.info(f"已加载前 {len(df)} 行数据（共 {total_rows} 行）")
                else:
                    df = pd.read_excel(uploaded_file)
            else:
                # 小文件直接读取
                df = pd.read_excel(uploaded_file)
            return None, df
            
    except Exception as e:
        logger.error(f"解析上传文件失败 ({uploaded_file.name}): {e}", exc_info=True)
        st.error(f"❌ 读取文件 '{uploaded_file.name}' 时发生错误，请检查文件格式是否正确。错误详情: {e}")
    return [], None


def _normalize_comment(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(.)\1{3,}", lambda m: m.group(1) * 2, t)
    t = re.sub(r"([\u4e00-\u9fff]{1,4})\1{2,}", r"\1", t)
    return t.strip()


def _is_spam_comment(text: str, min_len: int, unique_ratio_threshold: float):
    if not text:
        return True, "空文本"
    t = re.sub(r"\s+", "", text)
    if len(t) < int(min_len):
        return True, "过短"
    if re.fullmatch(r"\d+", t):
        return True, "纯数字"
    if re.fullmatch(r"[\W_]+", t):
        return True, "无有效字符"
    defaults = {"默认好评", "系统默认好评", "好评", "五星好评", "好评！", "好评!", "好评好评", "不错", "很好", "满意"}
    if t in defaults and len(t) <= 6:
        return True, "默认/模板"
    if re.search(r"(.)\1{4,}", t):
        return True, "重复字符"
    if re.fullmatch(r"([\u4e00-\u9fff]{1,3})\1{3,}", t):
        return True, "重复短语"
    uniq_ratio = (len(set(t)) / max(1, len(t)))
    if len(t) >= 8 and uniq_ratio < float(unique_ratio_threshold):
        return True, "重复度高"
    return False, ""


def preprocess_comments(comments, enable_dedup: bool, enable_spam_filter: bool, min_len: int, unique_ratio_threshold: float):
    stats = {
        "raw": len(comments),
        "empty_dropped": 0,
        "spam_dropped": 0,
        "dedup_dropped": 0,
    }
    cleaned = []
    for c in comments:
        t = _normalize_comment(c)
        if not t:
            stats["empty_dropped"] += 1
            continue
        if enable_spam_filter:
            is_spam, _ = _is_spam_comment(t, min_len=min_len, unique_ratio_threshold=unique_ratio_threshold)
            if is_spam:
                stats["spam_dropped"] += 1
                continue
        cleaned.append(t)

    if enable_dedup:
        seen = set()
        deduped = []
        for t in cleaned:
            if t in seen:
                stats["dedup_dropped"] += 1
                continue
            seen.add(t)
            deduped.append(t)
        cleaned = deduped

    stats["kept"] = len(cleaned)
    return cleaned, stats


def _stable_bytes_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _build_config_snapshot():
    keys = [
        "excel_target_col",
        "pre_dedup",
        "pre_spam",
        "pre_min_len",
        "pre_unique_ratio",
        "min_word_len",
        "max_word_len",
        "keyword_algo",
        "top_n_words",
        "pos_keep",
        "custom_words_input",
        "wc_bg_color",
        "wc_max_words",
        "wc_scale",
        "wc_random_state",
        "len_bins_mode",
        "len_bins_count",
        "len_xmax",
        "len_xmin",
        "len_log_y",
        "len_bar_color",
        "len_bar_alpha",
        "len_bar_edge_width",
        "len_bar_edge_color",
        "len_show_grid",
        "len_grid_alpha",
        "len_fig_w",
        "len_fig_h",
        "len_dpi",
        "len_bg_color",
        "len_show_density",
        "len_density_color",
        "len_show_stats",
        "len_stats_pos",
        "net_nodes",
        "net_min_weight",
        "net_layout",
        "net_seed",
        "net_iterations",
        "net_k",
        "net_max_edges",
        "net_keep_isolates",
        "net_fig_w",
        "net_fig_h",
        "net_dpi",
        "net_bg",
        "net_node_color",
        "net_edge_color",
        "net_node_alpha",
        "net_edge_alpha",
        "net_base_node_size",
        "net_node_scale",
        "net_node_exp",
        "net_min_edge_width",
        "net_edge_scale",
        "net_edge_exp",
        "net_show_labels",
        "net_label_size",
        "net_label_color",
        "net_label_alpha",
        "net_label_min_freq",
        "net_label_top_n",
        "topic_engine",
        "topic_engines",
        "auto_lda",
        "lda_topics",
        "lda_topics_range",
        "lda_max_iter",
        "lda_max_df",
        "lda_n_features",
        "lda_random_state",
        "lda_name_words",
        "sentiment_engine",
        "sentiment_engines",
        "score_mode",
        "pos_threshold",
        "neut_upper",
        "use_sent_wordlists",
        "pos_default_weight",
        "neg_default_weight",
        "enable_sent_cap",
        "sent_score_upper",
        "trend_order_mode",
        "trend_id_col",
        "trend_window",
        "active_view",
    ]
    snapshot = {k: st.session_state.get(k) for k in keys if k in st.session_state}
    snapshot["_schema"] = "comment-analyzer-config/v1"
    return snapshot


def _apply_config_snapshot(cfg: dict):
    if not isinstance(cfg, dict):
        return
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        st.session_state[k] = v


def _sanitize_session_state():
    def clamp_int(v, lo, hi, default):
        try:
            v = int(v)
        except Exception as e:
            logger.warning(f"配置解析失败: 无法将 '{v}' 转换为整数, 使用默认值 {default}。原因: {e}")
            return int(default)
        return int(min(hi, max(lo, v)))

    def clamp_float(v, lo, hi, default):
        try:
            v = float(v)
        except Exception as e:
            logger.warning(f"配置解析失败: 无法将 '{v}' 转换为浮点数, 使用默认值 {default}。原因: {e}")
            return float(default)
        return float(min(hi, max(lo, v)))

    def ensure_in(key, allowed, default):
        v = st.session_state.get(key)
        if v not in allowed:
            st.session_state[key] = default

    ensure_in("keyword_algo", ["词频", "TF-IDF", "TextRank"], "词频")
    ensure_in("wc_bg_color", ["white", "black", "gray"], "white")
    ensure_in("net_layout", ["Spring", "Kamada-Kawai", "Circular", "Shell"], "Spring")
    ensure_in("score_mode", ["原始", "按词数归一化", "按sqrt归一化", "按log归一化"], "按词数归一化")
    ensure_in("trend_order_mode", ["第一条=最远(正序)", "第一条=最近(倒序)", "按评论编号"], "第一条=最远(正序)")
    ensure_in("len_bins_mode", ["自动", "固定数量"], "自动")
    ensure_in("topic_engine", ["LDA", "BERTopic(高级)"], "LDA")

    if "pre_dedup" not in st.session_state:
        st.session_state["pre_dedup"] = True
    if "pre_spam" not in st.session_state:
        st.session_state["pre_spam"] = True

    if "sentiment_engines" not in st.session_state:
        old = st.session_state.get("sentiment_engine")
        st.session_state["sentiment_engines"] = [old] if isinstance(old, str) and old else ["词典+规则"]
    if not isinstance(st.session_state.get("sentiment_engines"), list):
        st.session_state["sentiment_engines"] = [str(st.session_state.get("sentiment_engines"))]
    allowed_sent = {"词典+规则", "SnowNLP (深度学习)"}
    st.session_state["sentiment_engines"] = [x for x in st.session_state["sentiment_engines"] if x in allowed_sent] or ["词典+规则"]

    if "topic_engines" not in st.session_state:
        old = st.session_state.get("topic_engine")
        st.session_state["topic_engines"] = [old] if isinstance(old, str) and old else ["LDA"]
    if not isinstance(st.session_state.get("topic_engines"), list):
        st.session_state["topic_engines"] = [str(st.session_state.get("topic_engines"))]
    allowed_topic = {"LDA", "BERTopic(高级)"}
    st.session_state["topic_engines"] = [x for x in st.session_state["topic_engines"] if x in allowed_topic] or ["LDA"]

    st.session_state["min_word_len"] = clamp_int(st.session_state.get("min_word_len"), 1, 5, 2)
    st.session_state["max_word_len"] = clamp_int(st.session_state.get("max_word_len"), 5, 20, 10)
    st.session_state["top_n_words"] = clamp_int(st.session_state.get("top_n_words"), 50, 500, 150)
    st.session_state["pre_min_len"] = clamp_int(st.session_state.get("pre_min_len"), 1, 20, 2)
    st.session_state["wc_max_words"] = clamp_int(st.session_state.get("wc_max_words"), 50, 1000, 200)
    st.session_state["wc_scale"] = clamp_int(st.session_state.get("wc_scale"), 1, 20, 10)
    st.session_state["wc_random_state"] = clamp_int(st.session_state.get("wc_random_state"), 0, 999999, 42)
    st.session_state["len_bins_count"] = clamp_int(st.session_state.get("len_bins_count"), 5, 200, 40)
    st.session_state["len_xmax"] = clamp_int(st.session_state.get("len_xmax"), 0, 1000000, 0)
    st.session_state["net_nodes"] = clamp_int(st.session_state.get("net_nodes"), 10, 100, 50)
    st.session_state["net_min_weight"] = clamp_int(st.session_state.get("net_min_weight"), 1, 100, 5)
    st.session_state["net_seed"] = clamp_int(st.session_state.get("net_seed"), 0, 9999, 42)
    st.session_state["net_iterations"] = clamp_int(st.session_state.get("net_iterations"), 10, 500, 200)
    st.session_state["net_max_edges"] = clamp_int(st.session_state.get("net_max_edges"), 0, 2000, 0)
    st.session_state["net_fig_w"] = clamp_int(st.session_state.get("net_fig_w"), 6, 30, 12)
    st.session_state["net_fig_h"] = clamp_int(st.session_state.get("net_fig_h"), 4, 20, 8)
    st.session_state["net_dpi"] = clamp_int(st.session_state.get("net_dpi"), 80, 300, 150)
    st.session_state["net_base_node_size"] = clamp_int(st.session_state.get("net_base_node_size"), 0, 1000, 200)
    st.session_state["net_node_scale"] = clamp_int(st.session_state.get("net_node_scale"), 1, 80, 10)
    st.session_state["net_label_size"] = clamp_int(st.session_state.get("net_label_size"), 5, 30, 10)
    st.session_state["net_label_min_freq"] = clamp_int(st.session_state.get("net_label_min_freq"), 1, 9999, 1)
    st.session_state["net_label_top_n"] = clamp_int(st.session_state.get("net_label_top_n"), 0, 100, 0)
    st.session_state["lda_topics"] = clamp_int(st.session_state.get("lda_topics"), 2, 15, 5)
    st.session_state["lda_max_iter"] = clamp_int(st.session_state.get("lda_max_iter"), 10, 500, 100)
    st.session_state["lda_n_features"] = clamp_int(st.session_state.get("lda_n_features"), 500, 10000, 2500)
    st.session_state["lda_random_state"] = clamp_int(st.session_state.get("lda_random_state"), 0, 999999, 42)
    st.session_state["lda_name_words"] = clamp_int(st.session_state.get("lda_name_words"), 1, 5, 2)
    st.session_state["pos_threshold"] = clamp_float(st.session_state.get("pos_threshold"), 0.0, 50.0, 10.0)
    st.session_state["neut_upper"] = clamp_float(st.session_state.get("neut_upper"), 0.0, 20.0, 5.0)
    st.session_state["pos_default_weight"] = clamp_float(st.session_state.get("pos_default_weight"), 0.0, 100.0, 1.0)
    st.session_state["neg_default_weight"] = clamp_float(st.session_state.get("neg_default_weight"), -100.0, 0.0, -1.0)
    st.session_state["sent_score_upper"] = clamp_float(st.session_state.get("sent_score_upper"), 0.0, 100000.0, 100.0)
    st.session_state["pre_unique_ratio"] = clamp_float(st.session_state.get("pre_unique_ratio"), 0.05, 0.5, 0.2)
    st.session_state["net_k"] = clamp_float(st.session_state.get("net_k"), 0.1, 2.0, 0.5)
    st.session_state["net_node_alpha"] = clamp_float(st.session_state.get("net_node_alpha"), 0.05, 1.0, 0.6)
    st.session_state["net_edge_alpha"] = clamp_float(st.session_state.get("net_edge_alpha"), 0.01, 1.0, 0.2)
    st.session_state["net_node_exp"] = clamp_float(st.session_state.get("net_node_exp"), 0.2, 2.0, 1.0)
    st.session_state["net_min_edge_width"] = clamp_float(st.session_state.get("net_min_edge_width"), 0.1, 5.0, 0.5)
    st.session_state["net_edge_scale"] = clamp_float(st.session_state.get("net_edge_scale"), 0.05, 5.0, 0.2)
    st.session_state["net_edge_exp"] = clamp_float(st.session_state.get("net_edge_exp"), 0.2, 2.0, 1.0)
    st.session_state["net_label_alpha"] = clamp_float(st.session_state.get("net_label_alpha"), 0.05, 1.0, 1.0)
    st.session_state["lda_max_df"] = clamp_float(st.session_state.get("lda_max_df"), 0.5, 1.0, 0.9)
    
    st.session_state["len_fig_w"] = clamp_int(st.session_state.get("len_fig_w"), 8, 24, 12)
    st.session_state["len_fig_h"] = clamp_int(st.session_state.get("len_fig_h"), 3, 12, 4)
    st.session_state["len_dpi"] = clamp_int(st.session_state.get("len_dpi"), 80, 300, 150)
    st.session_state["len_bar_alpha"] = clamp_float(st.session_state.get("len_bar_alpha"), 0.1, 1.0, 0.8)
    st.session_state["len_bar_edge_width"] = clamp_float(st.session_state.get("len_bar_edge_width"), 0.0, 3.0, 0.0)
    st.session_state["len_grid_alpha"] = clamp_float(st.session_state.get("len_grid_alpha"), 0.05, 1.0, 0.3)
    st.session_state["len_density_color"] = st.session_state.get("len_density_color", "#FF6B6B")
    st.session_state["trend_window"] = clamp_int(st.session_state.get("trend_window"), 3, 300, 30)

    lda_range = st.session_state.get("lda_topics_range")
    if isinstance(lda_range, (list, tuple)) and len(lda_range) == 2:
        a = clamp_int(lda_range[0], 2, 15, 2)
        b = clamp_int(lda_range[1], 2, 15, 8)
        if a > b:
            a, b = b, a
        st.session_state["lda_topics_range"] = (a, b)
    else:
        st.session_state["lda_topics_range"] = (2, 8)

    pos_keep = st.session_state.get("pos_keep")
    if not isinstance(pos_keep, list):
        st.session_state["pos_keep"] = ["n", "v"]
    else:
        allowed_pos = {"n", "v", "a", "d", "r"}
        st.session_state["pos_keep"] = [p for p in pos_keep if p in allowed_pos] or ["n", "v"]

@st.cache_data(show_spinner=False)
def load_stopwords_cached(file_path):
    stopwords = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word: stopwords.add(word)
    return stopwords

@st.cache_data(show_spinner=False)
def load_sentiment_dict_cached(file_path):
    sen_dict = defaultdict(float)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        sen_dict[parts[0]] = float(parts[1])
                    except ValueError:
                        logger.warning(f"加载情感词典失败 ({file_path} 行 {line_idx}): 无法将 '{parts[1]}' 转换为浮点数")
    else:
        logger.warning(f"情感词典文件未找到: {file_path}")
    return sen_dict

@st.cache_data(show_spinner=False)
def load_negation_words_cached(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"否定词词典文件未找到: {file_path}")
    return []

@st.cache_data(show_spinner=False)
def load_degree_words_cached(file_path):
    degree_dict = defaultdict(float)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        degree_dict[parts[0]] = float(parts[1])
                    except ValueError:
                        logger.warning(f"加载程度副词词典失败 ({file_path} 行 {line_idx}): 无法将 '{parts[1]}' 转换为浮点数")
    else:
        logger.warning(f"程度副词词典文件未找到: {file_path}")
    return degree_dict


@st.cache_data(show_spinner=False)
def load_wordlist_cached(file_path):
    words = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip()
                if w:
                    words.append(w)
    return words

# ---------------------- 核心分析逻辑 ----------------------
@st.cache_data(show_spinner=False)
def perform_segmentation(comments, stopwords, custom_words, pos_filter, min_word_len, max_word_len):
    """带缓存的并行分词函数"""
    # 小规模数据直接使用单线程，避免进程开销
    if len(comments) <= 1000:
        segmented_list = []
        for comment in comments:
            segmented_list.append(single_line_segment(comment, stopwords, custom_words=custom_words, 
                                                     pos_types=pos_filter, min_len=min_word_len, 
                                                     max_len=max_word_len))
        return segmented_list
    
    # 大规模数据使用进程池
    # Windows下需要在main函数中运行，这里做兼容性处理
    try:
        # 将数据分批，避免进程间传递大数据
        batch_size = min(5000, len(comments) // max(1, multiprocessing.cpu_count()))
        batches = [comments[i:i+batch_size] for i in range(0, len(comments), batch_size)]
        
        # 准备固定的参数
        process_func = partial(process_segmentation_batch, stopwords=stopwords, custom_words=custom_words,
                              pos_types=pos_filter, min_len=min_word_len, max_len=max_word_len)
        
        if multiprocessing.get_start_method() == 'spawn':
            # Windows环境下，使用spawn方式，需要特殊处理
            with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() // 2)) as executor:
                results = list(executor.map(process_func, batches))
        else:
            # Unix环境下，使用fork方式，更高效
            with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count())) as executor:
                results = list(executor.map(process_func, batches))
        
        # 合并结果
        segmented_list = []
        for batch_result in results:
            segmented_list.extend(batch_result)
        
        return segmented_list
    except Exception as e:
        logger.warning(f"进程池分词失败，回退到单线程: {e}")
        # 回退到单线程
        segmented_list = []
        for comment in comments:
            segmented_list.append(single_line_segment(comment, stopwords, custom_words=custom_words, 
                                                     pos_types=pos_filter, min_len=min_word_len, 
                                                     max_len=max_word_len))
        return segmented_list

@st.cache_data(show_spinner=False)
def extract_keywords(comments, algo, top_n, stopwords_list):
    """带缓存的关键词提取函数"""
    full_text = "\n".join(comments)
    if os.path.exists(STOPWORDS_FILE):
        jieba.analyse.set_stop_words(STOPWORDS_FILE)
    
    if algo == "TF-IDF":
        tags = jieba.analyse.extract_tags(full_text, topK=top_n, withWeight=True)
        if not tags:
            return pd.DataFrame(columns=["词语", "重要度 (TF-IDF)"])
        return pd.DataFrame(tags, columns=["词语", "重要度 (TF-IDF)"])
    elif algo == "TextRank":
        tags = jieba.analyse.textrank(full_text, topK=top_n, withWeight=True)
        if not tags:
            return pd.DataFrame(columns=["词语", "重要度 (TextRank)"])
        return pd.DataFrame(tags, columns=["词语", "重要度 (TextRank)"])
    else:
        words = []
        stop_set = set(stopwords_list)
        for line in comments:
            words.extend([w for w in jieba.cut(line) if w not in stop_set and len(w) >= 2])
        word_counts = Counter(words)
        top_words = word_counts.most_common(top_n)
        df = pd.DataFrame(top_words, columns=["词语", "出现次数"])
        total = float(df["出现次数"].sum()) if len(df) else 0.0
        df["频率占比"] = (df["出现次数"] / total).fillna(0.0) if total > 0 else 0.0
        df["频率占比"] = df["频率占比"].round(6)
        return df[["词语", "出现次数", "频率占比"]]

@st.cache_data(show_spinner=False)
def find_best_lda_k(segmented_list, k_min, k_max, max_df, n_features, random_state):
    """自动寻找最佳主题数（基于困惑度）"""
    vectorizer = CountVectorizer(max_features=n_features, max_df=max_df, min_df=1, token_pattern=r"(?u)\b\w+\b")
    texts_for_lda = [" ".join(s) for s in segmented_list]
    tf = vectorizer.fit_transform(texts_for_lda)
    best_perplexity = float('inf')
    best_k = k_min
    for k in range(k_min, k_max + 1):
        lda = LatentDirichletAllocation(n_components=k, max_iter=10, random_state=random_state)
        lda.fit(tf)
        perp = lda.perplexity(tf)
        if perp < best_perplexity:
            best_perplexity = perp
            best_k = k
    return best_k

@st.cache_data(show_spinner=False)
def run_lda_analysis(segmented_list, lda_topics, lda_max_iter, lda_max_df, lda_n_features, lda_random_state):
    """带缓存的 LDA 主题建模函数"""
    vectorizer = CountVectorizer(max_features=lda_n_features, max_df=lda_max_df, min_df=1, token_pattern=r"(?u)\b\w+\b")
    texts_for_lda = [" ".join(s) for s in segmented_list]
    tf = vectorizer.fit_transform(texts_for_lda)
    lda = LatentDirichletAllocation(n_components=lda_topics, max_iter=lda_max_iter, random_state=lda_random_state)
    doc_topic_dist = lda.fit_transform(tf)
    return lda, doc_topic_dist, tf, vectorizer

@st.cache_data(show_spinner=False)
def run_vector_topic_model(comments, n_topics, max_features, random_state):
    texts = [str(c).strip() for c in comments]
    vectorizer = TfidfVectorizer(
        max_features=int(max_features),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    if X.shape[0] <= 1 or X.shape[1] <= 1:
        topic_ids = [0 for _ in texts]
        topic_words = {0: []}
        info = pd.DataFrame([{"Topic": 0, "Count": len(texts)}])
        return topic_ids, topic_words, info

    n_topics = int(max(2, min(int(n_topics), X.shape[0])))
    n_components = int(min(100, max(2, X.shape[1] - 1)))
    X_red = TruncatedSVD(n_components=n_components, random_state=int(random_state)).fit_transform(X)
    labels = KMeans(n_clusters=n_topics, random_state=int(random_state), n_init=10).fit_predict(X_red)

    topic_words = {}
    rows = []
    for tid in range(n_topics):
        idx = np.where(labels == tid)[0]
        rows.append({"Topic": int(tid), "Count": int(len(idx))})
        if len(idx) <= 0:
            topic_words[int(tid)] = []
            continue
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:30]
        topic_words[int(tid)] = [(feature_names[i], float(mean_tfidf[i])) for i in top_idx if mean_tfidf[i] > 0]

    info = pd.DataFrame(rows).sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)
    return [int(x) for x in labels], topic_words, info

@st.cache_data(show_spinner=False)
def perform_sentiment_analysis(comments, _stopwords, _sen_dict, _not_words, _degree_dict, pos_threshold, neut_upper, score_mode, engine="词典+规则"):
    """带缓存且并行的情感倾向分析函数"""
    # 小规模数据直接使用单线程
    if len(comments) <= 1000:
        sentiments = []
        for comment in comments:
            sentiments.append(analyze_sentiment_logic(comment, _stopwords, _sen_dict, _not_words, _degree_dict, 
                                                     pos_threshold=pos_threshold, neut_upper=neut_upper, 
                                                     score_mode=score_mode, engine=engine))
        return sentiments
    
    # 大规模数据使用进程池
    try:
        batch_size = min(5000, len(comments) // max(1, multiprocessing.cpu_count()))
        batches = [comments[i:i+batch_size] for i in range(0, len(comments), batch_size)]
        
        process_func = partial(process_sentiment_batch, _stopwords=_stopwords, _sen_dict=_sen_dict,
                              _not_words=_not_words, _degree_dict=_degree_dict, pos_threshold=pos_threshold,
                              neut_upper=neut_upper, score_mode=score_mode, engine=engine)
        
        if multiprocessing.get_start_method() == 'spawn':
            with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() // 2)) as executor:
                results = list(executor.map(process_func, batches))
        else:
            with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count())) as executor:
                results = list(executor.map(process_func, batches))
        
        sentiments = []
        for batch_result in results:
            sentiments.extend(batch_result)
        
        return sentiments
    except Exception as e:
        logger.warning(f"进程池情感分析失败，回退到单线程: {e}")
        sentiments = []
        for comment in comments:
            sentiments.append(analyze_sentiment_logic(comment, _stopwords, _sen_dict, _not_words, _degree_dict, 
                                                     pos_threshold=pos_threshold, neut_upper=neut_upper, 
                                                     score_mode=score_mode, engine=engine))
        return sentiments

def process_sentiment_batch(batch_comments, _stopwords, _sen_dict, _not_words, _degree_dict, 
                           pos_threshold, neut_upper, score_mode, engine):
    """处理一批评论的情感分析"""
    results = []
    for comment in batch_comments:
        results.append(analyze_sentiment_logic(comment, _stopwords, _sen_dict, _not_words, _degree_dict,
                                              pos_threshold=pos_threshold, neut_upper=neut_upper,
                                              score_mode=score_mode, engine=engine))
    return results

def process_segmentation_batch(batch_comments, stopwords, custom_words, pos_types, min_len, max_len):
    """处理一批评论的分词"""
    results = []
    for comment in batch_comments:
        results.append(single_line_segment(comment, stopwords, custom_words, pos_types, min_len, max_len))
    return results

def single_line_segment(line, stopwords, custom_words=None, pos_types=('n', 'v'), min_len=2, max_len=10):
    word_pos_pairs = pseg.cut(line.strip())
    filtered_words = []
    custom_words_set = set(custom_words) if custom_words else set()
    for word, pos in word_pos_pairs:
        if word in custom_words_set:
            filtered_words.append(word)
        elif (pos.startswith(pos_types) and word not in stopwords and min_len <= len(word) <= max_len):
            filtered_words.append(word)
    return filtered_words

def analyze_sentiment_snownlp(text):
    text = text.strip()
    if not text:
        return "中性", 0.5, 0.5, 0, "无"
    try:
        s = SnowNLP(text)
        score = s.sentiments
        if score > 0.6:
            sentiment = "积极"
        elif score < 0.4:
            sentiment = "消极"
        else:
            sentiment = "中性"
        return sentiment, round(score, 4), round(score, 4), len(text), "无"
    except Exception as e:
        logger.warning(f"SnowNLP 分析失败: {e}")
        return "中性", 0.5, 0.5, len(text), "无"

def analyze_sentiment_logic(text, stopwords, sen_dict, not_words, degree_dict, pos_threshold=10.0, neut_upper=5.0, score_mode="原始", engine="词典+规则"):
    if engine == "SnowNLP (深度学习)":
        return analyze_sentiment_snownlp(text)

    seg_result = [word for word in jieba.cut(text.strip()) if word and word not in stopwords]
    token_count = len(seg_result)
    if not seg_result:
        return "中性", 0.0, 0.0, 0, "无"
    sen_word, not_word, degree_word = {}, {}, {}
    for i, word in enumerate(seg_result):
        if word in sen_dict and word not in not_words and word not in degree_dict:
            sen_word[i] = sen_dict[word]
        elif word in not_words: not_word[i] = -1
        elif word in degree_dict: degree_word[i] = degree_dict[word]
    if not sen_word:
        return "中性", 0.0, 0.0, token_count, "无"
    W, score = 1.0, 0.0
    sentiment_index_list = sorted(sen_word.keys())
    current_sentiment_idx = 0
    aspects = []

    for i in range(len(seg_result)):
        if i in sen_word:
            word_score = W * sen_word[i]
            score += word_score
            aspect_word = None
            for j in range(i - 1, max(-1, i - 4), -1):
                if len(seg_result[j]) >= 2 and seg_result[j] not in not_words and seg_result[j] not in degree_dict:
                     aspect_word = seg_result[j]
                     break
            if aspect_word:
                 aspects.append((aspect_word, word_score))

            W = 1.0
            current_sentiment_idx += 1
            if current_sentiment_idx < len(sentiment_index_list):
                next_pos = sentiment_index_list[current_sentiment_idx]
                for j in range(i + 1, next_pos):
                    if j in not_word: W *= -1
                    elif j in degree_word: W *= degree_word[j]
    raw_score = float(score)

    denom = 1.0
    if score_mode == "按词数归一化":
        denom = float(max(1, token_count))
    elif score_mode == "按sqrt归一化":
        denom = float(max(1.0, math.sqrt(max(1, token_count))))
    elif score_mode == "按log归一化":
        denom = float(max(1.0, math.log1p(max(1, token_count))))
    normalized_score = raw_score / denom

    sentiment = "积极" if normalized_score > pos_threshold else ("中性" if normalized_score > neut_upper else "消极")
    aspect_str = ", ".join([f"{a}({s:.1f})" for a, s in aspects]) if aspects else "无"

    return sentiment, round(normalized_score, 2), round(raw_score, 2), token_count, aspect_str

@st.cache_data(show_spinner=False)
def generate_topic_names(lda_components, feature_names, n_words):
    top_words_per_topic = []
    for topic in lda_components:
        top_indices = topic.argsort()[::-1][:n_words + 10]
        top_words_per_topic.append([feature_names[i] for i in top_indices])
    word_counts = Counter()
    for words in top_words_per_topic:
        word_counts.update(words)
    duplicates = {word for word, count in word_counts.items() if count > 1}
    final_names = {}
    for i, words in enumerate(top_words_per_topic):
        clean_words = [w for w in words if w not in duplicates]
        name = " | ".join(clean_words[:n_words]) if clean_words else f"主题 {i}"
        final_names[i] = name
    return final_names

@st.cache_data(show_spinner=False)
def build_semantic_network(segmented_list, top_n_nodes, min_edge_weight):
    all_words = [word for sublist in segmented_list for word in sublist]
    word_counts = Counter(all_words)
    top_words = [w for w, c in word_counts.most_common(top_n_nodes)]
    top_words_set = set(top_words)
    co_occurrence = defaultdict(int)
    for sentence in segmented_list:
        unique_words = sorted(list(set([w for w in sentence if w in top_words_set])))
        for i in range(len(unique_words)):
            for j in range(i + 1, len(unique_words)):
                co_occurrence[(unique_words[i], unique_words[j])] += 1
    G = nx.Graph()
    for (node1, node2), weight in co_occurrence.items():
        if weight >= min_edge_weight:
            G.add_edge(node1, node2, weight=weight)
    return G, word_counts


@st.cache_data(show_spinner=False)
def compute_network_layout(nodes, edges, layout_name, seed, k, iterations):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    if layout_name == "Circular":
        return nx.circular_layout(G)
    if layout_name == "Shell":
        return nx.shell_layout(G)
    if layout_name == "Kamada-Kawai":
        return nx.kamada_kawai_layout(G, weight="weight")
    return nx.spring_layout(G, k=k, seed=seed, iterations=iterations, weight="weight")


def render_semantic_network(semantic_net_res, my_font, net_max_edges, net_keep_isolates, net_layout, net_seed, net_k, net_iterations, net_fig_w, net_fig_h, net_dpi, net_bg, net_node_color, net_edge_color, net_node_alpha, net_edge_alpha, net_base_node_size, net_node_exp, net_node_scale, net_min_edge_width, net_edge_exp, net_edge_scale, net_show_labels, net_label_min_freq, net_label_top_n, net_label_size, net_label_color, net_label_alpha):
    G, counts = semantic_net_res
    if len(G.nodes) <= 0:
        st.info("语义网络无有效节点")
        return

    edges_all = [(u, v, float(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)]
    edges_all.sort(key=lambda x: x[2], reverse=True)
    edges = edges_all[:net_max_edges] if net_max_edges and net_max_edges > 0 else edges_all

    nodes = list({n for e in edges for n in e[:2]})
    if net_keep_isolates:
        nodes = list(G.nodes())

    pos = compute_network_layout(nodes, edges, net_layout, int(net_seed), float(net_k), int(net_iterations))

    fig, ax = plt.subplots(figsize=(net_fig_w, net_fig_h), dpi=net_dpi)
    fig.patch.set_facecolor(net_bg)
    ax.set_facecolor(net_bg)

    G_edges = nx.Graph()
    G_edges.add_nodes_from(nodes)
    for u, v, w in edges:
        G_edges.add_edge(u, v, weight=w)

    node_sizes = [net_base_node_size + (counts.get(n, 1) ** float(net_node_exp)) * net_node_scale for n in nodes]
    nx.draw_networkx_nodes(
        G_edges,
        pos,
        nodelist=nodes,
        node_size=node_sizes,
        node_color=net_node_color,
        alpha=float(net_node_alpha),
        ax=ax,
    )

    edge_widths = [max(float(net_min_edge_width), (float(d.get("weight", 1.0)) ** float(net_edge_exp)) * float(net_edge_scale)) for _, _, d in G_edges.edges(data=True)]
    nx.draw_networkx_edges(
        G_edges,
        pos,
        width=edge_widths,
        edge_color=net_edge_color,
        alpha=float(net_edge_alpha),
        ax=ax,
    )

    if net_show_labels:
        label_candidates = [n for n in nodes if counts.get(n, 0) >= int(net_label_min_freq)]
        label_candidates.sort(key=lambda n: counts.get(n, 0), reverse=True)
        if net_label_top_n and net_label_top_n > 0:
            label_candidates = label_candidates[:net_label_top_n]
        labels = {n: n for n in label_candidates}
        nx.draw_networkx_labels(
            G_edges,
            pos,
            labels=labels,
            font_size=int(net_label_size),
            font_color=net_label_color,
            alpha=float(net_label_alpha),
            font_family=my_font.get_name() if my_font else 'sans-serif',
            ax=ax,
        )

    plt.axis('off')
    st.pyplot(fig)

    # 下载按钮组
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # PNG下载
        net_png = io.BytesIO()
        fig.savefig(net_png, format="png", dpi=net_dpi, bbox_inches="tight", facecolor=net_bg)
        st.download_button("📥 下载PNG", net_png.getvalue(), "语义网络.png", mime="image/png")
    
    with col2:
        # SVG下载（矢量图）
        net_svg = io.BytesIO()
        fig.savefig(net_svg, format="svg", bbox_inches="tight", facecolor=net_bg)
        st.download_button("📥 下载SVG矢量图", net_svg.getvalue(), "语义网络.svg", mime="image/svg+xml")
    
    with col3:
        # PDF下载（矢量图）
        net_pdf = io.BytesIO()
        fig.savefig(net_pdf, format="pdf", bbox_inches="tight", facecolor=net_bg)
        st.download_button("📥 下载PDF矢量图", net_pdf.getvalue(), "语义网络.pdf", mime="application/pdf")

# ---------------------- Streamlit UI ----------------------
def main():
    st.title("📊 中文评论文本综合分析工具")

    # --- 1. 侧边栏配置 ---
    st.sidebar.header("⚙️ 参数配置")
    uploaded_file = st.sidebar.file_uploader("📤 上传评论文件", type=["txt", "docx", "xlsx", "xls"], key="comment_uploader")
    
    # 文件大小检查和限制
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        max_file_size_mb = 100  # 限制100MB
        
        if file_size_mb > max_file_size_mb:
            st.error(f"❌ 文件过大（{file_size_mb:.1f} MB）！请上传小于 {max_file_size_mb} MB 的文件。")
            st.info("对于大文件，建议使用文本格式（.txt）并按行分割数据")
            st.stop()
        
        # 大文件警告
        if file_size_mb > 20:
            st.warning(f"⚠️ 检测到较大文件（{file_size_mb:.1f} MB），处理可能需要较长时间。建议：\n"
                      "- 使用文本格式（.txt）而非 Excel\n"
                      "- 确保电脑有足够的内存（建议至少8GB）\n"
                      "- 关闭其他占用内存的程序")
    config_file = st.sidebar.file_uploader("🧩 上传配置文件", type=["json"], key="config_uploader")
    if not uploaded_file:
        st.sidebar.warning("请先上传文件")
        st.stop()

    if config_file is not None:
        cfg_bytes = config_file.getvalue()
        cfg_hash = _stable_bytes_hash(cfg_bytes)
        if st.session_state.get("_last_cfg_hash") != cfg_hash:
            try:
                cfg_obj = json.loads(cfg_bytes.decode("utf-8"))
                if isinstance(cfg_obj, dict) and cfg_obj.get("_schema") == "comment-analyzer-config/v1":
                    _apply_config_snapshot(cfg_obj)
                    _sanitize_session_state()
                    st.session_state["_last_cfg_hash"] = cfg_hash
                    logger.info("成功加载并应用了配置文件")
                    st.toast("✅ 配置文件加载成功")
                    st.rerun()
                else:
                    logger.warning("配置文件格式或 _schema 不正确")
                    st.sidebar.error("配置文件的格式不正确或版本不支持")
            except Exception as e:
                logger.error(f"配置文件解析失败: {e}", exc_info=True)
                st.sidebar.error(f"配置文件解析失败：{e}")
    
    comments_list, df_excel = load_data_from_file(uploaded_file)
    if df_excel is not None:
        st.sidebar.subheader("Excel 列选择")
        keywords = ['评论', '文本', '内容', 'text', 'comment', 'content']
        default_idx = 0
        for i, col in enumerate(df_excel.columns):
            if any(k in str(col).lower() for k in keywords):
                default_idx = i
                break
        current_target = st.session_state.get("excel_target_col")
        if current_target not in list(df_excel.columns):
            st.session_state["excel_target_col"] = df_excel.columns[default_idx]
        target_col = st.sidebar.selectbox("选择评论列", df_excel.columns, index=default_idx, key="excel_target_col")
        comments = df_excel[target_col].dropna().astype(str).str.strip().tolist()
    else:
        comments = comments_list

    _sanitize_session_state()

    raw_count = len(comments)
    if raw_count <= 0:
        st.error("未检测到有效数据")
        st.stop()

    with st.sidebar.expander("🧹 文本预处理", expanded=False):
        pre_dedup = st.checkbox("去重（完全相同文本）", value=bool(st.session_state.get("pre_dedup", True)), key="pre_dedup")
        pre_spam = st.checkbox("过滤垃圾/模板评论", value=bool(st.session_state.get("pre_spam", True)), key="pre_spam")
        pre_min_len = st.slider("最小长度", 1, 20, value=int(st.session_state.get("pre_min_len", 2)), key="pre_min_len")
        pre_unique_ratio = st.slider("重复度阈值（越大越严格）", 0.05, 0.5, value=float(st.session_state.get("pre_unique_ratio", 0.2)), key="pre_unique_ratio")

    comments, pre_stats = preprocess_comments(
        comments,
        enable_dedup=bool(pre_dedup),
        enable_spam_filter=bool(pre_spam),
        min_len=int(pre_min_len),
        unique_ratio_threshold=float(pre_unique_ratio),
    )

    # 在sidebar显示数据概览
    with st.sidebar:
        st.markdown("---")
        st.markdown("**📊 数据概览**")
        st.markdown(f"- 原始数据量: **{raw_count:,}** 条")
        st.markdown(f"- 清洗后数据量: **{len(comments):,}** 条")
        st.markdown(f"- 删除数据: **{raw_count - len(comments):,}** 条")
        if pre_stats:
            st.markdown(f"- 重复文本: **{pre_stats.get('duplicates_removed', 0):,}** 条")
            st.markdown(f"- 垃圾/模板: **{pre_stats.get('spam_removed', 0):,}** 条")
            st.markdown(f"- 过短文本: **{pre_stats.get('short_removed', 0):,}** 条")
    
    # 内存使用预估和分批处理建议
    data_size_mb = sum(len(c.encode('utf-8')) for c in comments) / (1024 * 1024)
    st.sidebar.caption(f"预估数据大小: {data_size_mb:.1f} MB")
    
    if len(comments) > 50000:
        st.warning(f"⚠️ 检测到大规模数据（{len(comments)} 条），处理时间可能较长。建议：\n"
                  "1. 确保至少有8GB可用内存\n"
                  "2. 关闭其他占用内存的程序\n"
                  "3. 考虑分批处理或使用采样")
    st.sidebar.caption(f"去空 {pre_stats['empty_dropped']} | 去重 {pre_stats['dedup_dropped']} | 过滤 {pre_stats['spam_dropped']}")
    st.sidebar.divider()

    if not comments:
        st.error("清洗后无可用数据，请调整预处理参数或检查输入文件")
        st.stop()

    # --- 2. 侧边栏配置与动态更新逻辑 ---
    with st.sidebar.expander("✂️ 分词与关键词配置", expanded=False):
        min_word_len = st.number_input("最小词长", 1, 5, value=int(st.session_state.get("min_word_len", 2)), key="min_word_len")
        max_word_len = st.number_input("最大词长", 5, 20, value=int(st.session_state.get("max_word_len", 10)), key="max_word_len")
        keyword_algo = st.selectbox(
            "关键词算法",
            ["词频", "TF-IDF", "TextRank"],
            index=["词频", "TF-IDF", "TextRank"].index(st.session_state.get("keyword_algo", "词频")) if st.session_state.get("keyword_algo", "词频") in ["词频", "TF-IDF", "TextRank"] else 0,
            key="keyword_algo",
        )
        top_n_words = st.slider("展示数量", 50, 500, value=int(st.session_state.get("top_n_words", 150)), key="top_n_words")
        selected_pos = st.multiselect(
            "保留词性",
            ["n", "v", "a", "d", "r"],
            default=st.session_state.get("pos_keep", ["n", "v"]),
            key="pos_keep",
        )
        pos_filter = tuple(selected_pos)
        custom_words_input = st.text_area("自定义保留词", value=st.session_state.get("custom_words_input", ""), key="custom_words_input")
        custom_words = [w.strip() for w in custom_words_input.split('\n') if w.strip()]

    with st.sidebar.expander("☁️ 词云样式配置", expanded=False):
        wc_bg_color = st.selectbox(
            "背景颜色",
            ["white", "black", "gray"],
            index=["white", "black", "gray"].index(st.session_state.get("wc_bg_color", "white")) if st.session_state.get("wc_bg_color", "white") in ["white", "black", "gray"] else 0,
            key="wc_bg_color",
        )
        wc_max_words = st.number_input("最大词数", 50, 1000, value=int(st.session_state.get("wc_max_words", 200)), key="wc_max_words")
        wc_scale = st.slider("精细度", 1, 20, value=int(st.session_state.get("wc_scale", 10)), key="wc_scale")
        wc_random_state = st.number_input("随机种子", 0, 999999, value=int(st.session_state.get("wc_random_state", 42)), key="wc_random_state")
        uploaded_mask = st.file_uploader("形状图上传", type=["png", "jpg", "jpeg"], key="wc_mask_uploader")

    with st.sidebar.expander("📊 文本内容长度分布", expanded=False):
        len_bins_mode = st.selectbox(
            "分箱方式",
            ["自动", "固定数量"],
            index=["自动", "固定数量"].index(st.session_state.get("len_bins_mode", "自动")) if st.session_state.get("len_bins_mode", "自动") in ["自动", "固定数量"] else 0,
            key="len_bins_mode",
        )
        len_bins_count = st.slider("分箱数量", 5, 200, value=int(st.session_state.get("len_bins_count", 40)), key="len_bins_count")
        len_xmax = st.number_input("横轴最大长度(0=不限制)", 0, 1000000, value=int(st.session_state.get("len_xmax", 0)), key="len_xmax")
        len_xmin = st.number_input("横轴最小长度(0=不限制)", 0, 1000000, value=int(st.session_state.get("len_xmin", 0)), key="len_xmin")
        len_log_y = st.checkbox("纵轴使用对数刻度", value=bool(st.session_state.get("len_log_y", False)), key="len_log_y")
        
        st.markdown("---")
        st.markdown("**📊 图表样式**")
        len_bar_color = st.color_picker("柱状图颜色", value=st.session_state.get("len_bar_color", "#1f77b4"), key="len_bar_color")
        len_bar_alpha = st.slider("柱状图透明度", 0.1, 1.0, value=float(st.session_state.get("len_bar_alpha", 0.8)), key="len_bar_alpha")
        len_bar_edge_width = st.slider("边框宽度", 0.0, 3.0, value=float(st.session_state.get("len_bar_edge_width", 0.0)), key="len_bar_edge_width")
        len_bar_edge_color = st.color_picker("边框颜色", value=st.session_state.get("len_bar_edge_color", "#FFFFFF"), key="len_bar_edge_color")
        len_show_grid = st.checkbox("显示网格线", value=bool(st.session_state.get("len_show_grid", True)), key="len_show_grid")
        len_grid_alpha = st.slider("网格线透明度", 0.05, 1.0, value=float(st.session_state.get("len_grid_alpha", 0.3)), key="len_grid_alpha")
        
        st.markdown("---")
        st.markdown("**📐 图形尺寸**")
        len_fig_w = st.slider("图表宽度", 8, 24, value=int(st.session_state.get("len_fig_w", 12)), key="len_fig_w")
        len_fig_h = st.slider("图表高度", 3, 12, value=int(st.session_state.get("len_fig_h", 4)), key="len_fig_h")
        len_dpi = st.slider("图表清晰度(DPI)", 80, 300, value=int(st.session_state.get("len_dpi", 150)), key="len_dpi")
        len_bg_color = st.color_picker("背景颜色", value=st.session_state.get("len_bg_color", "#FFFFFF"), key="len_bg_color")
        
        st.markdown("---")
        st.markdown("**📈 高级选项**")
        len_show_density = st.checkbox("显示密度曲线", value=bool(st.session_state.get("len_show_density", False)), key="len_show_density")
        len_density_color = st.color_picker("密度曲线颜色", value=st.session_state.get("len_density_color", "#FF6B6B"), key="len_density_color")
        len_show_stats = st.checkbox("显示统计信息", value=bool(st.session_state.get("len_show_stats", True)), key="len_show_stats")
        len_stats_pos = st.selectbox("统计信息位置", ["左上角", "右上角", "左下角", "右下角"], index=0, key="len_stats_pos")

    with st.sidebar.expander("🔗 语义网络配置", expanded=False):
        net_nodes = st.slider("节点数量", 10, 100, value=int(st.session_state.get("net_nodes", 50)), key="net_nodes")
        net_min_weight = st.number_input("最小共现次数", 1, 100, value=int(st.session_state.get("net_min_weight", 5)), key="net_min_weight")
        net_layout = st.selectbox(
            "布局算法",
            ["Spring", "Kamada-Kawai", "Circular", "Shell"],
            index=["Spring", "Kamada-Kawai", "Circular", "Shell"].index(st.session_state.get("net_layout", "Spring")) if st.session_state.get("net_layout", "Spring") in ["Spring", "Kamada-Kawai", "Circular", "Shell"] else 0,
            key="net_layout",
        )
        net_seed = st.number_input("布局随机种子", 0, 9999, value=int(st.session_state.get("net_seed", 42)), key="net_seed")
        net_iterations = st.slider("布局迭代次数", 10, 500, value=int(st.session_state.get("net_iterations", 200)), key="net_iterations")
        net_k = st.slider("布局疏散度", 0.1, 2.0, value=float(st.session_state.get("net_k", 0.5)), key="net_k")
        net_max_edges = st.slider("最大连线数", 0, 2000, value=int(st.session_state.get("net_max_edges", 0)), key="net_max_edges")
        net_keep_isolates = st.checkbox("保留孤立节点", value=bool(st.session_state.get("net_keep_isolates", False)), key="net_keep_isolates")

        net_fig_w = st.slider("画布宽度", 6, 30, value=int(st.session_state.get("net_fig_w", 12)), key="net_fig_w")
        net_fig_h = st.slider("画布高度", 4, 20, value=int(st.session_state.get("net_fig_h", 8)), key="net_fig_h")
        net_dpi = st.slider("导出清晰度(DPI)", 80, 300, value=int(st.session_state.get("net_dpi", 150)), key="net_dpi")
        net_bg = st.color_picker("背景颜色", value=st.session_state.get("net_bg", "#FFFFFF"), key="net_bg")

        net_node_color = st.color_picker("节点颜色", value=st.session_state.get("net_node_color", "#87CEEB"), key="net_node_color")
        net_edge_color = st.color_picker("连线颜色", value=st.session_state.get("net_edge_color", "#808080"), key="net_edge_color")
        net_node_alpha = st.slider("节点透明度", 0.05, 1.0, value=float(st.session_state.get("net_node_alpha", 0.6)), key="net_node_alpha")
        net_edge_alpha = st.slider("连线透明度", 0.01, 1.0, value=float(st.session_state.get("net_edge_alpha", 0.2)), key="net_edge_alpha")

        net_base_node_size = st.slider("节点基础大小", 0, 1000, value=int(st.session_state.get("net_base_node_size", 200)), key="net_base_node_size")
        net_node_scale = st.slider("节点缩放", 1, 80, value=int(st.session_state.get("net_node_scale", 10)), key="net_node_scale")
        net_node_exp = st.slider("节点权重指数", 0.2, 2.0, value=float(st.session_state.get("net_node_exp", 1.0)), key="net_node_exp")
        net_min_edge_width = st.slider("连线最小宽度", 0.1, 5.0, value=float(st.session_state.get("net_min_edge_width", 0.5)), key="net_min_edge_width")
        net_edge_scale = st.slider("连线缩放", 0.05, 5.0, value=float(st.session_state.get("net_edge_scale", 0.2)), key="net_edge_scale")
        net_edge_exp = st.slider("连线权重指数", 0.2, 2.0, value=float(st.session_state.get("net_edge_exp", 1.0)), key="net_edge_exp")

        net_show_labels = st.checkbox("显示标签", value=bool(st.session_state.get("net_show_labels", True)), key="net_show_labels")
        net_label_size = st.slider("字号", 5, 30, value=int(st.session_state.get("net_label_size", 10)), key="net_label_size")
        net_label_color = st.color_picker("标签颜色", value=st.session_state.get("net_label_color", "#111111"), key="net_label_color")
        net_label_alpha = st.slider("标签透明度", 0.05, 1.0, value=float(st.session_state.get("net_label_alpha", 1.0)), key="net_label_alpha")
        net_label_min_freq = st.number_input("标注最小词频", 1, 9999, value=int(st.session_state.get("net_label_min_freq", 1)), key="net_label_min_freq")
        net_label_top_n = st.slider("仅标注Top节点(0为全部)", 0, 100, value=int(st.session_state.get("net_label_top_n", 0)), key="net_label_top_n")

    with st.sidebar.expander("🧠 主题建模配置", expanded=False):
        topic_engines = st.multiselect(
            "主题建模算法",
            ["LDA", "BERTopic(高级)"],
            default=st.session_state.get("topic_engines", ["LDA"]),
            key="topic_engines",
        )
        if not topic_engines:
            topic_engines = ["LDA"]
            st.session_state["topic_engines"] = topic_engines

        auto_lda = st.checkbox("自动优化主题数", value=bool(st.session_state.get("auto_lda", False)), key="auto_lda")
        if auto_lda:
            lda_topics_range = st.slider("搜索范围", 2, 15, value=tuple(st.session_state.get("lda_topics_range", (2, 8))), key="lda_topics_range")
            lda_topics = int(st.session_state.get("lda_topics", 5))
        else:
            lda_topics = st.slider("主题个数", 2, 15, value=int(st.session_state.get("lda_topics", 5)), key="lda_topics")
            lda_topics_range = tuple(st.session_state.get("lda_topics_range", (2, 8)))
        lda_max_iter = st.number_input("迭代次数", 10, 500, value=int(st.session_state.get("lda_max_iter", 100)), key="lda_max_iter")
        lda_max_df = st.slider("Max DF", 0.5, 1.0, value=float(st.session_state.get("lda_max_df", 0.9)), key="lda_max_df")
        lda_n_features = st.number_input("特征上限", 500, 10000, value=int(st.session_state.get("lda_n_features", 2500)), key="lda_n_features")
        lda_random_state = st.number_input("种子", 0, 999999, value=int(st.session_state.get("lda_random_state", 42)), key="lda_random_state")
        lda_name_words = st.slider("命名词数", 1, 5, value=int(st.session_state.get("lda_name_words", 2)), key="lda_name_words")

    with st.sidebar.expander("❤️ 情感分析配置", expanded=False):
        sentiment_engines = st.multiselect(
            "分析引擎",
            ["词典+规则", "SnowNLP (深度学习)"],
            default=st.session_state.get("sentiment_engines", ["词典+规则"]),
            key="sentiment_engines",
        )
        if not sentiment_engines:
            sentiment_engines = ["词典+规则"]
            st.session_state["sentiment_engines"] = sentiment_engines

        # 初始化所有情感分析参数为默认值
        score_mode = st.selectbox(
            "词典得分模式",
            ["原始", "按词数归一化", "按sqrt归一化", "按log归一化"],
            index=["原始", "按词数归一化", "按sqrt归一化", "按log归一化"].index(st.session_state.get("score_mode", "按词数归一化")) if st.session_state.get("score_mode", "按词数归一化") in ["原始", "按词数归一化", "按sqrt归一化", "按log归一化"] else 1,
            key="score_mode",
        )
        pos_threshold = float(st.session_state.get("pos_threshold", 10.0))
        neut_upper = float(st.session_state.get("neut_upper", 5.0))
        use_sent_wordlists = bool(st.session_state.get("use_sent_wordlists", False))
        pos_default_weight = float(st.session_state.get("pos_default_weight", 1.0))
        neg_default_weight = float(st.session_state.get("neg_default_weight", -1.0))
        enable_sent_cap = bool(st.session_state.get("enable_sent_cap", False))
        sent_score_upper = float(st.session_state.get("sent_score_upper", 100.0))

        if "词典+规则" in sentiment_engines:
            st.markdown("---")
            st.markdown("**📚 词典配置**")
            use_sent_wordlists = st.checkbox("使用正负面词表补充情感词典", value=bool(st.session_state.get("use_sent_wordlists", False)), key="use_sent_wordlists")
            if use_sent_wordlists:
                pos_default_weight = st.number_input("正向默认权重", 0.0, 100.0, value=float(st.session_state.get("pos_default_weight", 1.0)), key="pos_default_weight")
                neg_default_weight = st.number_input("负向默认权重", -100.0, 0.0, value=float(st.session_state.get("neg_default_weight", -1.0)), key="neg_default_weight")
            
            st.markdown("---")
            st.markdown("**⚙️ 得分配置**")
            pos_threshold = st.number_input("积极阈值", 0.0, 50.0, value=float(st.session_state.get("pos_threshold", 10.0)), key="pos_threshold")
            neut_upper = st.number_input("中性上限", 0.0, 20.0, value=float(st.session_state.get("neut_upper", 5.0)), key="neut_upper")
            
            st.markdown("---")
            st.markdown("**🎯 高级筛选**")
            enable_sent_cap = st.checkbox("启用得分上限筛选", value=bool(st.session_state.get("enable_sent_cap", False)), key="enable_sent_cap")
            if enable_sent_cap:
                sent_score_upper = st.number_input("情感得分上限阈值(>=则归为超阈值)", 0.0, 100000.0, value=float(st.session_state.get("sent_score_upper", 100.0)), key="sent_score_upper")

        if "SnowNLP (深度学习)" in sentiment_engines:
            st.markdown("---")
            st.info("SnowNLP 得分范围为 0~1（积极概率）。>0.6 为积极，<0.4 为消极。")
        
        st.markdown("---")
        st.markdown("**📈 趋势分析**")
        trend_order_mode = st.selectbox(
            "趋势时间方向",
            ["第一条=最远(正序)", "第一条=最近(倒序)", "按评论编号"],
            index=["第一条=最远(正序)", "第一条=最近(倒序)", "按评论编号"].index(st.session_state.get("trend_order_mode", "第一条=最远(正序)")) if st.session_state.get("trend_order_mode", "第一条=最远(正序)") in ["第一条=最远(正序)", "第一条=最近(倒序)", "按评论编号"] else 0,
            key="trend_order_mode",
        )
        trend_id_col = None
        if trend_order_mode == "按评论编号" and df_excel is not None:
            id_keywords = ["编号", "序号", "id", "ID", "no", "No", "NO"]
            default_id_idx = 0
            for i, col in enumerate(df_excel.columns):
                if any(k.lower() in str(col).lower() for k in id_keywords):
                    default_id_idx = i
                    break
            trend_id_col = st.selectbox("选择评论编号列", df_excel.columns, index=default_id_idx, key="trend_id_col")

    # 自动分析检测：只要上传了文件就进行动态分析
    if not uploaded_file:
        st.info("请先上传文件")
        st.stop()

    # --- 3. 分析流程逻辑 (动态响应式) ---
    font_p = setup_matplotlib_font()
    my_font = None
    if font_p and os.path.exists(font_p):
        from matplotlib import font_manager
        my_font = font_manager.FontProperties(fname=font_p)
    
    stopwords = load_stopwords_cached(STOPWORDS_FILE)
    sen_dict = load_sentiment_dict_cached(SENTIMENT_DICT_FILE)
    not_words = load_negation_words_cached(NEGATION_WORDS_FILE)
    degree_dict = load_degree_words_cached(DEGREE_WORDS_FILE)

    random.seed(int(wc_random_state))
    np.random.seed(int(wc_random_state))

    effective_sen_dict = dict(sen_dict)
    pos_words = []
    neg_words = []
    if "词典+规则" in sentiment_engines and use_sent_wordlists:
        pos_words = load_wordlist_cached(POS_WORDS_FILE)
        neg_words = load_wordlist_cached(NEG_WORDS_FILE)
        pw = float(pos_default_weight)
        nw = float(neg_default_weight)
        for w in pos_words:
            if w not in effective_sen_dict:
                effective_sen_dict[w] = pw
        for w in neg_words:
            if w not in effective_sen_dict:
                effective_sen_dict[w] = nw
    
    # 应用自定义词典
    for word in custom_words:
        jieba.add_word(word)

    if "词典+规则" in sentiment_engines and use_sent_wordlists:
        wl_sig = (len(pos_words), len(neg_words))
        if st.session_state.get("jieba_sent_wordlists_sig") != wl_sig:
            for w in pos_words:
                jieba.add_word(w)
            for w in neg_words:
                jieba.add_word(w)
            st.session_state["jieba_sent_wordlists_sig"] = wl_sig

    # 1. 分词 (Cached)
    progress_holder = st.empty()
    progress_bar = progress_holder.progress(0, text="正在分析...")
    
    # 对大文件使用分批处理
    if len(comments) > 20000:
        st.info(f"📊 检测到大规模数据（{len(comments)} 条），正在分批处理以优化内存使用...")
        segmented_list = []
        batch_size = 5000
        total_batches = (len(comments) + batch_size - 1) // batch_size
        
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i+batch_size]
            batch_result = perform_segmentation(batch, stopwords, custom_words, pos_filter, min_word_len, max_word_len)
            segmented_list.extend(batch_result)
            
            # 更新进度
            batch_progress = int((i + len(batch)) / len(comments) * 20)
            progress_bar.progress(batch_progress, text=f"分词处理中: 第 {i//batch_size + 1}/{total_batches} 批")
    else:
        segmented_list = perform_segmentation(comments, stopwords, custom_words, pos_filter, min_word_len, max_word_len)
    
    progress_bar.progress(20, text="分词完成")
    
    # 2. 关键词提取 (Cached)
    df_freq = extract_keywords(comments, keyword_algo, top_n_words, list(stopwords))
    progress_bar.progress(35, text="关键词完成")
    
    # 3. 词云生成 (Cached by parameter change)
    @st.cache_data
    def generate_wc_img(seg_list, bg, max_w, scale, mask_bytes, font_path, random_state):
        mask = imageio.imread(io.BytesIO(mask_bytes)) if mask_bytes else None
        wc = wordcloud.WordCloud(
            width=1000,
            height=700,
            background_color=bg,
            font_path=font_path,
            mask=mask,
            scale=scale,
            max_words=max_w,
            random_state=int(random_state),
        )
        wc.generate(" ".join([" ".join(s) for s in seg_list]))
        return wc.to_image()
    
    mask_bytes = uploaded_mask.getvalue() if uploaded_mask else None
    wc_img = generate_wc_img(segmented_list, wc_bg_color, wc_max_words, wc_scale, mask_bytes, font_p, wc_random_state)
    progress_bar.progress(50, text="词云完成")

    # 4. 语义网络 (Cached)
    semantic_net_res = build_semantic_network(segmented_list, net_nodes, net_min_weight)
    progress_bar.progress(60, text="语义网络完成")

    topic_outputs = {}

    if "BERTopic(高级)" in topic_engines:
        try:
            from bertopic import BERTopic
            t_holder = st.empty()
            t_bar = t_holder.progress(0, text="正在运行 BERTopic 主题建模...")
            topic_model = BERTopic()
            topics, _ = topic_model.fit_transform(comments)
            bertopic_info = topic_model.get_topic_info()
            bertopic_topic_words = {}
            for tid in bertopic_info["Topic"].tolist():
                bertopic_topic_words[int(tid)] = topic_model.get_topic(int(tid)) or []
            topic_ids_bt = [int(t) for t in topics]
            topic_name_map_bt = {}
            for tid in bertopic_info["Topic"].tolist():
                tid = int(tid)
                if tid == -1:
                    topic_name_map_bt[tid] = "离群"
                    continue
                words = [w for w, _ in (bertopic_topic_words.get(tid) or [])][: int(lda_name_words)]
                topic_name_map_bt[tid] = " | ".join(words) if words else f"主题 {tid}"
            topic_outputs["BERTopic(高级)"] = {
                "topic_ids": topic_ids_bt,
                "topic_name_map": topic_name_map_bt,
                "info": bertopic_info,
                "topic_words": bertopic_topic_words,
            }
            t_bar.progress(100, text="BERTopic 完成")
            t_holder.empty()
        except ModuleNotFoundError as e:
            logger.info(f"BERTopic 未安装，将使用向量聚类近似实现: {e}")
            st.info("未安装 bertopic，已使用向量聚类近似实现 BERTopic。")
            t_holder = st.empty()
            t_bar = t_holder.progress(0, text="正在运行向量聚类主题建模(近似 BERTopic)...")
            try:
                topic_ids_bt, topic_words_bt, bt_info = run_vector_topic_model(
                    comments,
                    n_topics=int(lda_topics),
                    max_features=int(lda_n_features),
                    random_state=int(lda_random_state),
                )
                topic_name_map_bt = {}
                for tid in bt_info["Topic"].tolist():
                    tid = int(tid)
                    words = [w for w, _ in (topic_words_bt.get(tid) or [])][: int(lda_name_words)]
                    topic_name_map_bt[tid] = " | ".join(words) if words else f"主题 {tid}"
                topic_outputs["BERTopic(高级)"] = {
                    "topic_ids": topic_ids_bt,
                    "topic_name_map": topic_name_map_bt,
                    "info": bt_info,
                    "topic_words": topic_words_bt,
                }
            finally:
                t_bar.progress(100, text="向量聚类主题建模完成")
                t_holder.empty()
        except Exception as e:
            logger.warning(f"BERTopic 运行失败，将使用向量聚类近似实现: {e}")
            st.warning(f"BERTopic 运行失败，将使用向量聚类近似实现。原因：{e}")
            t_holder = st.empty()
            t_bar = t_holder.progress(0, text="正在运行向量聚类主题建模(近似 BERTopic)...")
            try:
                topic_ids_bt, topic_words_bt, bt_info = run_vector_topic_model(
                    comments,
                    n_topics=int(lda_topics),
                    max_features=int(lda_n_features),
                    random_state=int(lda_random_state),
                )
                topic_name_map_bt = {}
                for tid in bt_info["Topic"].tolist():
                    tid = int(tid)
                    words = [w for w, _ in (topic_words_bt.get(tid) or [])][: int(lda_name_words)]
                    topic_name_map_bt[tid] = " | ".join(words) if words else f"主题 {tid}"
                topic_outputs["BERTopic(高级)"] = {
                    "topic_ids": topic_ids_bt,
                    "topic_name_map": topic_name_map_bt,
                    "info": bt_info,
                    "topic_words": topic_words_bt,
                }
            finally:
                t_bar.progress(100, text="向量聚类主题建模完成")
                t_holder.empty()

    if "LDA" in topic_engines or not topic_outputs:
        curr_k = lda_topics
        if auto_lda:
            k_holder = st.empty()
            k_bar = k_holder.progress(0, text="正在搜索最佳主题数...")
            curr_k = find_best_lda_k(segmented_list, lda_topics_range[0], lda_topics_range[1], lda_max_df, lda_n_features, lda_random_state)
            k_bar.progress(100, text=f"最佳主题数：{curr_k}")
            k_holder.empty()
        lda_holder = st.empty()
        lda_bar = lda_holder.progress(0, text="正在运行 LDA 主题建模...")
        lda_res = run_lda_analysis(segmented_list, curr_k, lda_max_iter, lda_max_df, lda_n_features, lda_random_state)
        lda, dist, tf, vec = lda_res
        names = generate_topic_names(lda.components_, vec.get_feature_names_out(), int(lda_name_words))
        topic_ids_lda = [int(i) for i in dist.argmax(axis=1)]
        topic_name_map_lda = {int(k): str(v) for k, v in names.items()}
        topic_outputs["LDA"] = {
            "lda_res": lda_res,
            "topic_ids": topic_ids_lda,
            "topic_name_map": topic_name_map_lda,
        }
        lda_bar.progress(100, text="LDA 完成")
        lda_holder.empty()
    progress_bar.progress(75, text="主题建模完成")

    sentiment_dfs = {}
    df_sent_over = None
    df_sent_rest = None

    sent_holder = st.empty()
    sent_bar = sent_holder.progress(0, text="正在进行情感分析...")
    progress_step = 0
    for eng in sentiment_engines:
        progress_step += 1
        sent_bar.progress(int(progress_step * 100 / max(1, len(sentiment_engines))), text=f"正在进行情感分析：{eng}")
        
        # 对大文件使用分批情感分析
        if len(comments) > 20000:
            sentiments = []
            batch_size = 5000
            total_batches = (len(comments) + batch_size - 1) // batch_size
            
            for i in range(0, len(comments), batch_size):
                batch = comments[i:i+batch_size]
                batch_result = perform_sentiment_analysis(
                    batch,
                    stopwords,
                    effective_sen_dict,
                    not_words,
                    degree_dict,
                    pos_threshold,
                    neut_upper,
                    score_mode,
                    engine=eng,
                )
                sentiments.extend(batch_result)
                
                # 更新进度
                batch_progress = int(progress_step * 100 / max(1, len(sentiment_engines)) + 
                                   (i + len(batch)) / len(comments) * 100 / max(1, len(sentiment_engines)))
                sent_bar.progress(batch_progress, text=f"情感分析处理中: {eng} - 第 {i//batch_size + 1}/{total_batches} 批")
        else:
            sentiments = perform_sentiment_analysis(
                comments,
                stopwords,
                effective_sen_dict,
                not_words,
                degree_dict,
                pos_threshold,
                neut_upper,
                score_mode,
                engine=eng,
            )
        
        df_sent_tmp = pd.DataFrame(
            {
                "序号": list(range(1, len(comments) + 1)),
                "评论内容": comments,
                "情感倾向": [s[0] for s in sentiments],
                "情感得分": [s[1] for s in sentiments],
                "情感得分(原始)": [s[2] for s in sentiments],
                "有效词数": [s[3] for s in sentiments],
                "方面级情感(ABSA)": [s[4] for s in sentiments],
                "得分模式": score_mode if eng == "词典+规则" else "SnowNLP",
                "分析引擎": "词典+规则" if eng == "词典+规则" else "SnowNLP",
            }
        )
        sentiment_dfs[eng] = df_sent_tmp

    sent_bar.progress(100, text="情感分析完成")
    sent_holder.empty()
    progress_bar.progress(90, text="情感分析完成")

    df_sent_for_ipa = sentiment_dfs.get("词典+规则")
    if df_sent_for_ipa is None:
        df_sent_for_ipa = sentiment_dfs.get("SnowNLP (深度学习)")
    if df_sent_for_ipa is None:
        df_sent_for_ipa = next(iter(sentiment_dfs.values()))
    if enable_sent_cap and "词典+规则" in sentiment_engines:
        df_sent_over = df_sent_for_ipa[df_sent_for_ipa["情感得分"] >= float(sent_score_upper)].copy()
        df_sent_rest = df_sent_for_ipa[df_sent_for_ipa["情感得分"] < float(sent_score_upper)].copy()

    # 7. IPA 指标汇总
    def calculate_topic_metrics(topic_ids_in, topic_name_map_in, df_sent_in, comments_len):
        df = df_sent_in.copy()
        df["所属主题"] = [topic_name_map_in.get(int(i), str(i)) for i in topic_ids_in]
        m = df.groupby("所属主题").agg({"情感得分": "mean", "评论内容": "count"})
        m.columns = ["满意度", "评论数量"]
        m["重要性"] = m["评论数量"] / comments_len
        return m, df

    topic_metrics_map = {}
    df_sent_with_topics_map = {}
    for tname, tout in topic_outputs.items():
        m, df_with = calculate_topic_metrics(tout["topic_ids"], tout["topic_name_map"], df_sent_for_ipa, len(comments))
        topic_metrics_map[tname] = m
        df_sent_with_topics_map[tname] = df_with
    progress_bar.progress(100, text="分析完成")
    progress_holder.empty()

    # --- 4. 渲染 ---
    nav_options = ["📈 基础分析"]
    if "LDA" in topic_engines:
        nav_options.append("🧠 主题建模LDA")
    if "BERTopic(高级)" in topic_engines:
        nav_options.append("🧠 主题建模BERTopic")
    if "词典+规则" in sentiment_engines:
        nav_options.append("❤️ 情感分析-词典+规则")
    if "SnowNLP (深度学习)" in sentiment_engines:
        nav_options.append("❤️ 情感分析-SnowNLP")

    active_view = st.radio(
        "导航",
        nav_options,
        horizontal=True,
        label_visibility="collapsed",
        key="active_view",
    )

    if active_view == "📈 基础分析":
        st.subheader("关键词排行与词云")
        st.dataframe(df_freq, use_container_width=True, height=300)

        st.subheader("词云")
        st.image(wc_img, use_container_width=True)
        wc_png = io.BytesIO()
        wc_img.save(wc_png, format='PNG')
        st.download_button("📥 下载词云图片", wc_png.getvalue(), "词云.png", mime="image/png")

        lengths = pd.Series([len(str(c)) for c in comments], name="内容长度(字符)")
        if len(lengths) > 0:
            st.subheader("文本内容长度分布")
            
            # 使用Plotly创建交互式图表
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                values = lengths.values
                if int(len_xmax) > 0:
                    values = values[values <= int(len_xmax)]
                if int(len_xmin) > 0:
                    values = values[values >= int(len_xmin)]
                
                # 创建子图
                if len_show_density and len(values) > 1:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                else:
                    fig = go.Figure()
                
                # 添加直方图
                bins = "auto" if len_bins_mode == "自动" else int(len_bins_count)
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        nbinsx=bins if isinstance(bins, int) else None,
                        name="文本数量",
                        marker_color=len_bar_color,
                        opacity=float(len_bar_alpha),
                    )
                )
                
                # 添加密度曲线
                if len_show_density and len(values) > 1:
                    from scipy import stats
                    try:
                        kde = stats.gaussian_kde(values)
                        x_range = np.linspace(min(values), max(values), 200)
                        density = kde(x_range) * len(values)
                        if len_log_y:
                            density = np.log1p(density)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=density,
                                mode='lines',
                                name='密度曲线',
                                line=dict(color=len_density_color, width=2),
                                opacity=0.7,
                            ),
                            secondary_y=True,
                        )
                    except Exception as e:
                        logger.warning(f"密度曲线计算失败: {e}")
                
                # 更新布局
                fig.update_layout(
                    title="文本内容长度分布图",
                    xaxis_title="内容长度（字符）",
                    yaxis_title="文本数量",
                    height=500,
                    showlegend=True,
                )
                
                if len_show_grid:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                if len_log_y:
                    fig.update_yaxes(type="log")
                
                # 添加统计信息
                if len_show_stats and len(values) > 0:
                    mean_len = np.mean(values)
                    median_len = np.median(values)
                    std_len = np.std(values)
                    max_len = np.max(values)
                    min_len = np.min(values)
                    
                    stats_text = f"均值: {mean_len:.1f}<br>中位数: {median_len:.1f}<br>标准差: {std_len:.1f}<br>最大值: {max_len}<br>最小值: {min_len}<br>样本数: {len(values)}"
                    
                    # 根据位置参数添加注释
                    x_pos = 0.02 if '左' in len_stats_pos else 0.98
                    y_pos = 0.98 if '上' in len_stats_pos else 0.02
                    
                    fig.add_annotation(
                        text=stats_text,
                        xref="paper",
                        yref="paper",
                        x=x_pos,
                        y=y_pos,
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1,
                        opacity=0.8,
                        font=dict(size=10),
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 下载按钮
                col1, col2 = st.columns(2)
                with col1:
                    # PNG下载
                    try:
                        len_png = io.BytesIO()
                        fig.write_image(len_png, format='png', width=1200, height=600, scale=2)
                        st.download_button("📥 下载PNG图片", len_png.getvalue(), "文本长度分布.png", mime="image/png")
                    except Exception as e:
                        if "kaleido" in str(e).lower():
                            st.warning("导出图片需要安装 kaleido: pip install kaleido")
                        else:
                            st.error(f"导出图片失败: {str(e)}")
                with col2:
                    # SVG下载
                    try:
                        len_svg = io.BytesIO()
                        fig.write_image(len_svg, format='svg', width=1200, height=600)
                        st.download_button("📥 下载SVG矢量图", len_svg.getvalue(), "文本长度分布.svg", mime="image/svg+xml")
                    except Exception as e:
                        if "kaleido" in str(e).lower():
                            st.warning("导出图片需要安装 kaleido: pip install kaleido")
                        else:
                            st.error(f"导出图片失败: {str(e)}")
                
            except ImportError:
                st.info("安装 plotly 可获得交互式图表: pip install plotly")
                # 回退到matplotlib
                fig, ax = plt.subplots(figsize=(int(len_fig_w), int(len_fig_h)), dpi=int(len_dpi))
                fig.patch.set_facecolor(len_bg_color)
                ax.set_facecolor(len_bg_color)
                
                values = lengths.values
                if int(len_xmax) > 0:
                    values = values[values <= int(len_xmax)]
                if int(len_xmin) > 0:
                    values = values[values >= int(len_xmin)]
                
                bins = "auto" if len_bins_mode == "自动" else int(len_bins_count)
                
                edge_width = float(len_bar_edge_width)
                edge_color = len_bar_edge_color if edge_width > 0 else "none"
                
                n, bins_edges, patches = ax.hist(
                    values, 
                    bins=bins, 
                    color=len_bar_color, 
                    alpha=float(len_bar_alpha),
                    edgecolor=edge_color,
                    linewidth=edge_width
                )
                
                if len_show_density and len(values) > 1:
                    from scipy import stats
                    try:
                        kde = stats.gaussian_kde(values)
                        x_range = np.linspace(min(values), max(values), 200)
                        density = kde(x_range) * len(values)
                        if len_log_y:
                            density = np.log1p(density)
                        ax2 = ax.twinx()
                        ax2.plot(x_range, density, color=len_density_color, linewidth=2, alpha=0.7, label='密度曲线')
                        ax2.set_ylabel('密度', fontproperties=my_font if my_font else None, color=len_density_color)
                        ax2.tick_params(axis='y', labelcolor=len_density_color)
                        ax2.set_ylim(bottom=0)
                    except Exception as e:
                        logger.warning(f"密度曲线计算失败: {e}")
                
                if len_log_y:
                    ax.set_yscale("log")
                
                if len_show_grid:
                    ax.grid(True, alpha=float(len_grid_alpha), linestyle='--', linewidth=0.5)
                
                if my_font:
                    ax.set_xlabel("内容长度（字符）", fontproperties=my_font)
                    ax.set_ylabel("文本数量", fontproperties=my_font)
                    ax.set_title("文本内容长度分布图", fontproperties=my_font, fontsize=14, pad=15)
                else:
                    ax.set_xlabel("内容长度（字符）")
                    ax.set_ylabel("文本数量")
                    ax.set_title("文本内容长度分布图", fontsize=14, pad=15)
                
                if len_show_stats and len(values) > 0:
                    mean_len = np.mean(values)
                    median_len = np.median(values)
                    std_len = np.std(values)
                    max_len = np.max(values)
                    min_len = np.min(values)
                    
                    stats_text = f"均值: {mean_len:.1f}\n中位数: {median_len:.1f}\n标准差: {std_len:.1f}\n最大值: {max_len}\n最小值: {min_len}\n样本数: {len(values)}"
                    
                    pos_map = {"左上角": (0.02, 0.98), "右上角": (0.98, 0.98), "左下角": (0.02, 0.02), "右下角": (0.98, 0.02)}
                    pos = pos_map.get(len_stats_pos, (0.02, 0.98))
                    
                    ax.text(pos[0], pos[1], stats_text, transform=ax.transAxes, fontsize=9,
                            verticalalignment='top' if '上' in len_stats_pos else 'bottom',
                            horizontalalignment='left' if '左' in len_stats_pos else 'right',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))
                
                st.pyplot(fig, use_container_width=True)
                
                # 添加下载按钮
                len_png = io.BytesIO()
                fig.savefig(len_png, format='png', dpi=int(len_dpi), bbox_inches='tight', facecolor=len_bg_color)
                st.download_button("📥 下载长度分布图", len_png.getvalue(), "文本长度分布.png", mime="image/png")

        st.subheader("语义网络")
        render_semantic_network(
            semantic_net_res,
            my_font,
            net_max_edges,
            net_keep_isolates,
            net_layout,
            net_seed,
            net_k,
            net_iterations,
            net_fig_w,
            net_fig_h,
            net_dpi,
            net_bg,
            net_node_color,
            net_edge_color,
            net_node_alpha,
            net_edge_alpha,
            net_base_node_size,
            net_node_exp,
            net_node_scale,
            net_min_edge_width,
            net_edge_exp,
            net_edge_scale,
            net_show_labels,
            net_label_min_freq,
            net_label_top_n,
            net_label_size,
            net_label_color,
            net_label_alpha,
        )

    elif "主题建模" in active_view:
        ipa_sent_name = str(df_sent_for_ipa["分析引擎"].iloc[0]) if len(df_sent_for_ipa) else ""

        if "LDA" in topic_outputs and (active_view == "🧠 主题建模LDA" or active_view == "🧠 主题建模"):
            st.subheader("主题建模LDA")
            lda, dist, tf, vec = topic_outputs["LDA"]["lda_res"]
            mask = tf.getnnz(axis=1) > 0
            if mask.any():
                vis = pyLDAvis.prepare(
                    lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis],
                    dist[mask],
                    tf[mask].sum(axis=1).A1,
                    vec.get_feature_names_out(),
                    tf[mask].sum(axis=0).A1,
                    mds='mmds',
                )
                components.html(pyLDAvis.prepared_data_to_html(vis), height=800, scrolling=True)

            st.subheader("IPA 分析")
            st.caption(f"基于情感分析：{ipa_sent_name}")
            m = topic_metrics_map.get("LDA")
            if m is not None and len(m) > 0:
                fig, ax = plt.subplots(figsize=(10, 7))
                im, sm = m["重要性"].mean(), m["满意度"].mean()
                ax.axvline(im, color='gray', ls='--')
                ax.axhline(sm, color='gray', ls='--')
                for name, row in m.iterrows():
                    ax.scatter(row["重要性"], row["满意度"], s=300, alpha=0.7)
                    ax.text(row["重要性"], row["满意度"] + 0.05, name, fontproperties=my_font, ha='center')
                if my_font:
                    ax.set_title("IPA 决策象限", fontproperties=my_font)
                    ax.set_xlabel("重要性", fontproperties=my_font)
                    ax.set_ylabel("满意度", fontproperties=my_font)
                else:
                    ax.set_title("IPA 决策象限")
                    ax.set_xlabel("重要性")
                    ax.set_ylabel("满意度")
                st.pyplot(fig)
                
                # 添加下载按钮
                ipa_png = io.BytesIO()
                fig.savefig(ipa_png, format='png', dpi=150, bbox_inches='tight')
                st.download_button("📥 下载IPA分析图", ipa_png.getvalue(), "IPA分析图(LDA).png", mime="image/png")

        if "BERTopic(高级)" in topic_outputs and (active_view == "🧠 主题建模BERTopic" or active_view == "🧠 主题建模"):
            st.subheader("主题建模BERTopic")
            bt_info = topic_outputs["BERTopic(高级)"].get("info")
            bt_words = topic_outputs["BERTopic(高级)"].get("topic_words") or {}
            bt_name_map = topic_outputs["BERTopic(高级)"].get("topic_name_map") or {}
            if bt_info is not None:
                st.dataframe(bt_info, use_container_width=True, height=350)
            if bt_words:
                rows = []
                for tid, words in bt_words.items():
                    if int(tid) == -1:
                        continue
                    top_words = [w for w, _ in words][:10]
                    rows.append({"主题": bt_name_map.get(int(tid), str(tid)), "Top词": "、".join(top_words)})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

            st.subheader("IPA 分析")
            st.caption(f"基于情感分析：{ipa_sent_name}")
            m = topic_metrics_map.get("BERTopic(高级)")
            if m is not None and len(m) > 0:
                fig, ax = plt.subplots(figsize=(10, 7))
                im, sm = m["重要性"].mean(), m["满意度"].mean()
                ax.axvline(im, color='gray', ls='--')
                ax.axhline(sm, color='gray', ls='--')
                for name, row in m.iterrows():
                    ax.scatter(row["重要性"], row["满意度"], s=300, alpha=0.7)
                    ax.text(row["重要性"], row["满意度"] + 0.05, name, fontproperties=my_font, ha='center')
                if my_font:
                    ax.set_title("IPA 决策象限", fontproperties=my_font)
                    ax.set_xlabel("重要性", fontproperties=my_font)
                    ax.set_ylabel("满意度", fontproperties=my_font)
                else:
                    ax.set_title("IPA 决策象限")
                    ax.set_xlabel("重要性")
                    ax.set_ylabel("满意度")
                st.pyplot(fig)
                
                # 添加下载按钮
                ipa_png = io.BytesIO()
                fig.savefig(ipa_png, format='png', dpi=150, bbox_inches='tight')
                st.download_button("📥 下载IPA分析图", ipa_png.getvalue(), "IPA分析图(BERTopic).png", mime="image/png")

    elif active_view == "❤️ 情感分析-词典+规则":
        eng = "词典+规则"
        df_sent = sentiment_dfs.get(eng)
        if df_sent is not None:
            st.subheader("情感分析-词典+规则")
            st.caption(f"得分模式：{score_mode}（阈值判断/趋势图/筛选均使用该模式得分）")

            st.dataframe(df_sent, use_container_width=True, height=400)
            st.bar_chart(df_sent["情感倾向"].value_counts())

            if enable_sent_cap:
                df_over = df_sent[df_sent["情感得分"] >= float(sent_score_upper)].copy()
                over_count = int(len(df_over))
                st.caption(f"超阈值数量：{over_count} / {len(df_sent)}（阈值：{float(sent_score_upper):g}）")
                if over_count > 0:
                    with st.expander("查看超阈值文本", expanded=False):
                        st.dataframe(df_over, use_container_width=True, height=300)

            if len(df_sent) >= 3:
                st.subheader("情感趋势图")
                max_window = min(300, len(df_sent))
                default_window = min(30, max_window)
                window = st.slider("趋势窗口(越大越平滑)", 3, max_window, value=int(st.session_state.get("trend_window", default_window)), key=f"trend_window_{eng}")

                base_scores = df_sent["情感得分"].copy()
                if enable_sent_cap and eng == "词典+规则":
                    sent_score_upper_val = float(st.session_state.get("sent_score_upper", 100.0))
                    base_scores = base_scores.clip(upper=sent_score_upper_val)
                    st.caption(f"趋势图已对得分做上限截断：>= {sent_score_upper_val:g} 的点显示为 {sent_score_upper_val:g}")

                trend = df_sent[["序号"]].copy()
                trend["情感得分"] = base_scores
                x_name = "评论序号"
                x_note = None
                title_name = "情感得分趋势"
                if trend_order_mode == "第一条=最近(倒序)":
                    trend = trend.sort_values("序号", ascending=False).reset_index(drop=True)
                    trend["趋势序号"] = list(range(1, len(trend) + 1))
                    x_col = "趋势序号"
                    x_name = "时间序(1=最近)"
                    x_note = "左=最近 → 右=最远"
                elif trend_order_mode == "按评论编号" and trend_id_col and df_excel is not None:
                    ids = df_excel[trend_id_col].reindex(df_excel[target_col].dropna().index).astype(str).str.strip().tolist()
                    if len(ids) == len(trend):
                        trend["评论编号"] = ids
                        trend["评论编号(数值)"] = pd.to_numeric(trend["评论编号"], errors="coerce")
                        if trend["评论编号(数值)"].notna().any():
                            trend = trend.sort_values(["评论编号(数值)", "序号"], ascending=True).reset_index(drop=True)
                        else:
                            trend = trend.sort_values(["评论编号", "序号"], ascending=True).reset_index(drop=True)
                        trend["趋势序号"] = list(range(1, len(trend) + 1))
                        x_col = "趋势序号"
                        x_name = "按编号排序序列"
                        x_note = "左=编号小 → 右=编号大"
                    else:
                        x_col = "序号"
                else:
                    trend = trend.sort_values("序号", ascending=True).reset_index(drop=True)
                    x_col = "序号"
                    x_note = "左=最远 → 右=最近"

                trend["情感得分(滚动均值)"] = trend["情感得分"].rolling(window=window, min_periods=1).mean()

                fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
                ax.plot(trend[x_col], trend["情感得分"], color="#9AA0A6", alpha=0.25, linewidth=1, label="原始得分")
                ax.plot(trend[x_col], trend["情感得分(滚动均值)"], color="#1f77b4", linewidth=2, label="滚动均值")
                ax.axhline(0, color="#666666", linewidth=1, alpha=0.5)
                x_label = x_name if not x_note else f"{x_name}（{x_note}）"
                if my_font:
                    ax.set_title(title_name, fontproperties=my_font)
                    ax.set_xlabel(x_label, fontproperties=my_font)
                    ax.set_ylabel("情感得分", fontproperties=my_font)
                    ax.legend(prop=my_font)
                else:
                    ax.set_title(title_name)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("情感得分")
                    ax.legend()
                st.pyplot(fig, use_container_width=True)
                
                # 添加下载按钮
                sent_png = io.BytesIO()
                fig.savefig(sent_png, format='png', dpi=150, bbox_inches='tight')
                st.download_button("📥 下载情感趋势图", sent_png.getvalue(), "情感趋势图(词典+规则).png", mime="image/png")
        else:
            st.info("未启用词典+规则情感分析引擎")

    elif active_view == "❤️ 情感分析-SnowNLP":
        eng = "SnowNLP (深度学习)"
        df_sent = sentiment_dfs.get(eng)
        if df_sent is not None:
            st.subheader("情感分析-SnowNLP (深度学习)")
            st.caption("得分范围：0~1（积极概率），>0.6 为积极，<0.4 为消极")

            st.dataframe(df_sent, use_container_width=True, height=400)
            st.bar_chart(df_sent["情感倾向"].value_counts())

            if len(df_sent) >= 3:
                st.subheader("情感趋势图")
                max_window = min(300, len(df_sent))
                default_window = min(30, max_window)
                window = st.slider("趋势窗口(越大越平滑)", 3, max_window, value=int(st.session_state.get("trend_window", default_window)), key=f"trend_window_{eng}")

                base_scores = df_sent["情感得分"].copy()

                trend = df_sent[["序号"]].copy()
                trend["情感得分"] = base_scores
                x_name = "评论序号"
                x_note = None
                title_name = "情感得分趋势"
                if trend_order_mode == "第一条=最近(倒序)":
                    trend = trend.sort_values("序号", ascending=False).reset_index(drop=True)
                    trend["趋势序号"] = list(range(1, len(trend) + 1))
                    x_col = "趋势序号"
                    x_name = "时间序(1=最近)"
                    x_note = "左=最近 → 右=最远"
                elif trend_order_mode == "按评论编号" and trend_id_col and df_excel is not None:
                    ids = df_excel[trend_id_col].reindex(df_excel[target_col].dropna().index).astype(str).str.strip().tolist()
                    if len(ids) == len(trend):
                        trend["评论编号"] = ids
                        trend["评论编号(数值)"] = pd.to_numeric(trend["评论编号"], errors="coerce")
                        if trend["评论编号(数值)"].notna().any():
                            trend = trend.sort_values(["评论编号(数值)", "序号"], ascending=True).reset_index(drop=True)
                        else:
                            trend = trend.sort_values(["评论编号", "序号"], ascending=True).reset_index(drop=True)
                        trend["趋势序号"] = list(range(1, len(trend) + 1))
                        x_col = "趋势序号"
                        x_name = "按编号排序序列"
                        x_note = "左=编号小 → 右=编号大"
                    else:
                        x_col = "序号"
                else:
                    trend = trend.sort_values("序号", ascending=True).reset_index(drop=True)
                    x_col = "序号"
                    x_note = "左=最远 → 右=最近"

                trend["情感得分(滚动均值)"] = trend["情感得分"].rolling(window=window, min_periods=1).mean()

                fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
                ax.plot(trend[x_col], trend["情感得分"], color="#9AA0A6", alpha=0.25, linewidth=1, label="原始得分")
                ax.plot(trend[x_col], trend["情感得分(滚动均值)"], color="#1f77b4", linewidth=2, label="滚动均值")
                ax.axhline(0, color="#666666", linewidth=1, alpha=0.5)
                x_label = x_name if not x_note else f"{x_name}（{x_note}）"
                if my_font:
                    ax.set_title(title_name, fontproperties=my_font)
                    ax.set_xlabel(x_label, fontproperties=my_font)
                    ax.set_ylabel("情感得分", fontproperties=my_font)
                    ax.legend(prop=my_font)
                else:
                    ax.set_title(title_name)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("情感得分")
                    ax.legend()
                st.pyplot(fig, use_container_width=True)
                
                # 添加下载按钮
                sent_png = io.BytesIO()
                fig.savefig(sent_png, format='png', dpi=150, bbox_inches='tight')
                st.download_button("📥 下载情感趋势图", sent_png.getvalue(), "情感趋势图(SnowNLP).png", mime="image/png")
        else:
            st.info("未启用SnowNLP情感分析引擎")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_freq.to_excel(writer, sheet_name='关键词排行', index=False)
        df_sent_for_ipa.to_excel(writer, sheet_name='情感结果', index=False)

        if "词典+规则" in sentiment_dfs:
            sentiment_dfs["词典+规则"].to_excel(writer, sheet_name='情感_词典', index=False)
        if "SnowNLP (深度学习)" in sentiment_dfs:
            sentiment_dfs["SnowNLP (深度学习)"].to_excel(writer, sheet_name='情感_SnowNLP', index=False)

        primary_topic = "LDA" if "LDA" in topic_metrics_map else (next(iter(topic_metrics_map.keys())) if topic_metrics_map else None)
        if primary_topic:
            topic_metrics_map[primary_topic].to_excel(writer, sheet_name='IPA', index=True)
        if "LDA" in topic_metrics_map:
            topic_metrics_map["LDA"].to_excel(writer, sheet_name='IPA_LDA', index=True)
        if "BERTopic(高级)" in topic_metrics_map:
            topic_metrics_map["BERTopic(高级)"].to_excel(writer, sheet_name='IPA_BERTopic', index=True)

        if enable_sent_cap and "词典+规则" in sentiment_dfs:
            base_df = sentiment_dfs["词典+规则"]
            df_over = base_df[base_df["情感得分"] >= float(sent_score_upper)].copy()
            df_rest = base_df[base_df["情感得分"] < float(sent_score_upper)].copy()
            (df_over if df_over is not None else base_df.iloc[0:0]).to_excel(
                writer,
                sheet_name='超阈值文本',
                index=False,
            )
            (df_rest if df_rest is not None else base_df).to_excel(
                writer,
                sheet_name='剩余文本',
                index=False,
            )
    st.sidebar.download_button("📥 下载 Excel 报告", output.getvalue(), "分析报告.xlsx", use_container_width=True)
    
    if st.sidebar.button("🗑️ 清空分析缓存", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    cfg_snapshot = _build_config_snapshot()
    try:
        cfg_snapshot["_meta"] = {
            "comment_file_name": getattr(uploaded_file, "name", None),
            "comment_file_hash": _stable_bytes_hash(uploaded_file.getvalue()) if uploaded_file is not None else None,
        }
    except Exception as e:
        logger.warning(f"生成配置文件元数据时出错: {e}")
    st.sidebar.download_button(
        "🧩 下载配置文件",
        data=json.dumps(cfg_snapshot, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="配置.json",
        mime="application/json",
        use_container_width=True,
    )

if __name__ == "__main__":
    main()
