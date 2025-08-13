import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import statsmodels.api as sm

# ========= Config =========
DATA_PATH = Path("your_dataframe.parquet")  # or .csv
LOG_PATH = Path("judgments.parquet")
TEXT_COL = "predictions"
GROUP_COL = "Position of Oracle"          # 5 groups (e.g., 1,10,20,30,40)
TOPIC_ID_COL = "topic_id"
TOPIC_TXT_COL = "topic"
REFS_COL = "references"                   # optional
SHOW_REFS_DEFAULT = False

st.set_page_config(page_title="Pairwise Judging UI", layout="wide")

# ========= Data loading =========
@st.cache_data
def load_df(path):
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # add a unique row id
    if "_row_id" not in df.columns:
        df = df.copy()
        df["_row_id"] = np.arange(len(df))
    return df

def load_log():
    if LOG_PATH.exists():
        return pd.read_parquet(LOG_PATH)
    return pd.DataFrame(columns=[
        "ts","judge_id","topic_id","topic",
        "pair","left_group","right_group",
        "left_row_id","right_row_id",
        "choice","rt_ms"
    ])

def append_log(row_dict):
    log = load_log()
    log = pd.concat([log, pd.DataFrame([row_dict])], ignore_index=True)
    log.to_parquet(LOG_PATH, index=False)

df = load_df(DATA_PATH)
groups = sorted(df[GROUP_COL].unique())
pair_list = [(a, b) for i, a in enumerate(groups) for b in groups[i+1:]]

# ========= Sampling helpers =========
def seen_pair_exact(log, topic_id, a_id, b_id):
    if log.empty: return False
    m = (log["topic_id"] == topic_id) & (
            ((log["left_row_id"] == a_id) & (log["right_row_id"] == b_id)) |
            ((log["left_row_id"] == b_id) & (log["right_row_id"] == a_id))
    )
    return bool(m.any())

def next_balanced_group_pair(log):
    """Pick the least-used group pair overall (keeps the 10 pairs even)."""
    counts = log["pair"].value_counts().to_dict() if not log.empty else {}
    counts = {str(p): counts.get(str(p), 0) for p in pair_list}
    minc = min(counts.values()) if counts else 0
    candidates = [eval(k) for k, v in counts.items() if v == minc]
    return candidates[np.random.randint(len(candidates))]

def sample_two_within_topic(topic_id, g1, g2, log):
    sub = df[df[TOPIC_ID_COL] == topic_id]
    a_pool = sub[sub[GROUP_COL] == g1]
    b_pool = sub[sub[GROUP_COL] == g2]
    # try to avoid repeating the exact same pair
    for _ in range(250):
        a = a_pool.sample(1).iloc[0]
        b = b_pool.sample(1).iloc[0]
        if not seen_pair_exact(log, topic_id, int(a["_row_id"]), int(b["_row_id"])):
            return a, b
    return a_pool.sample(1).iloc[0], b_pool.sample(1).iloc[0]  # fallback

# ========= Bradley–Terry fitting (per topic) =========
def fit_bt_for_topic(topic_log):
    """
    topic_log: rows with columns left_group,right_group,choice in {Left,Right,Tie,Unsure}
    Returns: DataFrame with 'group','beta','rank'
    """
    if topic_log.empty:
        return pd.DataFrame(columns=["group","beta","rank"])

    # Prepare comparisons as binary outcomes (winner=1 for first item in row)
    # We’ll duplicate Ties as half-wins both ways (common pragmatic choice)
    rows = []
    for _, r in topic_log.iterrows():
        lg, rg = r["left_group"], r["right_group"]
        ch = r["choice"]
        if ch == "Left":
            rows.append((lg, rg, 1))
        elif ch == "Right":
            rows.append((rg, lg, 1))
        elif ch == "Tie":
            rows.append((lg, rg, 1))
            rows.append((rg, lg, 1))
        else:
            # Unsure -> ignore
            pass
    if not rows:
        return pd.DataFrame(columns=["group","beta","rank"])
    comp = pd.DataFrame(rows, columns=["winner","loser","y"])

    # Build design matrix for logistic regression: one column per group (minus ref)
    items = sorted(set(comp["winner"]).union(set(comp["loser"])))
    ref = items[0]  # reference item
    cols = [g for g in items if g != ref]

    X = []
    y = comp["y"].values  # all ones; we build as winner vs loser entries
    # For each (winner, loser) we encode +1 for winner, -1 for loser
    for _, r in comp.iterrows():
        v = {c: 0.0 for c in cols}
        for c in cols:
            if r["winner"] == c: v[c] = +1.0
            if r["loser"]  == c: v[c] = -1.0
        X.append([v[c] for c in cols])
    X = np.asarray(X)

    model = sm.GLM(y, X, family=sm.families.Binomial(), offset=None)
    res = model.fit()
    betas = {ref: 0.0}
    for c, b in zip(cols, res.params):
        betas[c] = b

    out = pd.DataFrame({"group": list(betas.keys()), "beta": list(betas.values())})
    out["rank"] = out["beta"].rank(ascending=False, method="average").astype(int)
    out = out.sort_values("rank", kind="mergesort").reset_index(drop=True)
    return out

def aggregate_average_rank(bt_tables):
    """bt_tables: dict {topic_id: bt_df} -> DataFrame of average rank per group."""
    frames = []
    for tid, df_ in bt_tables.items():
        if df_.empty: continue
        tmp = df_[["group","rank"]].copy()
        tmp["topic_id"] = tid
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["group","avg_rank","n_topics"])
    ranks = pd.concat(frames, ignore_index=True)
    agg = ranks.groupby("group")["rank"].agg(["mean","count"]).reset_index()
    agg.columns = ["group","avg_rank","n_topics"]
    return agg.sort_values("avg_rank")

# ========= UI Tabs =========
tab1, tab2 = st.tabs(["Judge", "Results"])

with tab1:
    st.title("Pairwise comparison (within topic)")

    with st.sidebar:
        st.header("Session")
        judge_id = st.text_input("Judge ID (required)", value=st.session_state.get("judge_id",""))
        st.session_state["judge_id"] = judge_id
        topic_ids = sorted(df[TOPIC_ID_COL].unique())
        topic_id = st.selectbox("Topic", options=topic_ids)
        show_refs = st.checkbox("Show references", value=SHOW_REFS_DEFAULT)
        st.markdown("---")
        if "done" not in st.session_state: st.session_state["done"] = 0
        st.metric("This‑session decisions", st.session_state["done"])

    if not judge_id:
        st.warning("Enter a Judge ID in the sidebar to begin.")
        st.stop()

    topic_text = df.loc[df[TOPIC_ID_COL]==topic_id, TOPIC_TXT_COL].iloc[0]
    st.subheader(f"Topic: {topic_text}")

    log = load_log()

    # pick a least-used group pair overall (you could also balance *within this topic* if preferred)
    g1, g2 = next_balanced_group_pair(log)

    # sample two responses within this topic
    a, b = sample_two_within_topic(topic_id, g1, g2, log)

    # randomize left/right
    if np.random.rand() < 0.5:
        left, right = a, b
        left_group, right_group = g1, g2
    else:
        left, right = b, a
        left_group, right_group = g2, g1

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.caption(f"Group: {left_group}")
        st.text_area("Left response", left[TEXT_COL], height=360)
        if show_refs and REFS_COL in df.columns:
            with st.expander("References"):
                st.write(left.get(REFS_COL,""))

    with col2:
        st.caption(f"Group: {right_group}")
        st.text_area("Right response", right[TEXT_COL], height=360)
        if show_refs and REFS_COL in df.columns:
            with st.expander("References"):
                st.write(right.get(REFS_COL,""))

    st.markdown("### Your judgment")
    if "start_ts" not in st.session_state:
        st.session_state["start_ts"] = time.time()

    def submit(choice_label):
        now = time.time()
        row = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "judge_id": judge_id,
            "topic_id": topic_id,
            "topic": topic_text,
            "pair": str(tuple(sorted([int(g1), int(g2)]))),
            "left_group": int(left_group),
            "right_group": int(right_group),
            "left_row_id": int(left["_row_id"]),
            "right_row_id": int(right["_row_id"]),
            "choice": choice_label,
            "rt_ms": int((now - st.session_state["start_ts"])*1000),
        }
        append_log(row)
        st.session_state["start_ts"] = time.time()
        st.session_state["done"] += 1
        st.experimental_rerun()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.button("Left better", use_container_width=True, on_click=lambda: submit("Left"))
    with c2: st.button("Right better", use_container_width=True, on_click=lambda: submit("Right"))
    with c3: st.button("Tie", use_container_width=True, on_click=lambda: submit("Tie"))
    with c4: st.button("Unsure", use_container_width=True, on_click=lambda: submit("Unsure"))

with tab2:
    st.title("Results")

    log = load_log()
    if log.empty:
        st.info("No judgments yet.")
        st.stop()

    st.write("Raw judgments:", log.shape[0])

    # ----- Per-topic Bradley–Terry -----
    bt_tables = {}
    for tid, tlog in log.groupby("topic_id"):
        bt_tables[tid] = fit_bt_for_topic(tlog)

    # Show a selector to inspect a topic’s ranking table
    topic_ids = sorted(bt_tables.keys())
    sel_tid = st.selectbox("Inspect topic ranking", options=topic_ids)
    bt_df = bt_tables[sel_tid]
    st.subheader(f"Topic {sel_tid} ranking")
    if bt_df.empty:
        st.warning("Not enough data yet for this topic.")
    else:
        st.dataframe(bt_df, use_container_width=True)

    # ----- Average rank across topics -----
    agg = aggregate_average_rank(bt_tables)
    st.subheader("Average rank across topics")
    if agg.empty:
        st.warning("Not enough topic rankings yet.")
    else:
        st.dataframe(agg, use_container_width=True)

    # Optional: export CSVs
    colA, colB = st.columns(2)
    if colA.button("Export per-topic BT tables (CSV)"):
        all_rows = []
        for tid, df_ in bt_tables.items():
            if df_.empty: continue
            tmp = df_.copy()
            tmp["topic_id"] = tid
            all_rows.append(tmp)
        if all_rows:
            out = pd.concat(all_rows, ignore_index=True)
            out.to_csv("bt_per_topic.csv", index=False)
            st.success("Saved bt_per_topic.csv")
    if colB.button("Export average rank (CSV)"):
        if not agg.empty:
            agg.to_csv("avg_rank_by_group.csv", index=False)
            st.success("Saved avg_rank_by_group.csv")
