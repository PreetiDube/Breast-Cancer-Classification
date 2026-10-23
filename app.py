import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Breast Cancer Classification | ML Demo", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.stApp {
 background: radial-gradient(circle at 85% 8%, rgba(173,132,255,.12), transparent 28%),
             radial-gradient(circle at 10% 20%, rgba(232,157,190,.10), transparent 25%), #0b0911;
 color: #f5eff7;
}
[data-testid="stHeader"] {background:rgba(11,9,17,.88)}
.block-container {max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.eyebrow {color:#efa5c2;text-transform:uppercase;letter-spacing:.18em;font-size:.75rem;font-weight:800}
.hero-title {font-size:clamp(2.3rem,5vw,4.3rem);line-height:1.05;font-weight:850;letter-spacing:-.045em;margin:.6rem 0 1rem}
.gradient {background:linear-gradient(100deg,#efa5c2,#c4a6ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtle {color:#b8afc4;font-size:1.05rem;line-height:1.7}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#191522,#12101a);border:1px solid #332a40;border-radius:16px;padding:18px 20px}
div[data-testid="stForm"] {border:1px solid #332a40;border-radius:16px;padding:1.1rem;background:rgba(23,19,31,.7)}
.notice {border-left:3px solid #efa5c2;background:#191320;border-radius:8px;padding:14px 16px;color:#e9dfef}
.stButton button[kind="primary"] {background:linear-gradient(100deg,#efa5c2,#c4a6ff);color:#17101d;border:0;font-weight:800;border-radius:10px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    X = frame.drop(columns="target")
    y = frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return dataset, X, model, accuracy_score(y_test, pred), confusion_matrix(y_test, pred, labels=[0, 1]), len(X_test)

dataset, X_all, model, accuracy, matrix, test_count = train_model()
features = list(X_all.columns)
defaults = X_all.median(numeric_only=True)

st.markdown('<div class="eyebrow">PD. / MACHINE LEARNING · INTERACTIVE DEMO</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Breast Cancer<br><span class="gradient">Classification.</span></div>', unsafe_allow_html=True)
st.markdown('<p class="subtle">Explore a Logistic Regression model trained on Scikit-learn’s Wisconsin Diagnostic Breast Cancer benchmark dataset. Enter numeric feature measurements to see how the model classifies a sample.</p>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Held-out test accuracy", f"{accuracy:.2%}")
m2.metric("Test samples", f"{test_count:,}")
m3.metric("Input features", f"{len(features)}")

st.markdown("---")
st.subheader("Try a prediction")
st.markdown("Enter the 30 numeric measurements below. Defaults are dataset medians for a neutral starting point; they are not measurements from a real person.")

feature_groups = [
    ("Mean measurements", [c for c in features if c.startswith("mean ")]),
    ("Standard-error measurements", [c for c in features if c.endswith(" error")]),
    ("Worst measurements", [c for c in features if c.startswith("worst ")]),
]
grouped = {name for _, cols in feature_groups for name in cols}
remaining = [c for c in features if c not in grouped]
if remaining:
    feature_groups.append(("Other measurements", remaining))

values = {}
with st.form("prediction_form"):
    st.caption("Feature names follow Scikit-learn's dataset terminology.")
    for group_title, cols in feature_groups:
        with st.expander(group_title, expanded=(group_title == "Mean measurements")):
            columns = st.columns(2)
            for i, col in enumerate(cols):
                with columns[i % 2]:
                    values[col] = st.number_input(
                        col.title(), min_value=0.0, value=float(defaults[col]),
                        step=max(float(defaults[col]) * 0.01, 0.001),
                        format="%.5f", key=f"feature_{col}", help=f"Dataset feature: {col}"
                    )
    submitted = st.form_submit_button("Run classification demo", type="primary", use_container_width=True)

if submitted:
    sample = pd.DataFrame([[values[c] for c in features]], columns=features)
    prediction = int(model.predict(sample)[0])
    scores = model.predict_proba(sample)[0]
    score = float(scores[list(model.classes_).index(prediction)])
    label = {0: "Malignant", 1: "Benign"}[prediction]
    st.markdown("### Model output")
    left, right = st.columns([1, 1])
    with left:
        if prediction == 0:
            st.error(f"Predicted class: {label}")
        else:
            st.success(f"Predicted class: {label}")
        st.metric("Model score for predicted class", f"{score:.1%}")
        st.caption("This score is the model's output for this input, not a clinically validated probability.")
    with right:
        st.markdown('<div class="notice"><strong>Important limitation</strong><br>This is a machine-learning learning demo using a public benchmark dataset. It is not clinically validated and must not be used for diagnosis, screening, or treatment decisions. Do not enter identifiable or private health information.</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("How the model works")
c1, c2, c3 = st.columns(3)
c1.markdown("**01 · Dataset**  \nScikit-learn's built-in Wisconsin Diagnostic Breast Cancer dataset.")
c2.markdown("**02 · Preprocessing**  \nA `StandardScaler` is fitted inside a Pipeline using training data.")
c3.markdown("**03 · Classification**  \nLogistic Regression; held-out split uses stratification and random state 42.")

with st.expander("View held-out confusion matrix"):
    st.caption("Rows are actual classes; columns are predicted classes. Class order: malignant, benign.")
    st.dataframe(pd.DataFrame(matrix, index=["Actual malignant", "Actual benign"],
                              columns=["Predicted malignant", "Predicted benign"]),
                 use_container_width=True)
    st.caption("These results describe this benchmark split, not real-world clinical performance.")

st.markdown("---")
st.markdown('<div class="notice">Educational use only. This app is not a medical device and is not a substitute for professional medical advice or diagnosis.</div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9f95ad;padding-top:1.5rem;font-size:.85rem'>PD. / Machine Learning · Built with Streamlit and Scikit-learn</div>", unsafe_allow_html=True)
