import streamlit as st
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

models=["K-nearest Neighbors","Logistic Regression","Decision Tree","Random Forest"]
run_ready=False
k=5
log_regression_solver='lbfgs'
min_split_decision=2
max_depth_decision=0
n_trees=100
min_split_rf=2
max_depth_rf=0

# Show title and description.
st.title("AutoML for Classification Tasks")
st.write(
    "Upload a dataset, select your settings, and click 'Run' to process your dataset through various machine learning algorithms. Note this application only works for classification tasks."
)

# Let the user upload a file.
uploaded_file = st.file_uploader("Upload a data file with headers (.csv)", type=("csv"))

if uploaded_file:
    # Read data, list size and variables
    df=pd.read_csv(uploaded_file)
    st.write(f"Your file contains {len(df)} rows and {len(df.columns)} columns.")
    st.divider()

    target = st.selectbox("Which variable is the target?",df.columns,)
    possible_features=[x for x in df.columns if x!=target]
    if target:
        features = st.pills("Which features should be included in the model? All features will be treated as numerical.", possible_features, selection_mode="multi",key="feature_selection")
        st.divider()
        if features:
            split=st.radio("What train/test split should be used in the model?",['90/10','80/20','60/40','50/50'],horizontal=True, index=None)
            st.divider()
            if split:
                selected_models = st.pills("Which models would you like to run?", models, selection_mode="multi",key="model_selection")
                advanced=st.toggle("Advanced Mode (default model parameters used if disabled)")
                if advanced:
                    if "K-nearest Neighbors" in selected_models:
                        k=st.number_input("Number of neighbors in K-nearest Neighbors:",min_value=1, max_value=len(df),value=5)
                        st.divider()
                    if "Logistic Regression" in selected_models:
                        log_regression_solver=st.selectbox("Solver algorithm to use in Logistic Regression:",['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
                        st.divider()
                    if "Decision Tree" in selected_models:
                        min_split_decision=st.number_input("Minimum number of samples required to split a node in Decision Tree:",min_value=2, max_value=len(df),value=2)
                        max_depth_decision=st.number_input("Maximum depth of the Decision Tree (0 is infinite):",min_value=0, max_value=len(df),value=0)
                        st.divider()
                    if "Random Forest" in selected_models:
                        n_trees=st.number_input("Number of trees in Random Forest:",min_value=1,max_value=None,value=100,step=20)
                        min_split_rf=st.number_input("Minimum number of samples required to split a node in Random Forest:",min_value=2, max_value=len(df),value=2)
                        max_depth_rf=st.number_input("Maximum depth of the Random Forest (0 is infinite):",min_value=0, max_value=len(df),value=0)
                else:
                    k=5
                    log_regression_solver='lbfgs'
                    min_split_decision=2
                    max_depth_decision=0
                    n_trees=100
                    min_split_rf=2
                    max_depth_rf=0
                if max_depth_decision==0:
                    max_depth_decision=None
                if max_depth_rf==0:
                    max_depth_rf=None
                if len(selected_models)>0:
                    run_ready=True
                run_models=st.button('Run Models',disabled=not run_ready,type="primary",key="submit")

                if run_models:
                    my_progress=st.progress(0, text="Running models...")
                    y=df[target]
                    X=df[features]
                    split_dict={'90/10':.1,'80/20':.2,'60/40':.4,'50/50':.5}
                    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split_dict[split], random_state=0, stratify=y)
                    for i, model in enumerate(selected_models):
                        my_progress.progress(i/len(selected_models),text=f"Running models: {model}...")
                        time.sleep(1)
                        #"Decision Tree","Random Forest"
                        if model=="K-nearest Neighbors":
                            classifier=sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
                        elif model=="Logistic Regression":
                            classifier=sklearn.linear_model.LogisticRegression(solver=log_regression_solver,random_state=0)
                        elif model=="Decision Tree":
                            classifier=sklearn.tree.DecisionTreeClassifier(min_samples_split=min_split_decision,max_depth=max_depth_decision,random_state=0)
                        elif model=="Random Forest":
                            classifier=RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth_rf, min_samples_split=min_split_rf, random_state=0)
                        else:
                            continue
                        classifier.fit(X_train, y_train)
                        y_pred=classifier.predict(X_test)
                        st.write(classifier.score(X_test,y_test))
                        cm=ConfusionMatrixDisplay.from_predictions(y_test, y_pred,display_labels=df[target].unique())
                        cm.plot()
                        plt.xlabel('predict')
                        plt.ylabel('truth')
                        plt.title(model)
                        st.pyplot(plt.gcf())
                    my_progress.empty()#remove progress bar
