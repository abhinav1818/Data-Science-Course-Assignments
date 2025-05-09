st.subheader("Upload CSV File for Prediction")
uploaded_file = st.file_uploader("Titanic_test.py")

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    original = user_df.copy()
    
    # Check for target column
    has_target = 'Survived' in user_df.columns

    # Preprocess user data
    user_df = preprocess(user_df)

    if has_target:
        X_user = user_df.drop("Survived", axis=1)
        y_user = original['Survived']
        y_pred = model.predict(X_user)
        y_prob = model.predict_proba(X_user)[:, 1]

        # Evaluation metrics
        st.write("**Uploaded Data Evaluation:**")
        st.write("Accuracy:", accuracy_score(y_user, y_pred))
        st.write("Precision:", precision_score(y_user, y_pred))
        st.write("Recall:", recall_score(y_user, y_pred))
        st.write("F1 Score:", f1_score(y_user, y_pred))
        st.write("ROC AUC Score:", roc_auc_score(y_user, y_prob))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_user, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC Curve')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Uploaded File)")
        ax.legend()
        st.pyplot(fig)
    else:
        # Predict for unlabeled data
        y_pred = model.predict(user_df)
        y_prob = model.predict_proba(user_df)[:, 1]
        result_df = original.copy()
        result_df['Prediction'] = y_pred
        result_df['Survival Probability'] = y_prob
        st.write("### Prediction Results")
        st.dataframe(result_df)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name="predictions.csv", mime='text/csv')
