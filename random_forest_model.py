import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

HYPERPARAMETER_TUNING = True
USE_PCA = True
N_PCA_COMPONENTS_LIST = [10, 50, 100, 200, 300, 400]

TRAIN_PATH = "../data/trainset.csv"
TEST_PATH = "../data/testset.csv"


def plot_class_distribution(train_df, test_df, label_col, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    train_counts = train_df[label_col].value_counts()
    test_counts = test_df[label_col].value_counts()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    train_counts.plot(kind='bar', color='skyblue')
    plt.title('Train Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    for i, v in enumerate(train_counts):
        plt.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)
    plt.subplot(1, 2, 2)
    test_counts.plot(kind='bar', color='salmon')
    plt.title('Test Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    for i, v in enumerate(test_counts):
        plt.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "class_distribution_train_test.png"))
    plt.close()


def save_metrics(metrics, path):
    pd.DataFrame([metrics]).to_csv(path, index=False)


def plot_confusion_matrix(cm, labels, title, path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true, y_proba, labels, path, title):
    plt.figure(figsize=(8, 7))
    y_bin = label_binarize(y_true, classes=range(len(labels)))
    for i, label in enumerate(labels):
        if np.sum(y_bin[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        try:
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        except ValueError:
            auc = float('nan')
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_importance(importances, feature_names, path, title, top_n=20):
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.title(title)
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(indices)))
    plt.barh(range(len(indices)), importances[indices], color=colors, align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    filename = title.replace(" ", "_").lower() + ".png"
    save_path = os.path.join(path, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")


def run_training(
    train_df,
    test_df,
    label_col,
    feature_cols,
    result_base,
    use_pca=False,
    n_pca_components_list=None,
):
    df = train_df.copy()
    non_numeric = [col for col in feature_cols if not np.issubdtype(df[col].dtype, np.number)]
    if non_numeric:
        print("Dropping non-numeric columns:", non_numeric)
        feature_cols = [col for col in feature_cols if col not in non_numeric]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = df[feature_cols]
    y = df[label_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if use_pca and n_pca_components_list:
        for n_components in n_pca_components_list:
            print(f"\nRunning Random Forest with PCA, n_components={n_components}")
            pca_result_base = os.path.join(result_base, f"pca{n_components}")
            os.makedirs(pca_result_base, exist_ok=True)

            num_features = min(n_components, X.shape[1])
            feature_names = [f'PC{i+1}' for i in range(num_features)]

            run_test_split_and_model(
                df,
                X,
                y,
                label_encoder,
                feature_names,
                pca_result_base,
                test_df,
                feature_cols,
                use_pca=True,
                n_pca_components=n_components,
            )
    else:
        os.makedirs(result_base, exist_ok=True)
        run_test_split_and_model(
            df,
            X,
            y,
            label_encoder,
            feature_cols,
            result_base,
            test_df,
            feature_cols,
            use_pca=False,
        )


def run_test_split_and_model(
    df,
    X,
    y,
    label_encoder,
    feature_names,
    result_base,
    test_df,
    original_feature_cols,
    use_pca=False,
    n_pca_components=None,
):
    if 'test' in df.columns:
        test_names = df['test'].unique()
    else:
        test_names = ['all']

    for test_name in test_names:
        if test_name == 'all':
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            test_dir = os.path.join(result_base, 'all')
        else:
            train_mask = df['test'] != test_name
            val_mask = df['test'] == test_name
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            test_dir = os.path.join(result_base, str(test_name))

        os.makedirs(test_dir, exist_ok=True)

        val_df = pd.DataFrame({label_col: y_val})
        plot_class_distribution(
            pd.DataFrame({label_col: y_train}),
            val_df,
            label_col,
            test_dir,
        )

        def build_pipeline(model, use_pca, n_pca_components):
            steps = [('scaler', StandardScaler())]
            if use_pca:
                num_features = X.shape[1]
                pca_n_components = min(n_pca_components, num_features)
                steps.append(('pca', PCA(n_components=pca_n_components, random_state=42)))
            steps.append(('classifier', model))
            return Pipeline(steps)

        print(f"\nTraining Random Forest for test: {test_name}")
        rf_model = RandomForestClassifier(random_state=42, n_jobs=4)
        rf_pipe = build_pipeline(rf_model, use_pca, n_pca_components)

        param_grid_rf = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5],
        }

        if HYPERPARAMETER_TUNING:
            rf_grid = GridSearchCV(
                rf_pipe,
                param_grid_rf,
                cv=3,
                n_jobs=4,
                return_train_score=True,
            )
            rf_grid.fit(X_train, y_train)
            rf = rf_grid.best_estimator_
            pd.DataFrame([rf_grid.best_params_]).to_csv(
                os.path.join(test_dir, "rf_best_params.csv"),
                index=False,
            )
            pd.DataFrame(rf_grid.cv_results_).to_csv(
                os.path.join(test_dir, "rf_all_cv_results.csv"),
                index=False,
            )
        else:
            rf = rf_pipe.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_val)
        y_proba_rf = rf.predict_proba(X_val)
        cm_rf = confusion_matrix(y_val, y_pred_rf)
        labels = label_encoder.classes_

        metrics_rf = {
            'Accuracy': accuracy_score(y_val, y_pred_rf),
            'Precision': precision_score(y_val, y_pred_rf, average='macro'),
            'Recall': recall_score(y_val, y_pred_rf, average='macro'),
            'F1_Score': f1_score(y_val, y_pred_rf, average='macro'),
        }
        save_metrics(metrics_rf, os.path.join(test_dir, "rf_metrics.csv"))
        plot_confusion_matrix(
            cm_rf,
            labels,
            "Random Forest Confusion Matrix",
            os.path.join(test_dir, "rf_confusion_matrix.png"),
        )
        plot_roc_curve(
            y_val,
            y_proba_rf,
            labels,
            os.path.join(test_dir, "rf_roc_curve.png"),
            "Random Forest ROC Curve",
        )

        if not use_pca:
            importances = rf.named_steps['classifier'].feature_importances_
            plot_feature_importance(
                importances,
                feature_names,
                test_dir,
                "Random Forest Feature Importance",
            )
        else:
            print(
                "Feature importance for Random Forest with PCA is not directly interpretable from the model and is not plotted."
            )

        pd.DataFrame(cm_rf, index=labels, columns=labels).to_csv(
            os.path.join(test_dir, "rf_confusion_matrix.csv")
        )

        X_test_external = test_df[original_feature_cols]
        y_test_external = test_df[label_col]
        y_test_external = label_encoder.transform(y_test_external)

        y_pred_rf_test = rf.predict(X_test_external)
        y_proba_rf_test = rf.predict_proba(X_test_external)
        cm_rf_test = confusion_matrix(y_test_external, y_pred_rf_test)
        metrics_rf_test = {
            'Accuracy': accuracy_score(y_test_external, y_pred_rf_test),
            'Precision': precision_score(y_test_external, y_pred_rf_test, average='macro'),
            'Recall': recall_score(y_test_external, y_pred_rf_test, average='macro'),
            'F1_Score': f1_score(y_test_external, y_pred_rf_test, average='macro'),
        }
        save_metrics(metrics_rf_test, os.path.join(test_dir, "rf_metrics_test.csv"))
        plot_confusion_matrix(
            cm_rf_test,
            labels,
            "Random Forest Confusion Matrix (Test Set)",
            os.path.join(test_dir, "rf_confusion_matrix_test.png"),
        )
        plot_roc_curve(
            y_test_external,
            y_proba_rf_test,
            labels,
            os.path.join(test_dir, "rf_roc_curve_test.png"),
            "Random Forest ROC Curve (Test Set)",
        )
        pd.DataFrame(cm_rf_test, index=labels, columns=labels).to_csv(
            os.path.join(test_dir, "rf_confusion_matrix_test.csv")
        )

        print(f"Random Forest results for validation and test set saved in: {test_dir}")


def feature_selection_and_training(
    feature_csv,
    analysis_groups,
    result_folder,
    train_df,
    test_df,
    label_col,
):
    feature_df = pd.read_csv(feature_csv)
    for group_name, analysis_values in analysis_groups.items():
        print(f"\n=== Running Random Forest for feature group: {group_name} ===")
        filtered = feature_df[feature_df['analysis'].isin(analysis_values)]
        selected_features = set(filtered['feature'].values)
        print(f"Selected features ({group_name}): {len(selected_features)}")
        train_features = set(train_df.columns) - set([label_col, 'Sample_ID', 'test'])
        feature_cols = list(selected_features & train_features)
        print(
            f"Number of selected features after filtering and intersection: {len(feature_cols)}"
        )
        result_base = os.path.join(result_folder, group_name)
        run_training(
            train_df,
            test_df,
            label_col,
            feature_cols,
            result_base,
            USE_PCA,
            N_PCA_COMPONENTS_LIST if USE_PCA else None,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_df = pd.read_csv(TRAIN_PATH, index_col=0)
    test_df = pd.read_csv(TEST_PATH, index_col=0)
    test_df = test_df[train_df.columns]
    train_df = train_df.fillna('Unclassified')
    test_df = test_df.fillna('Unclassified')
    label_col = "prediction"
    plot_class_distribution(train_df, test_df, label_col, "result_rf")

    anova_groups = {
        "Excellent": ["Excellent"],
        "Good": ["Good"],
        "Excellent_Good": ["Excellent", "Good"],
    }
    feature_selection_and_training(
        "../features/anova_f_test.csv",
        anova_groups,
        "../result/anova_f_test_rf",
        train_df,
        test_df,
        label_col,
    )

    mutual_groups = {
        "Good": ["Good"],
        "Moderate": ["Moderate"],
        "Good_Moderate": ["Good", "Moderate"],
    }
    feature_selection_and_training(
        "../features/mutual_information.csv",
        mutual_groups,
        "../result/mutual_information_rf",
        train_df,
        test_df,
        label_col,
    )

    flc_groups = {
        "Good": ["Good"],
        "Moderate": ["Moderate"],
        "Good_Moderate": ["Good", "Moderate"],
    }
    feature_selection_and_training(
        "../features/feature_label_correlation.csv",
        flc_groups,
        "../result/feature_label_correlation_rf",
        train_df,
        test_df,
        label_col,
    )

