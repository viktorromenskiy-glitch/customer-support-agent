"""
Обучение финальных (продакшен) классификаторов категории и приоритета
на эмбеддингах Voyage вместо TF-IDF.

Переиспользует уже посчитанные эмбеддинги (kb_embeddings.npy + test_embeddings.npy
вместе покрывают все 20 000 записей) - новых вызовов API не делает.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

df = pd.read_csv("../archive/twitter_support_with_priority_FINAL.csv")
df["request_created_at"] = pd.to_datetime(df["request_created_at"], errors="coerce", utc=True)
df = df.sort_values("request_created_at").reset_index(drop=True)
print(f"Всего записей: {len(df):,}")

# Эмбеддинги в том же порядке (по времени) - train (первые 16000) + test (последние 4000)
train_vectors = np.load("../archive/kb_embeddings.npy")
test_vectors = np.load("../archive/test_embeddings.npy")
all_vectors = np.vstack([train_vectors, test_vectors])
assert len(all_vectors) == len(df), "Размеры не совпадают - проверьте файлы эмбеддингов"
print(f"Эмбеддингов загружено: {len(all_vectors):,} (без новых вызовов API)")

# Кодировщик компании - обучаем на всех данных для финальной продакшен-версии
ohe = OneHotEncoder(handle_unknown="ignore")
company_encoded = ohe.fit_transform(df[["company"]])


def build_features(embeddings, company_encoded_part, is_dm_escalation):
    """Единая функция сборки признаков - используется и здесь, и в агенте."""
    return hstack([
        csr_matrix(embeddings),
        company_encoded_part,
        csr_matrix(np.array(is_dm_escalation).reshape(-1, 1)),
    ])


X_full = build_features(all_vectors, company_encoded, df["is_dm_escalation"].values)

# Честная оценка на отложенной по времени части (как и раньше) - для отчётности
split_idx = int(len(df) * 0.8)
X_train_eval, X_test_eval = X_full[:split_idx], X_full[split_idx:]

for target_col, name, save_name in [
    ("category_llm", "КАТЕГОРИЯ", "category_classifier_embeddings.joblib"),
    ("priority_llm", "ПРИОРИТЕТ", "priority_classifier_embeddings.joblib"),
]:
    y_full = df[target_col]
    y_train_eval, y_test_eval = y_full.iloc[:split_idx], y_full.iloc[split_idx:]

    # Честная оценка (train на первых 80% по времени, тест на последних 20%)
    eval_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    eval_clf.fit(X_train_eval, y_train_eval)
    pred = eval_clf.predict(X_test_eval)
    print(f"\n=== {name} (честная оценка на отложенной по времени части) ===")
    print(f"Accuracy: {accuracy_score(y_test_eval, pred):.4f}  F1 macro: {f1_score(y_test_eval, pred, average='macro'):.4f}")

    # Финальная продакшен-версия - обучаем на ВСЕХ данных
    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    final_clf.fit(X_full, y_full)
    joblib.dump({"ohe": ohe, "classifier": final_clf}, save_name)
    print(f"✓ Финальный классификатор сохранён в {save_name}")

print("\n✓ Готово. Оба классификатора обучены на эмбеддингах и сохранены.")
