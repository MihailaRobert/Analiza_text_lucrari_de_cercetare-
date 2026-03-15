# 4_baseline_tfidf.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

#1. CONFIGURARE CALE
base_path = os.path.dirname(os.path.abspath(__file__))
input_filename = "1_output_all_articles_paragraph_dataset_with_sections.csv"
full_path = os.path.join(base_path, input_filename)

if not os.path.exists(full_path):
    print(f"EROARE: Nu găsesc {input_filename}. Rulează 1_creare_dataset.py.")
    exit()

print(f"--> Încărcăm datele din: {input_filename}")
paras = pd.read_csv(full_path)

# Definim secțiunile pe care lucrăm
PAIR_SECTIONS = ["Introduction", "Conclusion"]


# LOGICA MATEMATICĂ 
# Build binary dataset: Introduction vs Conclusion
paras_bin = paras[paras["section"].isin(PAIR_SECTIONS)].copy()
paras_bin["label"] = paras_bin["section"].map({"Introduction": 0, "Conclusion": 1})
X = paras_bin["paragraph_text"].astype(str)
y = paras_bin["label"].values

# Split 
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorizare
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train_p)
X_test_vec  = tfidf.transform(X_test_p)

# Antrenare
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train_p)
pred_lr = clf.predict(X_test_vec)

# Calcul Metrici
acc_lr = accuracy_score(y_test_p, pred_lr)
p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_test_p, pred_lr, average='binary')



# SALVARE REZULTATE CA IMAGINI (.PNG)

# --- 1. SALVARE TEXT METRICI CA IMAGINE ---
print("Generăm imaginea cu textul metricilor...")

plt.figure(figsize=(10, 6)) # Imagine albă goală
plt.axis('off') 

# Construim textul 
text_output = f"Model inițial — TF-IDF + Logistic Regression (paragraf)\n"
text_output += f"Accuracy:  {acc_lr}\n"
text_output += f"Precision: {p_lr}\n"
text_output += f"Recall:    {r_lr}\n"
text_output += f"F1:        {f1_lr}\n\n"
text_output += "Detailed Classification Report:\n"
text_output += classification_report(y_test_p, pred_lr)

# Scriem textul pe imagine 
plt.text(0.02, 0.95, text_output, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', fontfamily='monospace')

output_text_png = "4_output_metrics_text.png"
plt.savefig(os.path.join(base_path, output_text_png), bbox_inches='tight', dpi=150)
plt.close()
print(f" [Salvat] {output_text_png}")


#2. SALVARE MATRICE CONFUZIE CA IMAGINE
print("Generăm imaginea matricei de confuzie...")

cm = confusion_matrix(y_test_p, pred_lr)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — TF-IDF Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")

output_cm_png = "4_output_confusion_matrix_baseline.png"
plt.savefig(os.path.join(base_path, output_cm_png), bbox_inches='tight', dpi=300)
plt.close()
print(f" [Salvat] {output_cm_png}")

print("\n" + "="*50)
print(" Rezultatele în format .png în folder.")
print("="*50)