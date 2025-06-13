import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import re
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


#====================================================================================
# Process 1: Load Dataset
data = pd.read_csv("enronSpamSubset.csv")
df = data[['Body']].copy()  # dataframe with only the email text
print(df.head())

#=======================================================================
# Process 2: Study Dataset
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()

with open("P2.study_dataset", "w", encoding="utf-8") as f:
    f.write("\nHead:\n")
    for i, line in enumerate(df.head(3)[df.columns[0]]):
        f.write(f"{i}: {line}\n")
    f.write("\n\nShape:\n" + str(df.shape))
    f.write("\n\nColumn info:\n" + info)
    f.write("\n\nUnique values in columns:\n" + df.nunique().to_string())
    f.write("\n\nStatistics:\n" + df.describe().to_string())

#=============================================================================
# Process 3: Basic Preprocessing
with open("P3.Basic_Preprocessing", "w", encoding="utf-8") as f:
    f.write("\n\nMissing values: \n" + df.isnull().sum().to_string())
    f.write("\n\nDuplicates: \n" + str(df.duplicated().sum()))

#=================================================================================
# Process 4: Basic text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W+', ' ', text)     # Remove non-word characters
    return text

df['Body_clean'] = df['Body'].astype(str).apply(clean_text)
print(df.head())

with open("P4.After_cleaning", "w", encoding="utf-8") as f:
    f.write("\nHead: \n")
    for i, line in enumerate(df['Body_clean'].head(5)):
        f.write(f"{i}: {line}\n")

#=================================================================================

# Process 5: Labelling using pretrained model from Huggingface

if not os.path.exists("labeled_emails.csv"):

    # Load model and tokenizer
    model_name = "dima806/email-spam-detection-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to eval mode

    emails = df['Body_clean'].tolist()

    def predict_spam_labels_batch(email_texts, batch_size=32):
        labels = []
        for i in tqdm(range(0, len(email_texts), batch_size), desc="Batch predicting spam labels"):
            batch_texts = email_texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            batch_labels = torch.argmax(probs, dim=1).tolist()
            labels.extend(batch_labels)
        return labels

    labels = predict_spam_labels_batch(emails, batch_size=32)
    df['spam_label'] = labels

    print(df[['Body_clean', 'spam_label']].head())
    df.to_csv("labeled_emails.csv", index=False)

#=====================================================================

#Process 6: Load the labeled dataset and extract the desired columns

labeled_df = pd.read_csv("labeled_emails.csv")

df_labeled = labeled_df[['Body_clean', 'spam_label']].copy()

print(df_labeled.head())

buffer = io.StringIO()
df_labeled.info(buf=buffer)
info = buffer.getvalue()

with open("P6.Labeled_dataset", "w", encoding="utf-8") as f:
    f.write("\n\nShape:\n" + str(df_labeled.shape))
    f.write("\n\nColumn info:\n" + info)
    f.write("\n\nUnique values in columns:\n" + df_labeled.nunique().to_string())
    f.write("\n\nNo.of unique values in spam label column: \n"+df_labeled['spam_label'].value_counts().to_string())

#=========================================================
#Process 7: EDA

with open("P7.EDA", "w", encoding="utf-8") as f:
    # Spam vs Ham Distribution
    f.write("Spam Label Distribution:\n")
    f.write(df_labeled['spam_label'].value_counts().to_string())
    f.write("\n\nPercentage distribution:\n")
    f.write((df_labeled['spam_label'].value_counts(normalize=True) * 100).to_string())

    # Text length analysis
    df_labeled['text_length'] = df_labeled['Body_clean'].apply(len)
    f.write("\n\nText Length Statistics:\n")
    f.write(df_labeled['text_length'].describe().to_string())

#Bar plot for class distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df_labeled, x='spam_label')
plt.title('Spam vs Ham Distribution')
plt.xlabel('Label (0 = Ham, 1 = Spam)')
plt.ylabel('Count')
plt.savefig("P7.spam_distribution.png")
plt.close()

#Histogram of email lengths
plt.figure(figsize=(8,5))
sns.histplot(df_labeled['text_length'], bins=50, kde=True)
plt.title('Distribution of Email Lengths')
plt.xlabel('Length of Email (characters)')
plt.ylabel('Frequency')
plt.savefig("P7.email_length_distribution.png")
plt.close()

#========================================================================
#Process 8: Split into train and test set

tfidf = TfidfVectorizer(stop_words='english', max_features=3000)

X = tfidf.fit_transform(df_labeled['Body_clean'])
y = df_labeled['spam_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#=======================================================================

#Process 9: Train  and evaluate the model

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

# Save evaluation results
with open("P8.Model Evaluation", "w") as f:
    f.write("classification report: \n"+ classification_report(y_test, y_pred))
    f.write("\nAccuracy Score: {:.4f}\n".format(accuracy_score(y_test, y_pred)))

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("P9.ConfusionMatrix_LR.png")

#==========================================================================

#Process 10:save the model

joblib.dump(clf, "spam_classifier_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")


