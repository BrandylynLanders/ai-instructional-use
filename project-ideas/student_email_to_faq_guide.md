
# Student Email to FAQ Generator: Step-by-Step Guide

This guide helps instructors automate the process of collecting student questions from email, analyzing them for trends, and generating FAQ entries using Python and ChatGPT.

---

## ✅ STEP 1: Set Up Outlook Rule

**Purpose**: Automatically move student emails to a specific folder.

**Rule Logic**:
- Condition 1: Sender address includes "@student.university.edu"
- Condition 2: Body contains "question" OR "help" OR "confused"
- Action: Move message to folder “Student Questions”

---

## ✅ STEP 2: Build Power Automate Flow

**Excel Setup**:
Create an Excel file with a table named `StudentEmails` with columns:

| Timestamp | From | Subject | Body | Keywords |

**Power Automate Steps**:
1. **Trigger**: When a new email arrives in folder “Student Questions”
2. **Get email content**: From, Subject, Body, Timestamp
3. **Optional Condition**: Check if the sender’s email contains student domain
4. **Detect Keywords**: Tag emails using Compose or Expression step
5. **Append Row**: Write to Excel file stored in OneDrive or SharePoint

---

## ✅ STEP 3: Python Script for Analysis & FAQ Suggestions

**Install Required Libraries**:

```bash
pip install pandas openai matplotlib wordcloud nltk scikit-learn
```

**Python Script**:

```python
import pandas as pd
import openai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("student_emails.xlsx")

# Clean body text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

df['clean_body'] = df['Body'].apply(clean_text)

# Vectorize text
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['clean_body'])

# Clustering
k = 5
model = KMeans(n_clusters=k, random_state=42)
df['topic'] = model.fit_predict(X)

# Sample emails per topic
for i in range(k):
    print(f"--- TOPIC {i} EXAMPLES ---")
    print(df[df['topic'] == i]['Body'].head(2).to_string(index=False))

# Word Cloud for each topic
for i in range(k):
    text = ' '.join(df[df['topic'] == i]['clean_body'])
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic {i}')
    plt.show()

# Optional: Generate FAQ Suggestions using ChatGPT
openai.api_key = 'your-openai-key'

def suggest_faqs(emails):
    joined = "\n\n".join(emails)
    prompt = f"""The following are student questions from a college course. Grouped by topic, please summarize the 3 most frequently asked questions and provide a possible FAQ entry for each.

Questions:
{joined}

Respond in plain text format as:

1. Question:
   FAQ Answer:

2. Question:
   FAQ Answer:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600
    )
    return response['choices'][0]['message']['content']

emails_topic_0 = df[df['topic'] == 0]['Body'].tolist()
print("Suggested FAQs for Topic 0:")
print(suggest_faqs(emails_topic_0[:5]))
```

---

## ✅ Optional Enhancements

- Schedule Python script weekly
- Output FAQs to Word/HTML or LMS
- Save results back to Excel or SharePoint
- Build a dashboard with Streamlit or Power BI

---

## ✅ Best File Type for This Guide

- **Markdown (.md)**: Best for easy readability and GitHub/wiki use.
- **PDF**: Best for sharing with non-technical staff.
- **HTML or Streamlit**: Ideal for turning into interactive dashboards.

Let us know if you’d like it in another format.
