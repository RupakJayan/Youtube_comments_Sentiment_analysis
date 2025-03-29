import os
import time
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Keys from .env file (Create a .env file with keys)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Validate API Keys
if not GEMINI_API_KEY or not YOUTUBE_API_KEY:
    raise ValueError("❌ API Keys are missing! Set them as environment variables.")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-001")  # Using a lighter model to save quota

# Function to fetch comments from YouTube
def fetch_youtube_comments(video_id, max_results=100):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    nextPageToken = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=nextPageToken
        )
        response = request.execute()
        comments.extend(response.get('items', []))
        nextPageToken = response.get('nextPageToken', None)
        if not nextPageToken:
            break

    return comments

# Function to analyze sentiment using TextBlob
def analyze_sentiment(comment_text):
    blob = TextBlob(comment_text)
    return blob.sentiment.polarity

# Function to classify comments
def classify_comments(df):
    df['sentiment_score'] = df['text'].apply(analyze_sentiment)
    df['label'] = df['sentiment_score'].apply(lambda x: 1 if x > 0 else 0 if x < 0 else 2)  # 1: Positive, 0: Negative, 2: Neutral

    classified_df = pd.DataFrame({
        'Good Comments': df[df['label'] == 1]['text'].reset_index(drop=True),
        'Bad Comments': df[df['label'] == 0]['text'].reset_index(drop=True),
        'Neutral Comments': df[df['label'] == 2]['text'].reset_index(drop=True)
    })

    return classified_df

# Function to train classifiers
def train_classifiers(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    return y_test, y_pred_lr, y_pred_nb, lr_model, nb_model, vectorizer

# Function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊 {model_name} Model Performance:")
    print(f"🔹 Accuracy: {accuracy:.2f}")
    print(f"🔹 Precision: {precision:.2f}")
    print(f"🔹 Recall: {recall:.2f}")
    print(f"🔹 F1 Score: {f1:.2f}")

# Function to generate polite suggestions for bad comments
def generate_suggestions(df):
    suggestions = []
    bad_comments = df['Bad Comments'].dropna().head(10)  # Limit to 10 comments

    for comment in bad_comments:
        prompt = f"Rewrite this comment in a polite way without changing its meaning: {comment}"
        
        for attempt in range(3):  # Retry mechanism
            try:
                response = model.generate_content(prompt)
                suggestions.append(response.text)
                break  # Exit loop on success
            except Exception as e:
                print(f"⚠️ Error with Gemini AI (Attempt {attempt+1}): {e}")
                time.sleep(5)  # Wait before retrying
        else:
            suggestions.append("⚠️ Failed to generate suggestion.")

    suggestions_df = pd.DataFrame({'Bad Comments': bad_comments, 'Suggested Comments': suggestions})
    suggestions_df.to_excel("suggested_comments.xlsx", index=False)
    print("✅ Suggestions saved in suggested_comments.xlsx.")

# Function to generate PDF report
def generate_pdf_report(results):
    pdf_filename = "final_output.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    for result in results:
        elements.append(Paragraph(f"Analysis for {result['title']}", styles['Title']))
        elements.append(Paragraph("Model Performance:", styles['Heading2']))
        elements.append(Paragraph(result['text'], styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    print(f"📄 Final report saved as {pdf_filename}.")

# Main execution
if __name__ == "__main__":
    videos = [
        {"video_id": "mwKJfNYwvm8", "title": "I Built 100 Wells In Africa"},
        {"video_id": "CMlKCR7jVzM", "title": "Gaza ceasefire"},
        {"video_id": "rgSr1NkFV-g", "title": "YOU LAUGH YOU LOSE"}
    ]

    for video in videos:
        print(f"\n📥 Fetching comments for: {video['title']}...")
        comments = fetch_youtube_comments(video['video_id'])
        raw_df = pd.DataFrame([{'text': item['snippet']['topLevelComment']['snippet']['textDisplay']} for item in comments])

        classified_df = classify_comments(raw_df)
        y_test, y_pred_lr, y_pred_nb, _, _, _ = train_classifiers(raw_df)
        evaluate_model(y_test, y_pred_lr, "Logistic Regression")
        evaluate_model(y_test, y_pred_nb, "Naïve Bayes")

        generate_suggestions(classified_df)

    generate_pdf_report([])  # Empty report for now
