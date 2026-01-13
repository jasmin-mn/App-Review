# -----------------------------
# Google Play Reviews Sentiment Analysis (Fixed Version)
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# -----------------------------
# 0️⃣ Download NLTK Resources
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# -----------------------------
# 1️⃣ Initialize NLTK tools
# -----------------------------
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')  # avoids punkt issues

# -----------------------------
# 2️⃣ Create Sample Reviews
# -----------------------------
reviews_df = pd.DataFrame({
    "App": [
        "WhatsApp", "WhatsApp",
        "Instagram", "Instagram",
        "Facebook", "Snapchat",
        "TikTok", "TikTok"
    ],
    "Review": [
        "Great app, very useful!",
        "Excellent messaging experience",
        "Love this app but too many ads",
        "Too many ads and bugs",
        "Good but slow sometimes",
        "Fun to use and creative",
        "Very entertaining videos",
        "Needs improvement, app crashes"
    ]
})

print("Original Reviews:\n", reviews_df)

# -----------------------------
# 3️⃣ Text Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

reviews_df["Clean_Review"] = reviews_df["Review"].apply(preprocess)

# Show preprocessing examples
print("\nPreprocessing examples:")
for review, clean in zip(reviews_df["Review"], reviews_df["Clean_Review"]):
    print(f"Original: {review}")
    print(f"Cleaned:  {clean}\n")

# -----------------------------
# 4️⃣ Sentiment Polarity
# -----------------------------
reviews_df["Sentiment_Polarity"] = reviews_df["Clean_Review"].apply(lambda x: sia.polarity_scores(x)['compound'])

# -----------------------------
# 5️⃣ Convert to Sentiment Label (with better thresholds)
# -----------------------------
def sentiment_label(score):
    if score > 0.1:       # stronger positive
        return "Positive"
    elif score < -0.1:    # stronger negative
        return "Negative"
    else:
        return "Neutral"

reviews_df["Sentiment"] = reviews_df["Sentiment_Polarity"].apply(sentiment_label)

# -----------------------------
# 6️⃣ Aspect-Based Sentiment Extraction
# -----------------------------
aspects = ["ads", "bugs", "crash", "slow", "performance"]

def extract_aspects(review):
    found = []
    for aspect in aspects:
        if aspect.lower() in review.lower():
            found.append(aspect)
    return found

reviews_df["Aspects"] = reviews_df["Review"].apply(extract_aspects)

print("\nProcessed Reviews with Sentiment and Aspects:\n")
print(reviews_df)

# -----------------------------
# 7️⃣ Average Sentiment per App
# -----------------------------
app_sentiment = reviews_df.groupby("App")["Sentiment_Polarity"].mean()
print("\nAverage Sentiment per App:\n")
print(app_sentiment)

# -----------------------------
# 8️⃣ Sentiment Distribution Visualization
# -----------------------------
sentiment_counts = reviews_df["Sentiment"].value_counts()
categories = ["Positive", "Neutral", "Negative"]  # ensure all categories appear
counts = [sentiment_counts.get(cat, 0) for cat in categories]

plt.figure(figsize=(6,5))
plt.bar(categories, counts, color=['green','gray','red'])
plt.title("Sentiment Distribution (NLTK VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

# -----------------------------
# 9️⃣ Aspect Frequency Visualization
# -----------------------------
aspect_series = reviews_df.explode("Aspects")["Aspects"].dropna()
aspect_counts = aspect_series.value_counts()

plt.figure(figsize=(6,5))
plt.bar(aspect_counts.index, aspect_counts.values, color='orange')
plt.title("Most Common Complaint Aspects")
plt.xlabel("Aspect")
plt.ylabel("Frequency")
plt.show()
