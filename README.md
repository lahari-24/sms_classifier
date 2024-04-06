import re
import math

# SMS dataset (spam and non-spam)
sms_data = [
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", "spam"),
    ("Sorry, I'll call later", "ham"),
    ("Wanna grab a coffee later?", "ham"),
    ("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18", "spam"),
    ("Even my brother is not like to speak with me. They treat me like aids patent.", "ham")
]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Function to calculate word frequencies
def calculate_word_freq(data):
    word_freq = {}
    total_docs = len(data)
    for sms, label in data:
        processed_sms = preprocess_text(sms)
        words = processed_sms.split()
        for word in words:
            if word not in word_freq:
                word_freq[word] = {"spam": 0, "ham": 0}
            word_freq[word][label] += 1
    return word_freq, total_docs

# Function to calculate probabilities
def calculate_probabilities(word_freq, total_docs):
    probabilities = {}
    for word, freq in word_freq.items():
        prob_spam = (freq["spam"] + 1) / (total_docs / 2 + 1)  # Add-one smoothing
        prob_ham = (freq["ham"] + 1) / (total_docs / 2 + 1)  # Add-one smoothing
        probabilities[word] = {"spam": prob_spam, "ham": prob_ham}
    return probabilities

# Function to classify SMS
def classify_sms(sms, probabilities):
    processed_sms = preprocess_text(sms)
    words = processed_sms.split()
    prob_spam_sms = 0
    prob_ham_sms = 0
    for word in words:
        if word in probabilities:
            prob_spam_sms += math.log(probabilities[word]["spam"])
            prob_ham_sms += math.log(probabilities[word]["ham"])
    prob_spam_sms += math.log(0.5)  # Prior probability of spam
    prob_ham_sms += math.log(0.5)  # Prior probability of ham
    if prob_spam_sms > prob_ham_sms:
        return "spam"
    else:
        return "ham"

# Train the model
word_freq, total_docs = calculate_word_freq(sms_data)
probabilities = calculate_probabilities(word_freq, total_docs)

# Test the model
test_sms = "You have won a free trip! Claim now."
classification = classify_sms(test_sms, probabilities)
print("Test SMS:", test_sms)
print("Classification:", classification)
