from textblob import TextBlob

line = TextBlob("I would hate to learn AI")

print(f"Polarity: {line.sentiment.polarity}")

p = line.sentiment.polarity

if p == -1.0:
    print("100% negative")
elif p == 0.0:
    print("neutral")
if p < 1.0 and p > 0:
    print("somewhere positive")
elif p == 1.0:
    print("100% positive")
