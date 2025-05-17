from emotion_detector import EmotionDetector

def main():
    # Initialize the emotion detector
    detector = EmotionDetector()
    
    # Train the model
    print("Hi! I'm your friendly emotion detector. Let's see how you're feeling today.\n")
    detector.train()
    
    # Save the trained model
    detector.save_model()
    
    # Test some example texts
    test_texts = [
        "I'm feeling really happy today!",
        "This is making me so angry!",
        "I'm feeling quite sad and lonely",
        "What a wonderful day it is!",
        "I can't believe how frustrating this is!"
    ]
    
    reactions = {
        "happy": "That's awesome! Keep smiling 😊",
        "sad": "I'm here for you. Things will get better 💙",
        "angry": "Take a deep breath. It's okay to feel angry sometimes 😠"
    }
    
    print("\nLet's check your emotions:\n" + "-" * 50)
    
    for text in test_texts:
        result = detector.predict_emotion(text)
        emotion = result['predicted_emotion']
        print(f"\nYou said: \"{text}\"")
        print(f"I sense that you might be feeling: {emotion.upper()}")
        print(reactions.get(emotion, ""))
        # Optionally, show the most likely emotion only, or a friendly summary of scores
        # print(f"(Just so you know, here's how sure I am: {int(result['confidence_scores'][emotion]*100)}%)")

if __name__ == "__main__":
    main() 