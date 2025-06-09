import fasttext
import os

class LanguageDetector:
    """
    Handles language detection using a pre-trained fastText model.
    """
    def __init__(self, model_path="lid.176.bin"):
        """
        Initializes the LanguageDetector, loading the fastText model.

        Args:
            model_path (str): The path to the pre-trained fastText model file.
        
        Raises:
            FileNotFoundError: If the model file cannot be found.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FastText model not found at '{model_path}'. "
                "Please download 'lid.176.bin' from the fastText website "
                "and place it in the project root directory."
            )
        
        print("Loading fastText language detection model...")
        self.model = fasttext.load_model(model_path)
        print("Language detection model loaded.")

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            str: The detected language code (e.g., 'en', 'te').
                 Returns 'en' as a fallback if detection fails.
        """
        try:
            # The model returns predictions like '__label__en'
            predictions = self.model.predict(text.replace("\n", " "), k=1)
            language_code = predictions[0][0].split('__')[-1]
            return language_code
        except Exception as e:
            print(f"Language detection failed: {e}")
            return "en" # Fallback to English

if __name__ == '__main__':
    # You must have the lid.176.bin model in your project root to run this
    try:
        detector = LanguageDetector()
        
        text_en = "Hello, how are you?"
        lang_en = detector.detect_language(text_en)
        print(f"Text: '{text_en}' -> Detected Language: {lang_en}")

        text_te = "మీరు ఎలా ఉన్నారు?"
        lang_te = detector.detect_language(text_te)
        print(f"Text: '{text_te}' -> Detected Language: {lang_te}")

        text_mixed = "That's a very మంచి idea."
        lang_mixed = detector.detect_language(text_mixed)
        print(f"Text: '{text_mixed}' -> Detected Language: {lang_mixed} (dominant)")

    except FileNotFoundError as e:
        print(e) 