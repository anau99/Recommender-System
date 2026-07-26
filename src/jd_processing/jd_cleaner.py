import re

class JDCleaner:
    def __init__(self):
        pass

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()

        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'http\S+', '', text)
        
        text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    cleaner = JDCleaner()

    raw_jd_english = """
    ***URGENT HIRING***: Senior Backend Engineer (Remote) 🚀
    Contact: careers@tech.com | Req: Python 3.9, C++, .NET Core
    """
    
    print("-" * 50)
    print("ORIGINAL:")
    print(raw_jd_english)
    print("-" * 50)
    print("CLEANED:")
    print(cleaner.clean(raw_jd_english))