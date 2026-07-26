
import re

class ExperienceMapper:
    def __init__(self):
        self.LEVEL_THRESHOLDS = {
            'Beginner': 0,
            'Intermediate': 2,
            'Advanced': 5,
            'Expert': 10
        }

        self.year_pattern = re.compile(
            r"""
            (?P<min>\d+)\s*[-–]\s*(?P<max>\d+)\s*(?:\+)?\s*(?:year|years|yrs|yoe)    
            |
            (?P<single>\d+)\s*\+\s*(?:year|years|yrs|yoe)                              
            |
            (?P<plain>\d+)\s*(?:year|years|yrs|yoe)                                   
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def map_experience(self, clean_jd_text):
        text = clean_jd_text.lower()

        years_found = []

        for match in self.year_pattern.finditer(text):
            if match.group("min") and match.group("max"):
                years_found.append(int(match.group("min")))
                years_found.append(int(match.group("max")))
            elif match.group("single"):
                years_found.append(int(match.group("single")))
            elif match.group("plain"):
                years_found.append(int(match.group("plain")))

        max_years = max(years_found) if years_found else 0

        if max_years >= self.LEVEL_THRESHOLDS['Expert']:
            level = 'Expert'
        elif max_years >= self.LEVEL_THRESHOLDS['Advanced']:
            level = 'Advanced'
        elif max_years >= self.LEVEL_THRESHOLDS['Intermediate']:
            level = 'Intermediate'
        else:
            level = 'Beginner'

        if max_years == 0:
            if "senior" in text or "lead" in text:
                level = 'Advanced'
            elif "mid" in text or "intermediate" in text:
                level = 'Intermediate'
            elif "junior" in text or "entry" in text or "fresher" in text:
                level = 'Beginner'

        return {
            "years": max_years,
            "level": level
        }
