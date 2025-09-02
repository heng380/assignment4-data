import re

def gopher_quality_filter(text: str) -> bool:
    words = re.findall(r"\b\w[\w']*\b", text)
    num_words = len(words)

    if num_words < 50 or num_words > 100_000:
        return False
    
    total_length = sum(len(word) for word in words)
    mean_length = total_length / num_words
    if mean_length < 3 or mean_length > 10:
        return False

    lines = text.splitlines()
    if lines:
        ellipsis_line = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_line / len(lines) > 0.3:
            return False
        
    alph_words = sum(1 for word in words if re.search(r"[A-Za-z]", word))
    if alph_words / num_words < 0.8:
        return False
    
    return True