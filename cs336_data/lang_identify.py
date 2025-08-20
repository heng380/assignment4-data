import fasttext
from typing import Tuple

model = fasttext.load_model("/home/ubuntu/repos/assignment4-data/cs336_data/lid.176.bin")

def identify_language(text: str) -> Tuple[str, float]:
    labels, probs = model.predict("".join(text.split()), k=1)
    lang = labels[0].replace("__label__", "")
    prob = float(probs[0])

    return lang, prob


if __name__ == "__main__":
    labels, probs = (model.predict("hello", k=5))
    print (labels)
    print (probs)
