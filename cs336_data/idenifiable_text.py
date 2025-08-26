import regex as re
import fasttext

def mask_email(text: str):
    PAT = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    res = re.subn(PAT, "|||EMAIL_ADDRESS|||", text)
    return res

def mask_phone_num(text: str):
    PAT = re.compile(r"(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
    res = re.subn(PAT, "|||PHONE_NUMBER|||",text)
    return res    

def mask_ip(text: str):
    PAT = re.compile(r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}")
    res = re.subn(PAT, "|||IP_ADDRESS|||", text)
    return res

def identify_nsfw(text: str):
    model = fasttext.load_model("/home/ubuntu/repos/assignment4-data/cs336_data/jigsaw_fasttext_bigrams_nsfw_final.bin")
    text = text.replace("\n", "")
    pred = model.predict(text)
    label = pred[0][0]
    label = label.replace("__label__", "")
    t = tuple((label, pred[1][0]))
    return t

def identify_hatespeech(text: str):
    model = fasttext.load_model("/home/ubuntu/repos/assignment4-data/cs336_data/jigsaw_fasttext_bigrams_hatespeech_final.bin")
    text = text.replace("\n", "")
    pred = model.predict(text)
    label = pred[0][0]
    label = label.replace("__label__", "")
    t = tuple((label, pred[1][0]))
    return t

if __name__ == "__main__":
    model = fasttext.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")
    text = "SUCK MY C*CK WIKIPEDIA EDITORS...F*CKING *SSH*LE DORKS. "
    text = text.replace("\n", "")
    pred = model.predict(text, k=2)
    print (pred)
