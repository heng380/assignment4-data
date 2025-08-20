import regex as re


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