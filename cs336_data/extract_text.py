from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType
import gzip
from .lang_identify import identify_language
from .idenifiable_text import *
from .gopher import *
def extract_text(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding, errors="replace")

    return extract_plain_text(html_str)


if __name__ == "__main__":
    warc_path = "/home/ubuntu/repos/assignment4-data/webdata/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

    count = 0
    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response and record.content_length > 0:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                print (f"warc id: {record.record_id}")

                print (text[:100].replace(" ",""))
                print (identify_language(text))
                print (identify_hatespeech(text))
                print (identify_nsfw(text))
                print (gopher_quality_filter(text))
                print ("----------------------------")
                count += 1
                if count == 20:
                    break