import argparse
import gzip
import os
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text
from cs336_data.lang_identify import identify_language
from cs336_data.idenifiable_text import *
from cs336_data.gopher import gopher_quality_filter

WARC_PATH = "/home/ubuntu/repos/assignment4-data/cs336_data/quality_classifier/unfiltered_positive_samples.warc.gz"
OUTPUT_PATH = "/home/ubuntu/repos/assignment4-data/cs336_data/quality_classifier/positive.txt"

def main():
    training_samples = []
    count = 0
    with gzip.open(WARC_PATH, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response and record.content_length > 0:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                count += 1
                if count % 100 == 0:
                    print (f"processed count: {count}")
                if not text.strip():
                    continue
                lang, score = identify_language(text)

                if lang != "en":
                    continue

                nsfw_label, nsfw_conf = identify_nsfw(text)
                if nsfw_label == "nsfw" or (nsfw_label == "non-nsfw" and nsfw_conf < 0.9):
                    continue

                toxic_label, toxic_conf = identify_hatespeech(text)
                if toxic_label == "toxic" or (toxic_label == "non-toxic" and toxic_conf < 0.9):
                    continue

                if gopher_quality_filter(text) == False:
                    continue

                training_sample = f"__label__positive {text.replace("\n", " ")}"
                training_samples.append(training_sample)

    with open(OUTPUT_PATH, "w") as f:
        f.writelines(training_samples)

    print (f"wrote {len(training_samples)} positive training samples into file.")

if __name__ == "__main__":
    main()