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

WARC_PATH = "/home/ubuntu/repos/assignment4-data/webdata/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
OUTPUT_PATH = "/home/ubuntu/repos/assignment4-data/cs336_data/quality_classifier/negative.txt"

def main():
    training_samples = []
    count = 0
    with gzip.open(WARC_PATH, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response and record.content_length > 0:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                if not text.strip():
                    continue
                count += 1
                if count % 100 == 0:
                    print (f"{count} samples are processed")

                training_sample = f"__label__negative {text.replace("\n", " ")}"
                training_samples.append(training_sample)
                if count == 10000:
                    break

    with open(OUTPUT_PATH, "w") as f:
        f.writelines(training_samples)

    print (f"wrote {len(training_samples)} positive training samples into file.")

if __name__ == "__main__":
    main()