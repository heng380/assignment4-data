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

count = 0
with gzip.open(WARC_PATH, "rb") as stream:
    for record in ArchiveIterator(stream):
        if record.record_type == WarcRecordType.response and record.content_length > 0:
            html_bytes = record.reader.read()
            count += 1

print (count)