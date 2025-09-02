import argparse
import gzip
import random


MAX_URLS = 1e5
INPATH = "/home/ubuntu/repos/assignment4-data/cs336_data/enwiki-20240420-extracted_urls.txt"
OUTPATH = "/home/ubuntu/repos/assignment4-data/cs336_data/quality_classifier/sampled_positive_urls.txt"


def sample_positive_urls(inpath: str, outpath: str, max_urls: int = 1e1, max_to_process: int | None = None):
    reservoir = []
    random.seed(42)

    with open(inpath, "rt") as f:
        processed = 0

        for line in f:
            processed += 1
            if processed % 1e6 == 0:
                print(f"Processed {processed:,} lines", end="\r")

            url = line.strip()
            if not url:
                continue

            if len(reservoir) < max_urls:
                reservoir.append(url)
            else:
                r = random.randrange(processed)
                if r < max_urls:
                    reservoir[r] = url

            if max_to_process is not None and processed >= max_to_process:
                break

    with open(outpath, "w") as f:
        for url in reservoir:
            f.write(url + "\n")

    print(f"Wrote {len(reservoir)} URLs to {outpath}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-urls", type=int, default=MAX_URLS)
    parser.add_argument("--max-to-process", type=int, default=None)
    parser.add_argument("--inpath", type=str, default=INPATH)
    parser.add_argument("--outpath", type=str, default=OUTPATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_positive_urls(args.inpath, args.outpath, args.max_urls, args.max_to_process)