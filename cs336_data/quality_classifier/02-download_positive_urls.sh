wget --timeout=5 \
  --tries=3 \
  -i sampled_positive_urls.txt \
  --warc-file=unfiltered_positive_samples \
  -O /dev/null