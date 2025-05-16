#!/usr/bin/env bash
#
# upload_all_to_s3.sh
# -------------------
# Fully automated uploader for logs, models, and metrics to S3,
# with IMDSv2 metadata and structured S3 prefixes.
#
set -euo pipefail

###
### 1. Determine Base Directory & Data Paths
###
# Resolve this script’s directory so we can reference data subfolders reliably
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                     # script location  [oai_citation_attribution:4‡Man7](https://man7.org/linux/man-pages/man1/date.1.html?utm_source=chatgpt.com)
LOG_DIR="$BASE_DIR/logs"                                                     # logs folder  [oai_citation_attribution:5‡AWS Documentation](https://docs.aws.amazon.com/snowball/latest/developer-guide/imds-code-examples.html?utm_source=chatgpt.com)
MODEL_DIR="$BASE_DIR/models"                                                 # models folder  [oai_citation_attribution:6‡AWS Documentation](https://docs.aws.amazon.com/snowball/latest/developer-guide/imds-code-examples.html?utm_source=chatgpt.com)
METRIC_DIR="$BASE_DIR/metrics"                                               # metrics folder  [oai_citation_attribution:7‡AWS Documentation](https://docs.aws.amazon.com/snowball/latest/developer-guide/imds-code-examples.html?utm_source=chatgpt.com)

###
### 2. Fetch EC2 Metadata (IMDSv2)
###
# Acquire a session token (valid 6 h) for secure metadata access
TOKEN=$(curl -sSX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")                          # IMDSv2 token  [oai_citation_attribution:8‡AWS Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html?utm_source=chatgpt.com)

# Retrieve Instance ID and local IPv4
INSTANCE_ID=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)                       # EC2 Instance ID  [oai_citation_attribution:9‡AWS Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html?utm_source=chatgpt.com)

IP_ADDRESS=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/local-ipv4)                       # Private IPv4  [oai_citation_attribution:10‡AWS Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html?utm_source=chatgpt.com)

###
### 3. Compute Prefix Variables
###
DATE=$(date +%F)                                                             # ISO date (YYYY‑MM‑DD)  [oai_citation_attribution:11‡Stack Overflow](https://stackoverflow.com/questions/1401482/yyyy-mm-dd-format-date-in-shell-script?utm_source=chatgpt.com)
RUN_ID="run_$(date +%s)"                                                     # Unique run ID by epoch  [oai_citation_attribution:12‡Stack Overflow](https://stackoverflow.com/questions/1401482/yyyy-mm-dd-format-date-in-shell-script?utm_source=chatgpt.com)

S3_BUCKET="aws-gpu-monitoring-logs"

# Construct the common S3 path prefix
S3_PREFIX="$DATE/$RUN_ID/$INSTANCE_ID/$IP_ADDRESS"

###
### 4. Sync to S3
###
# 4.1 Upload logs (all files)
aws s3 sync "$LOG_DIR/" \
    "s3://$S3_BUCKET/logs/$S3_PREFIX/" \
    --delete                                                                # mirror & remove stale  [oai_citation_attribution:13‡AWS Documentation](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html?utm_source=chatgpt.com)

# 4.2 Upload models (encrypt at rest with KMS)
aws s3 sync "$MODEL_DIR/" \
    "s3://$S3_BUCKET/models/$S3_PREFIX/" \
    --delete \
    --sse aws:kms --sse-kms-key-id alias/myKey                               # SSE‑KMS encryption  [oai_citation_attribution:14‡AWS CLI](https://awscli.amazonaws.com/v2/documentation/api/2.13.19/reference/s3/sync.html?utm_source=chatgpt.com)

# 4.3 Upload metrics (only JSONL files)
aws s3 sync "$METRIC_DIR/" \
    "s3://$S3_BUCKET/metrics/$S3_PREFIX/" \
    --exclude="*" --include="*.jsonl" \
    --delete                                                                # filter metrics  [oai_citation_attribution:15‡AWS Documentation](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html?utm_source=chatgpt.com)

###
### 5. Completion
###
echo "✅ Upload complete: run=$RUN_ID, date=$DATE, instance=$INSTANCE_ID, ip=$IP_ADDRESS"  # status message  [oai_citation_attribution:16‡Home](https://www.adyxax.org/blog/2024/10/12/shell-script-for-gathering-imdsv2-instance-metadata-on-aws-ec2/?utm_source=chatgpt.com)