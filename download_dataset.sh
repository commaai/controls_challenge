#!/bin/bash

if [ ! -d "data" ]; then
    mkdir data
fi

cd data

URL="https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"

echo "Downloading dataset from $URL"
wget "$URL"

ZIP_FILE="${URL##*/}"

echo "Unzipping $ZIP_FILE"
unzip -jq "$ZIP_FILE"
rm "$ZIP_FILE"
