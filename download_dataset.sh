#!/bin/bash

if [ ! -d "data" ]; then
    mkdir data
fi

cd data

URL="https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"

echo "Downloading dataset from $URL"

ZIP_FILE="dataset.zip"

curl -L "$URL" -o $ZIP_FILE

echo "Unzipping $ZIP_FILE"
unzip -jq "$ZIP_FILE"
rm "$ZIP_FILE"
