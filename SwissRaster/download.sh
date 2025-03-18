#!/bin/bash

url_file="maps.csv"

while IFS= read -r url; do
	echo "Download: $url"
	curl -O "$url"
done < "$url_file"

echo "All files downloaded!"
