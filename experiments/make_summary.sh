#!/bin/bash

# Initialize an empty array
combined_list=()

# Loop over all .json files in the current directory
for file in *.json; do
  # Read the content of the file and append it to the combined_list
  content=$(cat "$file")
  combined_list+=($(echo $content | jq -c '.[]'))
done

# Convert the combined list into a valid JSON array and save it to summary.json
echo "${combined_list[@]}" | jq -s '.' > _summary.json

echo "Combined JSON list saved to summary.json"
