#!/bin/bash

# Define the source and destination directories
mkdir -p "Downloads/OSCD"
destination_dir="Downloads/OSCD"

source_dir="Downloads/Onera Satellite Change Detection dataset - Images"
source_dir_label_train="Downloads/Onera Satellite Change Detection dataset - Train Labels"
source_dir_label_test="Downloads/Onera Satellite Change Detection dataset - Test Labels"
train_file="$destination_dir/train.txt"
test_file="$destination_dir/test.txt"

# Check if train.txt exists
if [ ! -f "$train_file" ]; then
    echo "Error: $train_file does not exist."
    exit 1
fi

# Check if test.txt exists
if [ ! -f "$test_file" ]; then
    echo "Error: $test_file does not exist."
    exit 1
fi

# Read train.txt and process each word
while IFS=',' read -r -a words; do
    for word in "${words[@]}"; do
        echo "Processing $word"
        # Example operation: create directories for each word
        mkdir -p "$train_dir/$word"
        mkdir -p "$train_dir/$word/pair"
        mkdir -p "$train_dir/$word/cm"
        
        # Example operation: copy files based on the word
        cp "$source_dir/$word.jpg" "$train_dir/$word/pair/img1.jpg"
        cp "$source_dir/$word.jpg" "$train_dir/$word/pair/img2.jpg"
        cp "$source_dir_label_train/$word.png" "$train_dir/$word/cm/cm.png"
    done
done < "$train_file"

# Read test.txt and process each word
while IFS=',' read -r -a words; do
    for word in "${words[@]}"; do
        echo "Processing $word"
        # Example operation: create directories for each word
        mkdir -p "$test_dir/$word"
        mkdir -p "$test_dir/$word/pair"
        mkdir -p "$test_dir/$word/cm"
        
        # Example operation: copy files based on the word
        cp "$source_dir/$word.jpg" "$test_dir/$word/pair/img1.jpg"
        cp "$source_dir/$word.jpg" "$test_dir/$word/pair/img2.jpg"
        cp "$source_dir_label_test/$word.png" "$test_dir/$word/cm/cm.png"
    done
done < "$test_file"