#!/bin/bash

# Define the source and destination directories
mkdir -p "Downloads/OSCD"
destination_dir="Downloads/OSCD"

source_dir="Downloads/Onera Satellite Change Detection dataset - Images"
source_dir_label_train="Downloads/Onera Satellite Change Detection dataset - Train Labels"
source_dir_label_test="Downloads/Onera Satellite Change Detection dataset - Test Labels"
train_file="$source_dir/train.txt"
test_file="$source_dir/test.txt"

# Read train.txt and process each word
while IFS=',' read -r -a words; do
    for word in "${words[@]}"; do
        echo "Processing $word"
        # Example operation: create directories for each word
        mkdir -p "$destination_dir/training/$word"
        mkdir -p "$destination_dir/training/$word/pair"
        mkdir -p "$destination_dir/training/$word/cm"
        
        # Example operation: copy files based on the word
        cp "$source_dir/$word/pair/img1.png" "$destination_dir/training/$word/pair/img1.png"
        cp "$source_dir/$word/pair/img2.png" "$destination_dir/training/$word/pair/img2.png"
        cp "$source_dir_label_train/$word/cm/cm.png" "$destination_dir/training/$word/cm/cm.png"
    done
done < "$train_file"

# Read train.txt and process each word
while IFS=',' read -r -a words; do
    for word in "${words[@]}"; do
        echo "Processing $word"
        # Example operation: create directories for each word
        mkdir -p "$destination_dir/testing/$word"
        mkdir -p "$destination_dir/testing/$word/pair"
        mkdir -p "$destination_dir/testing/$word/cm"
        
        # Example operation: copy files based on the word
        cp "$source_dir/$word/pair/img1.png" "$destination_dir/testing/$word/pair/img1.png"
        cp "$source_dir/$word/pair/img2.png" "$destination_dir/testing/$word/pair/img2.png"
        cp "$source_dir_label_test/$word/cm/cm.png" "$destination_dir/testing/$word/cm/cm.png"
    done
done < "$test_file"