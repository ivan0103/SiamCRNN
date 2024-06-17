#!/bin/bash

# Define the source and destination directories
mkdir -p "Downloads/DSIFN"
destination_dir="Downloads/DSIFN"


source_dir_A="Downloads/DSIFN/train/t1"
source_dir_B="Downloads/DSIFN/train/t2"
source_dir_Label="Downloads/DSIFN/train/mask_256"
train_dir="$destination_dir/training"
output_file="$destination_dir/train.txt"

# Loop through each file in source_dir_A
for file in "$source_dir_A"/*.jpg; do
    # Get the base filename without the directory and extension
    filename=$(basename "$file" .jpg)

    # Create the destination directories
    mkdir -p "$train_dir/$filename"
    mkdir -p "$train_dir/$filename/pair"
    mkdir -p "$train_dir/$filename/cm"

    # # Copy files from source_dir_A and source_dir_B to the 'pair' folder
    cp "$source_dir_A/$filename.jpg" "$train_dir/$filename/pair/img1.png"
    cp "$source_dir_B/$filename.jpg" "$train_dir/$filename/pair/img2.png"
    
    # # Convert .tif to .png and copy to the 'cm' folder
    cp "$source_dir_Label/$filename.png" "$train_dir/$filename/cm/cm.png"
        # Write the file name to the output file
    echo "$filename" >> "$output_file"
done

source_dir_A="Downloads/DSIFN/val/t1"
source_dir_B="Downloads/DSIFN/val/t2"
source_dir_Label="Downloads/DSIFN/val/mask_256"
train_dir="$destination_dir/testing"
output_file="$destination_dir/test.txt"

# Loop through each file in source_dir_A
for file in "$source_dir_A"/*.jpg; do
    # Get the base filename without the directory and extension
    filename=$(basename "$file" .jpg)

    # Create the destination directories
    mkdir -p "$train_dir/$filename"
    mkdir -p "$train_dir/$filename/pair"
    mkdir -p "$train_dir/$filename/cm"

    # # Copy files from source_dir_A and source_dir_B to the 'pair' folder
    cp "$source_dir_A/$filename.jpg" "$train_dir/$filename/pair/img1.png"
    cp "$source_dir_B/$filename.jpg" "$train_dir/$filename/pair/img2.png"
    
    # # Convert .tif to .png and copy to the 'cm' folder
    cp "$source_dir_Label/$filename.png" "$train_dir/$filename/cm/cm.png"
        # Write the file name to the output file
    echo "$filename" >> "$output_file"
done