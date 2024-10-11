#!/bin/bash

# Create Directories
mkdir weights
mkdir datasets

# Download the datasets
cd datasets
kaggle datasets download -d andrewmvd/drive-digital-retinal-images-for-vessel-extraction
wget -O stare.tar "https://www.dropbox.com/scl/fi/mfodfr67sqag047jkhntp/stare-DatasetNinja.tar?rlkey=z8kwhb5us58phsydquokt8kq1&dl=1"
wget -O hrf.zip https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
wget https://researchdata.kingston.ac.uk/96/1/CHASEDB1.zip

# Extract the datasets
unzip drive-digital-retinal-images-for-vessel-extraction.zip
tar -xvf stare.tar
mv ds stare
unzip hrf.zip -d hrf
unzip CHASEDB1.zip -d chase

# Remove the archive files
rm drive-digital-retinal-images-for-vessel-extraction.zip
rm stare.tar
rm hrf.zip
rm CHASEDB1.zip