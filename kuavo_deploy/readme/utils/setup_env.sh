#!/bin/bash

ENV_NAME="kdc_v0"
ENV_DIR="./${ENV_NAME}"

#Create a directory and unzip the environment package
mkdir -p "${ENV_DIR}"
tar -xzf "${ENV_NAME}.tar.gz" -C "${ENV_DIR}"

echo "Environment decompression is completed, directory: ${ENV_DIR}"

#Here is a demonstration of running it directly with the decompressed python
echo "Run the test using python from the unpacked environment:"
"${ENV_DIR}/bin/python" --version

#activate environment
echo "Activate the environment:"
source "${ENV_DIR}/bin/activate"

#After entering the activation environment, run python
echo "Environment activation, run python:"
python --version

#Run conda-unpack to clean up prefixes
echo "Run conda-unpack to clean up the prefixes:"
conda-unpack

echo "The script execution is completed."

