#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
ENVS_DIR=$(realpath "$SCRIPT_DIR/../../.envs")
TOOL_NAME="nnenum"
CONDA_PREFIX=$ENVS_DIR/$TOOL_NAME
ENV_FILE_PATH=$ENVS_DIR/nnenum.txt




# Check if the first argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_processes>"
    exit 1
fi

# Check if the argument is a valid integer
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: The input must be a valid integer."
    exit 1
fi

# Variables
FILE_PATH="$LIBS_DIR/nnenum/src/nnenum/settings.py"
LINE_NUMBER=37

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found at $FILE_PATH"
    exit 1
fi

# Extract the current line and validate its format
CURRENT_LINE=$(sed -n "${LINE_NUMBER}p" "$FILE_PATH")
if [[ "$CURRENT_LINE" =~ ^([[:space:]]*cls\.NUM_PROCESSES[[:space:]]*=[[:space:]]*.*)(#.*)?$ ]]; then
    INDENTATION="${BASH_REMATCH[1]%%cls*}" # Capture leading spaces
    COMMENT="${BASH_REMATCH[2]}"           # Capture any inline comment
    NEW_LINE="${INDENTATION}cls.NUM_PROCESSES = $1${COMMENT}" # Construct the new line
else
    echo "Error: Line $LINE_NUMBER does not match the expected format."
    echo "Current line: $CURRENT_LINE"
    exit 1
fi

# Edit the target line
sed -i "${LINE_NUMBER}s|.*|$NEW_LINE|" "$FILE_PATH"

echo "nnenum num_cores $LINE_NUMBER updated to:"
echo "$NEW_LINE"
