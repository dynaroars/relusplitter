import os

def check_logs_for_cuda():
    # Get the current working directory
    cwd = os.getcwd()

    # List all files in the directory
    files_in_directory = os.listdir(cwd)

    # Filter out only log files (assuming they have a .log extension)
    log_files = [f for f in files_in_directory if f.endswith('.log')]

    # List to store files that contain "CUDA"
    files_with_cuda = []

    for log_file in log_files:
        try:
            # Open the file and read through its content
            with open(log_file, 'r') as f:
                for line in f:
                    if "CUDA" in line:
                        files_with_cuda.append(log_file)
                        break  # No need to read further, move to the next file

        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    # Report files that contain "CUDA"
    if files_with_cuda:
        print("The following log files contain 'CUDA':")
        for file in files_with_cuda:
            print(f" - {file}")
    else:
        print("No log files contain 'CUDA'.")

if __name__ == "__main__":
    check_logs_for_cuda()
