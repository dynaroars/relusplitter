def get_all_files(directory):
    """
    Recursively get all files in the given directory and its subdirectories.

    Args:
        directory (str): The root directory to search.

    Returns:
        list: A list of file paths.
    """
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files