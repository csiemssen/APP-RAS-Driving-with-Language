import json
import os
from pathlib import Path

src_dir = Path(__file__).parent.resolve()
data_dir = src_dir / ".." / "data/test_server_eval"
readme_path = src_dir / ".." / "README.md"


# Generate Markdown table
def generate_table(data_files):
    headers = ["Folder", "File"] + list(data_files[0]["data"].keys())
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for entry in data_files:
        folder = entry["folder"]
        file = entry["file"]
        row = [folder, file] + [str(entry["data"].get(key, "")) for key in headers[2:]]
        table += "| " + " | ".join(row) + " |\n"

    return table


# Read JSON files
def read_json_files(directory):
    data_files = []
    print(directory)
    for root, _, files in os.walk(directory):
        print(f"Reading files in: {root}")
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                data_files.append(
                    {
                        "folder": os.path.basename(root),
                        "file": file,
                        "data": data,
                    }
                )
    return data_files


# Update README.md
def update_readme(table):
    with open(readme_path, "r") as f:
        content = f.read()

    # Replace the Eval results section
    start_marker = "#### Results"
    end_marker = "\n#### "
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        end_idx = len(content)

    new_content = (
        content[: start_idx + len(start_marker)]
        + "\n\n"
        + table
        + "\n"
        + content[end_idx:]
    )
    with open(readme_path, "w") as f:
        f.write(new_content)


# Main
if __name__ == "__main__":
    data_files = read_json_files(data_dir)
    if data_files:
        table = generate_table(data_files)
        update_readme(table)
