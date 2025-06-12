import json
import sys
import os
import fcntl

def write_dict_to_json_file(file_path, data_dict):
    """Safely appends a dictionary to the file using file locking."""
    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock for writing
        try:
            json.dump(data_dict, f)
            f.write('\n')
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def pop_last_dict_from_json_file(file_path):
    """Safely reads and removes the last JSON dictionary line from the file."""
    if not os.path.exists(file_path):
        print("File not found.")
        return {}

    # Lock the file for both reading and rewriting
    with open(file_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock
        try:
            lines = f.readlines()
            if not lines:
                print("File is empty.")
                return {}

            last_line = lines.pop().strip()

            try:
                last_dict = json.loads(last_line)
            except json.JSONDecodeError:
                print("Last line is not valid JSON.")
                return {}

            # Rewrite the file without the last line
            f.seek(0)
            f.truncate()
            f.writelines(lines)

            return last_dict
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def pop_dict_by_job_id(file_path, target_job_id):
    """Pops the first dictionary that matches the given job_id."""
    if not os.path.exists(file_path):
        print("File not found.")
        return {}

    with open(file_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.readlines()
            new_lines = []
            found_dict = None

            for line in lines:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if found_dict is None and entry.get("job_id") == target_job_id:
                    found_dict = entry
                else:
                    new_lines.append(line)

            if found_dict is None:
                print(f"No entry with job_id {target_job_id} found.")
                return {}

            # Rewrite file without the popped entry
            f.seek(0)
            f.truncate()
            f.writelines(new_lines)
            return found_dict
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  To write: python dict_json_io.py write <file_path> '{\"key\": \"value\"}'")
        print("  To pop:   python dict_json_io.py pop <file_path>")
        sys.exit(1)

    mode = sys.argv[1]
    file_path = sys.argv[2]

    if mode == "write":
        if len(sys.argv) < 4:
            print("Error: No dictionary data provided to write.")
            sys.exit(1)

        try:
            data_dict = json.loads(sys.argv[3])
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            sys.exit(1)

        write_dict_to_json_file(file_path, data_dict)
        print(f"\tWrote to {file_path}") #: {data_dict}")

    elif mode == "pop":
        result = pop_last_dict_from_json_file(file_path)
        print(f"Popped from {file_path}")#: {result}")

    elif mode == "pop_by_id":
        if len(sys.argv) < 4:
            print("Error: job_id required for pop_by_id.")
            sys.exit(1)
        try:
            job_id = int(sys.argv[3])
        except ValueError:
            print("Error: job_id must be an integer.")
            sys.exit(1)

        result = pop_dict_by_job_id(file_path, job_id)
        print(result)
        #print(f"Popped job_id {job_id} from {file_path}")#: {result}")

    else:
        print("Invalid mode. Use 'write' or 'pop'.")