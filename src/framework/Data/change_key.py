import json


def rename_question_key(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            json_objects = []
            for line in f_in:
                # Remove possible newlines and spaces
                line = line.strip()
                if not line:
                    continue
                # Parse single JSON object
                data = json.loads(line)
                # Check if "Question" key exists
                if "Question" in data:
                    # Create new dictionary, rename key and keep other keys
                    new_data = {"question": data.pop("Question")}
                    new_data.update(data)
                    json_objects.append(new_data)
                else:
                    json_objects.append(data)  # Keep original data (if no key needs to be replaced)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for obj in json_objects:
                # Write processed JSON objects line by line, maintaining original multi-line format
                json.dump(obj, f_out, ensure_ascii=False)
                f_out.write('\n')  # Maintain format of one JSON object per line

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed, position: {e.pos}, message: {e.msg}")
    except Exception as e:
        print(f"Unknown error occurred: {str(e)}")


if __name__ == "__main__":
    input_file = "./textbook/Question.json"  # Please replace with actual input file path
    output_file = "./textbook/output.json"  # Output file path after processing
    rename_question_key(input_file, output_file)
    print("Key name replacement completed, results saved to", output_file)