def print_dict_structure(d, indent=0):
    spacing = "  " * indent
    for key, value in d.items():
        print(f"{spacing}- {key}: {type(value).__name__}")

        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)