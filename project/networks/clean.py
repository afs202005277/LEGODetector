import json
import sys

def clear_outputs(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    for cell in notebook.get('cells', []):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clear_outputs.py <notebook_filename>")
        sys.exit(1)
    
    notebook_filename = sys.argv[1]
    clear_outputs(notebook_filename)
    print(f"Cleared outputs in notebook saved to {notebook_filename}")
