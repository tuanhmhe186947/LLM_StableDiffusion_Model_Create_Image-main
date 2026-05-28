import json

def extract_code(notebook_path, output_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        code_cells = []
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                code_cells.append(source)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Extracted Code from Notebook\n\n" + "\n# %% [CELL]\n".join(code_cells))
            
        print(f"Extracted {len(code_cells)} cells to {output_path}")
    except Exception as e:
        print(f"Error reading notebook: {e}")

if __name__ == "__main__":
    extract_code("source.ipynb", "extracted_source.py")
