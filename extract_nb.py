import json
import sys

notebook_path = r"c:\Users\ironh\Downloads\LLM_StableDiffusion_Model_Create_Image-main\LLM2 (1).ipynb"
output_path = r"c:\Users\ironh\Downloads\LLM_StableDiffusion_Model_Create_Image-main\extracted_code.py"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
code_parts = []
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if source.strip():
            code_parts.append(f"# === CELL {i} ===\n{source}")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(code_parts))

print(f"Done! Extracted {len(code_parts)} code cells to {output_path}")
