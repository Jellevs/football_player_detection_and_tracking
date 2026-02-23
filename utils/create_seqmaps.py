from pathlib import Path

data_root = Path(r"D:\TUE\Thesis\Code\data\tracking-2023\test\test")
output_file = Path(r"D:\TUE\Thesis\Code\data\tracking-2023\SNMOT-test.txt")

sequences = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

output_file.write_text("\n".join(sequences))
print(f"Written {len(sequences)} sequences to {output_file}")