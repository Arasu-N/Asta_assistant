from pathlib import Path

file_path = "data/documents/HRDataset_v13.csv.xls"

path = Path(file_path)

print("Name:", path.name)
print("Suffix:", path.suffix)

with open(file_path, "rb") as f:
    print(f.read(20))