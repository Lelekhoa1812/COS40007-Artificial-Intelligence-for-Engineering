import pandas as pd

# Try different encodings if the first one doesn't work
encodings = ['latin1']
delimiter = ','  # Change this to the correct delimiter if necessary

for encoding in encodings:
    try:
        df = pd.read_csv("Book1.csv", encoding=encoding, delimiter=delimiter, on_bad_lines='warn')
        print(f"Successfully read the file with encoding: {encoding}")
        print(df.head())
        break
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        print(f"Failed to read the file with encoding {encoding}: {e}")

# Inspect the first few lines of the CSV file to diagnose issues
with open("/Users/khoale/Downloads/COS40007/CCPP/Folds5x2_pp.csv", encoding='latin1') as file:
    for _ in range(10):
        print(file.readline())
