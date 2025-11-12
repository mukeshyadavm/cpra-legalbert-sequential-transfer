

pretty_print_dir = "/mnt/gdrive/MyDrive/OPP-115/pretty_print"

# Collect all CSV files
csv_files = [f for f in os.listdir(pretty_print_dir) if f.endswith(".csv")]
all_policies = []

for file in csv_files:
    file_path = os.path.join(pretty_print_dir, file)
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] >= 4:
            policies = df.iloc[:, 3].dropna().tolist()
            for policy in policies:
                all_policies.append({"filename": file, "policy_text": policy})
        else:
            print(f" Skipped {file} — fewer than 4 columns")
    except Exception as e:
        print(f" Error reading {file}: {e}")

# Convert to DataFrame
df_all = pd.DataFrame(all_policies)

# Save
df_all.to_csv("opp115_all_policies_combined.csv", index=False)
print(f" Extracted {len(df_all)} policy entries into 'opp115_all_policies_combined.csv'")
