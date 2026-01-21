import json
from huggingface_hub import HfApi, hf_hub_download

# Configuration
repo_id = "easyminnn/pi05_4tasks_final"
new_version = "v2.0"  # This is the version LeRobot expects

api = HfApi()

# 1. Download the current info.json
print("Downloading info.json...")
info_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")

# 2. Read and modify the JSON
with open(info_path, "r") as f:
    data = json.load(f)

print(f"Current version: {data.get('codebase_version')}")
data["codebase_version"] = new_version
print(f"New version set to: {new_version}")

# 3. Save the modified JSON locally
with open(info_path, "w") as f:
    json.dump(data, f, indent=4)

# 4. Upload the updated file back to the Hub
print("Uploading updated info.json...")
api.upload_file(
    path_or_fileobj=info_path,
    path_in_repo="meta/info.json",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=f"Update codebase_version to {new_version}"
)

# 5. Create the matching Git tag
# Note: We tag the commit we just made
print(f"Creating tag '{new_version}'...")
try:
    api.create_tag(
        repo_id=repo_id,
        tag=new_version,
        repo_type="dataset"
    )
    print("Success! Tag created.")
except Exception as e:
    print(f"Tag creation warning (it might already exist): {e}")

print("Done. Try running your training/stats script again.")