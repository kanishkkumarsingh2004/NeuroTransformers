import os
import base64
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================
# Configuration
# =====================================================

REPOSITORIES = [
    "LINT TO ALL THE REPO FOR DOWNLOAD THE ALL .MD FILES",

]



OUTPUT_FOLDER = "data3"

# Optional GitHub Token
GITHUB_TOKEN = ""

MAX_WORKERS = 10

# =====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HEADERS = {
    "Accept": "application/vnd.github+json"
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def extract_repo(url):
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    parts = url.split("/")

    if len(parts) < 5:
        raise ValueError(f"Invalid URL: {url}")

    return parts[-2], parts[-1]


def github_get(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def get_default_branch(owner, repo):
    repo_info = github_get(f"https://api.github.com/repos/{owner}/{repo}")
    return repo_info["default_branch"]


def list_markdown_files(owner, repo, branch):
    tree = github_get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    )

    files = []

    for item in tree["tree"]:
        if item["type"] != "blob":
            continue

        if item["path"].lower().endswith(".md"):
            files.append(item["path"])

    return files


def download_file(owner, repo, path):

    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    data = github_get(api)

    content = base64.b64decode(data["content"]).decode(
        "utf-8",
        errors="ignore"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_path = path.replace("/", "__").replace("\\", "__")

    filename = f"{owner}_{repo}_{safe_path}_{timestamp}.txt"

    filepath = os.path.join(OUTPUT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def process_repository(repo_url):

    owner, repo = extract_repo(repo_url)

    print(f"\nProcessing {owner}/{repo}")

    try:

        branch = get_default_branch(owner, repo)

        md_files = list_markdown_files(owner, repo, branch)

        print(f"Found {len(md_files)} markdown files")

        for md in md_files:
            try:
                saved = download_file(owner, repo, md)
                print("Saved:", saved)
            except Exception as e:
                print(f"Failed {md}: {e}")

    except Exception as e:
        print(f"Repository failed: {e}")


def main():

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(process_repository, repo)
            for repo in REPOSITORIES
        ]

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()