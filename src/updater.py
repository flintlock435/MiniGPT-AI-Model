import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request


# ============================================================
# MINIGPT AUTO UPDATER
# ============================================================

GITHUB_REPO = "https://github.com/flintlock435/MiniGPT-AI-Model/"


# Current installed model information.
LOCAL_VERSION_FILE = "data/version.json"


# Current model files.
LOCAL_MODEL_FILE = "checkpoints/v14_custom_best_model.pt"

LOCAL_TOKENIZER_FILE = "data/v12_tokenizer.json"


# Files that will be downloaded from a GitHub Release.
MODEL_ASSET_NAME = "model.pt"

TOKENIZER_ASSET_NAME = "tokenizer.json"


# ============================================================
# GITHUB API
# ============================================================

API_URL = (
    "https://api.github.com/repos/"
    + GITHUB_REPO
    + "/releases/latest"
)


# GitHub recommends the github+json Accept header.
HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "MiniGPT-Updater"
}


# ============================================================
# VERSION PARSING
# ============================================================

def parse_version(version):

    version = str(
        version
    ).strip().lower()


    version = version.lstrip(
        "v"
    )


    # --------------------------------------------------------
    # Extract numeric semantic-version components.
    # --------------------------------------------------------

    match = re.match(
        r"^(\d+)\.(\d+)\.(\d+)",
        version
    )


    if not match:

        raise ValueError(
            f"Invalid version: {version}"
        )


    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3))
    )


# ============================================================
# LOCAL VERSION
# ============================================================

def get_local_version():

    if not os.path.exists(
        LOCAL_VERSION_FILE
    ):

        return "0.0.0"


    try:

        with open(
            LOCAL_VERSION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        return str(
            data.get(
                "version",
                "0.0.0"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
        TypeError
    ):

        return "0.0.0"


# ============================================================
# SAVE LOCAL VERSION
# ============================================================

def save_local_version(
    version
):

    os.makedirs(
        os.path.dirname(
            LOCAL_VERSION_FILE
        ),
        exist_ok=True
    )


    data = {
        "version": version
    }


    temporary_file = (
        LOCAL_VERSION_FILE
        + ".tmp"
    )


    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


    os.replace(
        temporary_file,
        LOCAL_VERSION_FILE
    )


# ============================================================
# INITIAL VERSION FILE
# ============================================================

def ensure_local_version():

    if os.path.exists(
        LOCAL_VERSION_FILE
    ):

        return


    # --------------------------------------------------------
    # This should match the model currently shipped with
    # this application.
    # --------------------------------------------------------

    save_local_version(
        "14.0.0"
    )


# ============================================================
# HTTP GET
# ============================================================

def http_get_json(
    url
):

    request = urllib.request.Request(
        url,
        headers=HEADERS
    )


    with urllib.request.urlopen(
        request,
        timeout=15
    ) as response:

        raw = response.read()


    return json.loads(
        raw.decode(
            "utf-8"
        )
    )


# ============================================================
# GET LATEST RELEASE
# ============================================================

def get_latest_release():

    try:

        return http_get_json(
            API_URL
        )

    except urllib.error.HTTPError as error:

        print(
            f"Update check failed: HTTP {error.code}"
        )

    except urllib.error.URLError as error:

        print(
            "Update check failed:",
            error.reason
        )

    except Exception as error:

        print(
            "Update check failed:",
            error
        )


    return None


# ============================================================
# FIND RELEASE ASSET
# ============================================================

def find_asset(
    release,
    asset_name
):

    assets = release.get(
        "assets",
        []
    )


    for asset in assets:

        if asset.get(
            "name"
        ) == asset_name:

            return asset


    return None


# ============================================================
# SHA256
# ============================================================

def calculate_sha256(
    filepath
):

    digest = hashlib.sha256()


    with open(
        filepath,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )


            if not chunk:

                break


            digest.update(
                chunk
            )


    return digest.he