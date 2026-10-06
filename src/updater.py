import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request


# ============================================================
# MINIGPT UPDATE CONFIGURATION
# ============================================================

GITHUB_REPO = "flintlock435/MiniGPT-AI-Model"

LOCAL_VERSION_FILE = "data/version.json"

LOCAL_MODEL_FILE = "checkpoints/v14_custom_best_model.pt"

LOCAL_TOKENIZER_FILE = "data/v12_tokenizer.json"

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

        return "14.0.0"


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
                "14.0.0"
            )
        )


    except (
        OSError,
        json.JSONDecodeError,
        TypeError
    ):

        return "14.0.0"


# ============================================================
# SAVE LOCAL VERSION
# ============================================================

def save_local_version(
    version
):

    directory = os.path.dirname(
        LOCAL_VERSION_FILE
    )


    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )


    data = {
        "version": str(
            version
        )
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
# ENSURE VERSION FILE
# ============================================================

def ensure_local_version():

    if os.path.exists(
        LOCAL_VERSION_FILE
    ):

        return


    save_local_version(
        "14.0.0"
    )


# ============================================================
# GET JSON FROM GITHUB
# ============================================================

def http_get_json(
    url
):

    request = urllib.request.Request(
        url,
        headers=HEADERS,
        method="GET"
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


    return digest.hexdigest()


# ============================================================
# VERIFY SHA256
# ============================================================

def verify_asset(
    filepath,
    expected_digest
):

    if not expected_digest:

        print(
            "No SHA256 digest provided by GitHub."
        )

        return False


    expected_digest = str(
        expected_digest
    ).strip().lower()


    if expected_digest.startswith(
        "sha256:"
    ):

        expected_digest = (
            expected_digest[
                len("sha256:"):
            ]
        )


    actual_digest = calculate_sha256(
        filepath
    ).lower()


    print(
        "Expected SHA256:",
        expected_digest
    )


    print(
        "Actual SHA256:",
        actual_digest
    )


    return (
        actual_digest
        == expected_digest
    )


# ============================================================
# DOWNLOAD FILE
# ============================================================

def download_file(
    url,
    destination
):

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MiniGPT-Updater"
        },
        method="GET"
    )


    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:

        with open(
            destination,
            "wb"
        ) as file:

            while True:

                chunk = response.read(
                    1024 * 1024
                )


                if not chunk:

                    break


                file.write(
                    chunk
                )


# ============================================================
# SAFE FILE REPLACEMENT
# ============================================================

def replace_file(
    source,
    destination
):

    destination_directory = os.path.dirname(
        destination
    )


    if destination_directory:

        os.makedirs(
            destination_directory,
            exist_ok=True
        )


    backup = (
        destination
        + ".backup"
    )


    # --------------------------------------------------------
    # Keep the current file as a backup.
    # --------------------------------------------------------

    if os.path.exists(
        destination
    ):

        shutil.copy2(
            destination,
            backup
        )


    os.replace(
        source,
        destination
    )


# ============================================================
# PERFORM UPDATE
# ============================================================

def perform_update(
    release,
    latest_version
):

    model_asset = find_asset(
        release,
        MODEL_ASSET_NAME
    )


    tokenizer_asset = find_asset(
        release,
        TOKENIZER_ASSET_NAME
    )


    if model_asset is None:

        print(
            f"Update aborted: '{MODEL_ASSET_NAME}' "
            "was not found in the GitHub release."
        )

        return False


    if tokenizer_asset is None:

        print(
            f"Update aborted: '{TOKENIZER_ASSET_NAME}' "
            "was not found in the GitHub release."
        )

        return False


    model_url = model_asset.get(
        "browser_download_url"
    )

    tokenizer_url = tokenizer_asset.get(
        "browser_download_url"
    )


    if not model_url:

        print(
            "Update aborted: model download URL missing."
        )

        return False


    if not tokenizer_url:

        print(
            "Update aborted: tokenizer download URL missing."
        )

        return False


    temporary_directory = None


    try:

        temporary_directory = tempfile.mkdtemp(
            prefix="minigpt_update_"
        )


        temporary_model = os.path.join(
            temporary_directory,
            "model.pt"
        )


        temporary_tokenizer = os.path.join(
            temporary_directory,
            "tokenizer.json"
        )


        # ====================================================
        # DOWNLOAD MODEL
        # ====================================================

        print()

        print(
            "Downloading model..."
        )


        download_file(
            model_url,
            temporary_model
        )


        print(
            "Model downloaded."
        )


        # ====================================================
        # VERIFY MODEL
        # ====================================================

        print(
            "Verifying model..."
        )


        if not verify_asset(
            temporary_model,
            model_asset.get(
                "digest"
            )
        ):

            print(
                "Model verification failed."
            )

            return False


        print(
            "Model verification passed."
        )


        # ====================================================
        # DOWNLOAD TOKENIZER
        # ====================================================

        print(
            "Downloading tokenizer..."
        )


        download_file(
            tokenizer_url,
            temporary_tokenizer
        )


        print(
            "Tokenizer downloaded."
        )


        # ====================================================
        # VERIFY TOKENIZER
        # ====================================================

        print(
            "Verifying tokenizer..."
        )


        if not verify_asset(
            temporary_tokenizer,
            tokenizer_asset.get(
                "digest"
            )
        ):

            print(
                "Tokenizer verification failed."
            )

            return False


        print(
            "Tokenizer verification passed."
        )


        # ====================================================
        # REPLACE MODEL
        # ====================================================

        print(
            "Installing model..."
        )


        replace_file(
            temporary_model,
            LOCAL_MODEL_FILE
        )


        # ====================================================
        # REPLACE TOKENIZER
        # ====================================================

        print(
            "Installing tokenizer..."
        )


        replace_file(
            temporary_tokenizer,
            LOCAL_TOKENIZER_FILE
        )


        # ====================================================
        # SAVE VERSION
        # ====================================================

        save_local_version(
            latest_version
        )


        print()

        print(
            "=========================================="
        )

        print(
            "MiniGPT update successful."
        )

        print(
            "Installed version:",
            latest_version
        )

        print(
            "=========================================="
        )


        return True


    except Exception as error:

        print()

        print(
            "Update failed:",
            error
        )

        return False


    finally:

        if temporary_directory:

            shutil.rmtree(
                temporary_directory,
                ignore_errors=True
            )


# ============================================================
# CHECK FOR UPDATE
# ============================================================

def check_for_update():

    ensure_local_version()


    current_version = get_local_version()


    print(
        "Current MiniGPT version:",
        current_version
    )


    release = get_latest_release()


    if release is None:

        return False


    tag_name = release.get(
        "tag_name"
    )


    if not tag_name:

        print(
            "GitHub release did not contain a tag."
        )

        return False


    latest_version = str(
        tag_name
    ).lstrip(
        "vV"
    )


    try:

        current_tuple = parse_version(
            current_version
        )


        latest_tuple = parse_version(
            latest_version
        )


    except ValueError as error:

        print(
            "Version comparison failed:",
            error
        )

        return False


    # ========================================================
    # UP TO DATE
    # ========================================================

    if latest_tuple <= current_tuple:

        print(
            "MiniGPT is up to date."
        )

        return False


    # ========================================================
    # UPDATE AVAILABLE
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        "MiniGPT update available!"
    )

    print(
        "Current version:",
        current_version
    )

    print(
        "Latest version:",
        latest_version
    )

    print(
        "=========================================="
    )


    release_name = release.get(
        "name"
    )


    if release_name:

        print(
            "Release:",
            release_name
        )


    answer = input(
        "Update now? [Y/n]: "
    ).strip().lower()


    if answer not in {
        "",
        "y",
        "yes"
    }:

        print(
            "Update skipped."
        )

        return False


    return perform_update(
        release,
        latest_version
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    check_for_update()