#!/usr/bin/env python3
"""Push the reproducible Notebook to Kaggle as a public kernel.

Uses the registered custom.kaggle connection (same credential pattern as
kaggle_api.py: Bearer surrogate on www.kaggle.com).

Why not the old /kernels/push multipart endpoint: the current Kaggle API
client (kaggle 2.2.5 / kagglesdk) no longer uses it — it calls the
kernels.KernelsApiService/SaveKernel service with a JSON body, and the
multipart endpoint now 400s ("Error reading JToken"). This script mirrors
the official client exactly, including stripping cell outputs (the server
re-runs the notebook on push and regenerates them).

Usage:
  push_kaggle_kernel.py push     # create the public kernel (save + run)
  push_kaggle_kernel.py status   # show the kernel session status

Kernel: dewang2025/nfl-big-data-bowl-2027-beyond-the-stopwatch-reproducible-companion
  - public, internet enabled (the notebook clones the public GitHub repo),
  - competition data nfl-big-data-bowl-2027 attached.
Does NOT submit anything to the competition.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response

BASE = "https://www.kaggle.com/api/v1"
CONNECTION = "custom.kaggle"
ALLOWED_HOSTS = ["www.kaggle.com"]

KERNEL_SLUG = "nfl-big-data-bowl-2027-beyond-the-stopwatch"
KERNEL_ID = "dewang2025/" + KERNEL_SLUG
TITLE = "NFL Big Data Bowl 2027 — Beyond the Stopwatch"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "nfl_big_data_bowl_2027_reproducible.ipynb")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def _post_json(path, payload):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE + path, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    add_surrogate_to_request(request, CONNECTION, allowed_hosts=ALLOWED_HOSTS)
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=300) as response:
            return read_json_response(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("Kaggle API %s %s: %s" % (
            exc.code, exc.reason, exc.read()[:2000].decode("utf-8", "replace")))


def _notebook_text():
    """Notebook JSON with outputs stripped and list sources joined,
    exactly like the official client does before pushing."""
    with open(NOTEBOOK, encoding="utf-8") as handle:
        nb = json.load(handle)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and "outputs" in cell:
            cell["outputs"] = []
        if "source" in cell and isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    return json.dumps(nb)


def push():
    payload = {
        "slug": KERNEL_ID,
        "newTitle": TITLE,
        "text": _notebook_text(),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": False,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": True,
        "datasetDataSources": [],
        "competitionDataSources": ["nfl-big-data-bowl-2027"],
        "kernelDataSources": [],
        "modelDataSources": [],
    }
    result = _post_json("/kernels.KernelsApiService/SaveKernel", payload)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
    print("public URL: https://www.kaggle.com/code/%s" % KERNEL_ID)
    return result


def status():
    result = _post_json("/kernels.KernelsApiService/GetKernelSessionStatus",
                        {"userName": "dewang2025", "kernelSlug": KERNEL_SLUG})
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    return result


def main(argv):
    if not argv or argv[0] not in ("push", "status"):
        print(__doc__)
        return 2
    if argv[0] == "push":
        if not os.path.exists(NOTEBOOK):
            print("notebook not found:", NOTEBOOK, file=sys.stderr)
            return 2
        print("pushing", NOTEBOOK, os.path.getsize(NOTEBOOK), "bytes")
        push()
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
