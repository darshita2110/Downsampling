"""
DATA STEP 3 - Download the standard hyperspectral benchmark datasets (with labels).

These are the datasets most band-selection papers report results on, so our numbers can be
compared with theirs. All are real images and free to use for research.

    Indian Pines   AVIRIS (aircraft)       145 x 145,  200 bands, 16 classes   ~6 MB
    Pavia Univ.    ROSIS (aircraft)        610 x 340,  103 bands,  9 classes  ~33 MB
    Salinas        AVIRIS (aircraft)       512 x 217,  204 bands, 16 classes  ~25 MB
    KSC            AVIRIS (aircraft)       512 x 614,  176 bands, 13 classes  ~57 MB
    Botswana       Hyperion (satellite)   1476 x 256,  145 bands, 14 classes  ~79 MB

Original source: EHU "Hyperspectral Remote Sensing Scenes" page (blocked from India, 403).
Downloaded from Hugging Face mirrors of the same files. If a download fails, open the URL
printed for it in a browser and put the file in data/darshita/ with the name shown.

Run (from the repo folder):
    python darshita/data_3_download_benchmarks.py                 # all of them
    python darshita/data_3_download_benchmarks.py pavia_u salinas  # only some
"""
import os
import sys
import urllib.request

DATA_DIR = "data/darshita"
HF = "https://huggingface.co/datasets/"            # Hugging Face mirrors of the EHU files
# (the EHU site itself answers 403 Forbidden from India, so we use mirrors)
FILES = {
    "indian_pines": [("Indian_pines_corrected.mat", "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_corrected.mat"),
                     ("Indian_pines_gt.mat", "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_gt.mat")],
    "pavia_u": [("PaviaU.mat", HF + "danaroth/pavia/resolve/main/PaviaU.mat"),
                ("PaviaU_gt.mat", HF + "danaroth/pavia/resolve/main/PaviaU_gt.mat")],
    "salinas": [("Salinas_corrected.mat", HF + "danaroth/salinas/resolve/main/Salinas_corrected.mat"),
                ("Salinas_gt.mat", HF + "danaroth/salinas/resolve/main/Salinas_gt.mat")],
    "ksc": [("KSC.mat", HF + "Tanishq165/HSI_Datasets/resolve/main/KSC/KSC_data.mat"),
            ("KSC_gt.mat", HF + "Tanishq165/HSI_Datasets/resolve/main/KSC/KSC_gt.mat")],
    "botswana": [("Botswana.mat", HF + "danaroth/botswana/resolve/main/Botswana.mat"),
                 ("Botswana_gt.mat", HF + "danaroth/botswana/resolve/main/Botswana_gt.mat")],
}
opener = urllib.request.build_opener()               # some sites refuse Python's default name
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36")]
urllib.request.install_opener(opener)

wanted = sys.argv[1:] or list(FILES)
os.makedirs(DATA_DIR, exist_ok=True)
failed = []
for name in wanted:
    for filename, url in FILES[name]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"{filename:28s} already there")
            continue
        print(f"{filename:28s} downloading ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
            print(f"{os.path.getsize(path) / 1e6:.1f} MB")
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append((filename, url))
            if os.path.exists(path):
                os.remove(path)

if failed:
    print(f"\nDownload these in a browser and save them in {DATA_DIR}/ with the name on the left:")
    for filename, url in failed:
        print(f"  {filename:28s} {url}")
else:
    print("\nAll done. Check them with:  python darshita/run_all.py --list")