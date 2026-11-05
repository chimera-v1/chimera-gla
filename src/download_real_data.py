import os
import urllib.request

def download_wikitext():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "real_pretrain")
    os.makedirs(data_dir, exist_ok=True)
    
    # Stable raw text link for WikiText-2 (Wikipedia data)
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    dst_train = os.path.join(data_dir, "wikipedia_pretrain.txt")
    
    print("[System] Fetching real Wikipedia Pre-Training dataset...")
    try:
        urllib.request.urlretrieve(url, dst_train)
        print(f"[+] SUCCESS: Real Pre-Training dataset saved to {dst_train}")
    except Exception as e:
        print(f"[-] FAILED to download: {e}")

if __name__ == "__main__":
    download_wikitext()
