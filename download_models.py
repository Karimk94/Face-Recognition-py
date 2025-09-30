import os
import requests
from tqdm import tqdm
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def download_file_robustly(url, local_path):
    """
    Downloads a file with a progress bar, supports resuming, and verifies the final size.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Get the total size of the file from the server
            with requests.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))

            # Check if a partially downloaded file exists
            current_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

            if current_size >= total_size and total_size > 0:
                print(f"{os.path.basename(local_path)} is already complete. Skipping.")
                return True

            print(f"Downloading {os.path.basename(local_path)}... (Attempt {attempt + 1}/{max_retries})")
            headers = {'Range': f'bytes={current_size}-'}
            
            with requests.get(url, headers=headers, stream=True, timeout=20) as response:
                response.raise_for_status()
                
                with open(local_path, 'ab') as f, tqdm(
                    desc=os.path.basename(local_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=current_size,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)

            # --- Final Verification Step ---
            final_size = os.path.getsize(local_path)
            if final_size >= total_size:
                print(f"\n✅ Download of {os.path.basename(local_path)} successful and verified.")
                return True
            else:
                print(f"\n🚨 Download incomplete. Expected {total_size} bytes, got {final_size} bytes. Retrying...")
                time.sleep(2) # Wait before retrying

        except requests.exceptions.RequestException as e:
            print(f"\nA network error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    
    print(f"\n❌ Failed to download {os.path.basename(local_path)} after {max_retries} attempts.")
    return False

def setup_offline_models():
    """
    Ensures all necessary AI models are downloaded directly into the project's 'models' folder for offline use.
    """
    print("--- Setting up models for offline use ---")
    
    models_path = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_path, exist_ok=True)
    print(f"Models will be saved to: {models_path}")

    models = {
        "vgg_face_weights.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
        "retinaface.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"
    }

    all_successful = True
    for filename, url in models.items():
        # The local path is now inside the project's 'models' folder
        local_path = os.path.join(models_path, filename)
        if not download_file_robustly(url, local_path):
            all_successful = False

    if all_successful:
        print("\n✅ All models are downloaded and ready for offline use.")
    else:
        print("\n❌ Some models failed to download. Please try running the setup again.")

if __name__ == "__main__":
    setup_offline_models()