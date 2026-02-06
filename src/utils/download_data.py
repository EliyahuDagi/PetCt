import os
import sys
import argparse
import concurrent.futures
from tqdm import tqdm

# Add current directory to path to allow importing gdrive_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gdrive_loader import GoogleDriveLoader
except ImportError:
    # Fallback if running from project root
    try:
        from src.utils.gdrive_loader import GoogleDriveLoader
    except ImportError:
        # If running from src/utils directly
        from gdrive_loader import GoogleDriveLoader

def download_worker(args):
    """
    Worker function to download a single file.
    Creates its own loader instance to ensure thread safety with Google API.
    """
    file_id, file_name, destination_folder, file_size, credentials_file, token_file = args
    
    # Create a new instance per thread for safety
    # The token should be valid from main thread auth
    try:
        loader = GoogleDriveLoader(credentials_file=credentials_file, token_file=token_file)
        # We suppress the progress bar for individual files to keep the console clean
        # and only rely on the main progress bar.
        loader.download_file(file_id, file_name, destination_folder, expected_size=file_size, silent=True)
        return True
    except Exception as e:
        # print(f"Error downloading {file_name}: {e}")
        return False

def collect_files(loader, folder_id, local_path, file_list):
    """
    Recursively list files to build a complete download manifest.
    """
    items = loader.list_files_in_folder(folder_id)
    for item in items:
        name = item['name']
        item_id = item['id']
        mime_type = item['mimeType']
        
        if mime_type == 'application/vnd.google-apps.folder':
            sub_path = os.path.join(local_path, name)
            os.makedirs(sub_path, exist_ok=True)
            # Recursively collect
            collect_files(loader, item_id, sub_path, file_list)
        else:
            size = item.get('size')
            file_list.append((item_id, name, local_path, size))

def download_sequential(loader, folder_id, local_path):
    """
    Recursively download files one by one without pre-scanning the whole tree.
    This limits memory usage and avoids long scanning timeouts.
    """
    try:
        items = loader.list_files_in_folder(folder_id)
    except Exception as e:
        print(f"Failed to list folder {folder_id}: {e}")
        return

    # Create local dir if needed (though list_files requires it to be valid? No, just the ID)
    os.makedirs(local_path, exist_ok=True)

    for item in items:
        name = item['name']
        item_id = item['id']
        mime_type = item['mimeType']
        
        if mime_type == 'application/vnd.google-apps.folder':
            sub_path = os.path.join(local_path, name)
            download_sequential(loader, item_id, sub_path)
        else:
            size = item.get('size')
            loader.download_file(item_id, name, local_path, expected_size=size)

def main():
    parser = argparse.ArgumentParser(description='Download from Google Drive')
    parser.add_argument('--single-thread', action='store_true', help='Use single threaded sequential download (no pre-scan)')
    args = parser.parse_args()

    target_folder_id = '1T4LsR5QOGwwFyGnoQFCXtGBJq4qf8HO2'
    destination_root = r'C:\Users\eli.dagi\Projects\PetCt\data\gdrive_downloads\1T4LsR5QOGwwFyGnoQFCXtGBJq4qf8HO2'
    
    # Paths to credentials
    # Assuming script is run from project root or src/utils
    if os.path.exists('credentials.json'):
        creds_file = 'credentials.json'
        token_file = 'token.json'
    elif os.path.exists('../../credentials.json'): # If running from src/utils
        creds_file = '../../credentials.json'
        token_file = '../../token.json'
    else:
        # Fallback to absolute paths or blindly assume current dir
        creds_file = 'credentials.json'
        token_file = 'token.json'

    # 1. Authenticate (Main Thread)
    # This prepares the token.json so workers don't have to do the full auth flow
    try:
        loader = GoogleDriveLoader(credentials_file=creds_file, token_file=token_file)
        print("Authenticating...")
        loader.authenticate() 
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please ensure credentials.json is present.")
        return

    if args.single_thread:
        print("Running in Single-Threaded Sequential Mode...")
        print("This mode downloads files as they are discovered, avoiding long pre-scan times.")
        download_sequential(loader, target_folder_id, destination_root)
        print("\nDownload complete!")
        return

    # --- HYBRID MODE: Process top-level folders sequentially, but download their contents in parallel ---
    print(f"Listing top-level contents of directory {target_folder_id}...")
    try:
        top_level_items = loader.list_files_in_folder(target_folder_id)
    except Exception as e:
        print(f"Failed to list remote root folder: {e}")
        return

    print(f"Found {len(top_level_items)} top-level items. Processing them one by one...")

    # Ensure root exists
    os.makedirs(destination_root, exist_ok=True)

    # Process each top-level item individually
    for item in top_level_items:
        name = item['name']
        item_id = item['id']
        mime_type = item['mimeType']
        
        if mime_type == 'application/vnd.google-apps.folder':
            sub_path = os.path.join(destination_root, name)
            os.makedirs(sub_path, exist_ok=True)
            
            print(f"\n--- Processing Folder: {name} ---")
            
            # 1. Collect files JUST for this subfolder
            current_folder_files = []
            print(f"Scanning files in {name}...")
            try:
                collect_files(loader, item_id, sub_path, current_folder_files)
            except Exception as e:
                print(f"Error scanning folder {name}: {e}")
                continue
                
            total_sub_files = len(current_folder_files)
            if total_sub_files == 0:
                print(f"Folder {name} is empty or no files found.")
                continue

            # 2. Download files in parallel for THIS folder
            tasks = [
                (f_id, f_name, f_dest, f_size, creds_file, token_file) 
                for (f_id, f_name, f_dest, f_size) in current_folder_files
            ]
            
            max_workers = 16
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(download_worker, task): task for task in tasks}
                for future in tqdm(concurrent.futures.as_completed(futures), total=total_sub_files, desc=f"Downloading {name}", unit="file"):
                    try:
                        future.result()
                    except Exception as e:
                        # print(f"Worker exception: {e}")
                        pass
            
            print(f"Finished processing parent sub-directory: {name}")

        else:
            # It's a top-level file, download it directly (single thread is fine for few top-level files)
            print(f"Downloading top-level file: {name}")
            size = item.get('size')
            loader.download_file(item_id, name, destination_root, expected_size=size)

    print("\nBatch download complete!")

if __name__ == "__main__":
    main()
