
import os
import io
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

class GoogleDriveLoader:
    """
    Helper class to authenticate with Google Drive and download/sync datasets.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        
    def authenticate(self):
        """
        Authenticate using Chrome/Browser flow.
        Saves the token to 'token.json' for future use.
        """
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
                
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Credentials file '{self.credentials_file}' not found. "
                        "Please download it from Google Cloud Console (OAuth Client ID) "
                        "and place it in the project root."
                    )
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
                
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)
        print("Successfully authenticated with Google Drive.")

    def list_files_in_folder(self, folder_id):
        """
        List all files in a specific Google Drive folder.
        """
        if not self.service:
            self.authenticate()
            
        results = []
        page_token = None
        
        query = f"'{folder_id}' in parents and trashed = false"
        
        retry_count = 0
        max_retries = 5

        while True:
            try:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, size)',
                    pageToken=page_token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True
                ).execute()
                
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken', None)
                retry_count = 0 # Reset on success
                if page_token is None:
                    break
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise e
                print(f"Error listing files (Attempt {retry_count}/{max_retries}): {e}. Retrying in 2 seconds...")
                import time
                time.sleep(2)
                
        return results

    def download_file(self, file_id, file_name, destination_folder, expected_size=None, silent=True):
        """
        Download a file from Google Drive to a local folder.
        """
        if not self.service:
            self.authenticate()
            
        destination_path = os.path.join(destination_folder, file_name)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Skip if already exists and size matches
        if os.path.exists(destination_path):
            local_size = os.path.getsize(destination_path)
            if expected_size is not None:
                if int(expected_size) == local_size:
                    if not silent:
                        print(f"File {file_name} exists and size matches. Skipping.")
                    return destination_path
                else:
                    if not silent:
                        print(f"File {file_name} exists but size differs ({local_size} vs {expected_size}). Re-downloading.")
            else:
                if not silent:
                    print(f"File {file_name} already exists. Skipping.")
                return destination_path

        request = self.service.files().get_media(fileId=file_id)
        
        # Direct write to file with larger chunk size (10MB)
        with open(destination_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=10*1024*1024)
            done = False
            
            if not silent:
                print(f"Downloading {file_name}...")
            
            with tqdm(total=100, unit='pct', disable=silent, leave=False) as pbar:
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        pbar.update(int(status.progress() * 100) - pbar.n)
            
        return destination_path

    def download_folder_recursive(self, folder_id, local_root_dir):
        """
        Recursively download a Google Drive folder.
        """
        items = self.list_files_in_folder(folder_id)
        
        for item in items:
            name = item['name']
            item_id = item['id']
            mime_type = item['mimeType']
            
            if mime_type == 'application/vnd.google-apps.folder':
                new_local_dir = os.path.join(local_root_dir, name)
                os.makedirs(new_local_dir, exist_ok=True)
                self.download_folder_recursive(item_id, new_local_dir)
            else:
                self.download_file(item_id, name, local_root_dir)

