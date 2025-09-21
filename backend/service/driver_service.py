from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from config import Config
import json
import io
import os
import logging

logger = logging.getLogger(__name__)

class GoogleDriveService:
    def __init__(self):
        self.service = None
        self.credentials = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Drive service"""
        try:
            # Parse credentials from environment variable
            creds_json = Config.GOOGLE_DRIVE_CREDENTIALS_JSON
            if not creds_json:
                raise ValueError("Google Drive credentials not found in environment")
            
            # Parse JSON credentials
            creds_info = json.loads(creds_json)
            
            # Create credentials
            self.credentials = Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            # Build service
            self.service = build('drive', 'v3', credentials=self.credentials)
            logger.info("✅ Google Drive service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Drive service: {e}")
            raise
    
    def list_folder_files(self, folder_id):
        """List files in a Google Drive folder"""
        try:
            if not self.service:
                self._initialize_service()
            
            results = self.service.files().list(
                q=f"parents in '{folder_id}'",
                pageSize=50,
                fields="nextPageToken, files(id, name, mimeType, size)"
            ).execute()
            
            items = results.get('files', [])
            
            logger.info(f"✅ Found {len(items)} files in folder {folder_id}")
            return items
            
        except Exception as e:
            logger.error(f"❌ Error listing folder files: {e}")
            return []
    
    def download_file(self, file_id, local_path):
        """Download file from Google Drive"""
        try:
            if not self.service:
                self._initialize_service()
            
            # Get file metadata
            file_metadata = self.service.files().get(fileId=file_id).execute()
            
            # Download file
            request = self.service.files().get_media(fileId=file_id)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with io.FileIO(local_path, 'wb') as file_handle:
                downloader = MediaIoBaseDownload(file_handle, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        logger.info(f"Download progress: {progress}%")
            
            logger.info(f"✅ Downloaded {file_metadata['name']} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading file: {e}")
            return False
    
    def download_model_files(self, folder_id, local_model_dir):
        """Download all model files from a Google Drive folder"""
        try:
            # List files in the folder
            files = self.list_folder_files(folder_id)
            
            if not files:
                logger.warning(f"No files found in folder {folder_id}")
                return False
            
            # Create local directory
            os.makedirs(local_model_dir, exist_ok=True)
            
            downloaded_files = []
            
            # Download each file
            for file_info in files:
                file_name = file_info['name']
                local_file_path = os.path.join(local_model_dir, file_name)
                
                logger.info(f"Downloading {file_name}...")
                
                if self.download_file(file_info['id'], local_file_path):
                    downloaded_files.append(file_name)
                    logger.info(f"✅ {file_name} downloaded successfully")
                else:
                    logger.error(f"❌ Failed to download {file_name}")
            
            logger.info(f"✅ Downloaded {len(downloaded_files)}/{len(files)} files to {local_model_dir}")
            return len(downloaded_files) > 0
            
        except Exception as e:
            logger.error(f"❌ Error downloading model files: {e}")
            return False
    
    def get_file_info(self, file_id):
        """Get information about a specific file"""
        try:
            if not self.service:
                self._initialize_service()
            
            file_info = self.service.files().get(
                fileId=file_id,
                fields="id, name, mimeType, size, createdTime, modifiedTime"
            ).execute()
            
            return file_info
            
        except Exception as e:
            logger.error(f"❌ Error getting file info: {e}")
            return None
    
    def check_model_availability(self):
        """Check if both model folders are accessible"""
        try:
            t5_folder_id = Config.T5_MODEL_FOLDER_ID
            bart_folder_id = Config.BART_MODEL_FOLDER_ID
            
            status = {
                't5': False,
                'bart': False,
                'errors': []
            }
            
            # Check T5 model folder
            if t5_folder_id:
                t5_files = self.list_folder_files(t5_folder_id)
                status['t5'] = len(t5_files) > 0
                if not status['t5']:
                    status['errors'].append("T5 model folder is empty or inaccessible")
            else:
                status['errors'].append("T5 model folder ID not configured")
            
            # Check DistilBART model folder
            if bart_folder_id:
                bart_files = self.list_folder_files(bart_folder_id)
                status['bart'] = len(bart_files) > 0
                if not status['bart']:
                    status['errors'].append("DistilBART model folder is empty or inaccessible")
            else:
                status['errors'].append("DistilBART model folder ID not configured")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking model availability: {e}")
            return {
                't5': False,
                'bart': False,
                'errors': [f"Drive service error: {str(e)}"]
            }

# Global drive service instance
drive_service = GoogleDriveService()