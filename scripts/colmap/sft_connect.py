import paramiko
import os

# SFTP connection details
hostname = "140.112.94.123"
port = 20200  # default SFTP port
username = "noahh"
password = "Didgerid00"
remote_dir = "/volume1/UAV_Data/00_Data/01_plants_raw_data/2024-7-30-melon/2025-07-30"
local_dir = r"C:\Users\BBLab\Documents\nerf-preprocessing\data"

def download_files():
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        os.makedirs(local_dir, exist_ok=True)
        for filename in sftp.listdir(remote_dir):
            remote_filepath = os.path.join(remote_dir, filename)
            local_filepath = os.path.join(local_dir, filename)
            sftp.get(remote_filepath, local_filepath)
            print(f"Downloaded: {filename}")
    finally:
        sftp.close()
        transport.close()

download_files()