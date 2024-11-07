import os

class Config:
    # Konfigurasi dasar
    SECRET_KEY = os.urandom(24)
    
    # Konfigurasi MySQL
    MYSQL_HOST = 'localhost'
    MYSQL_PORT = 3306
    MYSQL_USER = 'root'  # Ganti dengan username MySQL Anda
    MYSQL_PASSWORD = 'root'  # Ganti dengan password MySQL Anda
    MYSQL_DB = 'sistem_absensi'  # Nama database yang digunakan
