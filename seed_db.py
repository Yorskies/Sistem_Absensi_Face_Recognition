from werkzeug.security import generate_password_hash
import mysql.connector
from config import Config

# Inisialisasi koneksi ke database
mysql = mysql.connector.connect(
    host=Config.MYSQL_HOST,
    user=Config.MYSQL_USER,
    password=Config.MYSQL_PASSWORD,
    database=Config.MYSQL_DB,
    port=Config.MYSQL_PORT
)

def add_admin_user(username, password):
    # Periksa apakah pengguna dengan username ini sudah ada
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    
    if user:
        print(f"User '{username}' already exists.")
    else:
        # Jika tidak ada, tambahkan pengguna baru
        password_hash = generate_password_hash(password)
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)", (username, password_hash, 'admin'))
        mysql.commit()
        print(f"Admin user '{username}' has been added successfully.")
    
    cur.close()

# Tambahkan admin dengan username dan password yang diinginkan
if __name__ == "__main__":
    add_admin_user('admin', 'password123')
    mysql.close()
