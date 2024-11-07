# from flask import Flask, render_template

# app = Flask(__name__, template_folder='views/templates')

# @app.route('/')
# def dashboard():
#     students = [
#         {'name': 'John Doe'},
#         {'name': 'Jane Smith'},
#         {'name': 'Alex Johnson'}
#     ]
#     return render_template('dashboard.html', students=students)

# if __name__ == '__main__':
#     app.run(debug=True)



# from controllers.yolo_controller import YOLOController

# # Inisialisasi YOLOController
# yolo = YOLOController(weights='yolov8n_100e.pt', stream_url='http://192.168.1.35')

# # Mulai deteksi wajah
# yolo.start_detection()

from flask import Flask, session, redirect, url_for, render_template, flash, request
from config import Config
from controllers.user_controller import login, logout
from controllers.student_controller import get_all_students, add_student
from controllers.attendance_controller import record_attendance, get_attendance
import mysql.connector

app = Flask(__name__, template_folder='views/templates')
app.config.from_object(Config)

# Inisialisasi koneksi MySQL
mysql = mysql.connector.connect(
    host=Config.MYSQL_HOST,
    user=Config.MYSQL_USER,
    password=Config.MYSQL_PASSWORD,
    database=Config.MYSQL_DB,
    port=Config.MYSQL_PORT
)

@app.route('/login', methods=['GET', 'POST'])
def login_view():
    return login(mysql)

@app.route('/logout')
def logout_view():
    return logout()

@app.route('/')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_view'))
    return render_template('dashboard.html')

@app.route('/students')
def student_list():
    students = get_all_students(mysql)
    return render_template('student_list.html', students=students)

@app.route('/attendance')
def attendance_list():
    attendance = get_attendance(mysql)
    return render_template('attendance_list.html', attendance=attendance)

if __name__ == '__main__':
    app.secret_key = Config.SECRET_KEY
    app.run(debug=True)

