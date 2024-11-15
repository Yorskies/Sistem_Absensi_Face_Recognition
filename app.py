import threading
from flask import Flask, session, redirect, url_for, render_template, flash, request, Response
from config import Config
from controllers.user_controller import login, logout
from controllers.student_controller import delete_student_by_id, get_all_students, add_student, get_student_by_id, update_student
from controllers.attendance_controller import record_attendance, get_attendance
from controllers.stream_controller import stream_bp
import mysql.connector
from controllers.yolo_controller import YOLOController

app = Flask(__name__, template_folder='views/templates')
app.config.from_object(Config)

# Inisialisasi YOLOController dengan URL stream ESP32
yolo = YOLOController(weights='yolov8n_100e.pt', stream_url=0)  # Menggunakan `0` untuk webcam laptop

def get_db_connection():
    """Create a new database connection for each request."""
    return mysql.connector.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB,
        port=Config.MYSQL_PORT
    )

@app.route('/login', methods=['GET', 'POST'])
def login_view():
    with get_db_connection() as mysql:
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
    with get_db_connection() as mysql:
        students = get_all_students(mysql)
    return render_template('student_list.html', students=students)

@app.route('/attendance')
def attendance_list():
    with get_db_connection() as mysql:
        attendance = get_attendance(mysql)
    return render_template('attendance_list.html', attendance=attendance)

@app.route('/students/add', methods=['GET', 'POST'])
def add_student_view():
    if request.method == 'POST':
        name = request.form['name']
        student_class = request.form['class']
        nis = request.form['nis']
        
        with get_db_connection() as mysql:
            add_student(mysql, name, student_class, nis)

        flash("Data siswa berhasil ditambahkan. Memulai scan wajah...")

        # Simpan `name` dan `nis` ke session untuk digunakan di video stream
        session['student_name'] = name
        session['student_nis'] = nis

        # Arahkan ke halaman video stream
        return redirect(url_for('start_video', name=name, nis=nis))
    
    return render_template('add_student.html')


@app.route('/students/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student_view(student_id):
    with get_db_connection() as mysql:
        if request.method == 'POST':
            name = request.form['name']
            student_class = request.form['class']
            nis = request.form['nis']
            update_student(mysql, student_id, name, student_class, nis)
            return redirect(url_for('student_list'))
        
        student = get_student_by_id(mysql, student_id)
    return render_template('edit_student.html', student=student)

@app.route('/students/delete/<int:student_id>', methods=['GET'])
def delete_student(student_id):
    with get_db_connection() as mysql:
        delete_student_by_id(mysql, student_id)
    return redirect(url_for('student_list'))

@app.route('/frame')
def get_frame():
    """Send single frame for AJAX request."""
    frame = yolo.generate_frame(name=session.get('student_name'), nis=session.get('student_nis'))
    if frame is None:
        return Response(status=404)
    return Response(frame, mimetype='image/jpeg')

@app.route('/video_stream')
def video_stream():
    """Streaming endpoint untuk halaman HTML."""
    return render_template('video_stream.html', name=session.get('student_name'), nis=session.get('student_nis'))

@app.route('/start_video/<name>/<nis>')
def start_video(name, nis):
    """Start video streaming with face detection."""
    session['student_name'] = name
    session['student_nis'] = nis
    return redirect(url_for('video_stream'))

@app.route('/video_feed')
def video_feed():
    """Streaming video dengan deteksi wajah dan bounding box."""
    # Pastikan nama dan NIS tersedia di session
    name = session.get('student_name')
    nis = session.get('student_nis')
    
    # Jika session tidak memiliki nama dan NIS, kirim 404
    if not name or not nis:
        return Response(status=404)

    # Menghasilkan stream video menggunakan YOLOController
    return Response(yolo.generate_frame(name=name, nis=nis), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.secret_key = Config.SECRET_KEY
    app.run(debug=True, host='0.0.0.0', port=5001)
