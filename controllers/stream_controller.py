from flask import Blueprint, render_template

# Membuat blueprint untuk stream controller
stream_bp = Blueprint('stream', __name__)

@stream_bp.route('/view_stream')
def view_stream():
    """Menampilkan halaman dengan stream dari ESP32."""
    return render_template('view_stream.html')
