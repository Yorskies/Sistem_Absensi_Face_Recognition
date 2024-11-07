from datetime import date

def record_attendance(mysql, student_id, status='present'):
    cur = mysql.cursor()
    cur.execute("INSERT INTO attendance_records (student_id, attendance_date, status) VALUES (%s, %s, %s)", (student_id, date.today(), status))
    mysql.commit()
    cur.close()

def get_attendance(mysql):
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT a.id, s.name, a.attendance_date, a.status FROM attendance_records a JOIN students s ON a.student_id = s.id")
    attendance = cur.fetchall()
    cur.close()
    return attendance
