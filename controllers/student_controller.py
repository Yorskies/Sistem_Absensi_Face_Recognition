def get_all_students(mysql):
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM students")
    students = cur.fetchall()
    cur.close()
    return students

def add_student(mysql, name, student_class, nis):
    cur = mysql.cursor()
    cur.execute("INSERT INTO students (name, class, nis) VALUES (%s, %s, %s)", (name, student_class, nis))
    mysql.commit()
    student_id = cur.lastrowid
    cur.close()
    return student_id  # Mengembalikan ID siswa atau informasi unik lainnya


def get_student_by_id(mysql, student_id):
    cur = mysql.cursor(dictionary=True)
    cur.execute("SELECT * FROM students WHERE id = %s", (student_id,))
    student = cur.fetchone()
    cur.close()
    return student

def update_student(mysql, student_id, name, student_class, nis):
    cur = mysql.cursor()
    cur.execute("UPDATE students SET name = %s, class = %s, nis = %s WHERE id = %s", (name, student_class, nis, student_id))
    mysql.commit()
    cur.close()

def delete_student_by_id(mysql, student_id):
    cur = mysql.cursor()
    cur.execute("DELETE FROM students WHERE id = %s", (student_id,))
    mysql.commit()
    cur.close()

