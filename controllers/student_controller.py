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
    cur.close()
