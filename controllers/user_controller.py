from flask import session, redirect, url_for, flash, request, render_template
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector

def login(mysql):
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash("Username atau password salah.")
            return redirect(url_for('login_view'))
    return render_template('login.html')

def logout():
    session.clear()
    return redirect(url_for('login_view'))
