import mysql.connector

conn = None


def get_new_cursor():
    global conn
    if conn is None or not conn.is_connected():
        conn = mysql.connector.connect(user='root', password='lyp82ndlf', host='localhost', database='trading')
    return conn.cursor()
