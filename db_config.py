import pymysql

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='stroke_db1',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.Cursor
    )
