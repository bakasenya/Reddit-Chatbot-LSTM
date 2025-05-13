import sqlite3
import json
from datetime import datetime
import time

# Configuration
timeframe = '2019-12'
sql_transaction = []
start_row = 0
cleanup = 1000000
db_path = f'C:/Users/swapn.LAPTOP-FETEEA2D/{timeframe}.db'
file_path = f'C:/Users/swapn.LAPTOP-FETEEA2D/RC_{timeframe}'

# Connect to SQLite database
connection = sqlite3.connect(db_path)
cursor = connection.cursor()


def create_table():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parent_reply (
            parent_id TEXT PRIMARY KEY, 
            comment_id TEXT UNIQUE, 
            parent TEXT, 
            comment TEXT, 
            subreddit TEXT, 
            unix INT, 
            score INT
        )
    """)


def format_data(data):
    return data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")


def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        cursor.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                cursor.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = f"""
            UPDATE parent_reply 
            SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? 
            WHERE parent_id = ?
        """
        transaction_bldr(sql, (parentid, commentid, parent, comment, subreddit, int(time), score, parentid))
    except Exception as e:
        print(f's0 insertion error: {e}')


def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = f"""
            INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) 
            VALUES ("{parentid}", "{commentid}", "{parent}", "{comment}", "{subreddit}", {int(time)}, {score})
        """
        transaction_bldr(sql)
    except Exception as e:
        print(f's0 insertion error: {e}')


def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        sql = f"""
            INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) 
            VALUES ("{parentid}", "{commentid}", "{comment}", "{subreddit}", {int(time)}, {score})
        """
        transaction_bldr(sql)
    except Exception as e:
        print(f's0 insertion error: {e}')


def acceptable(data):
    return len(data.split(' ')) <= 1000 and len(data) > 0 and len(data) <= 32000 and data not in ['[deleted]', '[removed]']


def find_parent(pid):
    try:
        sql = f"SELECT comment FROM parent_reply WHERE comment_id = '{pid}' LIMIT 1"
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0] if result else False
    except Exception as e:
        return False


def find_existing_score(pid):
    try:
        sql = f"SELECT score FROM parent_reply WHERE parent_id = '{pid}' LIMIT 1"
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0] if result else False
    except Exception as e:
        return False


if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0

    with open(file_path, buffering=1000) as f:
        for row in f:
            row_counter += 1

            if row_counter > start_row:
                try:
                    row_data = json.loads(row)
                    parent_id = row_data['parent_id'].split('_')[1]
                    body = format_data(row_data['body'])
                    created_utc = row_data['created_utc']
                    score = row_data['score']
                    comment_id = row_data['id']
                    subreddit = row_data['subreddit']

                    parent_data = find_parent(parent_id)
                    existing_comment_score = find_existing_score(parent_id)

                    if existing_comment_score:
                        if score > existing_comment_score:
                            if acceptable(body):
                                sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                    else:
                        if acceptable(body):
                            if parent_data:
                                if score >= 2:
                                    sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                                    paired_rows += 1
                            else:
                                sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)
                except Exception as e:
                    print(f'Error processing row {row_counter}: {e}')

            if row_counter % 100000 == 0:
                print(f'Total Rows Read: {row_counter}, Paired Rows: {paired_rows}, Time: {datetime.now()}')

            if row_counter > start_row and row_counter % cleanup == 0:
                print("Deleting null entries...")
                cursor.execute("DELETE FROM parent_reply WHERE parent IS NULL")
                connection.commit()
                cursor.execute("VACUUM")
                connection.commit()
