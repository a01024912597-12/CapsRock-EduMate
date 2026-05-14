import os
import sqlite3
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db.sqlite3")


def get_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()


def get_column_names(columns):
    return [column[1] for column in columns]


def main():
    print("DB 경로:", DB_PATH)

    if not os.path.exists(DB_PATH):
        print("db.sqlite3 파일을 찾을 수 없습니다.")
        return

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lectures_quiz'"
    )
    table = cursor.fetchone()

    if not table:
        print("lectures_quiz 테이블이 없습니다.")
        print("먼저 python manage.py makemigrations lectures 와 python manage.py migrate lectures 를 실행하세요.")
        connection.close()
        return

    before_columns = get_columns(cursor, "lectures_quiz")
    names = get_column_names(before_columns)

    print("\n[수정 전 lectures_quiz 컬럼]")
    for column in before_columns:
        print(column)

    if "quiz_text" not in names:
        cursor.execute(
            "ALTER TABLE lectures_quiz ADD COLUMN quiz_text text NOT NULL DEFAULT ''"
        )
        print("\nquiz_text 컬럼을 추가했습니다.")
    else:
        print("\nquiz_text 컬럼이 이미 있습니다.")

    if "created_at" not in names:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            f"ALTER TABLE lectures_quiz ADD COLUMN created_at datetime NOT NULL DEFAULT '{now}'"
        )
        print("created_at 컬럼을 추가했습니다.")
    else:
        print("created_at 컬럼이 이미 있습니다.")

    if "generation_number" not in names:
        cursor.execute(
            "ALTER TABLE lectures_quiz ADD COLUMN generation_number integer NOT NULL DEFAULT 1"
        )
        print("generation_number 컬럼을 추가했습니다.")
    else:
        print("generation_number 컬럼이 이미 있습니다.")

    connection.commit()

    after_columns = get_columns(cursor, "lectures_quiz")

    print("\n[수정 후 lectures_quiz 컬럼]")
    for column in after_columns:
        print(column)

    connection.close()
    print("\nDB 수정 작업이 끝났습니다.")


if __name__ == "__main__":
    main()