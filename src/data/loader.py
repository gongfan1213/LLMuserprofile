import logging
import pickle
import sqlite3
from typing import List

from src.models.user_profile import UserProfile, UserSegment

logger = logging.getLogger(__name__)


def save_user_profile(obj: UserProfile, user_id: str, db_path: str = "./user.db", table_name: str = "user_profile"):
    pickled_object = pickle.dumps(obj)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表 (如果不存在)
    # 存储键 (TEXT) 和序列化后的对象 (BLOB)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            data BLOB
        )
    ''')

    cursor.execute(f"INSERT OR REPLACE INTO {table_name} (id, data) VALUES (?, ?)", (user_id, pickled_object))
    conn.commit()
    conn.close()
    logger.info(f"对象已成功以键 '{user_id}' 保存到 '{db_path}' 的 '{table_name}' 表中。")


def load_user_profile(user_id: str, db_path: str = "./user.db", table_name: str = "user_profile") -> UserProfile | None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT data FROM {table_name} WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            pickled_object = row[0]
            obj = pickle.loads(pickled_object)
            logger.info(f"对象已成功从键 '{user_id}' 加载。")
            return obj
        else:
            logger.info(f"未在表 '{table_name}' 中找到键 '{user_id}'。")
            return None
    except sqlite3.OperationalError:
        logger.info(f"表 '{table_name}' 不存在或查询出错。")
        return None
    finally:
        conn.close()


def load_all_users(db_path: str = "./user.db", table_name: str = "user_profile") -> List[UserProfile]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    users = []

    try:
        cursor.execute(f"SELECT id, data FROM {table_name}")
        rows = cursor.fetchall()
        for row in rows:
            user_id, pickled_object = row
            obj = pickle.loads(pickled_object)
            users.append(obj)
        logger.info(f"已加载所有用户数据，共 {len(users)} 个用户。")
    except sqlite3.OperationalError:
        logger.info(f"表 '{table_name}' 不存在或查询出错。")
    finally:
        conn.close()

    return users


# --- 使用示例 ---

if __name__ == '__main__':
    # 定义一些要存储的 Python 对象
    my_dict = {'name': 'Alice', 'age': 30, 'cities': ['New York', 'London']}
    my_list = [1, 2, 'hello', {'value': 42}]
    my_custom_object = "这是一个简单的字符串对象"  # 几乎任何可 pickle 的对象都可以

    # 数据库和表信息
    DB_FILE = 'objects.db'
    TABLE = 'my_objects'

    # 1. 存储对象
    print("--- 正在存储对象 ---")
    save_user_profile(my_dict, DB_FILE, TABLE, 'user_profile')
    save_user_profile(my_list, DB_FILE, TABLE, 'data_list')
    save_user_profile(my_custom_object, DB_FILE, TABLE, 'greeting_message')

    print("\n" + "=" * 30 + "\n")

    # 2. 加载对象
    print("--- 正在加载对象 ---")
    loaded_dict = load_user_profile(DB_FILE, TABLE, 'user_profile')
    loaded_list = load_user_profile(DB_FILE, TABLE, 'data_list')
    loaded_message = load_user_profile(DB_FILE, TABLE, 'greeting_message')
    non_existent = load_user_profile(DB_FILE, TABLE, 'non_existent_key')

    # 3. 验证加载的对象
    print("\n--- 验证加载的对象 ---")
    print(f"加载的字典: {loaded_dict}, 类型: {type(loaded_dict)}")
    print(f"加载的列表: {loaded_list}, 类型: {type(loaded_list)}")
    print(f"加载的消息: {loaded_message}, 类型: {type(loaded_message)}")
    print(f"不存在的对象: {non_existent}")
