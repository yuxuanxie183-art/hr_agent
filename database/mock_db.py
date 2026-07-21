"""
==========================================================================
模拟数据库模块 (Mock Database Module)
==========================================================================
功能概述:
  1. 使用 SQLite 在本地创建一个轻量级的人力资源数据库
  2. 支持手动初始化建表 + 注入模拟测试数据
  3. 提供通用的数据库连接、查询、关闭等工具函数
  4. 供业务逻辑层调用，避免直接依赖重量级数据库（如 MySQL/PostgreSQL）

适用场景:
  - 本地开发调试
  - 单元测试
  - 原型验证 / PoC 演示

运行方式（手动初始化）:
  python database/mock_db.py        # 直接执行本脚本，建表并落盘测试数据

注意事项:
  - 每次执行 init_db() 会清空旧数据后重新插入，仅适用于 Mock 场景
  - 生产环境请替换为真实数据库连接
==========================================================================
"""

import sqlite3
from pathlib import Path

# ==========================================================================
# 路径常量
# ==========================================================================

# 项目根目录：当前文件向上两级即为项目根
# 例如：database/mock_db.py → 上级 database → 再上级 → 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据库文件完整路径
# 数据库文件存放在项目根目录下的 db/employees.db
DB_PATH = PROJECT_ROOT / 'db' / 'employees.db'


# ==========================================================================
# 数据库连接
# ==========================================================================

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """业务运行时获取数据库连接，仅负责连接并开启外键约束

    与 init_db() 的区别：
      - init_db() 用于首次手动建表 + 灌数据
      - get_connection() 用于日常业务读写，不执行 DDL

    参数:
        db_path (Path): 数据库文件路径，默认使用项目 DB_PATH

    返回:
        sqlite3.Connection: 已开启外键约束的数据库连接对象

    异常:
        FileNotFoundError: 当数据库文件不存在时抛出，提示用户先执行初始化脚本
    """
    # 防御性检查：如果 .db 文件不存在，说明还没初始化
    if not db_path.exists():
        raise FileNotFoundError(
            f'[错误] 数据库文件未找到: {db_path}\n'
            f'请先在终端手动运行一遍初始化脚本：python database/mock_db.py\n'
        )
    # 创建连接（check_same_thread 默认 True，适合单线程业务场景）
    conn = sqlite3.connect(str(db_path),check_same_thread=False)
    # 开启外键约束支持（SQLite 默认关闭外键检查）
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


# ==========================================================================
# 数据库初始化
# ==========================================================================

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """数据库初始化与测试数据落盘（仅手动单次运行，不可重复调用）

    执行流程:
      1. 创建 db 目录（如不存在）
      2. 建立数据库连接
      3. 创建 employees（员工表）和 leave_balances（假期余额表）
      4. 清空旧数据
      5. 插入预设的 4 条模拟员工 + 假期数据
      6. 提交事务并打印确认信息

    参数:
        db_path (Path): 数据库文件路径，默认使用项目 DB_PATH

    返回:
        sqlite3.Connection: 初始化完成后的数据库连接对象
    """
    # ---- 步骤1：确保存放数据库的目录存在 ----
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 步骤2：创建数据库连接 ----
    # check_same_thread=False：允许多线程共享连接（SQLite 默认只允许创建连接的线程使用）
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute('PRAGMA foreign_keys = ON')

    # ---- 步骤3：获取游标，用于执行 SQL 语句 ----
    # 游标（Cursor）是执行 SQL 并遍历结果集的句柄
    cursor = conn.cursor()

    # ---- 步骤4：创建 employees（员工表） ----
    # 字段说明:
    #   uid           : TEXT PRIMARY KEY — 员工唯一标识（如工号 "1001"）
    #   name          : TEXT NOT NULL    — 员工姓名
    #   rank          : TEXT NOT NULL    — 职级（如 P3 / P4 / P5 / P7）
    #   location      : TEXT NOT NULL    — 工作城市（如 "北京"）
    #   seniority     : INTEGER NOT NULL — 入职年限（单位：年）
    #   base_salary   : INTEGER NOT NULL — 基本工资（单位：元/月）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        uid         TEXT PRIMARY KEY,        -- 员工唯一标识
        name        TEXT NOT NULL,           -- 员工姓名
        rank        TEXT NOT NULL,           -- 职级（P3, P4, P5, P7 等）
        location    TEXT NOT NULL,           -- 工作地点（城市名称）
        seniority   INTEGER NOT NULL,        -- 入职年限（年）
        base_salary INTEGER NOT NULL         -- 基本工资（元/月）
    )
    """)

    # ---- 步骤5：创建 leave_balances（假期余额表） ----
    # 字段说明:
    #   uid                     : TEXT PRIMARY KEY — 员工唯一标识（外键关联 employees.uid）
    #   annual_leave_remaining  : INTEGER NOT NULL — 剩余年假天数
    #   sick_leave_remaining    : INTEGER NOT NULL — 剩余病假天数
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leave_balances (
        uid                     TEXT PRIMARY KEY,   -- 员工唯一标识（外键 → employees.uid）
        annual_leave_remaining  INTEGER NOT NULL,   -- 剩余年假天数
        sick_leave_remaining    INTEGER NOT NULL,   -- 剩余病假天数
        FOREIGN KEY (uid) REFERENCES employees (uid)
    )
    """)

    # ---- 步骤6：清空旧数据，避免重复插入或脏数据残留 ----
    # 注意：必须先删子表（leave_balances）再删父表（employees），否则会违反外键约束
    cursor.execute("DELETE FROM leave_balances")
    cursor.execute("DELETE FROM employees where 1=1")

    # ---- 步骤7：准备模拟测试数据 ----

    # 模拟员工数据（4 条）
    # 每条记录格式：(uid, name, rank, location, seniority, base_salary)
    test_employees = [
        ('1001', '张三', 'P5', '北京', 2, 18000),
        ('1002', '李四', 'P4', '成都', 4,  9000),
        ('1003', '王五', 'P7', '上海', 5, 35000),
        ('1004', '赵六', 'P3', '深圳', 6,  7500),
    ]

    # 模拟假期余额数据（4 条，与上面员工一一对应）
    # 每条记录格式：(uid, annual_leave_remaining, sick_leave_remaining)
    test_balances = [
        ('1001',  6, 10),   # 张三：年假 6 天，病假 10 天
        ('1002',  7, 12),   # 李四：年假 7 天，病假 12 天
        ('1003', 14, 15),   # 王五：年假 14 天，病假 15 天（高P，假期更多）
        ('1004',  2,  5),   # 赵六：年假 2 天，病假 5 天
    ]

    # ---- 步骤8：批量插入数据 ----
    # executemany() 比循环调用 execute() 效率更高，一次性绑定多组参数
    # ? 是 SQLite 的参数占位符，可防止 SQL 注入
    cursor.executemany(
        'INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)',
        test_employees
    )
    cursor.executemany(
        'INSERT INTO leave_balances VALUES (?, ?, ?)',
        test_balances
    )

    # ---- 步骤9：提交事务 ----
    # 所有 DML 操作（INSERT/UPDATE/DELETE）必须 commit 才能真正落盘
    conn.commit()

    print('✅ 初始化成功！模拟数据已落盘')
    print(f'📁 数据库路径: {db_path}')
    return conn


# ==========================================================================
# 通用工具函数
# ==========================================================================

def query_db(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """通用查询函数 —— 执行 SELECT 并将结果转为字典列表

    数据转换流程:
      cursor.description → 提取列名列表
      cursor.fetchall()  → 获取所有数据行
      zip(columns, row)  → 将列名与每行值配对
      dict()             → 转为 {列名: 值} 字典

    参数:
        conn (sqlite3.Connection): 数据库连接对象
        sql (str):                 要执行的 SQL 查询语句（仅支持 SELECT）
        params (tuple):            SQL 参数，使用 ? 占位符绑定，防注入

    返回:
        list[dict]: 查询结果列表，每行是一个字典
                    例如: [{"uid": "1001", "name": "张三", ...}, ...]

    使用示例:
        results = query_db(conn, "SELECT * FROM employees WHERE rank = ?", ("P5",))
        # results → [{"uid": "1001", "name": "张三", "rank": "P5", ...}]
    """
    cursor = conn.cursor()
    cursor.execute(sql, params)

    # 从游标的 description 属性中提取列名
    # description 格式: (('uid', None, ...), ('name', None, ...), ...)
    # 每个元素是一个 7 元组，第 0 项为列名
    columns = [col[0] for col in cursor.description]

    # 遍历所有行，将每行与列名 zip 后转为字典
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def close_db(conn: sqlite3.Connection):
    """安全关闭数据库连接

    设计理念:
      - 防止重复关闭（先判断 conn 非空）
      - 关闭后打印确认信息，方便调试排错

    参数:
        conn (sqlite3.Connection): 要关闭的数据库连接对象
    """
    if conn:
        conn.close()
        print('🔒 数据库连接已安全关闭')


# ==========================================================================
# 手动初始化入口
# ==========================================================================

# 当直接运行本脚本时（python database/mock_db.py），自动执行初始化流程
if __name__ == '__main__':
    print('🚀 正在执行数据库手动初始化操作 ...')
    standalone_conn = init_db()       # 建表 + 灌数据
    close_db(standalone_conn)         # 关闭连接，释放资源
