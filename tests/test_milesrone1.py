import sys
from pathlib import Path

# 为了防止ModuleNotFoundError 我们加入下面两行代码 可以自动将项目根目录挂在到系统环境变量中，调高鲁棒性
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))



from tools.hr_tools import (get_employee_profile,
                            get_leave_balance,generate_employment_certificate)


def test_get_employee_profile():
    """测试1：查看张三（1001）的信息"""
    result = get_employee_profile.invoke({'uid':'1001'})
    # 断言
    assert '张三' in result
    assert 'P5' in result


def test_get_leave_balance():
    """测试2：李四剩余假期"""
    result = get_leave_balance.invoke({'uid': '1002'})
    assert '李四' in result
    assert '7' in result

def test_generate_employment_certificate_P5():
    """测试3：张三打印收入证明"""
    result = generate_employment_certificate.invoke({'uid': '1001','cer_type':'income'})
    assert '系统成功' in result
    assert '收入证明' in result

def test_generate_employment_certificate_P4():
    """测试4：李四打印收入证明"""
    result = generate_employment_certificate.invoke({'uid': '1002','cer_type':'income'})
    assert '无法' in result




if __name__ == '__main__':
    print('------------测试一，查看张三的档案')
    print(get_employee_profile.invoke({'uid':'1001'}))

    print('------------测试二，查看李四的剩余假期')
    print(get_leave_balance.invoke({'uid': '1002'}))

    print('------------测试三，查看张三（P5）的收入证明')
    print(generate_employment_certificate.invoke({'uid': '1001','cer_type':'income'}))

    print('------------测试四，查看李四（P4）的收入证明')
    print(generate_employment_certificate.invoke({'uid': '1002', 'cer_type': 'income'}))