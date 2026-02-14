"""
测试运行脚本
提供统一的测试入口，方便运行所有测试或特定模块的测试
"""

import sys
import argparse
from pathlib import Path

# 添加 src 到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def run_loader_test():
    """运行 loader 模块测试"""
    print("\n" + "=" * 70)
    print("运行 loader.py 模块测试")
    print("=" * 70 + "\n")
    
    # 直接导入测试脚本
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_loader",
        project_root / "tests" / "test_loader.py"
    )
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    return test_module.main()


def run_ocv_test():
    """运行 OCV 模块测试"""
    print("\n" + "=" * 70)
    print("运行 ocv.py 模块测试")
    print("=" * 70 + "\n")
    
    # 直接导入测试脚本
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_ocv",
        project_root / "tests" / "test_ocv.py"
    )
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    return test_module.main()


def run_ecm2rc_test():
    """运行 ECM2RC 模块测试"""
    print("\n" + "=" * 70)
    print("运行 ecm2rc.py 模块测试")
    print("=" * 70 + "\n")
    
    # 直接导入测试脚本
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_ecm2rc",
        project_root / "tests" / "test_ecm2rc.py"
    )
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    return test_module.main()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("运行所有测试")
    print("=" * 70 + "\n")
    
    results = {}
    
    # 测试 loader
    print("\n[1/3] 测试 loader 模块...")
    results['loader'] = run_loader_test()
    
    # 测试 OCV
    print("\n[2/3] 测试 OCV 模块...")
    results['ocv'] = run_ocv_test()
    
    # 测试 ECM2RC
    print("\n[3/3] 测试 ECM2RC 模块...")
    results['ecm2rc'] = run_ecm2rc_test()
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for module, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {module}")
    
    print(f"\n总计: {passed}/{total} 个模块测试通过")
    print("=" * 70 + "\n")
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="运行 ECM 项目测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'module',
        nargs='?',
        default='all',
        choices=['all', 'loader', 'ocv', 'ecm2rc'],
        help='要测试的模块 (默认: all)'
    )
    
    args = parser.parse_args()
    
    if args.module == 'all':
        success = run_all_tests()
    elif args.module == 'loader':
        success = run_loader_test()
    elif args.module == 'ocv':
        success = run_ocv_test()
    elif args.module == 'ecm2rc':
        success = run_ecm2rc_test()
    else:
        print(f"未知的模块: {args.module}")
        return False
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
