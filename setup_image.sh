#!/bin/bash
#
# ============================================================
# ECM-dflow 镜像构建脚本
# ============================================================
# 在玻尔平台节点中执行此脚本，然后保存节点为镜像。
#
# 使用方法（在节点终端中）：
#   1. 上传代码到 /personal/ECM-dflow/
#   2. bash /personal/ECM-dflow/setup_image.sh
#   3. 在平台上将此节点保存为镜像
#
# 此脚本做以下事情：
#   1. 将代码复制到 /opt/ECM-dflow/（会包含在镜像中）
#   2. 安装 Python 依赖
#   3. 【关键】创建 .pth 文件，让 Python 永久认识项目路径
# ============================================================

set -e

echo "======================================================================"
echo "ECM-dflow 镜像环境配置"
echo "======================================================================"

# ============================================================
# 1. 创建项目目录并复制代码
# ============================================================
echo ""
echo "[1/4] 复制项目代码到 /opt/ECM-dflow/ ..."
mkdir -p /opt/ECM-dflow
cp -a /personal/ECM-dflow/. /opt/ECM-dflow/
echo "  OK: 代码已复制"

# ============================================================
# 2. 安装 Python 依赖
# ============================================================
echo ""
echo "[2/4] 安装 Python 依赖 ..."
pip install --no-cache-dir \
    numpy scipy pandas matplotlib seaborn \
    scikit-learn h5py tqdm requests pydflow
echo "  OK: 依赖已安装"

# ============================================================
# 3. 【关键修复】将项目路径写入 .pth 文件
#    这样 Python 启动时会自动将 /opt/ECM-dflow 加入 sys.path
#    使得 "from src.xxx import yyy" 可以在任何地方工作
# ============================================================
echo ""
echo "[3/4] 配置 Python 路径 (.pth 文件) ..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "/opt/ECM-dflow" > "$SITE_PACKAGES/ecm-dflow.pth"
echo "  OK: 已创建 $SITE_PACKAGES/ecm-dflow.pth"

# ============================================================
# 4. 验证
# ============================================================
echo ""
echo "[4/4] 验证配置 ..."

# 测试 Python 路径
python3 -c "
import sys
# 验证 /opt/ECM-dflow 在 sys.path 中
assert '/opt/ECM-dflow' in sys.path, 'ERROR: /opt/ECM-dflow not in sys.path!'
print('  OK: /opt/ECM-dflow 在 sys.path 中')

# 验证 src 模块可以导入
from src.ecm.loader import load_discharge_cc_segment
print('  OK: src.ecm.loader 导入成功')

from src.ecm.ecm2rc import ECM2RCParams, simulate_voltage
print('  OK: src.ecm.ecm2rc 导入成功')

from src.identification.fit import fit_ecm_params
print('  OK: src.identification.fit 导入成功')

from src.analysis.ci import analyze_parameter_uncertainty
print('  OK: src.analysis.ci 导入成功')

from src.analysis.bootstrap import residual_bootstrap
print('  OK: src.analysis.bootstrap 导入成功')

from src.analysis.sensitivity import local_sensitivity_analysis
print('  OK: src.analysis.sensitivity 导入成功')

# 验证 dflow 可以导入
from dflow.python import OP
print('  OK: dflow 导入成功')

print()
print('  ALL CHECKS PASSED!')
"

echo ""
echo "======================================================================"
echo "  镜像配置完成！"
echo "  请在玻尔平台上将此节点保存为镜像。"
echo "  保存后，记住镜像地址并更新 dflow_workflow.py 中的 DEFAULT_IMAGE"
echo "======================================================================"
