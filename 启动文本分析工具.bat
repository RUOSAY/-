@echo off
title 文本分析工具
echo ========================================
echo    文本分析工具启动中...
echo ========================================
echo.

REM 切换到当前目录
cd /d "%~dp0"

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到Python环境，请先安装Python并添加到系统PATH
    echo.
    pause
    exit /b 1
)

REM 检查streamlit
where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装Streamlit...
    pip install streamlit
    echo.
)

REM 启动应用
echo 正在启动应用，请稍候...
echo 浏览器将自动打开...
echo.
echo 提示：关闭此窗口即可退出应用
echo ========================================
echo.

streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo 启动失败！请检查：
    echo 1. 依赖是否完整（运行: pip install -r requirements.txt）
    echo 2. app.py 文件是否存在
    echo ========================================
    pause
)
