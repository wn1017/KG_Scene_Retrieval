@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CHECK_ONLY=0"
set "START_SERVICES=1"
set "SERVICES_ONLY=0"
set "SERVICE_VARS_FILE=.startup_service_vars.cmd"
set "DOCKER_WAIT_ATTEMPTS=40"
set "PORT_WAIT_ATTEMPTS=45"
set "APP_PORT=7860"
set "ATTU_PORT=8000"
set "APP_URL=http://127.0.0.1:%APP_PORT%"
set "ATTU_URL=http://127.0.0.1:%ATTU_PORT%"
set "NEO4J_BROWSER_URL=http://127.0.0.1:7474"
set "ATTU_CONTAINER_NAME=attu"
set "ATTU_MILVUS_URL=host.docker.internal:19530"
set "DOCKER_ATTU_IMAGE=zilliz/attu:v2.6.3"

:parse_args
if "%~1"=="" goto after_args
if /I "%~1"=="--check" (
    set "CHECK_ONLY=1"
    set "START_SERVICES=0"
)
if /I "%~1"=="--services" set "START_SERVICES=1"
if /I "%~1"=="--services-only" (
    set "START_SERVICES=1"
    set "SERVICES_ONLY=1"
)
shift
goto parse_args

:after_args
echo [1/5] Workspace: %CD%

set "MISSING=0"
if not exist "config.py" (
    echo [ERROR] Missing config.py
    set "MISSING=1"
)
if not exist "app.py" (
    echo [ERROR] Missing app.py
    set "MISSING=1"
)
if not exist "src\milvus_utils.py" (
    echo [ERROR] Missing src\milvus_utils.py
    set "MISSING=1"
)
if not exist "src\kg_builder.py" (
    echo [ERROR] Missing src\kg_builder.py
    set "MISSING=1"
)
if not exist "src\nlp_parser.py" (
    echo [ERROR] Missing src\nlp_parser.py
    set "MISSING=1"
)
if not exist "src\nuscenes_metadata.py" (
    echo [ERROR] Missing src\nuscenes_metadata.py
    set "MISSING=1"
)
if not exist "embedEtcd.yaml" (
    echo [ERROR] Missing embedEtcd.yaml
    set "MISSING=1"
)
if not exist "user.yaml" (
    echo [ERROR] Missing user.yaml
    set "MISSING=1"
)
if not exist "csvdata\nuScenes_v1.0_mini.csv" (
    echo [ERROR] Missing csvdata\nuScenes_v1.0_mini.csv
    set "MISSING=1"
)
if not exist "models\engclip\config.json" (
    echo [ERROR] Missing models\engclip\config.json
    set "MISSING=1"
)
if not exist "models\engclip\pytorch_model.bin" (
    echo [ERROR] Missing models\engclip\pytorch_model.bin
    set "MISSING=1"
)
if not exist "models\chnclip\config.json" (
    echo [ERROR] Missing models\chnclip\config.json
    set "MISSING=1"
)
if not exist "models\chnclip\pytorch_model.bin" (
    echo [ERROR] Missing models\chnclip\pytorch_model.bin
    set "MISSING=1"
)

if "%MISSING%"=="1" (
    echo.
    echo Startup aborted because required files are missing.
    exit /b 1
)

set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
if defined NO_PROXY (
    set "NO_PROXY=127.0.0.1,localhost,::1,%NO_PROXY%"
) else (
    set "NO_PROXY=127.0.0.1,localhost,::1"
)
if defined no_proxy (
    set "no_proxy=127.0.0.1,localhost,::1,%no_proxy%"
) else (
    set "no_proxy=127.0.0.1,localhost,::1"
)
set "GRADIO_SERVER_NAME=127.0.0.1"
set "GRADIO_SERVER_PORT=%APP_PORT%"

call conda run -n kg python -c "from config import APP_PORT, ATTU_CONTAINER_NAME, ATTU_MILVUS_URL, ATTU_PORT, DOCKER_ATTU_IMAGE, DOCKER_MILVUS_HEALTH_PORT, DOCKER_MILVUS_IMAGE, DOCKER_NEO4J_IMAGE, MILVUS_PORT, NEO4J_HTTP_PORT, NEO4J_URI; bolt_port=int(NEO4J_URI.rsplit(':', 1)[1]); lines=[f'set \"APP_PORT={APP_PORT}\"', f'set \"ATTU_CONTAINER_NAME={ATTU_CONTAINER_NAME}\"', f'set \"ATTU_MILVUS_URL={ATTU_MILVUS_URL}\"', f'set \"ATTU_PORT={ATTU_PORT}\"', f'set \"DOCKER_ATTU_IMAGE={DOCKER_ATTU_IMAGE}\"', f'set \"DOCKER_MILVUS_IMAGE={DOCKER_MILVUS_IMAGE}\"', f'set \"DOCKER_NEO4J_IMAGE={DOCKER_NEO4J_IMAGE}\"', f'set \"MILVUS_PORT={MILVUS_PORT}\"', f'set \"NEO4J_HTTP_PORT={NEO4J_HTTP_PORT}\"', f'set \"NEO4J_BOLT_PORT={bolt_port}\"', f'set \"DOCKER_MILVUS_HEALTH_PORT={DOCKER_MILVUS_HEALTH_PORT}\"']; print(*lines, sep='\n')" > "%SERVICE_VARS_FILE%"
if errorlevel 1 (
    echo [ERROR] Failed to load Docker service variables from config.py.
    exit /b 1
)
call "%SERVICE_VARS_FILE%"
if errorlevel 1 (
    echo [ERROR] Failed to apply Docker service variables.
    exit /b 1
)
if exist "%SERVICE_VARS_FILE%" del /q "%SERVICE_VARS_FILE%" >nul 2>&1
set "APP_URL=http://127.0.0.1:%APP_PORT%"
set "ATTU_URL=http://127.0.0.1:%ATTU_PORT%"
set "NEO4J_BROWSER_URL=http://127.0.0.1:%NEO4J_HTTP_PORT%"
set "GRADIO_SERVER_PORT=%APP_PORT%"

if "%START_SERVICES%"=="0" goto after_service_setup

call :detect_running_services
if "%SERVICES_ALREADY_ONLINE%"=="1" (
    echo [INFO] Milvus and Neo4j are already reachable. Skipping Docker startup.
    call :ensure_attu_if_possible
    goto after_service_setup
)

echo [2/5] Ensuring local services
set /a DOCKER_WAIT_COUNT=0
:docker_wait_loop
docker version >nul 2>&1
if not errorlevel 1 goto docker_ready
if %DOCKER_WAIT_COUNT% GEQ %DOCKER_WAIT_ATTEMPTS% (
    echo [ERROR] Docker Desktop is not ready. Open Docker Desktop and wait for it to finish starting, then try again.
    exit /b 1
)
if %DOCKER_WAIT_COUNT% EQU 0 echo [INFO] Waiting for Docker Desktop to become ready...
timeout /t 3 /nobreak >nul
set /a DOCKER_WAIT_COUNT+=1
goto docker_wait_loop

:docker_ready
docker inspect neo4j >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating Neo4j container...
    docker run -d --name neo4j -p %NEO4J_HTTP_PORT%:7474 -p %NEO4J_BOLT_PORT%:7687 -e NEO4J_AUTH=none -v neo4j_data:/data -v neo4j_logs:/logs %DOCKER_NEO4J_IMAGE% >nul
    if errorlevel 1 (
        echo [ERROR] Failed to create or start Neo4j.
        exit /b 1
    )
) else (
    docker start neo4j >nul 2>&1
)

docker inspect milvus-standalone >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating Milvus container...
    set "WORKSPACE_UNIX=%CD:\=/%"
    docker run -d --name milvus-standalone --security-opt seccomp=unconfined -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml -e COMMON_STORAGETYPE=local -e DEPLOY_MODE=STANDALONE -v kg_scene_retrieval_milvus_data:/var/lib/milvus -v "%WORKSPACE_UNIX%/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml:ro" -v "%WORKSPACE_UNIX%/user.yaml:/milvus/configs/user.yaml:ro" -p %MILVUS_PORT%:19530 -p %DOCKER_MILVUS_HEALTH_PORT%:9091 %DOCKER_MILVUS_IMAGE% milvus run standalone >nul
    if errorlevel 1 (
        echo [ERROR] Failed to create or start Milvus.
        exit /b 1
    )
) else (
    docker start milvus-standalone >nul 2>&1
)

echo [INFO] Waiting for Neo4j on port %NEO4J_BOLT_PORT%...
set /a NEO4J_WAIT_COUNT=%PORT_WAIT_ATTEMPTS%
:neo4j_wait_loop
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %NEO4J_BOLT_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 goto neo4j_ready
if %NEO4J_WAIT_COUNT% LEQ 0 (
    echo [ERROR] Neo4j did not open port %NEO4J_BOLT_PORT% in time.
    exit /b 1
)
timeout /t 2 /nobreak >nul
set /a NEO4J_WAIT_COUNT-=1
goto neo4j_wait_loop

:neo4j_ready
echo [INFO] Waiting for Milvus on port %MILVUS_PORT%...
set /a MILVUS_WAIT_COUNT=%PORT_WAIT_ATTEMPTS%
:milvus_wait_loop
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %MILVUS_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 goto core_services_ready
if %MILVUS_WAIT_COUNT% LEQ 0 (
    echo [ERROR] Milvus did not open port %MILVUS_PORT% in time.
    exit /b 1
)
timeout /t 2 /nobreak >nul
set /a MILVUS_WAIT_COUNT-=1
goto milvus_wait_loop

:core_services_ready
call :ensure_attu_if_possible

:after_service_setup
if "%START_SERVICES%"=="0" echo [2/5] Service startup skipped in check mode.

echo [3/5] Runtime checks
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %MILVUS_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [WARN] Milvus is not listening on port %MILVUS_PORT%.
) else (
    echo [OK] Milvus is listening on port %MILVUS_PORT%.
)
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %NEO4J_BOLT_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [WARN] Neo4j is not listening on port %NEO4J_BOLT_PORT%.
) else (
    echo [OK] Neo4j is listening on port %NEO4J_BOLT_PORT%.
)
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %ATTU_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [WARN] Attu is not listening on port %ATTU_PORT%.
) else (
    echo [OK] Attu is listening on port %ATTU_PORT%.
)

if "%SERVICES_ONLY%"=="1" (
    call :print_access_urls
    echo Services started.
    exit /b 0
)

echo [4/5] Python environment check
call conda run -n kg python -c "from pathlib import Path; compile(Path(r'app.py').read_text(encoding='utf-8'), 'app.py', 'exec'); print('APP_SYNTAX_OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] Python environment check failed.
    exit /b 1
)

if "%CHECK_ONLY%"=="1" (
    echo.
    echo Check completed successfully.
    exit /b 0
)

call :stop_existing_app_listener
if errorlevel 1 exit /b 1

echo [5/5] Launching Gradio app
call :print_access_urls
echo [INFO] Web UI may take 20-60 seconds on a cold start while local models load.
call conda run -n kg python app.py
exit /b %errorlevel%

:stop_existing_app_listener
set "APP_LISTENER_PID="
for /f %%P in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort %APP_PORT% -State Listen -ErrorAction SilentlyContinue ^| Select-Object -First 1; if ($conn) { $conn.OwningProcess }"') do set "APP_LISTENER_PID=%%P"
if not defined APP_LISTENER_PID exit /b 0
echo [INFO] Port %APP_PORT% is already in use by PID %APP_LISTENER_PID%. Stopping the previous app process...
taskkill /PID %APP_LISTENER_PID% /F >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to stop the existing process on port %APP_PORT%.
    exit /b 1
)
set /a APP_PORT_RELEASE_WAIT=15
:app_port_release_loop
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %APP_PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 exit /b 0
if %APP_PORT_RELEASE_WAIT% LEQ 0 (
    echo [ERROR] Port %APP_PORT% is still busy after stopping the previous process.
    exit /b 1
)
timeout /t 1 /nobreak >nul
set /a APP_PORT_RELEASE_WAIT-=1
goto app_port_release_loop

:detect_running_services
set "SERVICES_ALREADY_ONLINE=0"
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %MILVUS_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 exit /b 0
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %NEO4J_BOLT_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 exit /b 0
set "SERVICES_ALREADY_ONLINE=1"
exit /b 0

:ensure_attu_if_possible
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %ATTU_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 exit /b 0
docker version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Docker CLI is unavailable; skipping Attu startup.
    exit /b 0
)
docker inspect %ATTU_CONTAINER_NAME% >nul 2>&1
if errorlevel 1 (
    echo [INFO] Creating Attu container...
    docker run -d --name %ATTU_CONTAINER_NAME% -p %ATTU_PORT%:3000 -e MILVUS_URL=%ATTU_MILVUS_URL% %DOCKER_ATTU_IMAGE% >nul
    if errorlevel 1 (
        echo [WARN] Failed to create or start Attu. Continuing without Attu.
        exit /b 0
    )
) else (
    docker start %ATTU_CONTAINER_NAME% >nul 2>&1
)
echo [INFO] Waiting for Attu on port %ATTU_PORT%...
set /a ATTU_WAIT_COUNT=%PORT_WAIT_ATTEMPTS%
:attu_wait_loop
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %ATTU_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 exit /b 0
if %ATTU_WAIT_COUNT% LEQ 0 (
    echo [WARN] Attu did not open port %ATTU_PORT% in time. Continuing without Attu.
    exit /b 0
)
timeout /t 2 /nobreak >nul
set /a ATTU_WAIT_COUNT-=1
goto attu_wait_loop

:print_access_urls
echo.
echo [INFO] Web UI: %APP_URL%
echo [INFO] Attu: %ATTU_URL%
echo [INFO] Neo4j Browser: %NEO4J_BROWSER_URL%
exit /b 0
