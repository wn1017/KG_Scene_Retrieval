@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CHECK_ONLY=0"
set "START_SERVICES=1"
set "SERVICES_ONLY=0"
set "SERVICE_VARS_FILE=.startup_service_vars.cmd"
set "DOCKER_WAIT_ATTEMPTS=40"
set "PORT_WAIT_ATTEMPTS=45"
set "GRADIO_WAIT_ATTEMPTS=45"
set "APP_PORT=7860"
set "ATTU_PORT=8000"
set "APP_URL=http://127.0.0.1:%APP_PORT%"
set "ATTU_URL=http://127.0.0.1:%ATTU_PORT%"
set "NEO4J_BROWSER_URL=http://127.0.0.1:7474"
set "ATTU_CONTAINER_NAME=attu"
set "ATTU_MILVUS_URL=host.docker.internal:19530"
set "DOCKER_ATTU_IMAGE=zilliz/attu:v2.6.3"
set "GRADIO_STATUS=NOT_STARTED"
set "GRADIO_STATUS_DETAIL=Gradio has not been launched yet."
set "KGSR_ENABLE_SHARE=0"
set "KGSR_SHARE_SELECTED="
set "KGSR_LAUNCH_INFO_FILE=%CD%\.gradio_launch_info.cmd"
set "GRADIO_LOCAL_URL=%APP_URL%"
set "GRADIO_SHARE_URL="
set "GRADIO_SHARE_STATUS=DISABLED"
set "GRADIO_SHARE_STATUS_DETAIL=Public share link is disabled."

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
if /I "%~1"=="--share" (
    set "KGSR_ENABLE_SHARE=1"
    set "KGSR_SHARE_SELECTED=1"
)
if /I "%~1"=="--no-share" (
    set "KGSR_ENABLE_SHARE=0"
    set "KGSR_SHARE_SELECTED=1"
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

call conda run -n kg python -c "from config import APP_PORT, ATTU_CONTAINER_NAME, ATTU_MILVUS_URL, ATTU_PORT, CHNCLIP_MODEL_DIR, DOCKER_ATTU_IMAGE, DOCKER_MILVUS_HEALTH_PORT, DOCKER_MILVUS_IMAGE, DOCKER_NEO4J_IMAGE, ENGCLIP_MODEL_DIR, IMAGE_CSV_PATH, MILVUS_PORT, NEO4J_HTTP_PORT, NEO4J_URI, NUSCENES_META_DIR; bolt_port=int(NEO4J_URI.rsplit(':', 1)[1]); engclip_config=ENGCLIP_MODEL_DIR / 'config.json'; engclip_model=ENGCLIP_MODEL_DIR / 'pytorch_model.bin'; chnclip_config=CHNCLIP_MODEL_DIR / 'config.json'; chnclip_model=CHNCLIP_MODEL_DIR / 'pytorch_model.bin'; meta_scene=NUSCENES_META_DIR / 'scene.json'; meta_sample=NUSCENES_META_DIR / 'sample.json'; meta_sample_data=NUSCENES_META_DIR / 'sample_data.json'; meta_annotation=NUSCENES_META_DIR / 'sample_annotation.json'; lines=[f'set \"APP_PORT={APP_PORT}\"', f'set \"ATTU_CONTAINER_NAME={ATTU_CONTAINER_NAME}\"', f'set \"ATTU_MILVUS_URL={ATTU_MILVUS_URL}\"', f'set \"ATTU_PORT={ATTU_PORT}\"', f'set \"DOCKER_ATTU_IMAGE={DOCKER_ATTU_IMAGE}\"', f'set \"DOCKER_MILVUS_IMAGE={DOCKER_MILVUS_IMAGE}\"', f'set \"DOCKER_NEO4J_IMAGE={DOCKER_NEO4J_IMAGE}\"', f'set \"MILVUS_PORT={MILVUS_PORT}\"', f'set \"NEO4J_HTTP_PORT={NEO4J_HTTP_PORT}\"', f'set \"NEO4J_BOLT_PORT={bolt_port}\"', f'set \"DOCKER_MILVUS_HEALTH_PORT={DOCKER_MILVUS_HEALTH_PORT}\"', f'set \"IMAGE_CSV_PATH={IMAGE_CSV_PATH}\"', f'set \"NUSCENES_META_DIR={NUSCENES_META_DIR}\"', f'set \"META_SCENE_PATH={meta_scene}\"', f'set \"META_SAMPLE_PATH={meta_sample}\"', f'set \"META_SAMPLE_DATA_PATH={meta_sample_data}\"', f'set \"META_ANNOTATION_PATH={meta_annotation}\"', f'set \"ENGCLIP_CONFIG_PATH={engclip_config}\"', f'set \"ENGCLIP_MODEL_PATH={engclip_model}\"', f'set \"CHNCLIP_CONFIG_PATH={chnclip_config}\"', f'set \"CHNCLIP_MODEL_PATH={chnclip_model}\"']; print(*lines, sep='\n')" > "%SERVICE_VARS_FILE%"
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

if not exist "%IMAGE_CSV_PATH%" (
    echo [ERROR] Missing %IMAGE_CSV_PATH%
    set "MISSING=1"
)
if not exist "%META_SCENE_PATH%" (
    echo [ERROR] Missing %META_SCENE_PATH%
    set "MISSING=1"
)
if not exist "%META_SAMPLE_PATH%" (
    echo [ERROR] Missing %META_SAMPLE_PATH%
    set "MISSING=1"
)
if not exist "%META_SAMPLE_DATA_PATH%" (
    echo [ERROR] Missing %META_SAMPLE_DATA_PATH%
    set "MISSING=1"
)
if not exist "%META_ANNOTATION_PATH%" (
    echo [ERROR] Missing %META_ANNOTATION_PATH%
    set "MISSING=1"
)
if not exist "%ENGCLIP_CONFIG_PATH%" (
    echo [ERROR] Missing %ENGCLIP_CONFIG_PATH%
    set "MISSING=1"
)
if not exist "%ENGCLIP_MODEL_PATH%" (
    echo [ERROR] Missing %ENGCLIP_MODEL_PATH%
    set "MISSING=1"
)
if not exist "%CHNCLIP_CONFIG_PATH%" (
    echo [ERROR] Missing %CHNCLIP_CONFIG_PATH%
    set "MISSING=1"
)
if not exist "%CHNCLIP_MODEL_PATH%" (
    echo [ERROR] Missing %CHNCLIP_MODEL_PATH%
    set "MISSING=1"
)
if "%MISSING%"=="1" (
    echo.
    echo [INFO] Run "conda run -n kg python scripts\prepare_trainval06_subset.py" to build the configured trainval subset metadata and CSV.
    echo Startup aborted because required files are missing.
    exit /b 1
)

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

call :prompt_for_share_mode
if errorlevel 1 exit /b 1
call :stop_existing_app_listener
if errorlevel 1 exit /b 1

echo [5/5] Launching Gradio app
call :launch_gradio_app
if errorlevel 1 exit /b 1
call :wait_for_gradio_http
call :wait_for_gradio_launch_info
call :print_launch_summary
call :hold_console_open
exit /b 0

:prompt_for_share_mode
if "%CHECK_ONLY%"=="1" exit /b 0
if "%SERVICES_ONLY%"=="1" exit /b 0
if "%KGSR_SHARE_SELECTED%"=="1" goto share_mode_ready
choice /C YN /N /M "Enable public Gradio share link? [Y/N]"
if errorlevel 2 (
    set "KGSR_ENABLE_SHARE=0"
    set "KGSR_SHARE_SELECTED=1"
    goto share_mode_ready
)
if errorlevel 1 (
    set "KGSR_ENABLE_SHARE=1"
    set "KGSR_SHARE_SELECTED=1"
    goto share_mode_ready
)
echo [ERROR] Failed to read the public share selection.
exit /b 1

:share_mode_ready
if "%KGSR_ENABLE_SHARE%"=="1" (
    set "GRADIO_SHARE_STATUS=REQUESTED"
    set "GRADIO_SHARE_STATUS_DETAIL=Waiting for Gradio to return a public share URL."
) else (
    set "GRADIO_SHARE_STATUS=DISABLED"
    set "GRADIO_SHARE_STATUS_DETAIL=Public share link disabled by user."
)
exit /b 0

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
if not defined GRADIO_LOCAL_URL set "GRADIO_LOCAL_URL=%APP_URL%"
if not defined GRADIO_SHARE_URL set "GRADIO_SHARE_URL=not available"
echo.
echo [INFO] Web UI: %GRADIO_LOCAL_URL%
echo [INFO] Share URL: %GRADIO_SHARE_URL%
echo [INFO] Attu: %ATTU_URL%
echo [INFO] Neo4j Browser: %NEO4J_BROWSER_URL%
exit /b 0

:launch_gradio_app
set "GRADIO_STATUS=STARTING"
set "GRADIO_STATUS_DETAIL=Launching Gradio in a separate window."
if exist "%KGSR_LAUNCH_INFO_FILE%" del /q "%KGSR_LAUNCH_INFO_FILE%" >nul 2>&1
echo [INFO] Web UI may take 20-60 seconds on a cold start while local models load.
if "%KGSR_ENABLE_SHARE%"=="1" (
    echo [INFO] Public share link requested. Startup may take a little longer.
) else (
    echo [INFO] Public share link disabled for this session.
)
start "KG Scene Retrieval App" /min cmd /d /c "cd /d ""%CD%"" && call conda run -n kg python app.py"
if errorlevel 1 (
    echo [ERROR] Failed to launch the Gradio app process.
    exit /b 1
)
exit /b 0

:wait_for_gradio_http
set /a GRADIO_WAIT_COUNT=%GRADIO_WAIT_ATTEMPTS%
echo [INFO] Waiting for Gradio HTTP endpoint on %APP_URL%...
:gradio_wait_loop
call :probe_gradio_http
if not errorlevel 1 (
    set "GRADIO_STATUS=AVAILABLE"
    set "GRADIO_STATUS_DETAIL=Gradio HTTP endpoint is reachable."
    exit /b 0
)
if %GRADIO_WAIT_COUNT% LEQ 0 (
    set "GRADIO_STATUS=TIMEOUT"
    set "GRADIO_STATUS_DETAIL=Gradio did not become reachable within the expected time."
    echo [WARN] Gradio did not become reachable on %APP_URL% in time.
    exit /b 0
)
timeout /t 2 /nobreak >nul
set /a GRADIO_WAIT_COUNT-=1
goto gradio_wait_loop

:probe_gradio_http
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%APP_URL%/gradio_api/info' -TimeoutSec 4; if ($response.StatusCode -eq 200) { exit 0 } } catch { }; exit 1"
exit /b %errorlevel%

:wait_for_gradio_launch_info
set /a GRADIO_INFO_WAIT_COUNT=%GRADIO_WAIT_ATTEMPTS%
:gradio_info_wait_loop
if exist "%KGSR_LAUNCH_INFO_FILE%" call "%KGSR_LAUNCH_INFO_FILE%"
if exist "%KGSR_LAUNCH_INFO_FILE%" (
    if defined GRADIO_LOCAL_URL set "APP_URL=%GRADIO_LOCAL_URL%"
    exit /b 0
)
if %GRADIO_INFO_WAIT_COUNT% LEQ 0 (
    if "%KGSR_ENABLE_SHARE%"=="1" (
        set "GRADIO_SHARE_STATUS=UNREPORTED"
        set "GRADIO_SHARE_STATUS_DETAIL=The app did not report a public share URL before the summary was printed."
    )
    exit /b 0
)
timeout /t 1 /nobreak >nul
set /a GRADIO_INFO_WAIT_COUNT-=1
goto gradio_info_wait_loop

:print_launch_summary
call :print_access_urls
if "%GRADIO_STATUS%"=="AVAILABLE" (
    echo [OK] Gradio status: AVAILABLE
) else (
    if "%GRADIO_STATUS%"=="TIMEOUT" (
        echo [WARN] Gradio status: TIMEOUT
    ) else (
        echo [WARN] Gradio status: %GRADIO_STATUS%
    )
)
echo [INFO] %GRADIO_STATUS_DETAIL%
echo [INFO] Share status: %GRADIO_SHARE_STATUS%
echo [INFO] %GRADIO_SHARE_STATUS_DETAIL%
echo [INFO] Leave this window open while the demo is running.
exit /b 0

:hold_console_open
if "%KGSR_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
