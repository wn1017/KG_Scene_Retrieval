import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
START_BAT = ROOT / "start.bat"


def run_start_bat(args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmd", "/d", "/c", f"start.bat {args}".strip()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class StartScriptTests(unittest.TestCase):
    def test_check_mode_completes_successfully(self):
        completed = run_start_bat("--check")

        message = "\n".join(
            [
                f"returncode={completed.returncode}",
                "stdout:",
                completed.stdout,
                "stderr:",
                completed.stderr,
            ]
        )
        self.assertEqual(completed.returncode, 0, message)
        self.assertIn("Check completed successfully.", completed.stdout, message)

    def test_check_mode_uses_resolved_ports(self):
        completed = run_start_bat("--check")
        combined = f"{completed.stdout}\n{completed.stderr}"

        self.assertNotIn("port .", combined, combined)
        self.assertNotIn("Missing an argument for parameter 'Port'", combined, combined)

    def test_start_script_sets_local_no_proxy(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertIn('set "NO_PROXY=127.0.0.1,localhost,::1', content)
        self.assertIn('set "no_proxy=127.0.0.1,localhost,::1', content)
        self.assertIn('set "HTTP_PROXY="', content)
        self.assertIn('set "HTTPS_PROXY="', content)
        self.assertIn('set "ALL_PROXY="', content)
        self.assertIn('set "http_proxy="', content)
        self.assertIn('set "https_proxy="', content)
        self.assertIn('set "all_proxy="', content)
        self.assertIn('set "GRADIO_SERVER_NAME=127.0.0.1"', content)
        self.assertIn('set "GRADIO_SERVER_PORT=%APP_PORT%"', content)

    def test_start_script_handles_existing_app_listener(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertIn('set "APP_PORT=7860"', content)
        self.assertIn('call :stop_existing_app_listener', content)
        self.assertIn('Get-NetTCPConnection -LocalPort %APP_PORT% -State Listen', content)
        self.assertIn('taskkill /PID %APP_LISTENER_PID% /F', content)

    def test_start_script_skips_docker_wait_when_services_are_already_online(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertIn('call :detect_running_services', content)
        self.assertIn('if "%SERVICES_ALREADY_ONLINE%"=="1" (', content)
        self.assertIn('echo [INFO] Milvus and Neo4j are already reachable. Skipping Docker startup.', content)
        self.assertIn(':detect_running_services', content)

    def test_start_script_prints_copyable_urls(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertIn('set "APP_URL=http://127.0.0.1:%APP_PORT%"', content)
        self.assertIn('set "ATTU_URL=http://127.0.0.1:%ATTU_PORT%"', content)
        self.assertIn('set "NEO4J_BROWSER_URL=http://127.0.0.1:%NEO4J_HTTP_PORT%"', content)
        self.assertIn('echo [INFO] Web UI: %APP_URL%', content)
        self.assertIn('echo [INFO] Attu: %ATTU_URL%', content)
        self.assertIn('echo [INFO] Neo4j Browser: %NEO4J_BROWSER_URL%', content)

    def test_start_script_uses_lightweight_app_check(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertNotIn("import app; print('APP_IMPORT_OK')", content)
        self.assertIn("compile(Path(r'app.py').read_text(encoding='utf-8'), 'app.py', 'exec')", content)
        self.assertIn("APP_SYNTAX_OK", content)

    def test_start_script_manages_attu_container(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertIn('docker inspect %ATTU_CONTAINER_NAME% >nul 2>&1', content)
        self.assertIn('docker run -d --name %ATTU_CONTAINER_NAME% -p %ATTU_PORT%:3000 -e MILVUS_URL=%ATTU_MILVUS_URL% %DOCKER_ATTU_IMAGE% >nul', content)
        self.assertIn('echo [INFO] Waiting for Attu on port %ATTU_PORT%...', content)

    def test_start_script_treats_attu_as_optional(self):
        content = START_BAT.read_text(encoding="ascii")

        self.assertIn(':ensure_attu_if_possible', content)
        self.assertIn('echo [WARN] Docker CLI is unavailable; skipping Attu startup.', content)
        self.assertIn('echo [WARN] Attu did not open port %ATTU_PORT% in time. Continuing without Attu.', content)

    def test_config_exposes_attu_settings(self):
        config_content = (ROOT / "config.py").read_text(encoding="utf-8")

        self.assertIn('DOCKER_ATTU_IMAGE = "zilliz/attu:v2.6.3"', config_content)
        self.assertIn('ATTU_CONTAINER_NAME = "attu"', config_content)
        self.assertIn('ATTU_MILVUS_URL = "host.docker.internal:19530"', config_content)


if __name__ == "__main__":
    unittest.main()
