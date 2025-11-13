import subprocess
from django.core.management.base import BaseCommand
import os

class Command(BaseCommand):
    help = "Run Django and all Streamlit dashboards together"

    def handle(self, *args, **options):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        streamlit_path = os.path.join(os.path.dirname(base_dir), "streamlit-app")

        # Streamlit dashboard scripts
        dashboards = [
            ("dashboard.py", 8502),
            ("engagement_rate.py", 8503),
            ("top_posts.py", 8504),
            ("instagram_engagement.py", 8505),
        ]

        processes = []

        try:
            # Start each Streamlit dashboard
            for script, port in dashboards:
                script_path = os.path.join(streamlit_path, script)
                cmd = [
                    "python", "-m", "streamlit", "run", script_path,
                    "--server.enableCORS", "false",
                    "--server.enableXsrfProtection", "false",
                    "--server.port", str(port),
                    "--server.headless", "true",
                ]

                proc = subprocess.Popen(cmd)
                processes.append(proc)
                self.stdout.write(self.style.SUCCESS(f"✅ Started {script} on port {port}"))

            # Start Django server
            self.stdout.write(self.style.SUCCESS("\n🚀 Starting Django server on port 8000...\n"))
            subprocess.call(["python", "manage.py", "runserver"])

        finally:
            self.stdout.write(self.style.WARNING("\n🛑 Stopping all Streamlit dashboards..."))
            for p in processes:
                p.terminate()
