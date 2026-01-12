import subprocess
from django.core.management.base import BaseCommand
import os
import sys

class Command(BaseCommand):
    help = "Run Django and all Streamlit dashboards together"

    def handle(self, *args, **options):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        streamlit_path = os.path.join(os.path.dirname(base_dir), "streamlit-app")

        # Use the SAME python that is running Django
        PYTHON = sys.executable   # <-- IMPORTANT FIX ???

        # Streamlit dashboard scripts
        dashboards = [
            ("dashboard.py", 8502),
            ("engagement_rate.py", 8503),
            ("top_posts.py", 8504),
            ("instagram_engagement.py", 8505),
            ("youtube-dashboard.py", 8506),
            ("youtube_sentiment.py",8507),
            ("youtube_top_posts.py", 8508),
            ("youtube_prediction.py", 8509),
            ("youtube_engagement.py", 8510),
            ("twitter_dashboard.py", 8511),
            ("twitter_prediction.py", 8512),
            ("twitter_engagement.py", 8513),
            ("twitter_top_tweets.py", 8514),
            ("linkedin_top_posts.py", 8515),
            ("linkedin_prediction.py", 8516),
            ("twitter_sentiment.py", 8517),
            ("linkedin_dashboard.py", 8518),
            # ("twitter_dashboard.py", 8519), 
            # ("instagram_worst_posts.py", 8520),
            ("ga_gender_report.py", 8520),
            ("ga_asset_report.py", 8521),
            ("ga_report.py", 8522),
            ("ga_schedule_day_hour.py", 8523),
            ("ga_schedule.py", 8524),
            ("ga_search_terms.py", 8525),
            ("ga_search_keyword.py", 8526),
            ("ga_device_report.py", 8527),
            ("ga_household_income.py", 8528),
            ("ga_landing_page.py", 8529),
            ("ga_location_report.py", 8530),
            ("ga_group_report.py", 8531),
            ("ga_age_report.py", 8532),
            ("ga_campaign.py", 8533),
            ("ga_targeted_content.py", 8534),
        ]

        processes = []

        try:
            # Start each Streamlit dashboard
            for script, port in dashboards:
                script_path = os.path.join(streamlit_path, script)
                cmd = [
                    PYTHON, "-m", "streamlit", "run", script_path,  # FIXED HERE ?
                    "--server.enableCORS", "false",
                    "--server.enableXsrfProtection", "false",
                    "--server.port", str(port),
                    "--server.headless", "true",
                ]

                proc = subprocess.Popen(cmd)
                processes.append(proc)
                self.stdout.write(self.style.SUCCESS(f"? Started {script} on port {port}"))

            # Start Django server
            self.stdout.write(self.style.SUCCESS("\n?? Starting Django server on port 8000...\n"))
            subprocess.call([PYTHON, "manage.py", "runserver"])  # FIXED HERE ?

        finally:
            self.stdout.write(self.style.WARNING("\n?? Stopping all Streamlit dashboards..."))
            for p in processes:
                p.terminate()


