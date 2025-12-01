from django.shortcuts import render, redirect
from django.conf import settings
import os
from dotenv import load_dotenv
import random
import string
import hashlib
import base64
import requests
from urllib.parse import urlencode
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from datetime import date, timedelta
from urllib.parse import urlencode
import requests
from django.shortcuts import render, redirect
import datetime
from django.http import JsonResponse
from datetime import datetime, timedelta
from datetime import datetime
from datetime import datetime, timedelta
from django.shortcuts import redirect, render
import requests
from sklearn.ensemble import RandomForestRegressor
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils.safestring import mark_safe
import requests
import json
from urllib.parse import urlencode
from urllib.parse import urlencode
from django.shortcuts import redirect, render
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from textblob import TextBlob
import requests
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils.safestring import mark_safe

from urllib.parse import urlencode

from urllib.parse import urlencode
import random
import string
from datetime import date, datetime, timedelta
import random
import string
from urllib.parse import urlencode
from django.shortcuts import render
from myapp.models import CustomUser
from django.contrib.auth import logout

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import requests
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils.safestring import mark_safe



from django.shortcuts import render, redirect
from .models import Feedback, CustomUser

from django.utils import timezone

from django.shortcuts import render
from .models import CustomUser

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from .models import CustomUser  # Importing CustomUser model
from django.conf import settings
import plotly.express as px
import pandas as pd

load_dotenv()  # Load environment variables from .env file

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# View for YouTube Tab
def youtube_tab(request):
    return render(request, 'youtube_tab.html')

# View for YouTube Dashboard
def youtube_dashboard(request):
    return render(request, 'youtube_dashboard.html')

# View for Instagram Tab
def instagram_tab(request):
    return render(request, 'instagram_tab.html')

# View for Instagram Analytics Dashboard
def instagram_analytics(request):
    return render(request, 'instagram_analytics.html')

# View for Instagram Engagement Prediction
def instagram_engagement(request):
    return render(request, 'instagram_engagement.html')

# View for Engagement Rate Analysis
def engagement_rate(request):
    return render(request, 'engagement_rate.html')

# View for Top Posts
def top_posts(request):
    return render(request, 'top_posts.html')
    
import os
from django.shortcuts import redirect
from urllib.parse import urlencode

def youtube_login(request):
    YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
    # redirect_uri = "http://127.0.0.1:8000/callback-youtube/"
    redirect_uri = "https://socialalytics.in/callback-youtube/"
    
    scope = "https://www.googleapis.com/auth/youtube.readonly"

    params = {
        "client_id": YOUTUBE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "scope": scope,
    }

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return redirect(auth_url)

import requests
from django.http import JsonResponse

def youtube_callback(request):
    code = request.GET.get("code")
    YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
    YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")
    # redirect_uri = "http://127.0.0.1:8000/callback-youtube/"
    redirect_uri = "https://socialalytics.in/callback-youtube/"

    token_url = "https://oauth2.googleapis.com/token"

    data = {
        "code": code,
        "client_id": YOUTUBE_CLIENT_ID,
        "client_secret": YOUTUBE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    r = requests.post(token_url, data=data)
    token_info = r.json()

    # Save access token in session
    request.session["yt_access_token"] = token_info["access_token"]
    request.session["yt_refresh_token"] = token_info.get("refresh_token")

    # return JsonResponse({
    #     "message": "YouTube Login Successful",
    #     "token_info": token_info
    # })

    # REDIRECT TO YOUTUBE AFTER SUCCESS
    return redirect("https://www.youtube.com/")

def get_youtube_channel(request):
    access_token = request.session.get("yt_access_token")

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    url = "https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&mine=true"

    r = requests.get(url, headers=headers)
    return JsonResponse(r.json())

def facebook_insights(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        access_token = request.POST.get('access_token')
        period = request.POST.get('period', 'day')  # Default to 'day'

        if not page_id or not access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)

        # Get page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)

        # Calculate default `since` and `until` dates based on the selected period
        since_date, until_date = calculate_dates(period)
        since = since_date.strftime('%Y-%m-%d') if since_date else None
        until = until_date.strftime('%Y-%m-%d')

        # Fetch insights
        page_insights = get_page_insights(page_access_token, page_id,'day', since, until)
#        return JsonResponse(page_insights)
        # Extract relevant data for graphing
        metrics_data = {}
        for item in page_insights.get('data', []):
            metric_name = item['name']
            metrics_data[metric_name] = [
                {"date": value["end_time"][:10], "value": value["value"]}
                  for value in item.get("values", [])
            ]

        return render(request, 'facebook_insights.html', {
            'metrics': metrics_data,
            'page_id': page_id,
            'period': period,
            'since': since,
            'until': until
        })

    # For GET requests, render a form for user input
    return render(request, 'insights_form.html')
def calculate_dates(period):
    """Calculate default since and until dates based on the selected period."""
    today = date.today()
    if period == 'day':
        since = today - timedelta(days=1)
    elif period == 'week':
        since = today - timedelta(days=7)
    elif period == 'days_28':
        since = today - timedelta(days=28)
    else:
        since = None

    until = today
    return since, until
def get_page_insights(page_access_token, page_id, period, since=None, until=None):
    insights_url = f"https://graph.facebook.com/v17.0/{page_id}/insights"
    params = {
        'metric': 'page_impressions,page_posts_impressions,page_fan_adds,page_fan_removes,page_post_engagements,page_views_total,page_daily_follows,page_daaily_unfollows_unique',        'period': period,
        'access_token': page_access_token
    }

    # Add default or calculated date range
    if since and until:
        params['since'] = since
        params['until'] = until

    response = requests.get(insights_url, params=params)
    return response.json()

def get_page_insightsp(page_access_token, post_id, period, since, until):
    """
    Fetch insights from Facebook Graph API with since/until.
    """
    insights_url = f"https://graph.facebook.com/v21.0/{post_id}/insights"
    params = {
        'metric': 'page_posts_impressions,page_posts_impressions_unique',
        'period': period,
        'access_token': page_access_token,
        'since': since,
        'until': until,
    }
    response = requests.get(insights_url, params=params)
    return response.json()
def post_metrics(request):
    if request.method == "POST":
        post_id = request.POST.get('post_id')  # Get post_id from the POST request
        time_period = request.POST.get('time_period', 'day')  # Default to 'day'

        # Calculate 'since' and 'until' based on the selected time period
        since_date, until_date = calculate_dates(time_period)
        since = since_date.strftime('%Y-%m-%d') if since_date else None
        until = until_date.strftime('%Y-%m-%d')

        # Get the page access token from the session
        page_access_token = request.session.get('facebook_page_token')

        # Fetch page insights (replace `get_page_insightsp` with your actual function)
        page_insights = get_page_insightsp(page_access_token, post_id,'day', since, until)

        # Return the insights as a JSON response
        return JsonResponse(page_insights)

    # If not POST, handle invalid access
    return JsonResponse({'error': 'Invalid request method'}, status=400)


# Request Password Reset (View)
"""def request_password_reset(request):
    if request.method == "POST":
        email = request.POST.get("email")
        try:
            # Fetch user by email from CustomUser model
            user = CustomUser.objects.get(email=email)

            # Generate token and email user the password reset link
            token = default_token_generator.make_token(user)
            uidb64 = urlsafe_base64_encode(str(user.pk).encode())

            # Create the reset URL
            reset_url = request.build_absolute_uri(f'/reset_password/{uidb64}/{token}/')

            # Send reset link to the user's email
            subject = "Password Reset Request"
            message = render_to_string('password_reset_email.html', {
                'user': user,
                'reset_url': reset_url,
            })
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])

            messages.success(request, "Password reset link has been sent to your email.")
            return redirect('signin')  # Redirect to sign in page after email is sent
                    except CustomUser.DoesNotExist:
            messages.error(request, "User with this email does not exist.")
            return redirect('request_password_reset')

    return render(request, 'request_password_reset.html') """

def request_password_reset(request):
    if request.method == "POST":
        email = request.POST.get("email")
        try:
            # Fetch user by email from CustomUser model
            user = CustomUser.objects.get(email=email)

            # Generate token and email user the password reset link
            token = default_token_generator.make_token(user)
            uidb64 = urlsafe_base64_encode(str(user.pk).encode())

            # Create the reset URL
            reset_url = request.build_absolute_uri(f'/reset_password/{uidb64}/{token}/')

            # Send reset link to the user's email
            subject = "Password Reset Request"
            message = f"""
                Hi {user.name},

                We received a request to reset your password. You can reset your password by clicking the link below:

                {reset_url}
                If you didn’t request this change, you can safely ignore this email.

                Thank you,
                Marketing Analytics Team
                """   
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])

            messages.success(request, "Password reset link has been sent to your email.")
            return redirect('signin')  # Redirect to sign in page after email is sent

        except CustomUser.DoesNotExist:
            messages.error(request, "User with this email does not exist.")
            return redirect('request_password_reset')

    return render(request, 'request_password_reset.html')
# Reset Password (View)
def reset_password(request, uidb64, token):
    try:
        # Decode the UID from base64
        uid = urlsafe_base64_decode(uidb64).decode()

        # Retrieve the user from CustomUser model
        user = CustomUser.objects.get(pk=uid)

        # Validate the token
        if default_token_generator.check_token(user, token):
            if request.method == 'POST':
                new_password = request.POST.get('new_password')
                confirm_password = request.POST.get('confirm_password')

                if new_password != confirm_password:
                    messages.error(request, "Passwords do not match.")
                else:
                    # Set the new password
                    user.set_password(new_password)
                    user.save()
                    messages.success(request, "Your password has been reset successfully.")
                    return render(request, 'sign_in.html')  # Render sign_in.html after password reset

            return render(request, 'reset_password.html', {'uidb64': uidb64, 'token': token})
        else:
            # Invalid or expired token
            messages.error(request, "Invalid or expired token.")
            return render(request, 'sign_in.html')  # Render sign_in.html instead of redirect

    except CustomUser.DoesNotExist:
        messages.error(request, "User not found.")
        return render(request, 'sign_in.html')  # Render sign_in.html instead of redirect
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
        return render(request, 'sign_in.html')  # Render sign_in.html instead of redirect
def feedback_list(request):
    users_with_feedback = CustomUser.objects.prefetch_related('feedbacks').all()
    context = {
        'users': users_with_feedback
    }
    return render(request, 'feedback_list.html', context)


def submit_feedback(request):
    if request.method == 'POST' and request.session.get('is_authenticated'):
        email = request.session.get('user_email')
        user = CustomUser.objects.get(email=email)
        experience = request.POST.get('experience')
        review = request.POST.get('review')

        # Save the feedback
        Feedback.objects.create(user=user, experience=experience, review=review)

        return redirect('feedback_success')  # Redirect to success page

    return render(request, 'feedback_form.html')

def feedback_success(request):
    return render(request, 'feedback_success.html')

def analyze_sentiment_over_time(posts_with_insights):

    # Step 1: Ensure necessary columns
    posts_df = pd.DataFrame(posts_with_insights)

    # Create a content column from the 'message', defaulting to an empty string for NaN values
    posts_df['content'] = posts_df['message'].fillna('')

    # Ensure engagement columns are numeric
    for col in ['likes', 'shares', 'number_of_comments']:
        posts_df[col] = pd.to_numeric(posts_df[col], errors='coerce').fillna(0)

    # Calculate total engagement
    posts_df['engagement'] = posts_df['likes'] + posts_df['shares'] + posts_df['number_of_comments']

    # Step 2: Perform sentiment analysis
    # Assuming `analyze_sentiment` is a predefined function that returns sentiment as a string
    posts_df['sentiment'] = posts_df['content'].apply(
        lambda x: analyze_sentiment(x) if isinstance(x, str) and x.strip() else 'neutral'
    )
    
     # Step 3: Group by time period (e.g., month)
    posts_df['created_time'] = pd.to_datetime(posts_df['created_time'], errors='coerce')
    posts_df['time_period'] = posts_df['created_time'].dt.to_period('M')  # Monthly aggregation

    # Step 4: Calculate average sentiment and engagement per time period
    sentiment_engagement = posts_df.groupby('time_period').apply(
        lambda x: pd.Series({
            'avg_sentiment': x['sentiment'].apply(
                lambda s: 1 if s == 'positive' else (-1 if s == 'negative' else 0)
            ).mean(),
            'avg_engagement': x['engagement'].mean()
        })
    ).reset_index()

    # Step 5: Calculate average engagement for positive sentiment posts
    positive_posts = posts_df[posts_df['sentiment'] == 'positive']

    avg_engagement_positive = (
        positive_posts['engagement'].mean() if not positive_posts.empty else 0
    )
    avg_engagement_all = posts_df['engagement'].mean()

    # Step 6: Generate recommendation
    if avg_engagement_positive > avg_engagement_all:
        return "Focus on creating positive sentiment posts for better engagement."
    else:
        return "Neutral or mixed sentiment content performs better."
    
def analyze_post_length(posts_with_insights):
    # Step 1: Ensure necessary columns
    posts_df = pd.DataFrame(posts_with_insights)
    posts_df['content'] = posts_df['message'].fillna('')
    posts_df['engagement'] = posts_df['likes'] + posts_df['shares'] + posts_df['number_of_comments']

    # Step 2: Calculate post length (in letters)
    posts_df['post_length'] = posts_df['content'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    # Step 3: Categorize posts by length
    bins = [0, 50, 100, 200, 300, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-300', '300+']
    posts_df['length_category'] = pd.cut(posts_df['post_length'], bins=bins, labels=labels)

    # Step 4: Calculate average engagement for each length category
    engagement_by_length = posts_df.groupby('length_category')['engagement'].mean().reset_index()

    # Step 5: Find the length category with the highest engagement
    best_length = engagement_by_length.loc[engagement_by_length['engagement'].idxmax(), 'length_category']

    # Step 6: Generate recommendation
    return f"Optimal post length: {best_length} letters."

def extract_keywords_and_analyze(posts_with_insights):
    # Step 1: Preprocess text data
    posts_df = pd.DataFrame(posts_with_insights)
    posts_df['content'] = posts_df['message'].fillna('')

    # Step 2: Apply TF-IDF to extract keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(posts_df['content'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Step 3: Calculate engagement for each post
    posts_df['engagement'] = posts_df['likes'] + posts_df['shares'] + posts_df['number_of_comments']

    # Step 4: Correlate keywords with engagement
    keyword_engagement = {}
    for idx, row in posts_df.iterrows():
        engagement = row['engagement']
        text_vector = tfidf_matrix[idx].toarray().flatten()

        for word_idx, tfidf_score in enumerate(text_vector):
            keyword = feature_names[word_idx]
            if keyword not in keyword_engagement:
                keyword_engagement[keyword] = []
            keyword_engagement[keyword].append(engagement * tfidf_score)
   # Step 5: Identify keywords that lead to higher engagement
    keyword_avg_engagement = {k: np.mean(v) for k, v in keyword_engagement.items()}
    top_keywords = sorted(keyword_avg_engagement, key=keyword_avg_engagement.get, reverse=True)[:10]

    return top_keywords


def optimal_posting_time(posts_with_insights):
    # Step 1: Extract hour and day of the week from post timestamps
    posts_df = pd.DataFrame(posts_with_insights)
    posts_df['created_time'] = pd.to_datetime(posts_df['created_time'])
    posts_df['hour'] = posts_df['created_time'].dt.hour
    posts_df['day_of_week'] = posts_df['created_time'].dt.weekday  # Monday=0, Sunday=6

    # Step 2: Calculate average engagement by hour and day
    posts_df['engagement'] = posts_df['likes'] + posts_df['shares'] + posts_df['number_of_comments']
    hourly_engagement = posts_df.groupby('hour')['engagement'].mean()
    weekday_engagement = posts_df.groupby('day_of_week')['engagement'].mean()

    # Step 3: Find the hour and day with the highest engagement
    best_hour = hourly_engagement.idxmax()
    best_day = weekday_engagement.idxmax()

    return best_hour, best_day


def youtube_loginn(request):
    youtube_client_id = os.getenv('YOUTUBE_CLIENT_ID')  
    redirect_uri = 'https://www.marketinganalytics.live/callbackyoutube/'
    scopes = ['https://www.googleapis.com/auth/youtube.readonly', 'https://www.googleapis.com/auth/analytics' ,'https://www.googleapis.com/auth/yt-analytics.readonly','https://www.googleapis.com/auth/yt-analytics-monetary.readonly','https://www.googleapis.c.com/auth/youtube','https://www.googleapis.com/auth/youtubepartner','https://www.googleapis.com/auth/youtube.force-ssl'] 
    
    # Construct the OAuth2 authorization URL
    auth_url = (

        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={youtube_client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope={' '.join(scopes)}&"
        f"access_type=offline"

    )

    # Pass the login URL to the template
    context = {
        'auth_url': auth_url
    }

    return render(request, 'youtube_login.html', context)

# This view handles the callback after the user authenticates

def youtube_callbackn(request):
    code = request.GET.get('code')


    if not code:
        return HttpResponse("Error: Authorization code is missing.")

    youtube_client_id =  os.getenv('YOUTUBE_CLIENT_ID')
    youtube_client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
    redirect_uri = 'https://www.marketinganalytics.live/callbackyoutube/'

    # Exchange authorization code for access token
    token_url = 'https://oauth2.googleapis.com/token'
    token_data = {
        'code': code,
        'client_id': youtube_client_id,
        'client_secret': youtube_client_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }
    token_response = requests.post(token_url, data=token_data)
    if token_response.status_code != 200:
        return HttpResponse(f"Error fetching access token: {token_response.json()}")
    token_json = token_response.json()
    
      # Check if access token is available
    access_token = token_json.get('access_token')
    if not access_token:
        return JsonResponse({'error': 'Failed to retrieve access token'})
    request.session['youtube_access_token'] = access_token
    return redirect('youtube_profile')
def youtube_analytics(request):
    access_token = request.session.get('youtube_access_token')
    if not access_token:
        return redirect('youtube_login')
    if request.method == 'GET':
        return render(request, 'your_template.html')  # Replace with your template name

    elif request.method == 'POST':
        data = json.loads(request.body)
        video_id = data.get('video_id')
        time_period = int(data.get('time_period'))  # e.g., 7, 28, 90, 365 days

        # Calculate the date range for the analytics
        today = date.today()
        start_date = today - timedelta(days=time_period)
        end_date = today
        
          # Fetch YouTube Analytics Data
        analytics_data = fetch_youtube_analytics(video_id, start_date, end_date,access_token)

        return JsonResponse({
            'analytics': analytics_data,
        })

def fetch_youtube_analytics(video_id, start_date, end_date,access_token):

    headers = {'Authorization': f'Bearer {access_token}'}
    video_analytics_params = {
            'ids': 'channel==MINE',
            'filters': f'video=={video_id}',
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'metrics': 'views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage',
            'dimensions': 'day',
            'sort': 'day',
        }

    analytics_url = "https://youtubeanalytics.googleapis.com/v2/reports"

    # Handle the request and check for errors
    video_analytics_response = requests.get(analytics_url, headers=headers, params=video_analytics_params)

    if video_analytics_response.status_code != 200:
        # You can handle error more gracefully here
        return {'error': 'Failed to fetch data from YouTube API'}

    video_analytics_data = video_analytics_response.json()
    analytics_data = {
        'labels': [],
        'views': [],
        'estimatedMinutesWatched': [],
        'averageViewDuration': [],
        'averageViewPercentage': []
    }

    # Parse the response to extract relevant data
    for row in video_analytics_data.get('rows', []):
        analytics_data['labels'].append(row[0])  # Date
        analytics_data['views'].append(row[1])  # Views
        analytics_data['estimatedMinutesWatched'].append(row[2])  # Estimated Minutes Watched
        analytics_data['averageViewDuration'].append(row[3])  # Average View Duration
        analytics_data['averageViewPercentage'].append(row[4])  # Average View Percentage

    return analytics_data

def youtube_videos(request):
    access_token = request.session.get('youtube_access_token')
    if not access_token:
        return redirect('youtube_login')

    headers = {'Authorization': f'Bearer {access_token}'}

    # Fetch uploaded videos
    videos_list_url = "https://www.googleapis.com/youtube/v3/search?part=snippet&forMine=true&type=video&maxResults=10"
    videos_list_response = requests.get(videos_list_url, headers=headers)
    videos_list = videos_list_response.json()

    context = {
        'videos': videos_list.get('items', []),
    }
    return render(request, 'youtube_videos.html', context)

def video_analytics1(request, video_id, time_period):
    access_token = request.session.get('youtube_access_token')
    if not access_token:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    headers = {'Authorization': f'Bearer {access_token}'}

    today = date.today()
    time_periods = {
        '7_days': today - timedelta(days=7),
        '28_days': today - timedelta(days=28),
        '90_days': today - timedelta(days=90),
        '365_days': today - timedelta(days=365),
    }
    start_date = time_periods.get(time_period, today - timedelta(days=28))

    # Fetch analytics for the given video
    analytics_url = "https://youtubeanalytics.googleapis.com/v2/reports"
    analytics_params = {
        'ids': 'channel==MINE',
        'filters': f'video=={video_id}',
        'startDate': start_date,
        'endDate': today,
        'metrics': 'views,estimatedMinutesWatched',
        'dimensions': 'day',
        'sort': 'day',
    }
    
    analytics_response = requests.get(analytics_url, headers=headers, params=analytics_params)
    analytics_data = analytics_response.json()

    return JsonResponse(analytics_data)
def youtube_profile(request):
    access_token = request.session.get('youtube_access_token')
    if not access_token:
        return redirect('youtube_login')

    headers = {'Authorization': f'Bearer {access_token}'}
    # Fetch channel information
    channel_info_url = "https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&mine=true"
    channel_info_response = requests.get(channel_info_url, headers=headers)
    channel_info = channel_info_response.json()

    # Fetch uploaded videos
    videos_list_url = "https://www.googleapis.com/youtube/v3/search?part=snippet&forMine=true&type=video&maxResults=50"
    videos_list_response = requests.get(videos_list_url, headers=headers)
    videos_list = videos_list_response.json()

    # Extract video IDs
    video_ids = [item['id']['videoId'] for item in videos_list.get('items', []) if 'videoId' in item['id']]

    # Fetch detailed information for each video
    video_details = []
    if video_ids:
        # Use comma-separated video IDs for batch processing
        video_ids_str = ','.join(video_ids)
        video_details_url = "https://www.googleapis.com/youtube/v3/videos"
        video_params = {
            'part': 'snippet,contentDetails,statistics,status,player,topicDetails,recordingDetails',
            'id': video_ids_str
        }
        video_details_response = requests.get(video_details_url, headers=headers, params=video_params)
        video_details = video_details_response.json()

    analytics_url = "https://youtubeanalytics.googleapis.com/v2/reports"
    today = date.today()
    start_date = today - timedelta(days=90)  # Past 28 days
    end_date = today # Yesterday to ensure data availability
    analytics_params = {
        'ids': 'channel==MINE',
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'metrics': 'views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost',
        'dimensions': 'day',
        'sort': 'day',
    }
    
    analytics_response = requests.get(analytics_url, headers=headers, params=analytics_params)
    def fetch_video_comments(video_id):
        comments = []
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            'part': 'snippet,replies',
            'videoId': video_id,
            'maxResults': 100,  # Fetch up to 100 comments per request

        }

        while True:
            response = requests.get(url, headers=headers, params=params,)
            data = response.json()

        # Process the comments
            for item in data.get('items', []):
                top_level_comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': top_level_comment.get('authorDisplayName'),
                    'text': top_level_comment.get('textDisplay'),
                    'likes': top_level_comment.get('likeCount'),
                    'published_at': top_level_comment.get('publishedAt'),
                    'replies': [reply['snippet']['textDisplay'] for reply in item.get('replies', {}).get('comments', [])]
                })
   # Check if there's another page of comments
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
            params['pageToken'] = next_page_token

        return comments
    video_data_combined = []
    for video in video_details.get('items', []):
        video_id = video['id']

        comments = fetch_video_comments(video_id)
        video_analytics_params = {
            'ids': 'channel==MINE',
            'filters': f'video=={video_id}',
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'metrics': 'views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage',
            'dimensions': 'day',
            'sort': 'day',
        }

        video_analytics_response = requests.get(analytics_url, headers=headers, params=video_analytics_params)
        video_analytics_data = video_analytics_response.json()
        
         # Prepare the combined video data
        combined_video = {
            'video_id': video_id,
            'details': video,
            'comments': comments,
            'analytics': {
                'labels': [row[0] for row in video_analytics_data.get('rows', [])],  # Dates
                'views': [row[1] for row in video_analytics_data.get('rows', [])],
                'estimatedMinutesWatched': [row[2] for row in video_analytics_data.get('rows', [])],
                'averageViewDuration': [row[3] for row in video_analytics_data.get('rows', [])],
                'averageViewPercentage': [row[4] for row in video_analytics_data.get('rows', [])],
            }
        }

        video_data_combined.append(combined_video)  # Store combined data in the list
        
    analytics_data = analytics_response.json()
    chart_data = {
        'labels': [row[0] for row in analytics_data.get('rows', [])],  # Dates
        'views': [row[1] for row in analytics_data.get('rows', [])],
        'estimatedMinutesWatched': [row[2] for row in analytics_data.get('rows', [])],
        'averageViewDuration': [row[3] for row in analytics_data.get('rows', [])],
        'averageViewPercentage': [row[4] for row in analytics_data.get('rows', [])],
        'subscribersGained': [row[5] for row in analytics_data.get('rows', [])],
        'subscribersLost': [row[6] for row in analytics_data.get('rows', [])],
    }
    context = {
        'channel': channel_info,
        'videos': videos_list,
        'video_details': video_details,
        'analytics': analytics_data,
        'chart_data': chart_data,
        'video_data_combined': video_data_combined,
    }

    return render(request, 'youtube_data3.html', context)

def facebook_login1(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'facebook_login.html')  # Create this template
    else :
        return render(request, 'sign_in.html')


def forgot_password(request):
    return render(request, 'forgot_password.html')
def instagram_login(request):
    if request.session.get('is_authenticated', False):
        auth_url = (
            "https://www.instagram.com/oauth/authorize"
            "?client_id=969834388286582"
            "&redirect_uri=https://socialalytics.in/callbacki/"
            "&scope=instagram_business_basic,instagram_business_manage_messages,"
            "instagram_business_manage_comments,instagram_business_content_publish"
            "&response_type=code"
            "&enable_fb_login=0"
            "&force_authentication=1"
        )
        return render(request, 'index.html', {"auth_url": auth_url})
    else:
        return render(request, "sign_in.html")
    
def twitter_login(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'login.html')  # Create this template
    else :
        return render(request, 'sign_in.html')
def reddit_login(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'indexr.html')
    else :
        return render(request, 'sign_in.html')
    
# def youtube_login(request):
#     if request.session.get('is_authenticated', False):
#         return render(request, 'youtube_login.html')
#     else :
#         return render(request, 'sign_in.html')
# def linkedin_login(request):
#     if request.session.get('is_authenticated', False):
#         return render(request, 'linkedin_login.html')
#     else :
#         return render(request, 'sign_in.html')
def dashboard(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'dashboard.html')
    else:
        return render(request, 'sign_in.html')
def dashboard1(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'dashboard1.html')
    else:
        return render(request, 'sign_in.html')
def account(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'account.html')
    else:
        return render(request, 'sign_in.html')
def account1(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'account1.html')
    else:
        return render(request, 'sign_in.html')
def report(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'report.html')
    else:
        return render(request, 'sign_in.html')
def report1(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'report1.html')
    else:
        return render(request, 'sign_in.html')
def analytics1(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'analytics1.html')
    else:
        return render(request, 'sign_in.html')
def analytics(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'analytics.html')
    else:
        return render(request, 'sign_in.html')
def support1(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'support1.html')
    else:
        return render(request, 'sign_in.html')
def support(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'support.html')
    else:
        return render(request, 'sign_in.html')
def logout_view(request):
    request.session.flush()
    return redirect('landing')
def getting_started(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'getting_started.html')
    else:
        return render(request, 'sign_in.html')

def troubleshooting(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'troubleshooting.html')  # Create a 'troubleshooting.html' template
    else:
        return render(request, 'sign_in.html')
def contact_support(request):
    if request.session.get('is_authenticated', False):
        return render(request, 'contact_support.html')  # Create a 'contact_support.html' template
    else:
        return render(request, 'sign_in.html')

def home(request):
    return render(request, 'home.html')  # Render the 'home.html' template


def ar(request):
    return render(request, 'all-reviews.html')  # Render the 'home.html' template

def landing_view(request):
    return render(request, 'landing.html')

# Sign Up Page
def signup(request):
    if request.method == "POST":
        # Process Sign Up Form Data (if any)
        pass
    return render(request, 'register.html')  # Render the signup.html template

# Sign In Page
def signin(request):
    if request.method == "POST":
        # Process Sign In Form Data (if any)
        pass
    return render(request, 'sign_in.html')  # Render the signin.html template

# Registration view

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, "Email is already registered.")
            return render(request, 'register.html')  # Render register template again

        # Create a new user (without hashing the password)
        CustomUser.objects.create(name=name, email=email, password=password)
        messages.success(request, "Registration successful! Please sign in.")
        return render(request, 'sign_in.html')  # Render the sign-in template

    return render(request, 'register.html')  # Render registration form

def sign_in(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            request.session['is_authenticated'] = True
            request.session['user_name'] = "Admin"
            request.session['user_email'] = ADMIN_EMAIL
            return render(request, 'dashboard1.html'  )
        try:
            # Fetch user by email
            user = CustomUser.objects.get(email=email)

            # Check password (without using check_password)
            if password == user.password:
                user.last_login = timezone.now()  # Set the current time
                user.save()  #
                request.session['user_name'] = user.name
                request.session['user_email'] = user.email
                request.session['is_authenticated'] = True
                return render(request, 'dashboard.html' , {'user': user})  # Render the home page
            else:
                messages.error(request, "Email or password is incorrect.")
                return render(request, 'sign_in.html')  # Render sign-in template again
        except CustomUser.DoesNotExist:
            messages.error(request, "Email or password is incorrect.")
            return render(request, 'sign_in.html')  # Render sign-in template again

def linkedin_login(request):
    linkedin_client_id = os.getenv('LINKEDIN_CLIENT_ID')
    # redirect_uri = 'http://127.0.0.1:8000/callbacklin/'
    redirect_uri = "https://socialalytics.in/callbacklin/"
    scope = 'email openid profile   '  # Corrected scope

    # Generate a random state parameter to prevent CSRF attacks
    state = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    # Prepare the parameters as a dictionary
    params = {
        'response_type': 'code',
        'client_id': linkedin_client_id,
        'redirect_uri': redirect_uri,
        'state': state,  # Add the state parameter for security
        'scope': scope,
    }

    # URL-encode the parameters using urlencode
    authorization_url = f"https://www.linkedin.com/oauth/v2/authorization?{urlencode(params)}"

    # Optionally, store the state in the session to verify on the callback
    request.session['oauth_state'] = state

    return render(request, 'linkedin_login.html', {'authorization_url': authorization_url})

from django.http import HttpResponse
from django.shortcuts import redirect
import requests

def linkedin_callback(request):
    # Get the authorization code from the query parameters
    code = request.GET.get('code')
    state = request.GET.get('state')

    if not code:
        return HttpResponse("Error: Authorization code is missing.")

    # Check if the state parameter matches the one stored in the session
    session_state = request.session.get('oauth_state')
    if not session_state or state != session_state:
        return HttpResponse("Error: Invalid state parameter. Possible CSRF attack.")

    linkedin_client_id = os.getenv('LINKEDIN_CLIENT_ID')
    linkedin_client_secret = os.getenv('LINKEDIN_CLIENT_SECRET')
     # Use the same redirect URI as in the login view
    # redirect_uri = 'http://127.0.0.1:8000/callbacklin/'
    redirect_uri = "https://socialalytics.in/callbacklin/"
    
    # Prepare token request data
    token_url = 'https://www.linkedin.com/oauth/v2/accessToken'
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri,
        'client_id': linkedin_client_id,
        'client_secret': linkedin_client_secret,
    }

    # Make a POST request to exchange the authorization code for an access token
    token_response = requests.post(token_url, data=token_data)

    # Check if the token request was successful
    if token_response.status_code != 200:
        return HttpResponse(f"Error fetching access token: {token_response.json()}")

    # Extract the access token from the response
    access_token = token_response.json().get('access_token')
    if not access_token:
        return HttpResponse("Error: Access token is missing.")

    # Save the access token in the session for future API calls
    request.session['access_token'] = access_token
    
    # # Redirect the user to their profile page
    # return redirect('profilel')

    # REDIRECT TO YOUTUBE AFTER SUCCESS
    return redirect("https://www.linkedin.com/")

def profile_view(request):
    access_token = request.session.get('access_token')
    if not access_token:
        return redirect('linkedin_login')

    headers = {'Authorization': f'Bearer {access_token}'}

    # Fetch Profile Data
    profile_url = 'https://api.linkedin.com/v2/userinfo'
    profile_response = requests.get(profile_url, headers=headers)
    if profile_response.status_code != 200:
        return HttpResponse(f"Error fetching profile data: {profile_response.json()}")

    profile_data = profile_response.json()

    # Fetch Posts

    return render(request, 'profilel.html', {
        'profile': profile_data,
    })
    
def facebook_login(request):
   

    # Define the Facebook Login URL with required parameters
    facebook_client_id = os.getenv('FACEBOOK_CLIENT_ID') 
    # Your Facebook App ID
    redirect_uri = 'https://www.marketinganalytics.live/callbackii/'
    scope = 'instagram_basic,instagram_content_publish,instagram_manage_comments,instagram_manage_insights,pages_show_list,pages_read_engagement'
    response_type = 'code'

    auth_url = (
        f"https://www.facebook.com/v14.0/dialog/oauth?"
        f"client_id={facebook_client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"scope={scope}&"
        f"response_type={response_type}"
    )
    return render(request, 'instagram_login.html', {'auth_url': auth_url}) 

def facebook_callback(request):
    """
    Callback view to handle Facebook Login response.
    Extracts the access token from the URL and fetches user data.
    """
    # Get the access token from the URL

    code = request.GET.get('code')
    if not code:
        return JsonResponse({'error': 'Authorization code not provided.'}, status=400)

    facebook_client_id = os.getenv('FACEBOOK_CLIENT_ID')
    facebook_client_secret = os.getenv('FACEBOOK_CLIENT_SECRET')
    redirect_uri = 'https://www.marketinganalytics.live/callbackii/'
    token_url = (
        f"https://graph.facebook.com/v14.0/oauth/access_token?"
        f"client_id={facebook_client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"client_secret={facebook_client_secret}&"
        f"code={code}"
    )

    response = requests.get(token_url)
    token_info = response.json()
    if 'access_token' not in token_info:
        return JsonResponse({'error': 'Could not obtain access token.'}, status=400)

    access_token = token_info['access_token']
    user_url = f"https://graph.facebook.com/v14.0/me/accounts"
    params = {
        'access_token': access_token
    }

    user_response = requests.get(user_url, params=params)
    user_data = user_response.json()
    return JsonResponse({'user_data': user_data}, status=200)

def about_view(request):
    return render(request, 'about.html')
# URL to start the OAuth2 process
# def youtube_login(request):
#     youtube_client_id = os.getenv('YOUTUBE_CLIENT_ID')
#     redirect_uri = 'https://www.marketinganalytics.live/callbackyoutube/'
#     scopes = ['https://www.googleapis.com/auth/youtube.readonly', 'https://www.googleapis.com/auth/analytics' , 'https://www.googleapis.com/auth/yt-analytics.readonly','https://www.googleapis.com/auth/yt-analytics-monetary.readonly','https://www.googleapis.com/auth/youtube','https://www.googleapis.com/auth/youtubepartner','https://www.googleapis.com/auth/youtube.force-ssl']       
#     # Construct the OAuth2 authorization URL
#     auth_url = (
#         f"https://accounts.google.com/o/oauth2/v2/auth?"
#         f"client_id={youtube_client_id}&"
#         f"redirect_uri={redirect_uri}&"
#         f"response_type=code&"
#         f"scope={' '.join(scopes)}&"
#         f"access_type=offline"
#     )


#     # Pass the login URL to the template
#     context = {
#         'auth_url': auth_url
#     }

#     return render(request, 'youtube_login.html', context)

# # This view handles the callback after the user authenticates

# def youtube_callback(request):
#     code = request.GET.get('code')
#     youtube_client_id = os.getenv('YOUTUBE_CLIENT_ID')
#     youtube_client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
#     redirect_uri = 'https://www.marketinganalytics.live/callbackyoutube/'

#     # Exchange authorization code for access token
#     token_url = 'https://oauth2.googleapis.com/token'
#     token_data = {
#         'code': code,
#         'client_id': youtube_client_id,
#         'client_secret': youtube_client_secret,
#         'redirect_uri': redirect_uri,
#         'grant_type': 'authorization_code'
#     }
#     token_response = requests.post(token_url, data=token_data)
#     token_json = token_response.json()

#     # Check if access token is available
#     access_token = token_json.get('access_token')
#     if not access_token:
#         return JsonResponse({'error': 'Failed to retrieve access token'})

#     headers = {'Authorization': f'Bearer {access_token}'}

#     # Fetch channel information
#     channel_info_url = "https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&mine=true"
#     channel_info_response = requests.get(channel_info_url, headers=headers)
#     channel_info = channel_info_response.json()

#     # Fetch uploaded videos
#     videos_list_url = "https://www.googleapis.com/youtube/v3/search?part=snippet&forMine=true&type=video&maxResults=50"
#     videos_list_response = requests.get(videos_list_url, headers=headers)
#     videos_list = videos_list_response.json()

#     # Extract video IDs
#     video_ids = [item['id']['videoId'] for item in videos_list.get('items', []) if 'videoId' in item['id']]

#     # Fetch detailed information for each video
#     video_details = []
#     if video_ids:
#         # Use comma-separated video IDs for batch processing
#         video_ids_str = ','.join(video_ids)
#         video_details_url = "https://www.googleapis.com/youtube/v3/videos"
#         video_params = {
#             'part': 'snippet,contentDetails,statistics,status,player,topicDetails,recordingDetails',
#             'id': video_ids_str
#         }
#         video_details_response = requests.get(video_details_url, headers=headers, params=video_params)
#         video_details = video_details_response.json()
        
#     analytics_url = "https://youtubeanalytics.googleapis.com/v2/reports"
#     today = date.today()
#     start_date = today - timedelta(days=28)  # Past 28 days
#     end_date = today # Yesterday to ensure data availability
#     analytics_params = {
#         'ids': 'channel==MINE',
#         'startDate': start_date.strftime('%Y-%m-%d'),
#         'endDate': end_date.strftime('%Y-%m-%d'),
#         'metrics': 'views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost',
#         'dimensions': 'day',
#         'sort': 'day',
#     }
#     analytics_response = requests.get(analytics_url, headers=headers, params=analytics_params)
#     def fetch_video_comments(video_id):
#         comments = []
#         url = "https://www.googleapis.com/youtube/v3/commentThreads"
#         params = {
#             'part': 'snippet,replies',
#             'videoId': video_id,
#             'maxResults': 100,  # Fetch up to 100 comments per request

#         }

#         while True:
#             response = requests.get(url, headers=headers, params=params,)
#             data = response.json()
#       # Process the comments
#             for item in data.get('items', []):
#                 top_level_comment = item['snippet']['topLevelComment']['snippet']
#                 comments.append({
#                     'author': top_level_comment.get('authorDisplayName'),
#                     'text': top_level_comment.get('textDisplay'),
#                     'likes': top_level_comment.get('likeCount'),
#                     'published_at': top_level_comment.get('publishedAt'),
#                     'replies': [reply['snippet']['textDisplay'] for reply in item.get('replies', {}).get('comments', [])]
#                 })

#         # Check if there's another page of comments
#             next_page_token = data.get('nextPageToken')
#             if not next_page_token:
#                 break
#             params['pageToken'] = next_page_token

#         return comments
#     video_data_combined = []
#     for video in video_details.get('items', []):
#         video_id = video['id']
        
#         comments = fetch_video_comments(video_id)
#         video_analytics_params = {
#             'ids': 'channel==MINE',
#             'filters': f'video=={video_id}',
#             'startDate': start_date.strftime('%Y-%m-%d'),
#             'endDate': end_date.strftime('%Y-%m-%d'),
#             'metrics': 'views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage',
#             'dimensions': 'day',
#             'sort': 'day',
#         }

#         video_analytics_response = requests.get(analytics_url, headers=headers, params=video_analytics_params)
#         video_analytics_data = video_analytics_response.json()
#   # Prepare the combined video data
#         combined_video = {
#             'video_id': video_id,
#             'details': video,
#             'comments': comments,
#             'analytics': {
#                 'labels': [row[0] for row in video_analytics_data.get('rows', [])],  # Dates
#                 'views': [row[1] for row in video_analytics_data.get('rows', [])],
#                 'estimatedMinutesWatched': [row[2] for row in video_analytics_data.get('rows', [])],
#                 'averageViewDuration': [row[3] for row in video_analytics_data.get('rows', [])],
#                 'averageViewPercentage': [row[4] for row in video_analytics_data.get('rows', [])],
#             }
#         }

#         video_data_combined.append(combined_video)  # Store combined data in the list




#     analytics_data = analytics_response.json()
#     chart_data = {
#         'labels': [row[0] for row in analytics_data.get('rows', [])],  # Dates
#         'views': [row[1] for row in analytics_data.get('rows', [])],
#         'estimatedMinutesWatched': [row[2] for row in analytics_data.get('rows', [])],
#         'averageViewDuration': [row[3] for row in analytics_data.get('rows', [])],
#         'averageViewPercentage': [row[4] for row in analytics_data.get('rows', [])],
#         'subscribersGained': [row[5] for row in analytics_data.get('rows', [])],
#         'subscribersLost': [row[6] for row in analytics_data.get('rows', [])],
#     }
#     context = {
#         'channel': channel_info,
#         'videos': videos_list,
#         'video_details': video_details,
#         'analytics': analytics_data,
#         'chart_data': chart_data,
#         'video_data_combined': video_data_combined,
#     }

#     return render(request, 'youtube_data.html', context)

import os
import random
import string
import hashlib
import base64
import requests
from urllib.parse import urlencode
from django.shortcuts import redirect, render
from django.http import HttpResponse

import os
import random
import string
import hashlib
import base64
import requests
from urllib.parse import urlencode
from django.shortcuts import redirect, render
from django.http import HttpResponse

# Your Twitter API credentials
TWITTER_CLIENT_ID = os.getenv('TWITTER_CLIENT_ID')
TWITTER_CLIENT_SECRET = os.getenv('TWITTER_CLIENT_SECRET')
REDIRECT_URI = 'https://www.marketinganalytics.live/callback/'  # Ensure this matches your Twitter App's callback URL
AUTH_URL = 'https://twitter.com/i/oauth2/authorize'
TOKEN_URL = 'https://api.twitter.com/2/oauth2/token'

# Utility functions for PKCE
def generate_code_verifier():
    charset = string.ascii_letters + string.digits + '-._~'
    return ''.join(random.choice(charset) for _ in range(128))

def generate_code_challenge(verifier):
 charset = string.ascii_letters + string.digits + '-._~'
 return ''.join(random.choice(charset) for _ in range(128))

def generate_code_challenge(verifier):
    hashed = hashlib.sha256(verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(hashed).decode('ascii').rstrip('=')

# Step 1: Display login page
def login_page(request):
    verifier = generate_code_verifier()
    challenge = generate_code_challenge(verifier)

    # Save the verifier in session to use later in the callback
    request.session['code_verifier'] = verifier
    params = {
        'response_type': 'code',
        'client_id': TWITTER_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'tweet.read users.read',  # Define the permissions you need
        'state': ''.join(random.choices(string.ascii_letters + string.digits, k=8)),  # Random state
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
    }

    auth_url = f"{AUTH_URL}?{urlencode(params)}"
    return render(request, 'login.html', {'auth_url': auth_url})

# Step 2: Handle Twitter OAuth callback
def twitter_callback(request):
    code = request.GET.get('code')
    code_verifier = request.session.get('code_verifier')

    if not code or not code_verifier:
        return HttpResponse("Error: Missing code or verifier", status=400)

    # Prepare client credentials for Basic Auth (client_id:client_secret)
    credentials = f"{TWITTER_CLIENT_ID}:{TWITTER_CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')

    # Step 3: Exchange authorization code for access token
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'code_verifier': code_verifier,
    }

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_credentials}',  # Include the Basic Auth header
    }

    token_response = requests.post(TOKEN_URL, data=token_data, headers=headers)
    if token_response.status_code != 200:
        return HttpResponse(f"Error fetching token: {token_response.text}", status=400)

    token_json = token_response.json()
    access_token = token_json.get('access_token')

    if not access_token:
        return HttpResponse("Error: No access token received", status=400)

    # Store access token in session
    request.session['access_token'] = access_token

    # Step 4: Redirect to profile page


    return redirect('profile')



# Helper function to fetch user insights
def fetch_user_insights(access_token, user_id):
    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    url = f"https://api.twitter.com/2/users/{user_id}"
    params = {
        'user.fields': 'public_metrics',  # Include metrics like followers count
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        return None, f"Error fetching user insights: {response.text}"

    return response.json(), None


# Helper function to fetch all tweets with metrics

# Helper function to fetch all tweets with metrics
def fetch_all_post_insights_with_metrics(access_token, user_id):
    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        'tweet.fields': 'public_metrics,created_at',  # Include metrics and creation timestamp
        'max_results': 100,                          # Maximum allowed per request
    }

    all_tweets_with_metrics = []  # Store all fetched tweets with metrics
    next_token = None

    while True:
        if next_token:
            params['pagination_token'] = next_token

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            return None, f"Error fetching tweets: {response.text}"

        response_data = response.json()
        tweets = response_data.get('data', [])
        all_tweets_with_metrics.extend(tweets)
        
         # Check if there's a next page
        next_token = response_data.get('meta', {}).get('next_token')
        if not next_token:  # Break if no more pages
            break

    return all_tweets_with_metrics, None


# Profile view
def profile(request):
    access_token = request.session.get('access_token')
    if not access_token:
        return redirect('login_page')

    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    # Step 1: Fetch user profile
    profile_response = requests.get('https://api.twitter.com/2/users/me', headers=headers)
    if profile_response.status_code != 200:
        return HttpResponse(f"Error fetching profile: {profile_response.text}", status=400)

    profile_data = profile_response.json()
    user_id = profile_data.get('data', {}).get('id')
    
    if not user_id:
        return HttpResponse("Error: Unable to retrieve user ID", status=400)

    # Step 2: Fetch user insights
    user_insights, user_error = fetch_user_insights(access_token, user_id)
    if user_error:
        return HttpResponse(user_error, status=400)

    # Step 3: Fetch all tweets with insights
    all_tweets_with_metrics, post_error = fetch_all_post_insights_with_metrics(access_token, user_id)
    if post_error:
        return HttpResponse(post_error, status=400)

    # Render the profile with insights
    return render(request, 'profile.html', {
        'profile': profile_data,
        'user_insights': user_insights,
        'post_insights': all_tweets_with_metrics,
    })
from django.shortcuts import render, redirect
from django.http import JsonResponse
import requests


import requests
from django.shortcuts import redirect, render
from django.conf import settings
from requests.auth import HTTPBasicAuth
from urllib.parse import urlencode
from django.utils.crypto import get_random_string


from django.shortcuts import render, redirect
from django.http import JsonResponse
import requests

import requests
from django.shortcuts import render, redirect
from django.http import JsonResponse

INSTAGRAM_CLIENT_ID = os.getenv('INSTAGRAM_CLIENT_ID')
INSTAGRAM_CLIENT_SECRET = os.getenv('INSTAGRAM_CLIENT_SECRET')
REDIRECT_URIi = 'https://www.marketinganalytics.live/callbacki/'
SCOPESi = 'user_profile,user_media'
ACCESS_TOKEN = None  # Store access token here
def index(request):
    if request.session.get('is_authenticated', False):
        auth_url = (
            "https://www.instagram.com/oauth/authorize"
            "?client_id=969834388286582"
            "&redirect_uri=https://socialalytics.in/callbacki/"
            "&scope=instagram_business_basic,instagram_business_manage_messages,"
            "instagram_business_manage_comments,instagram_business_content_publish"
            "&response_type=code"
            "&enable_fb_login=0"
            "&force_authentication=1"
        )
        return render(request, 'index.html', {"auth_url": auth_url})
    else:
        return render(request, "sign_in.html")

def instagram_callback(request):
    code = request.GET.get('code')

    if code:
        token_url = 'https://api.instagram.com/oauth/access_token'
        data = {
            'client_id': INSTAGRAM_CLIENT_ID,
            'client_secret': INSTAGRAM_CLIENT_SECRET,
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URIi,
            'code': code
        }
        response = requests.post(token_url, data=data)
        access_token_info = response.json()

        if 'access_token' in access_token_info:
            global ACCESS_TOKEN
            ACCESS_TOKEN = access_token_info['access_token']
            request.session['access_token'] = ACCESS_TOKEN
            return redirect('instagram_info')
        else:
            return JsonResponse({'error': 'Failed to get access token', 'response': access_token_info})
    else:
        return JsonResponse({'error': 'No code provided'})

def instagram_info(request):
    # Retrieve the access token from the session
    access_token = request.session.get('access_token')
    if not access_token:
        return JsonResponse({'error': 'Access token not available. Please login again.'})

    # Fetch user information
    user_info = fetch_instagram_user_info(access_token)

    # Fetch all media posts
    media_data = fetch_instagram_media(access_token)

    # Fetch insights for each post and store debug messages
    media_insights = []
    debug_info = []  # List to hold debug messages for display in HTML and console
    for media in media_data.get('data', []):
        insights = fetch_instagram_post_insights(media['id'], access_token)
        
           # Check if insights are empty and add a debug message
        if 'data' not in insights or not insights['data']:
            debug_info.append(f"No insights available for media ID {media['id']}")
        elif 'error' in insights:
            debug_info.append(f"Error fetching insights for media ID {media['id']}: {insights['error']}")

        # Add insights to the media object and append to media_insights list
        media['insights'] = insights
        media_insights.append(media)

    # Render the Instagram info page with user info, media insights, and debug information
    return render(request, 'instagram_info.html', {
        'user_info': user_info,
        'media_insights': media_insights,
        'debug_info': debug_info  # Pass debug messages to the template
    })

def fetch_instagram_user_info(access_token):
    # Fetch user info from the Instagram API
    url = f"https://graph.instagram.com/me?fields=id,username,account_type,media_count&access_token={access_token}"
    response = requests.get(url)
    return response.json()

def fetch_instagram_media(access_token):
    # Fetch user's media posts from the Instagram API
    url = f"https://graph.instagram.com/me/media?fields=id,caption,media_type,media_url,permalink&access_token={access_token}"
    all_media = []

    while url:
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            all_media.extend(data['data'])

        # Check for pagination
        if 'paging' in data and 'next' in data['paging']:
            url = data['paging']['next']
        else:
            url = None  # No more pages to fetch

    return {'data': all_media}

def fetch_instagram_post_insights(media_id, access_token):
    # Fetch insights for a specific media post
    url = f"https://graph.instagram.com/v21.0/{media_id}/insights"
    params = {
        'metric': 'impressions,reach,likes,comments,saved,shares,total_interactions',
        'access_token': access_token
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        # Return an error message if the request fails
        return {'error': 'Failed to fetch post insights', 'details': response.json()}

import random
import string
import requests
import json
import logging
from urllib.parse import urlencode
from django.shortcuts import render
from django.http import JsonResponse
import random
import string
import requests
import json
import logging
from urllib.parse import urlencode
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
import datetime
# Reddit OAuth settings
import requests
import json
import logging
import random
import string
from urllib.parse import urlencode
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
import datetime
# Reddit OAuth settings
import requests
import json
import logging
import random
import string
from urllib.parse import urlencode
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
import datetime
# Reddit OAuth settings
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDIRECT_URIR = 'https://www.marketinganalytics.live/callbackreddit/'
SCOPESR = 'identity read history'  # Adding 'history' scope to fetch user's posts
AUTH_URLR = 'https://www.reddit.com/api/v1/authorize'

def reddit_login(request):
    params = {
        'client_id': REDDIT_CLIENT_ID,
        'response_type': 'code',
        'state': ''.join(random.choices(string.ascii_letters + string.digits, k=8)),
        'redirect_uri': REDIRECT_URIR,
        'duration': 'temporary',
        'scope': SCOPESR,
    }
    auth_url = f"{AUTH_URLR}?{urlencode(params)}"
    return render(request, 'indexr.html', {'auth_url': auth_url})


def reddit_callback(request):
    code = request.GET.get('code')
    if code:
        token_url = 'https://www.reddit.com/api/v1/access_token'
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': REDIRECT_URIR,
        }
        auth = (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        headers = {'User-Agent': 'TestApp'}

        response = requests.post(token_url, data=data, auth=auth, headers=headers)

        try:
            access_token_info = response.json()
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON: {str(e)}. Response content: {response.content.decode()}")
            return JsonResponse({'error': 'Invalid response from Reddit', 'response': response.content.decode()})

        if 'access_token' in access_token_info:
            access_token = access_token_info['access_token']
            request.session['access_token'] = access_token  # Store access token in session
            user_info = fetch_reddit_user_info(access_token)

            if user_info:
                username = user_info.get('name')
                user_posts = fetch_all_user_posts(access_token, username)
                account_creation_time = user_info.get('created_utc')
                account_age = calculate_account_age(account_creation_time)
                return render(request, 'reddit_info.html', {'user_info': user_info, 'user_posts': user_posts})
            else:
                return JsonResponse({'error': 'Failed to get user information'})
        else:
            return JsonResponse({'error': 'Failed to get access token', 'response': access_token_info})
    else:
        return JsonResponse({'error': 'No code provided'})

def fetch_reddit_user_info(access_token):
    user_info_url = 'https://oauth.reddit.com/api/v1/me'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'TestApp'
    }
    response = requests.get(user_info_url, headers=headers)
    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error(f"Failed to decode user info JSON: {response.content.decode()}")
        return None

def calculate_account_age(created_utc):
    # Convert the 'created_utc' timestamp to a datetime object
    created_time = datetime.datetime.utcfromtimestamp(created_utc)

    # Make the 'created_time' aware using Django's timezone
    created_time = timezone.make_aware(created_time, timezone.get_current_timezone())

    # Get the current time
    current_time = timezone.now()

    # Calculate the difference between current time and account creation time
    age = current_time - created_time

    # Calculate years, months, days
    years = age.days // 365
    months = (age.days % 365) // 30
    days = (age.days % 365) % 30

    return f"{years} years, {months} months, {days} days"

def fetch_all_user_posts(access_token, username):
    url = f'https://oauth.reddit.com/user/{username}/submitted'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'TestApp'
    }
    all_posts = []
    after = None  # Pagination parameter
    while True:
        params = {'after': after} if after else {}
        response = requests.get(url, headers=headers, params=params)

        try:
            data = response.json()
            posts_data = data['data']['children']

            for post in posts_data:
                post_data = post['data']
                image_url = None
                engagement_rate = None
                datar = post_data.get('selftext', '')
                sentiment = analyze_sentiment(datar)

                # Check if the post has a smaller image available (thumbnail or preview)
                if post_data.get('post_hint') == 'image':
                    image_url = post_data.get('thumbnail')

                # Calculate an approximate engagement rate (upvotes divided by comments)
                if post_data.get('score') and post_data.get('num_comments'):
                    engagement_rate = post_data['score'] / (post_data['num_comments'] + 1)  # Avoid division by zero

                # Fetch comments for each post
                comments = fetch_post_comments(access_token, post_data['id'])

                # Fetch views (if available)
                views = post_data.get('view_count', 0)  # 'view_count' might not be available, so set default to 0
                  # Format the 'created_utc' timestamp into a readable date
                created_time = datetime.datetime.utcfromtimestamp(post_data['created_utc'])

                # Make the datetime aware using timezone.make_aware
                created_time = timezone.make_aware(created_time, timezone.get_current_timezone())  # Convert to aware datetime

                # Include insights data for each post
                all_posts.append({
                    'title': post_data['title'],
                    'score': post_data['score'],
                    'url': post_data['url'],
                    'num_comments': post_data['num_comments'],
                    'created': created_time,  # Use the formatted 'created' time
                    'selftext': post_data.get('selftext', ''),
                    'image_url': image_url,  # Small image URL (if available)
                    'engagement_rate': engagement_rate,
                    'upvote_ratio': post_data.get('upvote_ratio', 0),
                    'content_type': post_data.get('post_hint', 'text'),
                    'comments': comments,
                    'subreddit': post_data.get('subreddit', 'Unknown'),  # Subreddit of the post
                    'author': post_data.get('author', 'Unknown'),  # Author of the post
                    'views': views,  # Views count (if available)
                    'created_at': created_time,  # Created time (formatted)
                    'sentiment' : sentiment,
                })
            after = data['data']['after']
            if not after:  # Exit the loop if no more pages
                break
        except json.JSONDecodeError:
            logging.error("Failed to decode user posts JSON.")
            break

    return all_posts

def fetch_post_comments(access_token, post_id):
    url = f'https://oauth.reddit.com/comments/{post_id}'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'TestApp'
    }
    response = requests.get(url, headers=headers)

    try:
        comments_data = response.json()
        comments = []

        # The comments are usually in the second element of the returned list
        if isinstance(comments_data, list) and len(comments_data) > 1:
            comment_section = comments_data[1]
            if 'data' in comment_section and 'children' in comment_section['data']:
                for comment in comment_section['data']['children']:
                    comment_data = comment.get('data', {})
                    # Extract relevant fields
                    comments.append({
                        'author': comment_data.get('author', 'Unknown'),
                        'body': comment_data.get('body', 'No comment text available'),
                        'score': comment_data.get('score', 0),
                        'created': comment_data.get('created_utc', None),
                    })
        return comments
    except json.JSONDecodeError:
        logging.error("Failed to decode comments JSON.")
        return []
    except KeyError as e:
        logging.error(f"KeyError while fetching comments: {e}")
        return []

# Reddit OAuth settings
from django.utils.safestring import mark_safe  # for safely passing JSON data to template
import json

import requests

import requests

def fetch_facebook_data(page_id, access_token):
    url = f"https://graph.facebook.com/v12.0/{page_id}/posts"
    params = {
        'fields': 'id,message,created_time,likes.summary(true),comments.summary(true),shares',
        'access_token': access_token,
    }
    all_posts = []
    while url:
        response = requests.get(url, params=params if len(all_posts) == 0 else None)
        if response.status_code == 200:
            data = response.json()
            all_posts.extend(data.get('data', []))
            # Get the next page URL if available
            url = data.get('paging', {}).get('next')
        else:
            # Break the loop if there's an error
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break
    return all_posts

def analyze_facebook_data(data):
    total_posts = len(data)
    total_likes = sum(item.get('likes', {}).get('summary', {}).get('total_count', 0) for item in data)
    total_comments = sum(item.get('comments', {}).get('summary', {}).get('total_count', 0) for item in data)
    total_shares = sum(item.get('shares', {}).get('count', 0) for item in data)

    avg_engagement = (total_likes + total_comments + total_shares) / total_posts if total_posts else 0

    # Find the top-performing post
    top_post = max(
        data,
        key=lambda x: (
            x.get('likes', {}).get('summary', {}).get('total_count', 0) +
            x.get('comments', {}).get('summary', {}).get('total_count', 0) +
            x.get('shares', {}).get('count', 0)
        ),
        default=None
    )

    return {
        'total_posts': total_posts,
        'total_likes': total_likes,
        'total_comments': total_comments,
        'total_shares': total_shares,
        'avg_engagement': avg_engagement,
        'top_post': top_post,
    }
    
def predict_engagement(posts_with_insights):
    # Initialize counters for total engagement, likes, and shares
    total_engagements = 0
    total_likes = 0
    total_shares = 0

    # First, calculate the total likes, shares, and engagement
    for post in posts_with_insights:
        # Extracting insights data for each post
        insights_data = post['insights'].get('data', [])
        insights_dict = {}
        for metric in insights_data:
            insights_dict[metric['name']] = metric.get('values', [{}])[0].get('value', 0)

        # Assigning transformed insights back to the post
        post['insights'] = insights_dict

        # Fetching the relevant metrics
        shares = post.get('shares', 0)
        likes = post['insights'].get('post_reactions_like_total', 0)
        comments = post.get('number_of_comments', 0)
        clicks = post['insights'].get('post_clicks', 0)
        
        
        # Calculate engagement as the sum of shares, likes, comments, and clicks
        engagement = shares + likes + clicks + comments
        total_engagements += engagement
        total_likes += likes
        total_shares += shares

        # Store the engagement value back to the post
        post['engagement'] = engagement
        post['likes'] = likes
        post['shares'] = shares

    # Now calculate the engagement percentage for each post based on total values
    for post in posts_with_insights:
        # Calculate engagement percentage for this post
        post['engagement_percentage'] = (post['engagement'] / total_engagements) * 100 if total_engagements else 0

        # Calculate likes percentage for this post
        post['likes_percentage'] = (post['likes'] / total_likes) * 100 if total_likes else 0

        # Calculate shares percentage for this post
        post['shares_percentage'] = (post['shares'] / total_shares) * 100 if total_shares else 0

    return posts_with_insights

# Facebook Login View
def facebook_login_view(request):
    facebook_client_id2 = os.getenv('FACEBOOK_CLIENT_ID')  # Your Facebook App ID
    # redirect_uri = 'https://www.marketinganalytics.live/callbackfacebook/'
    redirect_uri = 'https://socialalytics.in/callbackfacebook/'
    scope = 'pages_read_engagement,pages_read_user_content,read_insights'
    response_type = 'code'

    auth_url = (
        f"https://www.facebook.com/v14.0/dialog/oauth?"
        f"client_id={facebook_client_id2}&"
        f"redirect_uri={redirect_uri}&"
        f"scope={scope}&"
        f"response_type={response_type}"
    )

    return render(request, 'facebook_login.html', {'auth_url': auth_url})

# Facebook Callback View
def get_shares_for_post(post_id, access_token):
    # Define the URL to fetch the shares data
    url = f"https://graph.facebook.com/{post_id}?fields=shares&access_token={access_token}"

    # Send the request to the Facebook Graph API
    response = requests.get(url)

    # Parse the JSON response
    data = response.json()

    # Check if shares data is available
    if 'shares' in data:
        return data['shares']['count']
    else:
        return 0
def facebook_callback_view(request):
    code = request.GET.get('code')
    print(code)  # Debugging line to check the received code
     # Handle the case where 'code' is None 
    if not code:
        return JsonResponse({'error': 'Authorization code not provided.'}, status=400)
    facebook_client_id2 = os.getenv('FACEBOOK_CLIENT_ID2')
    facebook_client_secret2 = os.getenv('FACEBOOK_CLIENT_SECRET2')
    redirect_uri = 'https://www.marketinganalytics.live/callbackfacebook/'
    token_url = (
        f"https://graph.facebook.com/v14.0/oauth/access_token?"
        f"client_id={facebook_client_id2}&"
        f"redirect_uri={redirect_uri}&"
        f"client_secret={facebook_client_secret2}&"
        f"code={code}"
    )
    print(token_url)  # Debugging line to check the token URL

    response = requests.get(token_url)
    token_info = response.json()

    access_token = token_info['access_token']

    pages = []
    pages_url = f"https://graph.facebook.com/me/accounts?access_token={access_token}"
    
    while pages_url:
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        if 'error' in pages_data:
            return JsonResponse({'error': pages_data['error']['message']}, status=400)

        # Add current page data to the list
        pages.extend(pages_data.get('data', []))

        # Update pages_url to the next page if available
        pages_url = pages_data.get('paging', {}).get('next')

    return render(request, 'pages.html', {'pages': pages, 'access_token': access_token})
# Function to Get Post Insights
def get_post_insights(page_access_token, post_id):
    insights_url = f"https://graph.facebook.com/v21.0/{post_id}/insights"
    params = {
        'metric': 'post_reactions_like_total,post_reactions_love_total,post_reactions_wow_total,post_reactions_haha_total,post_reactions_sorry_total,post_reactions_anger_total,post_clicks,post_impressions,post_impressions_unique,post_impressions_paid,post_impressions_paid_unique,post_impressions_fan,post_impressions_fan_unique,post_impressions_organic,post_impressions_organic_unique,post_impressions_viral,post_impressions_viral_unique,post_impressions_nonviral,post_impressions_nonviral_unique',                 
        'access_token': page_access_token
    }
    response = requests.get(insights_url, params=params)
    return response.json()

# Get Page Posts View with Insights Integration for Visualization
import requests
import json
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe
from textblob import TextBlob

def analyze_sentiment(text):
    if not text:  # Handle cases where text might be None
        return "Neutral"
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # Polarity score: [-1.0, 1.0]
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"
    
def get_page_posts_view(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        user_access_token = request.POST.get('access_token')

        if not page_id or not user_access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)

        # Fetch the page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={user_access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)
   # Fetch all posts using pagination
        all_posts = []
        posts_url = f"https://graph.facebook.com/v21.0/{page_id}/feed?access_token={page_access_token}"

        while posts_url:
            posts_response = requests.get(posts_url)
            posts_data = posts_response.json()

            if 'error' in posts_data:
                return JsonResponse({'error': posts_data['error']['message']}, status=400)

            # Append posts from the current batch
            all_posts.extend(posts_data.get('data', []))

            # Update posts_url to the next page, if available
            posts_url = posts_data.get('paging', {}).get('next')

        # Fetch insights for each post
        posts_with_insights = []
        for post in all_posts:
            post_id = post['id']
            insights_data = get_post_insights(page_access_token, post_id)
            post['insights'] = insights_data
            posts_with_insights.append(post)
            
           # Convert data to JSON for use in JavaScript
        posts_json = mark_safe(json.dumps(posts_with_insights))

        return render(request, 'reports.html', {
            'posts': posts_with_insights,
            'page_name': page_id,
            'posts_json': posts_json  # Pass JSON data to template for visualization
        })

    return redirect('facebook_login')

# Get Page Reviews View (Optional)
def get_page_reviews_view(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        user_access_token = request.POST.get('access_token')

        if not page_id or not user_access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)

        pages_url = f"https://graph.facebook.com/me/accounts?access_token={user_access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)

        reviews_url = f"https://graph.facebook.com/v21.0/{page_id}/ratings?access_token={page_access_token}"
        reviews_response = requests.get(reviews_url)
        reviews_data = reviews_response.json()
        if 'error' in reviews_data:
            return JsonResponse({'error': reviews_data['error']['message']}, status=400)

        return render(request, 'reviews.html', {'reviews': reviews_data.get('data', []), 'page_name': page_id})

    return redirect('facebook_login')

def get_page_postanalysis_view(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        user_access_token = request.POST.get('access_token')

        if not page_id or not user_access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)

        # Fetch page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={user_access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)
          # Fetch all posts for the selected page using pagination
        posts_with_insights = []
        fields = "id,message,created_time,attachments{media_type,media,subattachments}"
        posts_url = f"https://graph.facebook.com/v21.0/{page_id}/feed?fields={fields}&access_token={page_access_token}"
        while posts_url:
            response = requests.get(posts_url)
            posts_data = response.json()

            if 'error' in posts_data:
                return JsonResponse({'error': posts_data['error']['message']}, status=400)

            for post in posts_data.get('data', []):
                post_id = post['id']
                insights_data = get_post_insights(page_access_token, post_id)
                message = post.get('message', '')
                sentiment = analyze_sentiment(message)
                share_count = get_shares_for_post(post_id, page_access_token)
                post['shares'] = share_count
                post['sentiment'] = sentiment
                post['insights'] = insights_data
                media_data = None
                if 'attachments' in post:
                    attachments = post['attachments']['data'][0]
                    media_data = attachments.get('media')
                    media_type = attachments.get('media_type')

                    # Check for subattachments for albums
                    if 'subattachments' in attachments:
                        subattachments = attachments['subattachments']['data']
                        media_data = [sub.get('media') for sub in subattachments]

                    post['media_type'] = media_type
                    post['media'] = media_data
                comments_url = f"https://graph.facebook.com/{post_id}/comments?access_token={page_access_token}"
                comments_response = requests.get(comments_url)
                comments_data = comments_response.json()

                if 'data' in comments_data:
                    post['comments'] = comments_data['data']
                    post['number_of_comments'] = len(comments_data['data'])
                    nc=len(comments_data['data'])
                else:
                    post['comments'] = []
                    post['number_of_comments'] = 0
                    nc = 0
                posts_with_insights.append(post)

            # Get the next page URL from the pagination info
            posts_url = posts_data.get('paging', {}).get('next')
        posts_with_insights = predict_engagement(posts_with_insights)
        top_keywords = extract_keywords_and_analyze(posts_with_insights)
        
          # Calculate optimal posting time
        best_hour, best_day = optimal_posting_time(posts_with_insights)
        data = fetch_facebook_data(page_id ,  page_access_token )
        analytics = analyze_facebook_data(data)
        # Convert posts_with_insights to a DataFrame
        # Run analyses
        sentiment_recommendation = analyze_sentiment_over_time(posts_with_insights)
        length_recommendation = analyze_post_length(posts_with_insights)

        posts_json = mark_safe(json.dumps(posts_with_insights))
        return render(request, 'reports4.html', {
            'posts': posts_with_insights,
            'page_name': page_id,
            'posts_json': posts_json,
            'top_keywords': top_keywords,
            'best_hour': best_hour,
            'best_day': best_day,
            'sentiment_recommendation': sentiment_recommendation,
            'length_recommendation': length_recommendation,
            'analytics': analytics
        })

    return redirect('facebook_login')

def create_post_view(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        user_access_token = request.POST.get('access_token')
        post_message = request.POST.get('message')

        if not page_id or not user_access_token or not post_message:
            return JsonResponse({'error': 'Page ID, access token, or message missing.'}, status=400)

        # Fetch page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={user_access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)

        # If a photo is provided, upload it first
            # If no photo, create a text-only post
        post_data = {
               'message': post_message,
                'access_token': page_access_token
            }
        
        post_url = f"https://graph.facebook.com/{page_id}/feed"

        # Create the post
        post_response = requests.post(post_url, data=post_data)
        post_result = post_response.json()

        if 'error' in post_result:
            return render(request, 'failure.html')

        return  render(request, 'success.html')

    return redirect('select_page')

def graphicala_view(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        user_access_token = request.POST.get('access_token')

        if not page_id or not user_access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)

        # Fetch page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={user_access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)
         # Fetch all posts with pagination
        fields = "id,message,created_time,attachments{media_type,media,subattachments}"
        posts_url = f"https://graph.facebook.com/v21.0/{page_id}/feed?fields={fields}&access_token={page_access_token}"
        posts_with_insights = []
        while posts_url:
            posts_response = requests.get(posts_url)
            posts_data = posts_response.json()

            if 'error' in posts_data:
                return JsonResponse({'error': posts_data['error']['message']}, status=400)

            # Process posts and fetch insights
            for post in posts_data.get('data', []):
                post_id = post['id']
                insights_data = get_post_insights(page_access_token, post_id)
                media_data = None
                if 'attachments' in post:
                    attachments = post['attachments']['data'][0]
                    media_data = attachments.get('media')
                    media_type = attachments.get('media_type')

                    # Check for subattachments for albums
                    if 'subattachments' in attachments:
                        subattachments = attachments['subattachments']['data']
                        media_data = [sub.get('media') for sub in subattachments]

                    post['media_type'] = media_type
                    post['media'] = media_data
                    
                post['insights'] = insights_data
                posts_with_insights.append(post)

            # Check for next page in pagination
            posts_url = posts_data.get('paging', {}).get('next')

        # Convert data to JSON for JavaScript use
        posts_json = mark_safe(json.dumps(posts_with_insights))

        return render(request, 'reports2.html', {
            'posts': posts_with_insights,
            'page_name': page_id,
            'posts_json': posts_json  # Pass JSON data to template for visualization
        })

    return redirect('facebook_login')
def get_page_insights(page_access_token, page_id, period):
    insights_url = f"https://graph.facebook.com/v21.0/{page_id}/insights"
    params = {
        'metric': (
            'page_total_actions,page_post_engagements,page_fan_adds_by_paid_non_paid_unique,'
            'page_lifetime_engaged_followers_unique,page_daily_follows,page_daily_follows_unique,'
            'page_daily_unfollows_unique,page_follows,page_impressions,page_impressions_unique,'
            'page_impressions_paid,page_impressions_paid_unique,page_impressions_viral,'
            'page_impressions_viral_unique,page_impressions_nonviral,page_impressions_nonviral_unique'
        ),
        'period': period,
        'access_token': page_access_token
    }
    response = requests.get(insights_url, params=params)
    return response.json()

def get_page_insights_view(request):
    if request.method == 'POST':
        page_id = request.POST.get('page_id')
        access_token = request.POST.get('access_token')
        period = request.POST.get('period', 'day')  # Default to 'day' if no period is selected

        if not page_id or not access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)
  # Get page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)
   # Fetch page insights
        page_insights = get_page_insights(page_access_token, page_id, period)

        return render(request, 'page_insights.html', {
            'page_insights': page_insights,
            'page_name': page_id,
            'period': period
        })

    return redirect('facebook_login')

# Helper function to get page insights

# Fetch metrics from Facebook Graph API

def fetch_facebook_metrics(page_access_token, page_id):
    # Define the metrics to fetch with their respective periods
    metrics_with_period = {
        "page_post_engagements": "day,week,days_28",
        "page_fan_adds_by_paid_non_paid_unique": "day",
        "page_lifetime_engaged_followers_unique": "lifetime",
        "page_daily_follows": "day",
        "page_daily_follows_unique": "day,week,days_28",
        "page_daily_unfollows_unique": "day,week,days_28",
        "page_follows": "day",
    }

    metrics_data = {}
      # Fetch data for each metric
    for metric, period in metrics_with_period.items():
        url = f"https://graph.facebook.com/v17.0/{page_id}/insights"
        params = {
            "metric": metric,
            "access_token": page_access_token,
            "period": period,
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json().get("data", [])
            # Use the specified period directly
            metrics_data[metric] = [
                {
                    "period": period,
                    "value": value.get("value", 0),
                }
                for metric_data in data for value in metric_data.get("values", [])
            ]
        else:
            print(f"Error fetching metric {metric}: {response.status_code}, {response.text}")
            metrics_data[metric] = [{"period": "Error", "value": 0}]

    return metrics_data
# View to render the page
def facebook_insights(request):
     if request.method == 'POST':
        page_id = request.POST.get('page_id')
        access_token = request.POST.get('access_token')
        period = request.POST.get('period', 'day')  # Default to 'day' if no period is selected

        if not page_id or not access_token:
            return JsonResponse({'error': 'Page ID or access token missing.'}, status=400)

        # Get page access token
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={access_token}"
        pages_response = requests.get(pages_url)
        pages_data = pages_response.json()

        page_access_token = None
        for page in pages_data.get('data', []):
            if page['id'] == page_id:
                page_access_token = page.get('access_token')
                break

        if not page_access_token:
            return JsonResponse({'error': 'Page access token not found.'}, status=400)

    # Fetch metrics
     metrics_data = fetch_facebook_metrics(page_access_token, page_id)
     
        # Pass data to the template
     context = {

        "metrics": metrics_data,
        "metrics_json": metrics_data,  # Pass JSON for use in JavaScript
    }
     return render(request, "facebook_insights.html", context)