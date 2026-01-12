from django.urls import path
from . import views

urlpatterns = [
    path('post/metrics/', views.post_metrics, name='post_metrics'),
    path('request_password_reset/', views.request_password_reset, name='request_password_reset'),
    path('reset_password/<uidb64>/<token>/', views.reset_password, name='reset_password'),

    path('forgot_password/', views.forgot_password, name='forgot_password'),
    path('youtube/videos/', views.youtube_videos, name='youtube_videos'),
#    path('youtube/video-analytics/<str:video_id>/<str:time_period>/', views.video_analytics, name='video_analytics'),
    path('youtube/video-analytics/', views.youtube_analytics, name='youtube_analytics'),
    path('feedback-list/', views.feedback_list, name='feedback_list'),
    path('feedback/', views.submit_feedback, name='submit_feedback'),
    path('feedback-success/', views.feedback_success, name='feedback_success'),
    path('twitters/', views.login_page, name='twitters'),  # Login Page
    path('callback/', views.twitter_callback, name='twitter_callback'),  # Twitter OAuth callback
    path('profile/', views.profile, name='profile'),
    path('insta/', views.index, name='login_page'),  # Login Page
    path('callbacki/', views.instagram_callback, name='insta_callback'),  #
    path('instagram_info/', views.instagram_info, name='instagram_info'),
    path('', views.landing_view, name='landing'),  # Login Page

    # path('youtubel/', views.youtube_login, name='login_page'),  # Login Page
    
    path('callbackyoutube/', views.youtube_callbackn, name='youtube_callback'),
    path('about/', views.about_view, name='about'),
    path('reddit/', views.reddit_login, name='reddit_login'),  # Home page where the Reddit login link is displayed
    path('callbackreddit/', views.reddit_callback, name='reddit_callback'),
        # urls.py
    path('facebookl/', views.facebook_login_view, name='facebook_login'),
    path('callbackfacebook/', views.facebook_callback_view, name='facebook_callback'),
    path('get_page_posts/', views.get_page_posts_view, name='get_page_posts'),
    path('get_page_postanalysis/', views.get_page_postanalysis_view, name='get_page_postanalysis'),
    path('create_post/', views.create_post_view, name='create_post'),
    path('graphicala/', views.graphicala_view, name='graphicala'),
    path('get_page_reviews/', views.get_page_reviews_view, name='get_page_reviews'),
    path('facebook_login/', views.facebook_login, name='facebook_login'),
    path('callbackii/', views.facebook_callback, name='facebook_callback'),
    path('get_page_insights/', views.get_page_insights_view, name='get_page_insights'),
    path('linkedinl/', views.linkedin_login, name='linkedin_login'),
    path('callbacklin/', views.linkedin_callback, name='linkedin_callback'),  # Updated path
    path('profilel/', views.profile_view, name='profilel'),
    path('facebook_insights/', views.facebook_insights, name='facebook_insights'),
    path('register/', views.register, name='register'),  # URL for registration page
    path('sign_in/', views.sign_in, name='sign_in'),  # URL for sign-in page
    path('home/', views.home, name='home'),  # After successful login, redirect to the home page (or dashboard)
    path('all_reviews/', views.ar, name='all_reviews'),

    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    # path('dashboard1/', views.dashboard1, name='dashboard1'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('dashboard/', views.dashboard, name='dashboard'),  # Homepage or Dashboard
    path('account/', views.account, name='account'),  # Account page
    path('account1/', views.account1, name='account1'),
    path('report/', views.report, name='report'),  # Report page
    path('report1/', views.report1, name='report1'),  # Report page
    path('analytics/', views.analytics, name='analytics'),  # Analytics page
    path('support/', views.support, name='support'),  # Support page
    path('logout/', views.logout_view, name='logout_view'),
    path('analytics1/', views.analytics1, name='analytics1'),  # Analytics page
    path('getting_started/', views.getting_started, name='getting_started'),
    path('support1/', views.support1, name='support1'),  # Support page
    path('troubleshooting/', views.troubleshooting, name='troubleshooting'),
    path('contact_support/', views.contact_support, name='contact_support'),


    path('facebook-login1/', views.facebook_login1, name='facebook_login1'),
    path('instagram-login/', views.instagram_login, name='instagram_login'),
    path('twitter-login1/', views.twitter_login, name='twitter_login1'),
    path('reddit-login/', views.reddit_login, name='reddit_login'),

    path('facebook-tab/', views.facebook_tab, name='facebook_tab'),
    # path('youtube-login/', views.youtube_login, name='youtube_login'),

    path('linkedin-login/', views.linkedin_login, name='linkedin_login'),
    path('facebookll/', views.facebook_login_view, name='facebookll'),
    path('youtubell', views.youtube_loginn, name='youtube_login'),
    path('youtube/profile/', views.youtube_profile, name='youtube_profile'),

    # path('instagram-analytics/', views.instagram_analytics, name='instagram_analytics'),
    # path('instagram-engagement/', views.instagram_engagement, name='instagram_engagement'),
    # path('engagement-rate/', views.engagement_rate, name='engagement_rate'),
    # path('top-posts/', views.top_posts, name='top_posts'),

    path('instagram-analytics/', views.instagram_analytics, name='instagram_analytics'),
    path('instagram-insights/', views.instagram_engagement, name='instagram_insights'),
    path('instagram-engagement/', views.engagement_rate, name='instagram_engagement'),
    path('instagram-top-posts/', views.top_posts, name='instagram_top_posts'),
   
    path('youtube-login/', views.youtube_login, name='youtube_login'),
    path('callback-youtube/', views.youtube_callback, name='youtube_callback'),

    path('instagram-tab/', views.instagram_tab, name='instagram_tab'),

    path('youtube-tab/', views.youtube_tab, name='youtube_tab'),
    path('youtube-dashboard/', views.youtube_dashboard, name='youtube_dashboard'),
    path('youtube-sentiment/', views.youtube_sentiment, name='youtube_sentiment'),
    path('youtube-top-posts/', views.youtube_top_posts, name='youtube_top_posts'),
    path('youtube-prediction/', views.youtube_prediction, name='youtube_prediction'),
    path('youtube-engagement/', views.youtube_engagement, name='youtube_engagement'),

    path('twitter-tab/', views.twitter_tab, name='twitter_tab'),
    path('twitter-dashboard/', views.twitter_dashboard, name='twitter_dashboard'),  
    path('twitter-top-tweets/', views.twitter_top_tweets, name='twitter_top_tweets'),
    path('twitter-prediction/', views.twitter_prediction, name='twitter_prediction'),
    path('twitter-engagement/', views.twitter_engagement, name='twitter_engagement'),
    path('twitter-sentiment/', views.twitter_sentiment, name='twitter_sentiment'),  

    path('linkedin-tab/', views.linkedin_tab, name='linkedin_tab'),
    path('linkedin-dashboard/', views.linkedin_dashboard, name='linkedin_dashboard'),   
    path('linkedin-top-posts/', views.linkedin_top_posts, name='linkedin_top_posts'),
    path('linkedin-prediction/', views.linkedin_prediction, name='linkedin_prediction'),
    path('linkedin-engagement/', views.linkedin_engagement, name='linkedin_engagement'),

    path("login-tab/", views.login_tab, name="login_tab"),
    path('platform/<str:platform>/', views.platform_click, name='platform_click'),

    path('ga-campaign/', views.ga_campaign, name='ga_campaign'),
    path('ga-targeted-content/', views.ga_targeted_content, name='ga_targeted_content'),        
    path('ga-group-report/', views.ga_group_report, name='ga_group_report'),    
    path('ga-age-report/', views.ga_age_report, name='ga_age_report'),    
    path('ga-location-report/', views.ga_location_report, name='ga_location_report'),
    path('ga-landing-page/', views.ga_landing_page, name='ga_landing_page'),
    path('ga-household-income/', views.ga_household_income, name='ga_household_income'),
    path('ga-device-report/', views.ga_device_report, name='ga_device_report'),
    path('ga-search-keyword/', views.ga_search_keyword, name='ga_search_keyword'),
    path('ga-search-terms/', views.ga_search_terms, name='ga_search_terms'),
    path('ga-schedule/', views.ga_schedule, name='ga_schedule'),
    path('ga-schedule-day-hour/', views.ga_schedule_day_hour, name='ga_schedule_day_hour'),
    path('ga-report/', views.ga_report, name='ga_report'),
    path('ga-asset-report/', views.ga_asset_report, name='ga_asset_report'),
    path('ga-gender-report/', views.ga_gender_report, name='ga_gender_report'),
    path('google-ads-tab/', views.google_ads_tab, name='google_ads_tab'),
    path('ga-strategic-level/', views.ga_strategic_level, name='ga_strategic_level'),
    path('ga-tactical-level/', views.ga_tactical_level, name='ga_tactical_level'),
    path('ga-operational-level/', views.ga_operational_level, name='ga_operational_level'),

]