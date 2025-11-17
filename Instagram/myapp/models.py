from django.db import models

# Create your models here.
from django.db import models

class CustomUser(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True, primary_key=True)  # Set email as primary key
    password = models.CharField(max_length=255)
    last_login = models.DateTimeField(null=True, blank=True)  # Add this line
    def __str__(self):
        return self.name
    def get_email_field_name(self):
        return 'email'
    def set_password(self, raw_password):
        """ Custom method to set password without hashing (plain text) """
        self.password = raw_password

class Feedback(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='feedbacks')  # Reference to CustomUser
    experience = models.CharField(max_length=50, choices=[  # Dropdown options for feedback
        ('Excellent', 'Excellent'),
        ('Good', 'Good'),
        ('Average', 'Average'),
        ('Poor', 'Poor')
    ])
    review = models.TextField()  # Field to store the review text
    submitted_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the feedback was submitted

    def __str__(self):
        return f"Feedback from {self.user.name} - {self.experience}"