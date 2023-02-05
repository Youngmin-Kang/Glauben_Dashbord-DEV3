from django import forms
from .models import User

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'email','rol']
        labels = {
            'first_name':'Nombre', 
            'email':'Email',
            'rol':'Rol'
        }
        widgets = {
            'first_name':forms.TextInput(attrs={'class': 'form-control'}), 
            'email':forms.EmailInput(attrs={'class': 'form-control'}),
            'rol':forms.TextInput(attrs={'class': 'form-control'}),
        }