from .models import Task
from django.forms import ModelForm,TextInput

class TaskForm(ModelForm):
    class Meta:
        model = Task
        fields = ["fighter1","fighter2"]
        widgets = {"fighter1": TextInput(attrs={
            'class': 'form-control',
        }), "fighter2": TextInput(attrs={
            'class': 'form-control',})

        }

