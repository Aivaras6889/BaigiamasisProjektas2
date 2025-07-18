from flask_wtf import FlaskForm
from wtforms import SubmitField

class ClearModelsForm(FlaskForm):
    """Form for clearing all models"""
    submit = SubmitField('Clear Models')

class ClearDatasetForm(FlaskForm):
    """Form for clearing dataset"""
    submit = SubmitField('Clear Dataset')

class ClearAllForm(FlaskForm):
    """Form for clearing everything"""
    submit = SubmitField('Clear All')