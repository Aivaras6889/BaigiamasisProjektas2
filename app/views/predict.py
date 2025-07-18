from datetime import datetime
import os
from flask import Blueprint, current_app, render_template, flash, redirect, session, url_for, request
from flask_login import login_required, current_user
import numpy as np
from app.extensions import db
from app.forms.predictions import PredictionForm
from app.ml.predict import get_recent_predictions, predict_single_image, predict_with_model
from app.models.dataset import Dataset
from app.models.dataset_images import DatasetImage
from app.models.results import Results
from app.models.trained_models import TrainedModel
from app.services.predictions import get_predictions_with_offset, total_predictions_count
from werkzeug.utils import secure_filename
from app.utils.classes import get_class_name
from app.utils.images import extract_features_from_image
from app.utils.models import save_directory_model, save_model_performance, save_uploaded_model
from PIL import Image
bp = Blueprint('predict', __name__)


@bp.route('/')
# @bp.route('/predict', methods=['GET', 'POST'])

# def predict():
#     form = PredictionForm()
#     prediction_result = None
    
#     # Get recent predictions for display
#     try:
#         recent_predictions = get_recent_predictions(10)
#     except:
#         recent_predictions = []
    
#     if form.validate_on_submit():
#         try:
#             result, error = predict_single_image(
#                 form.file.data,
#                 form.model_type.data,
#                 form.model_name.data
#             )
            
#             if error:
#                 flash(error, 'error')
#             else:
#                 prediction_result = result
#                 flash('Prediction completed successfully!', 'success')
#                 # Refresh recent predictions
#                 recent_predictions = get_recent_predictions(10)
                
#         except Exception as e:
#             flash(f'Prediction error: {str(e)}', 'error')
#         finally:
#             db.session.close()
    
#     db.session.close()
#     return render_template('predict.html', 
#                          form=form, 
#                          result=prediction_result,
#                          recent_predictions=recent_predictions)

# @bp.route('/predict', methods=['GET', 'POST'])
@bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictionForm()

    if form.validate_on_submit():
        try:
            model_id = form.model_id.data
            prediction_type = form.prediction_type.data

            model = TrainedModel.query.get(model_id)
            if not model:
                flash('Selected model not found', 'danger')
                return render_template('predict.html', form=form)

            actual_class = None
            actual_class_name = None
            filepath = ''
            image_path = ''

            if prediction_type == 'upload':
                file = form.image.data
                if not file or not file.filename:
                    flash('Please select a file to upload', 'danger')
                    return render_template('predict.html', form=form)

                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                unique_filename = timestamp + filename

                upload_dir = os.path.join(current_app.static_folder, 'uploads')
                filepath = os.path.join(upload_dir, unique_filename)
                file.save(filepath)

                dataset = Dataset.query.first()
                if not dataset:
                    dataset = Dataset(name="Default Dataset", status='ready', total_images=0, num_classes=0)
                    db.session.add(dataset)
                    db.session.flush()

                uploaded_image = DatasetImage(
                    dataset_id=dataset.id,
                    image_path=f'uploads/{unique_filename}',
                    class_id=-1,
                    dataset_type='uploaded'
                )
                db.session.add(uploaded_image)
                db.session.commit()

                image_path = f'uploads/{unique_filename}'

            elif prediction_type == 'uploaded':
                uploaded_image = DatasetImage.query.get_or_404(form.uploaded_image_id.data)
                image_path = uploaded_image.image_path
                filename = os.path.basename(image_path)
                filepath = os.path.join(current_app.static_folder, 'uploads', filename)

            elif prediction_type == 'database':
                db_image = DatasetImage.query.get_or_404(form.database_image_id.data)
                actual_class = db_image.class_id
                actual_class_name = get_class_name(actual_class)
                image_path = db_image.image_path
                
                # Fix: Use the full stored path instead of assuming uploads folder
                filepath = os.path.join(current_app.static_folder, image_path)
            else:
                flash('Invalid prediction type.', 'danger')
                return render_template('predict.html', form=form)

            # Run prediction
            if model.framework in ['tensorflow', 'pytorch']:
                result = predict_with_model(model_id, filepath)
            else:
                # For RandomForest, use full flattened image
                from PIL import Image
                img = Image.open(filepath).convert('L')
                img = img.resize((96, 96))
                img_array = np.array(img) / 255.0
                features = img_array.flatten()
                result = predict_with_model(model_id, features)

            # Convert predicted class number to name
            predicted_class = result['prediction']
            predicted_class_name = get_class_name(predicted_class)
            
            # Update result with readable names
            result['actual_class'] = actual_class
            result['actual_class_name'] = actual_class_name
            result['predicted_class_name'] = predicted_class_name
            result['image_path'] = image_path

            new_result = Results(
                profile_id=current_user.id,
                prediction=str(result['prediction']),
                confidence=result.get('confidence'),
                actual_class=actual_class,
                model_name=model.name,
                image_path=image_path,
                prediction_time=result.get('prediction_time', 0)
            )
            db.session.add(new_result)
            db.session.commit()

            if actual_class is not None:
                save_model_performance(model_id)

            flash('Prediction completed successfully!', 'success')
            return render_template('predict.html', form=form, result=result)

        except Exception as e:
            db.session.rollback()
            flash(f'Prediction error: {str(e)}', 'danger')

    return render_template('predict.html', form=form)
# def predict():
#     form = PredictionForm()

#     if form.validate_on_submit():
#         try:
#             model_id = form.model_id.data
#             prediction_type = form.prediction_type.data

#             model = TrainedModel.query.get(model_id)
#             if not model:
#                 flash('Selected model not found', 'danger')
#                 return render_template('predict.html', form=form)

#             actual_class = None
#             filepath = ''
#             image_path = ''

#             if prediction_type == 'upload':
#                 file = form.image.data
#                 if not file or not file.filename:
#                     flash('Please select a file to upload', 'danger')
#                     return render_template('predict.html', form=form)

#                 filename = secure_filename(file.filename)
#                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
#                 unique_filename = timestamp + filename

#                 upload_dir = os.path.join(current_app.static_folder, 'uploads')
#                 filepath = os.path.join(upload_dir, unique_filename)
#                 file.save(filepath)

#                 dataset = Dataset.query.first()
#                 if not dataset:
#                     dataset = Dataset(name="Default Dataset", status='ready', total_images=0, num_classes=0)
#                     db.session.add(dataset)
#                     db.session.flush()

#                 uploaded_image = DatasetImage(
#                     dataset_id=dataset.id,
#                     image_path=f'uploads/{unique_filename}',
#                     class_id=-1,
#                     dataset_type='uploaded'
#                 )
#                 db.session.add(uploaded_image)
#                 db.session.commit()

#                 image_path = f'static/uploads/{unique_filename}'

#             elif prediction_type == 'uploaded':
#                 uploaded_image = DatasetImage.query.get_or_404(form.uploaded_image_id.data)
#                 image_path = uploaded_image.image_path
#                 filepath = os.path.join(current_app.static_folder, image_path.replace('static/', ''))

#             elif prediction_type == 'database':
#                 db_image = DatasetImage.query.get_or_404(form.database_image_id.data)
#                 actual_class = db_image.class_id
#                 image_path = db_image.image_path
#                 filepath = os.path.join(current_app.static_folder, image_path.replace('static/', ''))

#             else:
#                 flash('Invalid prediction type.', 'danger')
#                 return render_template('predict.html', form=form)

#             # Run prediction
#             if model.framework in ['tensorflow', 'pytorch']:
#                 result = predict_with_model(model_id, filepath)
#             else:
#                 features = extract_features_from_image(filepath)
#                 result = predict_with_model(model_id, features)

#             result['actual_class'] = actual_class
#             result['image_path'] = image_path

#             new_result = Results(
#                 profile_id=current_user.id,
#                 prediction=str(result['prediction']),
#                 confidence=result.get('confidence'),
#                 actual_class=actual_class,
#                 model_name=model.name,
#                 image_path=image_path,
#                 prediction_time=result.get('prediction_time', 0)
#             )
#             db.session.add(new_result)
#             db.session.commit()

#             if actual_class is not None:
#                 save_model_performance(model_id)

#             flash('Prediction completed successfully!', 'success')
#             return render_template('predict.html', form=form, result=result)

#         except Exception as e:
#             db.session.rollback()
#             flash(f'Prediction error: {str(e)}', 'danger')

#     return render_template('predict.html', form=form)

@bp.route('/predict-image', methods=['POST'])
def predict_image():
    """Make prediction on uploaded image"""
    form = ""
    try:
        # Get model from session
        model_id = session.get('selected_model_id')
        if not model_id:
            flash('Please load a model first', 'error')
            return redirect(url_for('predict.predict'))
        
        # Handle file upload
        if 'image_file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('predict.predict'))
        
        file = request.files['image_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('predict.predict'))
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = timestamp + filename
        
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, unique_filename)
        file.save(filepath)
        
        # Extract features
        features = extract_features_from_image(filepath)
        
        # Make prediction
        result = predict_with_model(model_id, features)
        
        flash('Prediction completed successfully!', 'success')
        return render_template('predict.html', 
                             form=form, 
                             result=result,
                             image_path=f'uploads/{unique_filename}')
        
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('predict.predict'))


# @bp.route('/history')
# def prediction_history():
#     """Simple prediction history"""
#     page = request.args.get('page', 1, type=int)
    
#     # Simple filters
#     query = PredictionHistory.query
    
#     model_id = request.args.get('model_id')
#     if model_id:
#         query = query.filter(PredictionHistory.model_id == model_id)
    
#     result_filter = request.args.get('result')
#     if result_filter == 'correct':
#         query = query.filter(PredictionHistory.is_correct == True)
#     elif result_filter == 'incorrect':
#         query = query.filter(PredictionHistory.is_correct == False)
    
#     predictions = query.order_by(PredictionHistory.created_at.desc()).paginate(
#         page=page, per_page=10, error_out=False
#     )
    
#     # Simple stats
#     total = PredictionHistory.query.count()
#     correct = PredictionHistory.query.filter_by(is_correct=True).count()
#     accuracy = (correct / total * 100) if total > 0 else 0
#     avg_time = db.session.query(db.func.avg(PredictionHistory.prediction_time)).scalar() or 0
    
#     most_used = db.session.query(TrainedModel.name).join(PredictionHistory).group_by(TrainedModel.id).order_by(db.func.count(PredictionHistory.id).desc()).first()
    
#     stats = {
#         'total_predictions': total,
#         'accuracy': accuracy,
#         'avg_time': avg_time,
#         'most_used_model': most_used.name if most_used else 'None'
#     }
    
#     available_models = TrainedModel.query.filter_by(is_active=True).all()
    
#     return render_template('prediction_history.html', 
#                          predictions=predictions, 
#                          stats=stats,
#                          available_models=available_models)



@bp.route('/debug_images')
def debug_images():
    """Temporary route to check image paths"""
    import os
    from flask import current_app
    
    output = []
    
    # Check what's in database
    results = Results.query.all()
    output.append("=== DATABASE PATHS ===")
    for r in results:
        output.append(f"ID: {r.id}, Path: '{r.image_path}'")
    
    output.append("\n=== FILE EXISTENCE CHECK ===")
    # Check if files exist
    for r in results:
        if r.image_path:
            full_path = os.path.join(current_app.static_folder, r.image_path)
            exists = os.path.exists(full_path)
            output.append(f"ID: {r.id}, Path: '{r.image_path}', Exists: {exists}, Full: '{full_path}'")
        else:
            output.append(f"ID: {r.id}, Path: NULL/EMPTY")
    
    return '<br>'.join(output)


@bp.route('/fix_paths')
def fix_paths():
    """Fix inconsistent image paths in database"""
    fixed_count = 0
    deleted_count = 0
    
    results = Results.query.all()
    for r in results:
        if not r.image_path:
            continue
            
        # Check if file exists first
        current_full_path = os.path.join(current_app.static_folder, r.image_path)
        
        if os.path.exists(current_full_path):
            # File exists - fix the path format if needed
            if r.image_path.startswith('app/static/uploads/'):
                # Fix: app/static/uploads/file.png -> uploads/file.png
                r.image_path = r.image_path.replace('app/static/uploads/', 'uploads/')
                fixed_count += 1
            elif r.image_path.startswith('static/uploads/'):
                # Fix: static/uploads/file.png -> uploads/file.png  
                r.image_path = r.image_path.replace('static/uploads/', 'uploads/')
                fixed_count += 1
        else:
            # File doesn't exist - delete the record
            db.session.delete(r)
            deleted_count += 1
    
    db.session.commit()
    return f"Fixed {fixed_count} paths, deleted {deleted_count} missing records"