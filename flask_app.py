from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template
import os
from ai import get_yolo_net, yolo_forward, yolo_save_img
import cv2
import numpy as np

# where we will store images
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# load the NN to memory
here = os.getcwd()
print(here)
names_path = os.path.join(here, 'yolo', 'rockfish_obj.names')
print(names_path)
LABELS = open(names_path).read().strip().split("\n")
print()
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

weights_path = os.path.join(here, 'yolo', 'rockfish_yolov3_final.weights')
cfg_path = os.path.join(here, 'yolo', 'rockfish_yolov3.cfg')
net = get_yolo_net(cfg_path, weights_path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes definitions
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))


    return render_template('index.html') 


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # read image file and make prediction
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)
    (class_ids, labels, boxes, confidences) = yolo_forward(net, LABELS, image, confidence_level=0.3)
    
    # format data for template rendering
    # found emotions, save images with bounding boxes.
    if len(class_ids) > 0:
        found = True
        new_filename = 'yolo_' + filename
        file_path = os.path.join(here, app.config['UPLOAD_FOLDER'], new_filename)
        yolo_save_img(image, class_ids, boxes, labels, confidences, COLORS, file_path=file_path)
        
        # help function to format result sentences.
        def and_syntax(alist):
            if len(alist) == 1:
                alist = "".join(alist)
                return alist
            elif len(alist) == 2:
                alist = " and ".join(alist)
                return alist
            elif len(alist) > 2:
                alist[-1] = "and " + alist[-1]
                alist = ", ".join(alist)
                return alist
            else:
                return

        # confidences: rounding and changing to percent, putting in function
        format_confidences = []
        for percent in confidences:
            format_confidences.append(str(round(percent*100)) + '%')
        format_confidences = and_syntax(format_confidences)
        # labels: sorting and capitalizing, putting into function
        labels = set(labels)
        labels = [emotion.capitalize() for emotion in labels]
        labels = and_syntax(labels)
        
        # return template with data
        return render_template('results.html', confidences = format_confidences, labels=labels, 
            old_filename = filename, 
            filename=new_filename) 
    else:
        found = False
        return render_template('results.html', labels='Wrong Format', old_filename = filename, filename=filename)

@app.route('/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
@app.route('/page3.html')
def gotoPage3():
    return render_template('page3.html') 



if __name__ == "__main__":
    app.run(debug=True)