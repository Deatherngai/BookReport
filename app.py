from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from DB_Mag.Book import QueryBook
from main import Recognition
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads/"

@app.route('/')
def upload_file():
    value = QueryBook("leading with")
    return render_template('index.html')

@app.route('/display', methods = ['GET', 'POST'])
def display_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(str(f.filename))
        f.save(app.config['UPLOAD_FOLDER'] + filename)
        file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        image = 'static/uploads/'+filename
        result = Recognition(image)
        #query = QueryBook("Leading With Principle")
        query = QueryBook(result)
    return render_template('content.html', len = len(query),query=query)

if __name__ == '__main__':
    app.run(debug = True)
