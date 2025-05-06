
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename
from model_utils import process_and_cluster

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            clustered_df, silhouette, plot_path = process_and_cluster(filepath)
            table = clustered_df.to_html(classes='table table-striped', index=False)

            return render_template('result.html', table=table, silhouette=silhouette, plot_url=plot_path)
        else:
            return "Only CSV files are supported.", 400
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
