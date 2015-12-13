
# Standard imports
import os
import sqlite3

# Flask imports
from flask import Flask, request, g, redirect, url_for, render_template, flash

# Import our classifier class
from yelp_classifier import YelpClf

# create the app as a global variable
app = Flask(__name__)

# Specify a basic configuration
# (Remember that this step happens in the application context)
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'reviews.db'),
    DEBUG=True,
    SECRET_KEY='my super secret string',
    CLF_PICKLE='classifier.p'
))

# Load our pickled classifier before servicing requests
clf = YelpClf(app.config['CLF_PICKLE'])

def connect_db():
    """Connects to the database defined in the application configuration.
    Returns a sqlite3 object that allows rows to be manipulated as python
    dictionaries.
    """
    rv = sqlite3.connect(app.config['DATABASE'])
    # allows a row to be accessed as a dictionary rather than a tuple.
    rv.row_factory = sqlite3.Row
    return rv
# end of connect_db()

def init_db():
    """Initialize the database from schema.sql.
    This needs to be run once manually, or whenever the database table 
    should be dropped. This is not executed during online operation.
    """
    # establish the application context
    with app.app_context():
        # Open a connection to the database
        db = get_db()
        # Open schema.sql
        with app.open_resource('schema.sql','r') as f:
            # Execute the commands in the schema file
            db.cursor().executescript(f.read())
        # Write changes to the database
        db.commit()
# end of init_db()

def get_db():
    """Opens a new db connection if none exists for the current app context."""
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db
# end of get_db()

@app.teardown_appcontext
def close_db(error):
    """Closes the db at the end of the request.
    The flask app.teardown_appcontext decorator releases and destroys
    the back-end server resources. We just need to close the database.
    """
    # Note that we were using the global application object, g, to store
    # our open sqlite handle.
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()
# end of close_db()

# Views - Note how simple these endpoints are!
# We hardly need to consider the fact that we are operating online!

@app.route('/')
def show_predictions():
    """Main view to display all rating predictions.
    """
    db = get_db()
    cur = db.execute('SELECT lyrics, artist FROM predictions ORDER BY id DESC')
    predictions = cur.fetchall()
    return render_template('show_predictions.html', predictions=predictions)
# end of show_predictions()

@app.route('/add', methods=['POST'])
def add_prediction():
    reviews = request.form['reviews']
    rating = clf.predictRating(reviews)
    db = get_db()
    # remember how we set our SQL driver to treat rows as dictionaries? win!
    # note: question mark notation safer than string replacement
    # helps to prevent SQLi attacks
    db.execute('INSERT INTO predictions (lyrics, artist) values (?, ?)',
        [reviews, str(rating)])
    db.commit()
    flash('Prediction was successfully posted')
    # another example where the framework is omnipotent: url_for()
    return redirect(url_for('show_predictions'))
# end of add_prediction()

if __name__=="__main__":
    # Tell Flask to run!
    app.run()
