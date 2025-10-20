Deployment Guide: Face Recognition Python App on macOS with Supervisor
This guide provides step-by-step instructions for deploying the Face Recognition Python web application on a macOS machine, specifically a Mac Studio. We will use supervisor to manage the application process, ensuring it runs continuously in the background.

This guide assumes you have Homebrew installed on your macOS system.

Step 1: Install Prerequisites
First, we need to install Python 3, pip, and Supervisor using Homebrew.

Install Python 3:

brew install python

Install Supervisor:

brew install supervisor

Start Supervisor Service:
Have supervisor start automatically at login:

brew services start supervisor

Step 2: Clone and Set Up the Project
Next, clone your project from its repository and set up the environment.

Clone the Repository:
Open Terminal and navigate to the directory where you want to store your project, then clone it.

# Replace with your actual repository URL
git clone [https://github.com/karimk94/face-recognition-py.git](https://github.com/karimk94/face-recognition-py.git)
cd face-recognition-py

Create a Virtual Environment:
It's best practice to create a virtual environment for your project to manage its dependencies separately.

python3 -m venv venv

Activate the Virtual Environment:
You must activate the environment before installing dependencies.

source venv/bin/activate

Your terminal prompt should now be prefixed with (venv).

Step 3: Install Dependencies and Download Models
Now, install the required Python packages and download the necessary machine learning models.

Install Python Dependencies:
Install all the packages listed in requirements.txt.

pip install --upgrade pip
pip install -r requirements.txt

Download the ML Models:
The project includes a script to download the required models. Run it using the Python interpreter from your virtual environment.

python download_models.py

This script will download the models and place them in the correct directory within your project. This step is crucial and must be completed before starting the application.

Step 4: Configure Supervisor
Supervisor will manage our Flask application, automatically starting it and restarting it if it crashes.

Create a Supervisor Configuration File:
Supervisor configuration files are typically located in /usr/local/etc/supervisor.d/. Create a new configuration file for your face recognition app.

nano /usr/local/etc/supervisor.d/face-recognition-py.conf

Add the Configuration:
Copy and paste the following configuration into the file. Make sure to replace /path/to/your/face-recognition-py with the actual absolute path to your project directory.

[program:face-recognition-py]
command=/Users/okool_arfelous/Developer/Face-Recognition-py/venv/bin/gunicorn -w 4 -b 127.0.0.1:8001 app:app
directory=/Users/okool_arfelous/Developer/Face-Recognition-py
autostart=true
autorestart=true
stderr_logfile=/var/log/face-recognition-py.err.log
stdout_logfile=/var/log/face-recognition-py.out.log

command: This is the command to start your app. We use Gunicorn, a robust Python WSGI server. It will execute it from within your virtual environment. We're binding it to port 8001.

directory: The absolute path to your project.

user: Your macOS username.

Log files: The paths where stdout and stderr logs will be stored. You may need to create the /var/log directory or grant permissions if it doesn't exist.

Save and Exit:
Press Ctrl + X, then Y, then Enter to save the file.

Step 5: Start and Manage the Application
With the configuration in place, you can now let Supervisor manage your application.

Update Supervisor:
Tell Supervisor to read the new configuration file.

supervisorctl reread
supervisorctl update

Start the Application:
Start your face recognition application using supervisorctl.

supervisorctl start face-recognition-py

Check the Status:
You can check the status of your application at any time.

supervisorctl status face-recognition-py

You should see that the process is RUNNING. If it's not, check the log files for errors:

tail -f /var/log/face-recognition-py.out.log
tail -f /var/log/face-recognition-py.err.log

Your application is now deployed and will run in the background, automatically restarting on boot or if it crashes. You can access it at http://127.0.0.1:8001.