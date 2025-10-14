This document provides a robust, step-by-step guide to deploy a Python Flask application with local AI models on a Windows Server using IIS. This guide is tailored for an offline server environment, where packages cannot be installed from the internet on the target machine.

Core Strategy
The strategy is to create a self-contained deployment package on a development machine (with internet) that includes the application code, AI models, and all required Python libraries as local files. This package is then transferred to the server, where a script creates a clean virtual environment and installs the libraries from the local files, ensuring all paths are correct for the server environment.

Part 1: Server Preparation (One-Time Setup)
This only needs to be done once on the Windows Server.

Step 1.1: A Proper Python Installation (Crucial)
You must have a full, standard installation of Python on the server.

Download: On a machine with internet access, download the "Windows installer (64-bit)" for your desired Python version (e.g., Python 3.9+) from the official Python website.

Transfer & Install: Copy the installer to the server and run it as an administrator.

Customize Installation:

On the first screen, check "Add python.exe to PATH".

Choose "Customize installation".

On the "Optional Features" screen, ensure all boxes are checked, especially pip and tcl/tk.

On the "Advanced Options" screen, check "Install for all users" and set the installation path to a simple, permission-friendly location like C:\Python39.

Step 1.2: Install IIS with HttpPlatformHandler
Install IIS Role: Use Server Manager to add the "Web Server (IIS)" role with default settings.

Download HttpPlatformHandler: Download the x64 installer (HttpPlatformHandler_amd64_en-US.msi) from the official Microsoft IIS page.

Install HttpPlatformHandler: Run the installer on the server.

Restart IIS: Open an Administrator Command Prompt and run iisreset.

Part 2: Preparing the Application Package (Local Machine)
Follow these steps on your development machine where you have internet access and have already run the initial setup.bat.

Step 2.1: Download Libraries for Offline Use
This is the key step for an offline deployment. We will download all the necessary Python packages as files into a local folder.

Open a Command Prompt in your project directory (face-recognition-py).

Run the following command:

pip download -r requirements.txt -d packages

This will create a new folder named packages in your project directory.

It will fill this folder with all the .whl (wheel) files for every library listed in requirements.txt, including all their dependencies.

Step 2.2: Create the Final Deployment Package (.zip)
Your project is now ready to be packaged. Create a .zip file that contains the following:

app.py

face_processor.py

requirements.txt

setup.bat

web.config

The models folder (with the downloaded AI models inside).

The new packages folder (with all the downloaded .whl library files).

DO NOT include the venv folder in your .zip file. A fresh one will be created on the server.

Part 3: Server Deployment
Step 3.1: Copy and Extract
Copy the final .zip package to the server.

Extract it to a permanent location, for example: C:\edms\face_recognition.

Step 3.2: Run the Offline Setup Script
Open a Command Prompt as an Administrator.

Navigate to your project folder: cd C:\edms\face_recognition

Run the setup script: setup.bat

This updated script will now:

Create a new, clean venv folder using the server's Python installation.

Install all the required libraries using the files from your ./packages folder, with no internet connection required.

Step 3.3: Configure the IIS Application
Open IIS Manager.

Navigate to your existing site (e.g., "Smart EDMS API").

Right-click the site and select "Add Application".

Alias: FaceRecognition (This is the name used in the URL, e.g., .../FaceRecognition).

Physical path: C:\edms\face_recognition (Point this to where you extracted the project).

Set Permissions:

In Windows File Explorer, right-click the C:\edms\face_recognition folder and go to Properties > Security > Edit.

Add the IIS_IUSRS group and grant it "Read & execute", "List folder contents", and "Read" permissions.

Restart IIS: Run iisreset from an Administrator Command Prompt.

Your Face Recognition API should now be running and accessible.

Part 4: Troubleshooting
Error: "The 'packages' folder with library files was not found"
Cause: This error occurs during the server setup (setup.bat) if the packages folder is missing from your deployment .zip file. This means Step 2.1 was likely skipped on the development machine before the project was zipped. * Solution:

Go back to your development machine.

Navigate to the project folder (face-recognition-py) in a command prompt.

Run the command: pip download -r requirements.txt -d packages

This will create the packages folder.

Create a new .zip file, making sure to include this new packages folder.

Transfer the new .zip file to the server, extract it (overwriting the old files), and run setup.bat again.