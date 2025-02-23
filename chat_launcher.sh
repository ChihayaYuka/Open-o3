#!/bin/bash

APP_NAME = "./src/o3_app.py"
VIRTUAL_ENV_NAME = "o3_env"

if [ ! -f "$APP_NAME" ]; then
  echo "Error: App file '$APP_NAME' not found."
  exit 1
fi

if [ -n "$VIRTUAL_ENV_NAME" ]; then
 if [ -d "$VIRTUAL_ENV_NAME" ]; then
   echo "Activating virtual environment: $VIRTUAL_ENV_NAME"
   source "$VIRTUAL_ENV_NAME/bin/activate"
 else
   echo "Error: Virtual environment '$VIRTUAL_ENV_NAME' not found."
   echo "Please create and activate the virtual environment manually, or remove VIRTUAL_ENV_NAME from this script."
   exit 1
 fi
fi

echo "Starting app: $APP_NAME"
streamlit run "$APP_NAME"

if [ -n "$VIRTUAL_ENV_NAME" ]; then
 deactivate
fi

echo "ChatUI stopped."

exit 0
